
"""
Worker Functions for the ABM Engine.

This module:
- loads and groups convict records by voyage,
- synthesizes historically constrained agent attributes,
- samples universe-level uncertainty,
- applies historical environmental stress,
- executes one 40-quarter simulation,
- returns one auditable universe-level result.
"""

import time

import numpy as np
import pandas as pd

from src.agent_brain import ABMConvictAgent

from src.config import (
    K_VOYAGES,
    MIN_COHORT_SIZE,
    AGE_MEAN,
    AGE_SD,
    AGE_CLIP_MIN,
    AGE_CLIP_MAX,
    SKILL_PREMIUM_CUTOFF_YEAR,
    SKILL_PREMIUM_MAX,
    REBELLION_BASE,
    REBELLION_SENTENCE_WEIGHT_RANGE,
    LIFE_SENTENCE_N,
    DESPAIR_RATE_RANGE,
    PROMOTION_DESPAIR_RETENTION,
    NATURAL_DEATH_RATE_QUARTERLY_RANGE,
    COLLAPSE_MUTINY_THRESHOLD,
    CARROT_THRESHOLD_MODIFIER_RANGE,
    STICK_THRESHOLD_MODIFIER_RANGE,
    CARROT_COVERAGE_MODIFIER_RANGE,
    STICK_COVERAGE_MODIFIER_RANGE,
    get_colony_stress_with_observed_fallback,
    sample_colony_stress,
    build_year_stress_map,
)


# =====================================================================
# DATA LOADING
# =====================================================================

def _sample_param(param_overrides, name, rng_sampler):
    """
    Return a fixed override value for `name` if one was supplied,
    otherwise draw a fresh sample via rng_sampler().

    This allows execute_universe_worker to support single-variable
    ablation studies (a named parameter held constant across every
    universe) without any change to its default random-sampling
    behaviour when param_overrides is None or does not contain `name`.

    Parameters
    ----------
    param_overrides : dict or None
        Mapping of parameter name -> fixed value for this run.
        None (the default) disables all overrides.

    name : str
        The parameter name to check for an override.

    rng_sampler : callable
        A zero-argument callable that draws one random sample when
        no override is present, e.g. `lambda: rng.uniform(0.01, 0.08)`.

    Returns
    -------
    float
        Either the override value or a freshly sampled value.
    """
    if param_overrides and name in param_overrides:
        return param_overrides[name]
    return rng_sampler()

def load_and_group_by_voyage(convicts_path):
    """
    Load convict records and dynamically construct voyage identifiers
    from ship and transportation year.
    """
    df = pd.read_csv(
        convicts_path,
        low_memory=False,
    )

    required_columns = [
        "ship",
        "transported_year",
        "sentence_n",
        "dp_id",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    df["transported_year"] = pd.to_numeric(
        df["transported_year"],
        errors="coerce",
    )

    df["sentence_n_clean"] = pd.to_numeric(
        df["sentence_n"],
        errors="coerce",
    ).fillna(LIFE_SENTENCE_N)

    df["voyage_id_dynamic"] = (
        df["ship"].astype(str)
        + " | "
        + df["transported_year"].astype(str)
    )

    voyage_col = "voyage_id_dynamic"

    grouped = {
        voyage_id: sub_df.reset_index(drop=True)
        for voyage_id, sub_df
        in df.groupby(voyage_col)
    }

    voyage_ids = list(grouped.keys())

    print(
        f"INFO: Loaded {len(voyage_ids):,} voyages."
    )

    return grouped, voyage_ids


# =====================================================================
# SYNTHETIC AGENT ATTRIBUTES
# =====================================================================

def synthesize_age(rng, n):
    """
    Synthesize age from a truncated normal distribution.
    """
    ages = rng.normal(
        AGE_MEAN,
        AGE_SD,
        size=n,
    )

    return np.clip(
        ages,
        AGE_CLIP_MIN,
        AGE_CLIP_MAX,
    )


def synthesize_skill(rng, transported_years):
    """
    Synthesize skill tiers according to the historically defined
    transportation-period cutoff.

    Convicts transported from the cutoff year onward receive a
    higher skill distribution.
    """
    transported_years = np.asarray(
        transported_years,
        dtype=float,
    )

    skills = np.ones(
        len(transported_years),
        dtype=float,
    )

    valid_years = ~np.isnan(transported_years)

    post_cutoff = (
        valid_years
        & (
            transported_years
            >= SKILL_PREMIUM_CUTOFF_YEAR
        )
    )

    n_post = int(post_cutoff.sum())

    if n_post > 0:
        skills[post_cutoff] = rng.uniform(
            1.0,
            SKILL_PREMIUM_MAX,
            size=n_post,
        )

    return skills


def compute_inherent_rebellion(
    sentence_n_clean,
    weight,
):
    """
    Compute inherent rebellion propensity from sentence length
    using the calibrated sentence-weight parameter.
    """
    raw = (
        REBELLION_BASE
        + weight
        * (
            sentence_n_clean
            / LIFE_SENTENCE_N
        )
    )

    return np.clip(
        raw,
        0.10,
        1.00,
    )


# =====================================================================
# DROUGHT MAPPING
# =====================================================================

def build_quarterly_drought_series(
    annual_drought_history,
    n_quarters=40,
):
    """
    Expand annual historical drought observations into quarterly
    simulation values.

    The historical drought data are annual.

    Therefore:
        Year Y Q1 -> annual drought value for Y
        Year Y Q2 -> annual drought value for Y
        Year Y Q3 -> annual drought value for Y
        Year Y Q4 -> annual drought value for Y

    The ABM nevertheless advances one quarter at a time.

    Returns
    -------
    list
        Quarterly drought values for the simulation horizon.
    """
    if annual_drought_history is None:
        raise ValueError(
            "Annual drought history cannot be None."
        )

    annual_values = list(
        annual_drought_history
    )

    if len(annual_values) == 0:
        raise ValueError(
            "Annual drought history is empty."
        )

    quarterly_values = []

    for annual_value in annual_values:
        quarterly_values.extend(
            [float(annual_value)] * 4
        )

        if len(quarterly_values) >= n_quarters:
            break

    if len(quarterly_values) < n_quarters:
        raise ValueError(
            "Historical drought series does not cover "
            f"the required {n_quarters} quarters."
        )

    return quarterly_values[:n_quarters]


# =====================================================================
# MAIN UNIVERSE WORKER
# =====================================================================

def execute_universe_worker(args):
    """
    Execute one complete 40-quarter simulation universe.

    Expected args (9-tuple):
        (
            universe_id,
            grouped_voyages,
            voyage_ids,
            annual_drought_history,
            year_stress_map,
            k_voyages,
            child_seed,
            model_version,
            param_overrides,
        )

    The universe_id identifies the statistical simulation replicate.
    It is intentionally independent of multiprocessing worker identity.

    param_overrides : dict or None
        Optional mapping of parameter name -> fixed value. When a
        parameter name is present in this dict, that parameter is
        held constant for the entire universe instead of being
        randomly sampled. Used for single-variable ablation studies;
        pass None (or omit entries) to preserve standard robustness-run
        behaviour.
    """
    (
        universe_id,
        grouped_voyages,
        voyage_ids,
        annual_drought_history,
        year_stress_map,
        k_voyages,
        child_seed,
        model_version,
        param_overrides,
    ) = args

    try:
        rng = np.random.default_rng(child_seed)

        # ================================================================
        # 1. SAMPLE VOYAGES
        # ================================================================
        n_available_voyages = len(voyage_ids)

        n_sampled_voyages = min(
            k_voyages,
            n_available_voyages,
        )

        sampled_voyage_ids = rng.choice(
            voyage_ids,
            size=n_sampled_voyages,
            replace=False,
        )

        sub_dfs = [
            grouped_voyages[voyage_id]
            for voyage_id in sampled_voyage_ids
        ]

        df_sample = pd.concat(
            sub_dfs,
            ignore_index=True,
        )

        n_agents = len(df_sample)

        if n_agents == 0:
            raise ValueError(
                "Sampled universe contains zero agents."
            )

        # ================================================================
        # 2. SAMPLE UNIVERSE-LEVEL PARAMETERS
        # ================================================================
        w1 = _sample_param(param_overrides, "w1", lambda: rng.uniform(0.01, 0.08))
        w2 = _sample_param(param_overrides, "w2", lambda: rng.uniform(0.2, 1.0))
        w3 = _sample_param(param_overrides, "w3", lambda: rng.uniform(0.1, 0.5))
        w4 = _sample_param(param_overrides, "w4", lambda: rng.uniform(0.8, 2.5))
        w5_social = _sample_param(param_overrides, "w5_social", lambda: rng.uniform(0.1, 1.0))

        despair_rate = _sample_param(
            param_overrides, "despair_rate", lambda: rng.uniform(*DESPAIR_RATE_RANGE)
        )

        natural_death_rate = _sample_param(
            param_overrides,
            "natural_death_rate",
            lambda: rng.uniform(*NATURAL_DEATH_RATE_QUARTERLY_RANGE),
        )

        rebel_threshold = _sample_param(
            param_overrides, "rebel_threshold", lambda: rng.uniform(0.58, 0.72)
        )

        rebellion_sentence_weight = _sample_param(
            param_overrides,
            "rebellion_sentence_weight",
            lambda: rng.uniform(*REBELLION_SENTENCE_WEIGHT_RANGE),
        )
        # ================================================================
        # 3. POLICY DESIGN
        #
        # Every universe receives exactly one intervention type.
        # There is intentionally no "no-policy" category.
        # ================================================================
        intervention_type = _sample_param(param_overrides, "intervention_type", lambda: rng.choice(["carrot", "stick"]))
              
        #  (supports Sobol param_overrides):
        intervention_quarter = int(_sample_param(param_overrides, "intervention_quarter", lambda: int(rng.integers(4, 37))))
        intervention_threshold = rng.uniform(20.0, 100.0)
        intervention_coverage = _sample_param(param_overrides, "intervention_coverage", lambda: rng.uniform(0.3, 1.0))

        # Initialize intervention state variables
        intervention_applied = False
        n_promoted = 0
        mean_despair_pre = 0.0
        mean_despair_post = 0.0

        # ================================================================
        # 4. DROUGHT ROBUSTNESS CONDITION
        #
        # Drought is independently switched on/off at the universe level.
        # This is intentional and is used to test whether drought is
        # sufficient to determine collapse.
        # ================================================================
        drought_enabled = bool(_sample_param(param_overrides, "drought_enabled", lambda: rng.choice([True, False])))
        
        # ================================================================
        # 5. SYNTHESIZE AGENT ATTRIBUTES
        # ================================================================
        transported_years = pd.to_numeric(
            df_sample[
                "transported_year"
            ],
            errors="coerce",
        ).to_numpy()

        ages = synthesize_age(
            rng,
            n_agents,
        )

        skills = synthesize_skill(
            rng,
            transported_years,
        )

        rebellion_baseline = compute_inherent_rebellion(
            df_sample[
                "sentence_n_clean"
            ].to_numpy(),
            rebellion_sentence_weight,
        )

        if "arrival_place_clean" in df_sample.columns:
            destinations = (
                df_sample[
                    "arrival_place_clean"
                ].to_numpy()
            )
        else:
            destinations = np.array(
                [None] * n_agents,
                dtype=object,
            )

        voyage_labels = (
            df_sample[
                "voyage_id_dynamic"
            ].to_numpy()
        )

        # ================================================================
        # 6. HISTORICAL COLONY STRESS
        #
        # Layer 1:
        # Destination-level historical uncertainty is sampled once
        # per destination per universe.
        #
        # Layer 2:
        # Individual-level heterogeneity is applied as an independent
        # ±20% multiplier.
        #
        # Missing destination records use the observed year-specific
        # historical fallback.
        # ================================================================
        unique_destinations = set(
            destination
            for destination in destinations
            if pd.notna(destination)
        )

        destination_stress_samples = {}

        for destination in unique_destinations:
            destination_stress_samples[
                destination
            ] = sample_colony_stress(
                destination,
                rng,
            )

        individual_stress_noise = rng.uniform(
            0.8,
            1.2,
            size=n_agents,
        )

        agents = []

        for i in range(n_agents):

            destination = destinations[i]
            transported_year = transported_years[i]

            if (
                pd.notna(destination)
                and destination in destination_stress_samples
            ):
                base_colony_stress = (
                    destination_stress_samples[
                        destination
                    ]
                )

            else:
                base_colony_stress = (
                    get_colony_stress_with_observed_fallback(
                        destination,
                        transported_year,
                        year_stress_map,
                    )
                )

            colony_stress = (
                base_colony_stress
                * individual_stress_noise[i]
            )

            colony_stress = max(
                0.5,
                min(
                    2.5,
                    colony_stress,
                ),
            )

            agents.append(
                ABMConvictAgent(
                    dp_id=df_sample[
                        "dp_id"
                    ].iloc[i],
                    age=ages[i],
                    skill_tier=skills[i],
                    inherent_rebellion=rebellion_baseline[i],
                    voyage_id=voyage_labels[i],
                    colony_stress=colony_stress,
                )
            )

        # ================================================================
        # 7. PREPARE HISTORICAL DROUGHT SERIES
        # ================================================================
        quarterly_drought = (
            build_quarterly_drought_series(
                annual_drought_history,
                n_quarters=40,
            )
        )

        # ================================================================
        # 8. SIMULATION STATE
        # ================================================================
        total_gdp_accumulated = 0.0

        peak_mutiny_rate = 0.0
        final_mortality_rate = 0.0

        collapse_quarter = None

        n_agents_initial = len(
            agents
        )

        cumulative_dead = 0

        # ================================================================
        # 9. 40-QUARTER SIMULATION
        # ================================================================
        for quarter in range(
            1,
            41,
        ):

            # ------------------------------------------------------------
            # 9.1 DROUGHT
            #
            # Annual historical drought values are held constant across
            # the four quarters belonging to that historical year.
            # ------------------------------------------------------------
            if drought_enabled:

                raw_drought_value = (
                    quarterly_drought[
                        quarter - 1
                    ]
                )

                current_drought = min(
                    1.0,
                    raw_drought_value / 4.5,
                )

            else:
                current_drought = 0.0

            # ------------------------------------------------------------
            # 9.2 GRANARY / RATION CALCULATION
            # ------------------------------------------------------------
            if quarter > 3:

                accumulated_shock = (
                    quarterly_drought[
                        quarter - 2
                    ]
                    + quarterly_drought[
                        quarter - 3
                    ]
                    + quarterly_drought[
                        quarter - 4
                    ]
                )

            else:

                accumulated_shock = (
                    quarterly_drought[0]
                    * 3
                    if drought_enabled
                    else 0.0
                )

            current_rations = max(
                1.2,
                min(
                    6.5,
                    6.5
                    * (
                        1.0
                        - 0.14
                        * accumulated_shock
                    ),
                ),
            )

            # ------------------------------------------------------------
            # 9.3 ALIVE AGENTS
            # ------------------------------------------------------------
            alive_agents = [
                agent
                for agent in agents
                if not agent.is_dead
            ]

            # ------------------------------------------------------------
            # 9.4 POLICY INTERVENTION
            # ------------------------------------------------------------
            if (
                quarter
                == intervention_quarter
            ):

                despair_values_pre = [
                    agent.quarters_without_promotion
                    for agent in alive_agents
                ]

                mean_despair_pre = (
                    float(
                        np.mean(
                            despair_values_pre
                        )
                    )
                    if despair_values_pre
                    else 0.0
                )

                if intervention_type == "carrot":

                    threshold_modifier = rng.uniform(
                        *CARROT_THRESHOLD_MODIFIER_RANGE
                    )

                    coverage_modifier = rng.uniform(
                        *CARROT_COVERAGE_MODIFIER_RANGE
                    )

                    effective_threshold = (
                        intervention_threshold
                        * threshold_modifier
                    )

                    effective_coverage = min(
                        1.0,
                        intervention_coverage
                        * coverage_modifier,
                    )

                else:

                    threshold_modifier = rng.uniform(
                        *STICK_THRESHOLD_MODIFIER_RANGE
                    )

                    coverage_modifier = rng.uniform(
                        *STICK_COVERAGE_MODIFIER_RANGE
                    )

                    effective_threshold = (
                        intervention_threshold
                        * threshold_modifier
                    )

                    effective_coverage = max(
                        0.0,
                        intervention_coverage
                        * coverage_modifier,
                    )

                eligible_agents = [
                    agent
                    for agent in alive_agents
                    if (
                        agent.policy_state
                        == "Assigned"
                        and agent.compliance_score
                        > effective_threshold
                    )
                ]

                n_to_promote = int(
                    len(eligible_agents)
                    * effective_coverage
                )

                if n_to_promote > 0:

                    selected_agents = rng.choice(
                        eligible_agents,
                        size=n_to_promote,
                        replace=False,
                    )

                    for agent in selected_agents:

                        agent.policy_state = (
                            "TicketOfLeave"
                        )

                        agent.quarters_without_promotion *= (
                            PROMOTION_DESPAIR_RETENTION
                        )

                        if intervention_type == "carrot":

                            agent.energy = min(
                                100.0,
                                agent.energy + 5.0,
                            )

                        else:

                            agent.quarters_without_promotion += 2

                    n_promoted = n_to_promote

                intervention_applied = True

                despair_values_post = [
                    agent.quarters_without_promotion
                    for agent in alive_agents
                ]

                mean_despair_post = (
                    float(
                        np.mean(
                            despair_values_post
                        )
                    )
                    if despair_values_post
                    else 0.0
                )

            # ------------------------------------------------------------
            # 9.5 VOYAGE-LEVEL SOCIAL CONTAGION
            # ------------------------------------------------------------
            voyage_rebel_counts = {}
            voyage_total_counts = {}

            for agent in alive_agents:

                voyage_total_counts[
                    agent.voyage_id
                ] = (
                    voyage_total_counts.get(
                        agent.voyage_id,
                        0,
                    )
                    + 1
                )

                if (
                    agent.policy_state
                    == "ActiveRebellion"
                ):

                    voyage_rebel_counts[
                        agent.voyage_id
                    ] = (
                        voyage_rebel_counts.get(
                            agent.voyage_id,
                            0,
                        )
                        + 1
                    )

            # ------------------------------------------------------------
            # 9.6 STEP EACH AGENT
            # ------------------------------------------------------------
            current_rebels = 0
            newly_dead_this_quarter = 0

            for agent in alive_agents:

                # Historical promotion pathway based on compliance.
                if (
                    agent.policy_state
                    == "Assigned"
                    and agent.compliance_score
                    > 115
                ):

                    agent.policy_state = (
                        "TicketOfLeave"
                    )

                    agent.quarters_without_promotion *= (
                        PROMOTION_DESPAIR_RETENTION
                    )

                cohort_size = voyage_total_counts.get(
                    agent.voyage_id,
                    0,
                )

                if (
                    cohort_size
                    >= MIN_COHORT_SIZE
                ):

                    peer_rate = (
                        voyage_rebel_counts.get(
                            agent.voyage_id,
                            0,
                        )
                        / cohort_size
                    )

                else:

                    peer_rate = 0.0

                result = agent.step(
                    current_drought,
                    current_rations,
                    peer_rate,
                    w1,
                    w2,
                    w3,
                    w4,
                    w5_social,
                    despair_rate,
                    natural_death_rate,
                    rebel_threshold,
                    rng,
                )

                total_gdp_accumulated += (
                    result["output"]
                )

                if agent.is_dead:

                    newly_dead_this_quarter += 1

                elif (
                    agent.policy_state
                    == "ActiveRebellion"
                ):

                    current_rebels += 1

            # ------------------------------------------------------------
            # 9.7 MORTALITY AND COLLAPSE METRICS
            # ------------------------------------------------------------
            cumulative_dead += (
                newly_dead_this_quarter
            )

            n_alive = (
                n_agents_initial
                - cumulative_dead
            )

            current_mutiny_rate = (
                (
                    current_rebels
                    / n_alive
                )
                * 100.0
                if n_alive > 0
                else 0.0
            )

            peak_mutiny_rate = max(
                peak_mutiny_rate,
                current_mutiny_rate,
            )

            if (
                collapse_quarter is None
                and current_mutiny_rate
                >= COLLAPSE_MUTINY_THRESHOLD
            ):

                collapse_quarter = quarter

            final_mortality_rate = (
                cumulative_dead
                / n_agents_initial
            ) * 100.0

        # ================================================================
        # 10. FINAL UNIVERSE RESULT
        # ================================================================
        return {
            "timestamp": int(
                time.time() * 1000
            ),

            "universe_id": universe_id,

            "model_version": model_version,

            "w1_energy": round(
                w1,
                4,
            ),

            "w2_drought": round(
                w2,
                4,
            ),

            "w3_rebellion": round(
                w3,
                4,
            ),

            "w4_policy": round(
                w4,
                4,
            ),

            "w5_social": round(
                w5_social,
                4,
            ),

            "despair_rate": round(
                despair_rate,
                5,
            ),

            "rebellion_sentence_weight": round(
                rebellion_sentence_weight,
                4,
            ),

            "natural_death_rate_quarterly": round(
                natural_death_rate,
                6,
            ),

            "threshold": round(
                rebel_threshold,
                3,
            ),

            "intervention_type": (
                1
                if intervention_type == "carrot"
                else 0
            ),

            "intervention_quarter": (
                intervention_quarter
            ),

            "intervention_threshold": round(
                intervention_threshold,
                2,
            ),

            "intervention_coverage": round(
                intervention_coverage,
                3,
            ),

            "intervention_applied": int(
                intervention_applied
            ),

            "n_promoted": n_promoted,

            "mean_despair_pre": round(
                mean_despair_pre,
                3,
            ),

            "mean_despair_post": round(
                mean_despair_post,
                3,
            ),

            "drought_enabled": int(
                drought_enabled
            ),

            "n_voyages_sampled": len(
                sampled_voyage_ids
            ),

            "n_agents": n_agents,

            "total_gdp": round(
                total_gdp_accumulated / 40.0,
                1,
            ),

            "peak_mutiny": round(
                peak_mutiny_rate,
                2,
            ),

            "collapse_quarter": (
                collapse_quarter
                if collapse_quarter is not None
                else -1
            ),

            "final_mortality": round(
                final_mortality_rate,
                2,
            ),
        }

    except Exception as exc:

        print(
            f"ERROR: Universe {universe_id} failed: {exc}"
        )

        return None