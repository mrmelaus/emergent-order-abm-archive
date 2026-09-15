
"""
Configuration Parameters for the ABM Engine.

All model parameters are centralized here to support:
- historical calibration,
- reproducibility,
- sensitivity analysis,
- robustness testing.
"""

import pandas as pd


# =====================================================================
# SAMPLING AND AGENT PARAMETERS
# =====================================================================

K_VOYAGES = 30
MIN_COHORT_SIZE = 5


# =====================================================================
# AGE SYNTHESIS PARAMETERS
# Historical reference: Nicholas (1988)
# =====================================================================

AGE_MEAN = 26.1
AGE_SD = 6.5

AGE_CLIP_MIN = 14
AGE_CLIP_MAX = 65


# =====================================================================
# SKILL SYNTHESIS PARAMETERS
# =====================================================================

SKILL_PREMIUM_CUTOFF_YEAR = 1820
SKILL_PREMIUM_MAX = 2.0


# =====================================================================
# REBELLION BASELINE PARAMETERS
# =====================================================================

REBELLION_BASE = 0.30

REBELLION_SENTENCE_WEIGHT_RANGE = (0.05,0.25,)

LIFE_SENTENCE_N = 99


# =====================================================================
# STRUCTURAL DESPAIR PARAMETERS
# =====================================================================

DESPAIR_RATE_RANGE = (0.003, 0.032)

PROMOTION_DESPAIR_RETENTION = 0.0


# =====================================================================
# NATURAL MORTALITY
#
# Historical mortality estimates are annual.
# The ABM operates quarterly.
#
# We therefore convert annual probability p_a into an equivalent
# quarterly probability p_q such that:
#
#     (1 - p_q)^4 = 1 - p_a
#
# Therefore:
#
#     p_q = 1 - (1 - p_a)^(1/4)
# =====================================================================

NATURAL_DEATH_RATE_ANNUAL_RANGE = (
    0.0048,
    0.015,
)


def annual_to_quarterly_probability(annual_probability):
    """
    Convert an annual event probability into the equivalent
    quarterly probability.

    This conversion preserves the annual probability over
    four independent quarterly periods.
    """
    if not 0.0 <= annual_probability <= 1.0:
        raise ValueError(
            "Annual probability must be between 0 and 1."
        )

    return 1.0 - (1.0 - annual_probability) ** 0.25


NATURAL_DEATH_RATE_QUARTERLY_RANGE = tuple(
    annual_to_quarterly_probability(rate)
    for rate in NATURAL_DEATH_RATE_ANNUAL_RANGE
)


# =====================================================================
# COLLAPSE THRESHOLD
# =====================================================================

COLLAPSE_MUTINY_THRESHOLD = 20.0


# =====================================================================
# HISTORICAL COLONY STRESS INTERVALS
#
# Each interval represents historical uncertainty around the
# destination-specific stress level.
# =====================================================================

COLONY_STRESS_INTERVALS = {
    "Norfolk Island": (1.5, 2.3),
    "Van Diemen's Land": (1.1, 1.7),
    "Moreton Bay": (1.0, 1.6),
    "Western Australia": (0.9, 1.3),
    "New South Wales": (0.8, 1.2),
    "Port Phillip": (0.8, 1.2),
    "Gibraltar": (1.0, 1.4),
}


# Midpoints are used for deterministic historical fallback
# calculations.
COLONY_STRESS_MIDPOINTS = {
    destination: (low + high) / 2.0
    for destination, (low, high)
    in COLONY_STRESS_INTERVALS.items()
}


def get_colony_stress_interval(destination):
    """
    Return the historical stress interval for a destination.

    Unknown or missing destinations receive a neutral fallback
    interval of (1.0, 1.0).
    """
    if pd.isna(destination):
        return (1.0, 1.0)

    destination_string = str(destination)

    for key, interval in COLONY_STRESS_INTERVALS.items():
        if key in destination_string:
            return interval

    return (1.0, 1.0)


def get_colony_stress_midpoint(destination):
    """
    Return the deterministic midpoint of the historical stress
    interval for a destination.
    """
    if pd.isna(destination):
        return 1.0

    destination_string = str(destination)

    for key, midpoint in COLONY_STRESS_MIDPOINTS.items():
        if key in destination_string:
            return midpoint

    return 1.0


def sample_colony_stress(destination, rng):
    """
    Sample a destination-specific stress value from its historical
    uncertainty interval.

    This is sampled once per destination within each universe.
    """
    low, high = get_colony_stress_interval(destination)

    return rng.uniform(low, high)


def get_colony_stress_with_observed_fallback(
    destination,
    transported_year,
    year_stress_map,
):
    """
    Determine deterministic baseline colony stress.

    Priority:
        1. Destination-specific historical stress
        2. Year-specific observed stress
        3. Global historical mean
        4. Neutral value of 1.0
    """
    if pd.notna(destination):
        return get_colony_stress_midpoint(destination)

    if transported_year in year_stress_map:
        return year_stress_map[transported_year]

    if year_stress_map:
        return sum(year_stress_map.values()) / len(year_stress_map)

    return 1.0


def build_year_stress_map(convicts_path):
    """
    Build a deterministic year-specific historical stress map.

    Only records with valid destination information are used.

    Returns
    -------
    dict
        Mapping:
            transported_year -> mean historical colony stress
    """
    df = pd.read_csv(
        convicts_path,
        low_memory=False,
    )

    matched = df[
        df["arrival_place_clean"].notna()
    ].copy()

    matched["stress"] = matched[
        "arrival_place_clean"
    ].apply(get_colony_stress_midpoint)

    year_stress_map = (
        matched
        .groupby("transported_year")["stress"]
        .mean()
        .to_dict()
    )

    return year_stress_map



# =====================================================================
# CARROT / STICK INTERVENTION MODIFIER RANGES
# =====================================================================

CARROT_THRESHOLD_MODIFIER_RANGE = (0.5, 0.8)
STICK_THRESHOLD_MODIFIER_RANGE = (1.2, 1.6)
CARROT_COVERAGE_MODIFIER_RANGE = (1.1, 1.5)
STICK_COVERAGE_MODIFIER_RANGE = (0.5, 0.8)


# ============================================================
# Parameter Bounds for Sobol Sensitivity Analysis
# Centralized here so worker.py and sobol_runner.py stay in sync.
# ============================================================
# =====================================================================
# PARAMETER SEMANTIC GLOSSARY
#
# w1 (Energy deprivation weight)      — how strongly physiological
#   energy shortfall drives rebellion probability
# w2 (Drought pressure weight)        — how strongly the current
#   drought index drives rebellion probability
# w3 (Baseline rebellion weight)      — how strongly an agent's
#   inherent, sentence-derived rebellion propensity drives the
#   decision function
# w4 (Institutional suppression weight) — how strongly policy state
#   (Assigned / TicketOfLeave / Emancipist) suppresses rebellion
# w5_social (Social contagion weight) — how strongly peer rebellion
#   rate within an agent's voyage cohort drives rebellion probability
# despair_rate                        — rate of structural despair
#   accumulation per quarter spent unpromoted in Assigned status
# rebel_threshold                     — probability threshold above
#   which an agent transitions to ActiveRebellion
#
# See MODEL_DOCUMENTATION.md (Submodels: Rebellion decision function)
# for the full logistic specification combining these weights.
# =====================================================================
PARAM_BOUNDS = {
    "w1": (0.01, 0.08),
    "w2": (0.2, 1.0),
    "w3": (0.1, 0.5),
    "w4": (0.8, 2.5),
    "w5_social": (0.1, 1.0),
    "rebel_threshold": (0.58, 0.72),
    "despair_rate": DESPAIR_RATE_RANGE,
    "natural_death_rate": NATURAL_DEATH_RATE_QUARTERLY_RANGE,
    "rebellion_sentence_weight": REBELLION_SENTENCE_WEIGHT_RANGE,

    "intervention_quarter": (4.0, 36.0),
    "intervention_coverage": (0.3, 1.0),
}