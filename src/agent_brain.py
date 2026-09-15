
"""
Agent-Based Model Core: ABMConvictAgent

Implements the micro-level behavioural dynamics of the ABM,
including:

- Non-linear logistic rebellion probability
- Metabolic energy depletion
- Policy-state transitions
- Social contagion
- Structural despair accumulation
- Colony-specific environmental stress
- Natural mortality
- Rebellion survival dynamics
"""

import numpy as np


class ABMConvictAgent:
    """
    Individual convict agent operating on a quarterly time step.

    One simulation step represents one quarter of a year.
    Therefore, age increases by 0.25 years per step.
    """

    def __init__(
        self,
        dp_id,
        age,
        skill_tier,
        inherent_rebellion,
        voyage_id,
        colony_stress,
    ):
        self.dp_id = dp_id
        self.age = age
        self.skill_tier = skill_tier
        self.inherent_rebellion = inherent_rebellion
        self.voyage_id = voyage_id
        self.colony_stress = colony_stress

        # Dynamic internal state
        self.energy = 100.0
        self.compliance_score = 100.0
        self.personal_wealth = 0.0

        # Historical governance state
        self.policy_state = "Assigned"

        # Structural promotion delay
        self.quarters_without_promotion = 0

        # Mortality state
        self.is_dead = False

    def step(
        self,
        global_drought_index,
        current_ration_level,
        peer_rebellion_rate,
        w1,
        w2,
        w3,
        w4,
        w5_social,
        despair_rate,
        natural_death_rate,
        rebel_threshold,
        rng,
    ):
        """
        Advance the agent by one quarter.

        Parameters
        ----------
        global_drought_index : float
            Historical drought intensity for the current year.

        current_ration_level : float
            Simulated ration availability for the current quarter.

        peer_rebellion_rate : float
            Rebellion rate among eligible peers in the agent's voyage cohort.

        w1-w5_social : float
            Model parameters sampled at the universe level.

        despair_rate : float
            Structural despair accumulation parameter.

        natural_death_rate : float
            Quarterly natural mortality probability.

        rebel_threshold : float
            Probability threshold for transition to ActiveRebellion.

        rng : numpy.random.Generator
            Universe-specific random number generator.

        Returns
        -------
        dict
            Agent-level outcome for the current quarter.
        """

        # ================================================================
        # 0. NATURAL MORTALITY
        # ================================================================
        if rng.random() < natural_death_rate:
            self.is_dead = True

            return {
                "output": 0.0,
                "rebelled": False,
                "status": "Deceased",
            }

        # ================================================================
        # 1. ACTIVE REBELLION SURVIVAL TRACK
        # ================================================================
        if self.policy_state == "ActiveRebellion":

            bush_foraging_yield = 3.5
            exposure_damage = 12.0 * global_drought_index

            self.energy = (
                self.energy
                - exposure_damage
                + bush_foraging_yield
            )

            self.energy = max(0.0, min(100.0, self.energy))

            # One quarter = 0.25 years
            self.age += 0.25

            if self.energy <= 0.0:
                self.is_dead = True

                return {
                    "output": 0.0,
                    "rebelled": True,
                    "status": "Deceased",
                }

            # Conditional return from rebellion under relatively favourable
            # environmental conditions.
            if (
                global_drought_index < 0.4
                and rng.random() < 0.20
            ):
                self.policy_state = "Assigned"
                self.compliance_score = 40.0

                return {
                    "output": 12.0,
                    "rebelled": False,
                    "status": "Amnesty_Return",
                }

            return {
                "output": 0.0,
                "rebelled": True,
                "status": "ActiveRebellion",
            }

        # ================================================================
        # 2. METABOLIC CONSUMPTION
        # ================================================================
        # Colony stress metabolic conversion rate.
        # Calibrated to 0.3 based on historical mortality differentials
        # between Norfolk Island (stress≈1.9, mortality 2-3× NSW) and NSW (stress≈1.0).
        # Sensitivity analysis (0.2-0.5) confirms Q13 collapse robustness.
        COLONY_STRESS_METABOLIC_WEIGHT = 0.3

        stress_penalty = (
            1.0
            + (self.colony_stress - 1.0) * COLONY_STRESS_METABOLIC_WEIGHT
        )

        self.energy = (
            self.energy
            - 4.0 * stress_penalty
            - 8.0 * global_drought_index
            + current_ration_level
        )

        self.energy = max(0.0, min(100.0, self.energy))

        # Quarterly time step
        self.age += 0.25

        if self.energy <= 0.0:
            self.is_dead = True

            return {
                "output": 0.0,
                "rebelled": False,
                "status": "Deceased",
            }

        # ================================================================
        # 3. STRUCTURAL DESPAIR
        # ================================================================
        if self.policy_state == "Assigned":
            self.quarters_without_promotion += 1

        # Colony stress acts as a catalyst, not the primary driver.
        # Only 30% of additional stress (relative to NSW baseline)
        # translates into accelerated despair accumulation.
        # Calibrated to Q13 historical collapse anchor.
        COLONY_STRESS_DESPAIR_WEIGHT = 0.3

        structural_despair = (
            despair_rate
            * self.quarters_without_promotion
            * (1.0 + (self.colony_stress - 1.0) * COLONY_STRESS_DESPAIR_WEIGHT)
        )

        # ================================================================
        # 4. COGNITIVE LOGISTIC REBELLION MODEL
        # ================================================================
        policy_multiplier = (
            2.0
            if self.policy_state == "TicketOfLeave"
            else 0.1
        )

        if self.policy_state == "Emancipist":
            policy_multiplier = 4.0

        net_distress = (
            (w1 * (100.0 - self.energy))
            + (w2 * global_drought_index)
            + (w3 * self.inherent_rebellion)
            + (w5_social * peer_rebellion_rate)
            + structural_despair
        ) - (
            w4 * policy_multiplier
            + 0.005 * self.personal_wealth
        )

        # Numerically stable logistic transformation
        p_rebel = 1.0 / (1.0 + np.exp(-np.clip(
            net_distress,
            -700.0,
            700.0,
        )))

        # ================================================================
        # 5. PHASE TRANSITION
        # ================================================================
        if p_rebel > rebel_threshold:
            self.policy_state = "ActiveRebellion"
            self.compliance_score = 0.0

            return {
                "output": 0.0,
                "rebelled": True,
                "status": "Rebelled",
            }

        # ================================================================
        # 6. MACRO PRODUCTION FUNCTION
        # ================================================================
        base_output = 55.0

        current_output = (
            base_output
            * self.skill_tier
            * (self.energy / 100.0)
            * (1.0 - 0.40 * global_drought_index)
        )

        if self.policy_state == "TicketOfLeave":
            self.personal_wealth += current_output * 0.40
            self.compliance_score += 1.5

        elif self.policy_state == "Emancipist":
            self.personal_wealth += current_output
            self.compliance_score += 2.0

        else:
            self.compliance_score += 0.8

        return {
            "output": current_output,
            "rebelled": False,
            "status": self.policy_state,
        }
