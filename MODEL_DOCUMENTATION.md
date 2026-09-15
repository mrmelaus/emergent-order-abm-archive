# Model Documentation: Emergent Order and Institutional Collapse

**An Agent-Based Model of Colonial Convict Society**

This document is the standalone narrative documentation for the agent-based
model in this repository, following the ODD (Overview, Design concepts,
Details) protocol (Grimm et al. 2006, 2010, 2020). It accompanies the source
code in `src/` and is intended to be sufficient, on its own, for another
computational modeler to understand and replicate the model without needing
to read the source code first.

For installation, usage, and command-line examples, see `README.md`. For the
full scientific motivation, results, and discussion, see the associated
manuscript (citation to be added on publication).

---

## Purpose and Patterns

### Purpose

The model's purpose is to determine whether system-level institutional
collapse in a historically calibrated agent population can be understood as
an emergent outcome of four interacting mechanisms — individual behaviour,
accumulated structural pressure, an exogenous environmental shock, and
institutional intervention — rather than as the consequence of any single
dominant driver. The model addresses four research questions: whether
structural pressure alone can drive collapse in the absence of drought;
whether drought is a necessary condition or an accelerant; whether
intervention timing and coverage, not only intervention type, shape
institutional trajectories; and to what extent collapse timing is
attributable to individual parameters versus their interaction.

### Patterns

The model is required to reproduce one primary structural pattern used for
calibration: the historically documented 1791 grain and resource crisis,
corresponding approximately to simulated quarter 13. This is used as a
**calibration anchor** for the study's headline outcomes rather than a
fitted target for them: the upper bound of the structural despair parameter
was itself set via a pilot grid search so that one representative parameter
configuration reproduces median collapse timing at quarter 13, but this
calibration fixes only one boundary of one sampling interval. It is not used
to tune the model to reproduce a specific aggregate collapse rate, and
`despair_rate` continues to be drawn freely across its full interval,
jointly with six other freely varying parameters, throughout the main
200,000-universe experiment.

A second pattern is the concentration of simulated collapses at quarter 13
across the full joint parameter space of the main 200,000-universe run
(34.14% of all collapse cases). Because the despair-rate upper bound was
itself calibrated so that one representative configuration reproduces this
timing, this concentration could plausibly result from the calibration
anchor itself rather than from the interacting mechanisms under study across
the full parameter space. The Q13 exclusion diagnostic is therefore not a
precautionary check but a necessary one: the model's principal qualitative
conclusion (that intervention type affects collapse rate) is required to
survive the removal of this cluster, which it does.

The model does not attempt to reproduce a specific historical rebellion
event as a validation target; its qualitative behaviour, not its numerical
fit to any single event, is the object of validation throughout.

---

## Entities, State Variables, and Scales

### Entities

The model contains three kinds of entities:

1. **Agents** — individual convicts, the model's only active
   decision-making entity;
2. **Voyages** — a passive collective entity grouping agents who arrived
   together, used to compute peer effects and to structure the sampling
   procedure;
3. **The environment** — a single global exogenous drought signal shared
   by all agents in a universe, with no individual agent-level
   environmental heterogeneity beyond the destination-specific colony
   stress coefficient described below.

### Agent state variables

| State variable | Type | Set at | Description |
|---|---|---:|---|
| `dp_id` | identifier | initialisation | Links the agent to its source historical record |
| `age` | float, years | initialisation; updated | Increments by 0.25 per quarter |
| `skill_tier` | float | initialisation, fixed | 1.0, or sampled from [1.0, `SKILL_PREMIUM_MAX`] for agents transported from 1820 onward |
| `inherent_rebellion` (R_i) | float, fixed | initialisation | Derived from sentence length |
| `voyage_id` | identifier, fixed | initialisation | Determines voyage-cohort membership |
| `colony_stress` (sigma_colony) | float, fixed | initialisation | Destination-specific pressure coefficient, with individual-level ±20% noise |
| `energy` (E_i,t) | float, [0, 100] | init = 100.0; updated each quarter | Physiological state; death occurs at 0 |
| `compliance_score` | float, [0, inf) | init = 100.0; updated each quarter | Institutional standing; governs both the automatic and intervention-triggered promotion pathways |
| `personal_wealth` (W_i,t) | float, [0, inf) | init = 0.0; updated | Accumulates only in `TicketOfLeave` and `Emancipist` states |
| `policy_state` | categorical | init = `Assigned`; updated | One of `Assigned`, `TicketOfLeave`, `ActiveRebellion`, `Emancipist` (see note below) |
| `quarters_without_promotion` (tau_i,t) | integer counter | init = 0; updated | Drives structural despair accumulation; resets on promotion |
| `is_dead` | boolean | init = False | Removes the agent from the active population once True |

**Note on `Emancipist`.** This state exists in the implementation, with a
distinct compliance-accrual rate and policy multiplier, because it
represents a genuine and historically distinct institutional category: an
emancipist held permanently restored civil rights (property ownership,
standing to sue, unrestricted movement) rather than the conditional,
revocable status of a ticket-of-leave holder. It is included for
representational fidelity to the historical institutional taxonomy.

No transition rule in the current model assigns an agent to `Emancipist`,
however. This is not a gap in the transition logic but a deliberate scope
boundary: full emancipation removes an individual from the institutional
system this model is designed to study. The model's object of analysis is
behaviour, structural pressure, and intervention *within* the convict
management system; an emancipist is, by historical and legal definition, no
longer a convict subject to that system. One could construct a scenario in
which an emancipated individual re-offends and re-enters convict status, but
this trajectory falls outside the scope of the underlying British
Transportation Registers data, which record transportation and colonial
convict administration rather than the subsequent life course of free
citizens. The `Emancipist` state is therefore retained in the state machine
as a documented boundary of the model's object of study, not silently
omitted.

### Collective- and environment-level variables

| Variable | Scope | Description |
|---|---|---|
| `PeerRate` | per voyage, per quarter | Proportion of the agent's alive voyage cohort currently in `ActiveRebellion`; computed only for cohorts of at least 5 alive agents, else fixed at 0 |
| D_t (drought index) | universe-wide, per quarter | Derived from the historical annual drought reconstruction, held constant across the four quarters of a historical year, and set to 0 in every quarter when drought is disabled |
| current ration level | universe-wide, per quarter | Derived from a lagged rolling accumulation of drought exposure (see Ration determination below) |

### Scales

- **Temporal extent**: 40 quarters (10 simulated years) per universe.
- **Temporal resolution**: one quarter per simulation step.
- **Spatial/population extent**: one universe = the pooled population of
  K = 30 sampled historical voyages; population size therefore varies by
  universe according to which voyages are drawn, rather than being fixed.
- **Replication structure**: the principal robustness experiment simulates
  200,000 independent universes; each universe is one independent draw of
  both the universe-level parameter vector and the sampled voyage set.

---

## Process Overview and Scheduling

### Universe-level setup (executed once per universe, before the quarterly loop)

1. Sample K = 30 historical voyages without replacement (or fewer, if
   unavailable); pool all agents from the sampled voyages.
2. Sample the universe-level continuous parameters (w1–w5, `despair_rate`,
   `natural_death_rate`, `rebel_threshold`, `rebellion_sentence_weight`),
   unless a parameter is held fixed by an external override (used for
   ablation and Sobol runs).
3. Sample the intervention design: `intervention_type` (Carrot/Stick,
   uniform), `intervention_quarter`, `intervention_threshold`,
   `intervention_coverage`.
4. Sample `drought_enabled` (uniform Boolean).
5. Synthesise fixed agent attributes: age, skill tier, baseline rebellion
   propensity, and colony stress (two-layer: a destination-level draw
   shared by all agents assigned to that destination within the universe,
   multiplied by an independent ±20% individual-level noise term).
6. Construct the 40-quarter drought series from the annual historical
   reconstruction; pre-fill the quarter-indexed drought history array with
   a uniform placeholder value (0.20) at every position, to be overwritten
   during the quarterly loop.
7. Initialise universe-level trackers: cumulative GDP, peak mutiny rate,
   collapse quarter (unset), cumulative deaths.

### Quarterly loop (repeated for quarters 1–40, in this order)

1. **Drought update** — compute the current quarter's drought index (0 if
   disabled); write it into the drought history array at the current
   quarter's position.
2. **Ration update** — compute current rations from the accumulated
   drought exposure (see Ration determination below).
3. **Filter to alive agents.**
4. **Policy intervention** (executed only in the single quarter equal to
   `intervention_quarter`) — apply the Carrot or Stick threshold/coverage
   modifiers, identify eligible `Assigned` agents above the effective
   compliance threshold, and promote a randomly selected subset to
   `TicketOfLeave` according to the effective coverage.
5. **Voyage-level aggregation** — compute each voyage's current alive
   count and `ActiveRebellion` count, used to derive `PeerRate` in the
   next step.
6. **Per-agent update**, for every alive agent, in this sub-order:
   - a. Check the independent automatic-promotion pathway (`Assigned`
     with `compliance_score` > 115 → `TicketOfLeave`);
   - b. Compute this agent's `PeerRate`;
   - c. Execute the agent's own quarterly update (natural mortality
     check; if in `ActiveRebellion`, apply the separate
     survival/foraging/amnesty-return dynamic; otherwise apply metabolic
     depletion, structural despair accumulation, the rebellion decision
     function, and — if not rebelling — the production function and
     compliance accrual).
7. **Aggregate quarter-level outcomes** — update cumulative deaths,
   compute the current mutiny rate (share of the currently alive
   population in `ActiveRebellion`), update peak mutiny rate, and — on
   the first quarter in which the mutiny rate reaches or exceeds the
   collapse threshold (20%) — record `collapse_quarter`.

### End-of-universe output

After 40 quarters, the universe returns one result record containing the
sampled parameter vector, the intervention configuration, and the principal
outcome measures — `collapse_binary`, `collapse_quarter`, and
`peak_mutiny` — together with the secondary humanitarian-cost indicator
`final_mortality` and several intervention-diagnostic fields (`n_promoted`,
`mean_despair_pre`/`post`).

---

## Design Concepts

### Basic principles

The model follows the generative, bottom-up tradition in agent-based social
science (Epstein & Axtell 1996): institutional collapse is not imposed as a
rule but is read off the aggregate state of a population of individually
simple agents interacting under four concurrently operating mechanisms —
individual behaviour, accumulated structural pressure, an exogenous
environmental shock, and institutional intervention. No single mechanism is
assumed a priori to dominate; the model's central methodological
contribution is to let the relative importance of each mechanism, and their
interactions, be an empirical output of the simulation rather than a
modelling assumption.

### Emergence

`collapse_quarter` and `peak_mutiny` are emergent, not imposed: no rule
directly sets either variable. Both arise from the accumulated
quarter-by-quarter outcome of individual rebellion decisions, which are
themselves driven by state variables (`energy`, `quarters_without_promotion`,
`PeerRate`) that are only updated through the interaction of the model's
four mechanisms. The large gap between first-order and total-order Sobol
indices for `collapse_quarter`, together with identified second-order
interactions and unresolved rank instability between `rebel_threshold` and
`intervention_quarter`, indicate that `collapse_quarter` is a substantially
more interaction-driven emergent property than `peak_mutiny`, which is
comparatively well explained by w1 alone.

### Adaptation, Objectives, Learning, Prediction

**Not applicable.** Agents do not adapt their decision rule, pursue an
explicit objective function, learn from experience, or form expectations
about future states. The rebellion decision function and the
production/compliance-accrual functions are fixed functional forms applied
identically to every agent at every quarter of its lifetime; only the
agent's *state* (not its decision rule) changes over time.

### Sensing

Agents have perfect, cost-free knowledge of their own private state
(`energy`, `compliance_score`, `personal_wealth`, `age`, `colony_stress`,
`policy_state`). Universe-level environmental variables — the current
drought index and current ration level — are sensed without error or delay
and are identical for every agent in the universe. The only socially
mediated signal an agent senses is `PeerRate`, an aggregate statistic
rather than information about any specific other agent; agents do not
observe individual peers' private state.

### Interaction

Agents interact only indirectly, through two channels. First, `PeerRate`
couples an agent's rebellion probability to the aggregate rebellion state
of its voyage cohort, implementing social contagion without direct
agent-to-agent messaging. Second, during a policy intervention quarter,
eligible agents implicitly compete for a capacity-limited promotion
opportunity: the number of agents promoted is capped by
`effective_coverage`, so an agent's probability of promotion depends on how
many other eligible agents exist in the same quarter, even though agents
themselves have no awareness of this competition.

### Stochasticity

Stochastic elements fall into two categories. **Universe-level draws**
(each fixed once per universe, at setup): the continuous parameter vector
(w1–w5, `despair_rate`, `natural_death_rate`, `rebel_threshold`,
`rebellion_sentence_weight`), the intervention design (`intervention_type`,
`intervention_quarter`, `intervention_threshold`, `intervention_coverage`),
`drought_enabled`, the destination-level colony stress draw, and each
agent's individual ±20% colony-stress noise term. **Per-quarter, per-agent
draws**: the natural mortality check, the conditional amnesty-return check
for agents in `ActiveRebellion`, and (in the single intervention quarter)
the random selection of which eligible agents are promoted. All stochastic
draws are made through a single universe-specific `numpy.random.Generator`
instance seeded deterministically, so every reported result is exactly
reproducible given its reported seed.

### Collectives

Voyages are an *imposed*, not emergent, collective: cohort membership is
fixed at initialisation from the historical record and does not change
during a run. Their behavioural consequence, however — the feedback loop
between cohort-level `ActiveRebellion` share and individual rebellion
probability via `PeerRate` — is an emergent dynamic property of the model,
not a designed outcome.

### Observation

The model outputs one result record per universe, comprising the sampled
parameter vector, the intervention configuration, and the outcome measures
(`collapse_binary`, `collapse_quarter`, `peak_mutiny`, the secondary
indicator `final_mortality`, cumulative GDP, and intervention-diagnostic
fields `n_promoted`, `mean_despair_pre`/`post`). No within-run agent-level
trajectory data is retained by default; all reported analyses (Sobol, SHAP,
robustness, ablation) operate on these per-universe summary outputs
aggregated across the relevant ensemble of independently seeded universes,
not on raw agent-level time series.

---

## Initialization

Every universe begins from the following fixed initial conditions, applied
identically regardless of the sampled parameter vector:

| Variable | Initial value | Source / sampling rule |
|---|---:|---|
| `energy` | 100.0 | Fixed |
| `compliance_score` | 100.0 | Fixed |
| `personal_wealth` | 0.0 | Fixed |
| `policy_state` | `Assigned` | Fixed |
| `quarters_without_promotion` | 0 | Fixed |
| `is_dead` | False | Fixed |
| `age` | — | Sampled from Normal(26.1, 6.5), then **clamped** (not resampled) to [14, 65] — values outside this range are set to the boundary rather than redrawn, so the boundary values carry a small excess probability mass relative to a true truncated distribution |
| `skill_tier` | 1.0 | 1.0 for agents transported before 1820; sampled from [1.0, `SKILL_PREMIUM_MAX` = 2.0] for agents transported from 1820 onward |
| `inherent_rebellion` (R_i) | — | Derived from sentence length via `rebellion_sentence_weight`, with a fixed baseline of 0.30 |
| `colony_stress` | — | Two-layer draw: a destination-level value shared by all agents sent to that destination in the universe, multiplied by an independent ±20% individual-level noise term. Agents with missing destination information receive a deterministic fallback value: the year-specific mean stress among matched records for the same transportation year, or the global mean if no matched records exist for that year. |

No agent is initialised in `TicketOfLeave`, `ActiveRebellion`, or
`Emancipist`; all state transitions away from `Assigned` occur only through
the quarterly process described above.

---

## Input Data

The model draws on three external, empirically sourced datasets, used
respectively to determine the initial agent population, the environmental
shock series, and (indirectly, through calibration) the bound on the
despair-accumulation parameter:

1. **Convict population** — 123,888 individual records, cross-referenced
   against an open convict register for destination information.
2. **Environmental shock series** — the Eastern Australia and New Zealand
   Drought Atlas, a tree-ring based reconstruction, supplying the annual
   drought index expanded to the model's quarterly resolution.
3. **Historical calibration anchor** — the documented 1791 resource
   crisis, used only to bound the upper end of the `despair_rate` sampling
   range, not as a fitted target.

No other external dataset informs any state variable, transition rule, or
parameter bound in the model. See `README.md` for the exact input file
schema expected by `src/data_loader.py`.

---

## Submodels

Each submodel below is a single, precisely defined process invoked at the
point in the quarterly schedule indicated by its heading.

### Ration determination

At the start of each quarter t, the current quarter's drought value is
written into a quarter-indexed history array before the ration calculation
is performed:

```
D_hist[t-1] <- D_t
```

where `D_t` is the current quarter's drought index — the historical value
if drought is enabled, or 0.0 otherwise. The accumulated shock used for the
ration calculation is then:

```
AccumulatedDroughtShock_t =
    D_hist[0] * 3                                     if t <= 3
    D_hist[t-2] + D_hist[t-3] + D_hist[t-4]            if t > 3

ration_t = clip(6.5 * (1 - 0.14 * AccumulatedDroughtShock_t), 1.2, 6.5)
```

Because the current quarter's drought value is written to `D_hist[t-1]`
*before* the ration calculation runs, `D_hist[0]` already holds quarter 1's
own drought value by the time any ration is computed. The `t <= 3` branch
therefore uses three copies of quarter 1's drought value as its
accumulated-shock estimate, rather than a fixed background constant — a
startup approximation used because a genuine three-quarter rolling window
does not yet exist this early in the run. The uniform 0.20 values the
history array is pre-filled with at initialisation are overwritten before
any ration calculation reads them, and never themselves enter a ration
calculation. From `t > 3` onward, the accumulated shock is a
one-quarter-lagged three-quarter rolling sum of the true recorded drought
history.

### Metabolic energy update (non-rebelling agents)

```
E_i,t = clip(E_i,t-1 - 4.0 * stress_penalty_i - 8.0 * D_t + ration_t, 0, 100)
stress_penalty_i = 1.0 + (sigma_colony,i - 1.0) * 0.3
```

Death occurs if `E_i,t <= 0`.

### Structural despair accumulation

The counter `tau_i,t` (`quarters_without_promotion`) increments by 1 each
quarter an agent remains `Assigned`, and is reset toward zero on any
promotion. Structural despair `Psi_i,t` is computed as `despair_rate * tau
* colony-stress-adjusted multiplier` (see `src/worker.py` for the exact
formula) and enters the rebellion decision function below.

### Rebellion decision function

The probability that agent i rebels at time t follows a logistic function
of energy deprivation, drought pressure, baseline rebellion propensity,
peer contagion, structural despair, and institutional suppression (see
`src/agent_brain.py` for the exact functional form). An agent whose
computed probability exceeds `rebel_threshold` transitions to
`ActiveRebellion` and its `compliance_score` is reset to 0.0.

### ActiveRebellion survival, foraging, and amnesty return

```
E_i,t = clip(E_i,t-1 - 12.0 * D_t + 3.5, 0, 100)
```

This replaces the standard metabolic rule for the duration of
`ActiveRebellion`; death occurs under the same `E_i,t <= 0` condition. In
each quarter, if `D_t < 0.4`, the agent returns to `Assigned` with
probability 0.20, its `compliance_score` reset to 40.0 (a partial, not
full, restoration), reflecting the historical practice of amnesty following
periods of reduced unrest.

### Production and compliance accrual (non-rebelling agents)

```
output_i,t = 55.0 * skill_tier_i * (E_i,t / 100) * (1 - 0.40 * D_t)
```

| `policy_state` | Wealth accrual | Compliance accrual |
|---|---|---:|
| `Assigned` | none | +0.8 |
| `TicketOfLeave` | +40% of output | +1.5 |
| `Emancipist`* | +100% of output | +2.0 |

\* Formula defined in the implementation for representational completeness
but unreachable under every transition rule in the current specification
(see the note on `Emancipist` above).

### Automatic promotion pathway

Independently of any policy intervention, an `Assigned` agent with
`compliance_score` > 115 is promoted to `TicketOfLeave` every quarter this
condition holds, with `quarters_without_promotion` reset to 0.

### Policy intervention (Carrot / Stick)

Executed once, in the single quarter equal to `intervention_quarter`.
Modifiers are drawn once per universe from:

| Modifier | Carrot range | Stick range |
|---|---:|---:|
| Threshold modifier | [0.5, 0.8] | [1.2, 1.6] |
| Coverage modifier | [1.1, 1.5] | [0.5, 0.8] |

`effective_threshold` and `effective_coverage` are derived by applying the
drawn modifier to `intervention_threshold`/`intervention_coverage`.
Eligible agents are `Assigned` agents with `compliance_score` above
`effective_threshold`; a random subset, sized to `effective_coverage` of
the eligible pool, is promoted to `TicketOfLeave` without replacement.
Promoted agents' `quarters_without_promotion` is reset toward zero;
Carrot-promoted agents additionally gain +5.0 `energy` (capped at 100),
while Stick-promoted agents retain a fixed +2 residual on
`quarters_without_promotion` after the reset, representing a smaller
structural-pressure relief than under Carrot.

### Natural mortality

A quarterly probability is applied independently of all other rules,
derived from an annual historical estimate `p_a` via
`p_q = 1 - (1 - p_a)^(1/4)`, sampled once per universe from the range
corresponding to `p_a` in [0.0048, 0.015].

### Collapse detection

```
current_mutiny_rate_t = (count of alive agents in ActiveRebellion / count of alive agents) * 100
```

`collapse_quarter` is set to the first quarter t at which
`current_mutiny_rate_t >= 20.0`, and remains fixed thereafter.
`peak_mutiny` is the maximum of `current_mutiny_rate_t` over all 40
quarters, recorded regardless of whether the collapse threshold is ever
crossed.

---

## References

- Epstein, J. M. & Axtell, R. L. (1996). *Growing Artificial Societies:
  Social Science from the Bottom Up*. Brookings Institution Press / MIT
  Press.
- Grimm, V. et al. (2006). A standard protocol for describing
  individual-based and agent-based models. *Ecological Modelling*,
  198(1-2), 115-126.
- Grimm, V. et al. (2010). The ODD protocol: A review and first update.
  *Ecological Modelling*, 221(23), 2760-2768.
- Grimm, V. et al. (2020). The ODD protocol for describing agent-based and
  other simulation models: A second update to improve clarity,
  replication, and structural realism. *JASSS*, 23(2), 7.
