# Emergent Order and Institutional Collapse: A Historical Agent-Based Model of Behaviour, Structural Pressure, External Shock, and Intervention

**CoMSES Net Archival Package**

An agent-based model (ABM) of colonial Australian convict society,
combining historical convict records with drought data to study how
endogenous structural despair, external environmental shocks,
micro-level behavioural thresholds, and institutional intervention
design interact to produce societal collapse.

This is the complete, self-contained archival package for the paper
*"Emergent Order and Institutional Collapse: A Historical Agent-Based
Model of Behaviour, Structural Pressure, External Shock, and
Intervention"* (submitted to JASSS — the Journal of Artificial
Societies and Social Simulation). It contains the simulation code, the
full input datasets, and the complete set of simulation outputs and
figures underlying the paper's results, so that the study can be
independently reproduced and verified without needing anything beyond
what is included here.

For a full description of the model's structure, entities, and
process logic, see `MODEL_DOCUMENTATION.md`, which follows the ODD
(Overview, Design concepts, Details) protocol.

## Requirements

```bash
pip install -r requirements.txt
```

## Project structure

```
main.py                     Unified command-line entry point for every analysis mode
src/
  worker.py                  Core per-universe simulation logic
  agent_brain.py              Individual agent behavioural model (ABMConvictAgent)
  config.py                   Centralized model parameters and historical calibration
  data_loader.py               Loads input data (see Data section below)
  plot_style.py                Shared matplotlib house style for all figures
  runners/                    One module per analysis mode (see Usage below)
scripts/
  prepare_jasss_figures.py    Validates/resizes figures for JASSS submission
data/
  raw/                        Original source datasets, as obtained (see Data below)
  processed/                  Cleaned datasets in the schema src/data_loader.py expects
analysis/
  robustness_results/         Main 200,000-universe run and targeted ablation/stress-test runs
  sensitivity_results/        K-voyage structural sensitivity runs (K=10/30/50)
  sobol_results/              Sobol global sensitivity analysis indices, raw evaluations,
                               and interaction results, plus resumable checkpoints
  shap_results/                SHAP surrogate-model explanation outputs and checkpoints
  diagnostic_results/          Q13 collapse-timing diagnostic outputs
  ztest_results/               Two-proportion z-test comparison results
  figures/submission/          The final curated figure set (Figure_1 through Figure_S8)
                               referenced in the manuscript, with submission_manifest.csv
                               mapping each figure to its exact source file
MODEL_DOCUMENTATION.md       Standalone ODD protocol model description
requirements.txt
```

## Usage

Every analysis mode is run through `main.py`:

```bash
# Large-scale robustness run (main 200,000-universe sweep)
python3 main.py robustness --n 200000 --batch 200 --seed 7777

# K-value structural sensitivity comparison (K=10/30/50)
python3 main.py sensitivity --n 5000 --k-values 10,30,50

# Q13 collapse-timing diagnostic check (post-hoc, no new simulation)
python3 main.py diagnostic --input analysis/robustness_results/simulation_results_K30_v1_20260803_024351.csv

# Sobol global sensitivity analysis
python3 main.py sobol --samples 2048

# Two-proportion z-tests on an existing results file
python3 main.py ztest --input analysis/robustness_results/simulation_results_K30_v1_20260803_024351.csv

# SHAP surrogate-model explanation
python3 main.py shap --input analysis/robustness_results/simulation_results_K30_v1_20260803_024351.csv

# Single-variable ablation study (hold one or more parameters constant)
python3 main.py ablation --fix w3=0.0 --n 50000
```

Each subcommand supports `--help` for its full argument list. Most
subcommands also support `--figures-only`, which regenerates figures
from an existing results CSV without rerunning any simulation.

The exact command, seed, and configuration used to produce each
reported figure or statistic in the manuscript is recorded in
`analysis/figures/submission/submission_manifest.csv` and in each
result file's own naming convention (K-value, seed, sample size, and
a configuration hash where applicable).

## Data

This package includes both the raw source data and the cleaned data
actually consumed by the model.

### `data/raw/`

The original source files as obtained from their respective
providers, prior to any cleaning or linkage:

- **Convict records** — British convict transportation records
  (Digital Panopticon, 1787-1867).
- **Queensland convict register** — used to recover destination
  information via record linkage (State Library of Queensland, open
  convict register).
- **Drought reconstruction** — the Eastern Australia and New Zealand
  Drought Atlas, a tree-ring based annual drought index (NOAA
  Paleoclimatology, Palmer et al. 2016).

### `data/processed/`

The cleaned, linked datasets in the exact schema `src/data_loader.py`
expects:

- `btr_enriched_with_dest.csv` — convict records enriched with
  destination information via record linkage (see `MODEL_DOCUMENTATION.md`,
  Input Data, for the linkage method and match rate).
- `quarterly_drought.csv` — the drought reconstruction expanded to
  the model's quarterly resolution.
- `voyage_stats_date_cleaned.csv` — voyage-level summary statistics
  derived from the convict records.

The data-cleaning and record-linkage scripts that produced
`data/processed/` from `data/raw/` are not included in this package;
`MODEL_DOCUMENTATION.md` and the manuscript's Historical and Data
Foundations section describe the linkage method and resulting match
rates in full.

## Results

`analysis/` contains the complete set of simulation outputs, sensitivity
analyses, and figures referenced in the manuscript, organized by
analysis mode. Every result file's name encodes its configuration
(K-value, seed, sample size, and/or a short configuration hash), so
the exact provenance of any reported number or figure can be traced
directly from its filename, cross-referenced against
`analysis/figures/submission/submission_manifest.csv` for the final
figures.

## Reproducibility

All stochastic components of the model use explicitly seeded random
number generators. Every reported result can be exactly reproduced
from the corresponding command-line invocation and random seed, as
documented in the Usage section above and in the manuscript's Methods
and Appendix G (Reproducibility).
