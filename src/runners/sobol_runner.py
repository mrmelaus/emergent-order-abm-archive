"""
Sobol Sensitivity Analysis Runner for the ABM Engine.

Executes a Saltelli-sampled global sensitivity analysis over a chosen
subset of continuous model parameters and computes first-order and
total-order Sobol indices for one or more scalar outcome variables.
Produces machine-readable CSVs and publication-quality figures (PDF +
300 DPI PNG) suitable for direct inclusion in a manuscript.

Design notes
------------
1. Parameter bounds are read from src.config.PARAM_BOUNDS so that the
   sampled parameter space always matches the bounds actually used by
   worker.execute_universe_worker's default random sampling. These two
   sources must be kept manually in sync; PARAM_BOUNDS is not derived
   automatically from worker.py.

2. Categorical variables (intervention_type, drought_enabled) are
   intentionally excluded from Sobol analysis. Standard Sobol variance
   decomposition assumes continuous inputs; categorical factors should
   instead be studied as separate groups via the 'ablation' subcommand
   (see main.py), comparing outcomes between fixed categorical values.

3. Random noise control: each sampled parameter vector is evaluated
   under a small, fixed set of replicate seeds (not one seed per
   Saltelli row). Outcomes are averaged across replicates before the
   variance decomposition, separating "variance explained by the
   sampled parameters" from "variance explained by the model's
   intrinsic stochasticity".

4. Checkpointing and resume. Saltelli sampling is deterministic given
   (--samples, --vars, --second-order), so an identical command line
   always regenerates an identical parameter matrix. Every completed
   batch of evaluations is appended immediately to a checkpoint CSV
   keyed by a signature of the run settings. Re-running the same
   command automatically resumes from the checkpoint. Use --fresh to
   discard it and start over. Checkpoints are NOT shared across
   different --samples values, since Saltelli matrices at different N
   are independent sample sets, not nested extensions of one another.

5. Performance: a single persistent worker pool is created once for
   the entire invocation (including every N in a --convergence sweep)
   and reused across all batches, instead of being rebuilt per batch.
   Large, read-only simulation inputs (voyage data, drought history,
   year-stress map) are transferred to each worker process exactly
   once via the pool initializer; individual tasks only carry a small
   (row_index, replicate_id, param_vector) tuple, eliminating repeated
   serialization of the full dataset on every evaluation.

6. Worker count control. By default every available CPU core is used
   (appropriate for an unattended NAS/server run). When running
   interactively on a personal machine (e.g. a MacBook used for other
   work at the same time), pass --reserve-cores N to leave N cores
   free for the rest of the system; this has no effect unless
   explicitly requested, so unattended runs are unaffected.

7. Second-order interactions. When --second-order is passed, pairwise
   interaction indices (S2, S2_conf) are additionally written to a
   dedicated *_interactions.csv, with an accompanying heatmap figure
   per target. Running a full second-order sweep over many parameters
   is expensive (N * (2D + 2) evaluations); for a specific, targeted
   interaction question (e.g. "how much do w1 and w4 interact?"),
   restrict --vars to just that pair, e.g. --vars w1,w4 --second-order,
   which is both cheaper and directly answers the question. Since
   several such targeted pairs are typically run in the same project
   (each producing figures with the same generic-looking filename
   otherwise), always pass --label for these restricted runs — see
   note 9 below.

8. Convergence sweeps. Pass --convergence-n "256,512,1024" to run a
   sequence of N values in one invocation (sharing the same worker
   pool and simulation inputs), and additionally produce a convergence
   CSV and a ST-vs-N line plot per target for the supplementary
   material, alongside each N's own regular indices/raw CSV and
   figures.

9. Figure filename labels. By default, figure filenames embed the
   sorted, underscore-joined parameter names (e.g.
   despair_rate_rebel_threshold_w1_w2_w3_w4_w5_social), which is
   always self-describing but can be long and does not necessarily
   match a name used elsewhere (e.g. in a manuscript's figure
   provenance table). Pass --label to override this with a short,
   stable tag (e.g. --label main7 or --label restricted_w1_w4);
   this affects ONLY figure filenames, never the run signature/hash
   used for the CSV filenames or checkpoint resumption. When
   reproducing a specific figure that already exists in a
   manuscript's figure directory, always pass the same --label used
   to originally generate it, so the new file overwrites the old one
   in place rather than producing a differently-named duplicate.
   
Outputs are written to shared, project-level directories:
    analysis/sobol_results/               -> indices/raw/interactions CSVs
    analysis/sobol_results/checkpoints/    -> resumable checkpoint CSVs
    analysis/figures/                      -> PDF and PNG figures

Usage examples:
    # Run Sobol analysis with default settings
    python main.py sobol --samples 2048

    # Regenerate figures from existing indices CSV (no simulation)
    python main.py sobol --figures-only --output analysis/sobol_results/sobol_*_indices.csv

    # Run with second-order interactions and custom label
    python main.py sobol --samples 1024 --vars w1,w4 --second-order --label restricted_w1_w4
"""

import hashlib
import multiprocessing
import os
import re
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from .plot_style import apply_house_style, heatmap_figsize, FIGSIZE_STANDARD, save_figure, COLORS

from src.data_loader import load_simulation_inputs, default_data_paths
from src.worker import execute_universe_worker
from src.config import K_VOYAGES as DEFAULT_K
from src.config import PARAM_BOUNDS

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol as sobol_analyze
except ImportError:
    saltelli = None
    sobol_analyze = None


# ================================================================
# DEFAULT SETTINGS
# ================================================================

DEFAULT_VARS = "w1,w2,w3,w4,w5_social,despair_rate"
DEFAULT_TARGETS = "peak_mutiny,collapse_quarter"
DEFAULT_SAMPLES = 1024
DEFAULT_SEEDS = 5
DEFAULT_GLOBAL_SEED = 7777
DEFAULT_MAX_RETRIES = 3
DEFAULT_MODEL_VERSION = "v1"

_SOBOL_SEED_SALT = 990_000_000


# ================================================================
# ARGUMENT REGISTRATION
# ================================================================

def build_sobol_parser(subparsers):
    """
    Register the 'sobol' subcommand.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = subparsers.add_parser(
        "sobol",
        help="Sobol global sensitivity analysis over continuous model parameters.",
    )

    parser.add_argument("--k", type=int, default=DEFAULT_K,
                         help=f"Number of voyages sampled per model evaluation (default: {DEFAULT_K}).")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                         help="Base sample size N for a single-N run. Ignored if --convergence-n is set. "
                              f"Total evaluations = N * (D + 2) * seeds (default: {DEFAULT_SAMPLES}).")
    parser.add_argument("--convergence-n", type=str, default=None,
                         help="Comma-separated list of N values to sweep, e.g. '256,512,1024'. "
                              "When set, --samples is ignored; each N gets its own resumable "
                              "checkpoint and its own indices/raw CSV and figures, plus an "
                              "aggregated convergence CSV and ST-vs-N plot per target.")
    parser.add_argument("--vars", type=str, default=DEFAULT_VARS,
                         help=f"Comma-separated list of continuous parameter names to analyze, "
                              f"each of which must exist in PARAM_BOUNDS (default: {DEFAULT_VARS}). "
                              f"For a targeted interaction question, restrict this to the pair of "
                              f"interest, e.g. --vars w1,w4 --second-order.")
    parser.add_argument("--label", type=str, default=None,
                         help="Short human-readable tag included in figure filenames so they "
                              "remain self-describing at a glance, e.g. --label main7 or "
                              "--label restricted_w1_w4. Does NOT affect the run signature/hash "
                              "or checkpoint resumption — only figure filenames. If omitted, "
                              "defaults to the sorted, underscore-joined parameter names.")
    parser.add_argument("--targets", type=str, default=DEFAULT_TARGETS,
                         help=f"Comma-separated list of scalar outcome fields to explain "
                              f"(default: {DEFAULT_TARGETS}).")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                         help="Number of fixed replicate seeds averaged per sampled parameter "
                              f"vector (default: {DEFAULT_SEEDS}).")
    parser.add_argument("--second-order", action="store_true",
                         help="Also compute second-order (pairwise interaction) Sobol indices, "
                              "written to a dedicated *_interactions.csv with a heatmap figure. "
                              "Increases evaluations to N * (2D + 2) * seeds.")
    parser.add_argument("--seed", type=int, default=DEFAULT_GLOBAL_SEED,
                         help=f"Global random seed for replicate seed generation (default: {DEFAULT_GLOBAL_SEED}).")
    parser.add_argument("--batch", type=int, default=200,
                         help="Number of evaluations submitted per checkpoint-flush batch.")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of multiprocessing workers. Default: all available cores "
                              "minus --reserve-cores.")
    parser.add_argument("--reserve-cores", type=int, default=0,
                         help="Number of CPU cores to leave free for the rest of the system. "
                              "Use e.g. --reserve-cores 1 on a machine you are using interactively. "
                              "Ignored if --workers is explicitly set. Default 0 (use every core), "
                              "correct for an unattended NAS/server run.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                         help=f"Maximum attempts for a failed evaluation (default: {DEFAULT_MAX_RETRIES}).")
    parser.add_argument("--version", type=str, default=DEFAULT_MODEL_VERSION,
                         help=f"Model version tag recorded in raw output (default: {DEFAULT_MODEL_VERSION}).")
    parser.add_argument("--output", type=str, default=None,
                         help="Output CSV path for Sobol indices (single-N mode only). "
                              "Default: automatically generated.")
    parser.add_argument("--figures-only", action="store_true",
                         help="Skip simulation entirely and (re)generate figures from an existing --output CSV. "
                              "Requires --output to point to an existing indices CSV file (e.g. sobol_*_indices.csv).")
    parser.add_argument("--no-figures", action="store_true",
                         help="Skip figure generation and only write the CSVs.")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from an existing checkpoint (default behaviour already "
                              "resumes automatically if a matching checkpoint exists).")
    parser.add_argument("--fresh", action="store_true",
                         help="Discard any existing checkpoint matching these settings and start over.")

    return parser


def _validate_args(args, param_names):
    """Validate resolved Sobol settings; exits with an error message on invalid input."""
    if saltelli is None or sobol_analyze is None:
        sys.exit(
            "ERROR: SALib is required for Sobol analysis. "
            "Install it with: pip install SALib --break-system-packages"
        )

    if args.k <= 0:
        sys.exit("ERROR: --k must be greater than zero.")
    if not args.convergence_n and args.samples <= 0:
        sys.exit("ERROR: --samples must be greater than zero.")
    if args.seeds <= 0:
        sys.exit("ERROR: --seeds must be greater than zero.")
    if args.batch <= 0:
        sys.exit("ERROR: --batch must be greater than zero.")
    if args.max_retries <= 0:
        sys.exit("ERROR: --max-retries must be greater than zero.")
    if args.workers is not None and args.workers <= 0:
        sys.exit("ERROR: --workers must be greater than zero.")
    if args.reserve_cores < 0:
        sys.exit("ERROR: --reserve-cores cannot be negative.")
    if args.resume and args.fresh:
        sys.exit("ERROR: --resume and --fresh cannot be used together.")

    missing = [name for name in param_names if name not in PARAM_BOUNDS]
    if missing:
        sys.exit(
            f"ERROR: the following --vars are not defined in PARAM_BOUNDS: {missing}. "
            f"Available parameters: {sorted(PARAM_BOUNDS.keys())}. "
            f"PARAM_BOUNDS must be kept manually in sync with worker.py's sampling bounds."
        )


def _resolve_worker_count(args):
    """
    Resolve the number of worker processes to use.

    Default: every available core minus --reserve-cores (which
    defaults to 0, i.e. use every core). --workers overrides this
    entirely if supplied.
    """
    cpu_count = multiprocessing.cpu_count()

    if args.workers is not None:
        return max(1, args.workers)

    return max(1, cpu_count - args.reserve_cores)


# ================================================================
# OUTPUT / CHECKPOINT PATHS
# ================================================================

def _sobol_results_dir(project_root):
    results_dir = os.path.join(project_root, "analysis", "sobol_results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _figures_dir(project_root):
    figures_dir = os.path.join(project_root, "analysis", "figures", "sobol")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def _run_signature(k_voyages, samples, seeds, global_seed, second_order, param_names):
    """Deterministic signature identifying one exact (N-specific) run configuration."""
    vars_key = ",".join(param_names)
    order_tag = "2nd" if second_order else "1st"
    raw = f"K{k_voyages}_N{samples}_seeds{seeds}_seed{global_seed}_{order_tag}_{vars_key}"
    short_hash = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"K{k_voyages}_N{samples}_seeds{seeds}_{order_tag}_{short_hash}"


def _checkpoint_path(results_dir, signature):
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"sobol_checkpoint_{signature}.csv")


def _generate_csv_paths(results_dir, signature, args_output):
    if args_output:
        indices_path = args_output
        base, ext = os.path.splitext(args_output)
        raw_path = f"{base}_raw{ext or '.csv'}"
        interactions_path = f"{base}_interactions{ext or '.csv'}"
        return indices_path, raw_path, interactions_path

    stem = os.path.join(results_dir, f"sobol_{signature}")
    return f"{stem}_indices.csv", f"{stem}_raw.csv", f"{stem}_interactions.csv"


# ================================================================
# CHECKPOINT READ / WRITE
# ================================================================

def _load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
        return {}

    df = pd.read_csv(checkpoint_path)
    results = {}
    for _, row in df.iterrows():
        key = (int(row["row_index"]), int(row["replicate_id"]))
        results[key] = row.to_dict()
    return results


def _append_checkpoint(checkpoint_path, batch_results):
    if not batch_results:
        return

    rows = []
    for (row_index, replicate_id), result in batch_results.items():
        row = {"row_index": row_index, "replicate_id": replicate_id}
        row.update(result)
        rows.append(row)

    df = pd.DataFrame(rows)
    file_exists = os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0
    df.to_csv(checkpoint_path, mode="a", index=False, header=not file_exists)


# ================================================================
# PERSISTENT WORKER POOL (module-level globals, set once per process
# via the pool initializer — this is what lets each task ship only a
# tiny (row_index, replicate_id, param_vector) tuple instead of the
# full simulation inputs on every call).
# ================================================================

_WORKER_INPUTS = None
_WORKER_SETTINGS = None


def _pool_initializer(inputs, settings):
    """Runs once per worker process when the persistent Pool starts up."""
    global _WORKER_INPUTS, _WORKER_SETTINGS
    _WORKER_INPUTS = inputs
    _WORKER_SETTINGS = settings


def _generate_replicate_seed(global_seed, replicate_id):
    """Deterministic seed for one Sobol replicate, salted away from robustness universe seeds."""
    return np.random.SeedSequence([int(global_seed), _SOBOL_SEED_SALT + int(replicate_id)])


def _build_worker_args(row_index, replicate_id, param_vector, param_names, inputs, settings):
    """Assemble the 9-field args tuple expected by execute_universe_worker."""
    param_overrides = dict(zip(param_names, param_vector))
    child_seed = _generate_replicate_seed(settings["global_seed"], replicate_id)
    evaluation_id = f"sobol_{row_index}_{replicate_id}"

    return (
        evaluation_id,
        inputs["grouped_voyages"],
        inputs["voyage_ids"],
        inputs["drought_history_source"],
        inputs["year_stress_map"],
        settings["k_voyages"],
        child_seed,
        settings["model_version"],
        param_overrides,
    )


def _evaluate_task(task):
    """
    Runs inside a worker process. Reconstructs the full worker args
    using the process-local globals set by _pool_initializer, so the
    only data actually shipped through the task queue is the small
    (row_index, replicate_id, param_vector) tuple.
    """
    row_index, replicate_id, param_vector = task
    worker_args = _build_worker_args(
        row_index, replicate_id, param_vector, _WORKER_SETTINGS["param_names"], _WORKER_INPUTS, _WORKER_SETTINGS
    )
    result = execute_universe_worker(worker_args)
    return row_index, replicate_id, result


def _run_batch_with_retry(pending_specs, pool, max_retries):
    """
    Evaluate a list of (row_index, replicate_id, param_vector) specs
    using the shared persistent pool, retrying failures in place.

    Returns
    -------
    dict
        Mapping of (row_index, replicate_id) -> result dict.
    """
    results = {}
    retry_counts = {(spec[0], spec[1]): 0 for spec in pending_specs}
    remaining = list(pending_specs)

    while remaining:
        tasks = [(row_index, replicate_id, param_vector) for row_index, replicate_id, param_vector in remaining]
        raw_results = pool.map(_evaluate_task, tasks)

        still_pending = []
        for (row_index, replicate_id, result) in raw_results:
            key = (row_index, replicate_id)

            if result is None:
                retry_counts[key] += 1
                if retry_counts[key] >= max_retries:
                    raise RuntimeError(
                        f"Sobol evaluation row={row_index}, replicate={replicate_id} "
                        f"failed {max_retries} times. Analysis aborted. "
                        f"Progress so far is preserved in the checkpoint file; "
                        f"rerun the same command to resume."
                    )
                param_vector = next(v for r, rep, v in remaining if r == row_index and rep == replicate_id)
                still_pending.append((row_index, replicate_id, param_vector))
                continue

            results[key] = result

        remaining = still_pending

    return results


# ================================================================
# FIGURE GENERATION (PUBLICATION QUALITY)
# ================================================================

def _plot_sobol_indices(indices_df, target_name, figures_dir, k_voyages, param_names, samples, file_hash, figure_label):
    """
    Bar chart of S1/ST with confidence intervals for one target.
    Returns (pdf_path, png_path) or None.
    """
    subset = indices_df[indices_df["target"] == target_name].sort_values("ST", ascending=True)
    if subset.empty:
        return None

    apply_house_style()

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    y_pos = np.arange(len(subset))

    ax.barh(
        y_pos - 0.2,
        subset["S1"],
        height=0.4,
        xerr=subset["S1_conf"],
        label="$S_1$ (first-order)",
        color=COLORS["blue"],
        capsize=2,
    )
    ax.barh(
        y_pos + 0.2,
        subset["ST"],
        height=0.4,
        xerr=subset["ST_conf"],
        label="$S_T$ (total-order)",
        color=COLORS["orange"],
        capsize=2,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(subset["parameter"])
    ax.set_xlabel("Sobol sensitivity index")
    ax.set_title(f"Sobol indices — {target_name}")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    stem = f"sobol_{target_name}_{figure_label}_K{k_voyages}_N{samples}_{file_hash}"
    pdf_path = os.path.join(figures_dir, f"{stem}.pdf")
    png_path = os.path.join(figures_dir, f"{stem}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)

    return pdf_path, png_path


def _plot_interaction_heatmap(interactions_df, target_name, param_names, figures_dir, k_voyages, samples, file_hash, figure_label):
    """
    Heatmap of pairwise S2 interaction indices for one target.
    Returns (pdf_path, png_path) or None.
    """
    subset = interactions_df[interactions_df["target"] == target_name]
    if subset.empty:
        return None

    n = len(param_names)
    matrix = np.full((n, n), np.nan)
    index_of = {name: i for i, name in enumerate(param_names)}

    for _, row in subset.iterrows():
        i, j = index_of[row["param_a"]], index_of[row["param_b"]]
        matrix[i, j] = row["S2"]
        matrix[j, i] = row["S2"]

    apply_house_style()

    # Use the shared heatmap_figsize from plot_style
    fig, ax = plt.subplots(figsize=heatmap_figsize)
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.set_yticklabels(param_names)
    ax.set_title(f"Second-order interactions ($S_2$) — {target_name}")

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]) and i != j:
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > np.nanmax(matrix) / 2 else "black",
                    fontsize=9,
                )

    fig.colorbar(im, ax=ax, label="$S_2$", shrink=0.8)
    fig.tight_layout()

    stem = f"sobol_interactions_{target_name}_{figure_label}_K{k_voyages}_N{samples}_{file_hash}"
    pdf_path = os.path.join(figures_dir, f"{stem}.pdf")
    png_path = os.path.join(figures_dir, f"{stem}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)

    return pdf_path, png_path


def _plot_convergence(convergence_df, target_name, figures_dir, k_voyages, param_names, n_values, file_hashes, figure_label):
    """
    Line plot of ST vs N per parameter, for the convergence sweep.
    Returns (pdf_path, png_path) or None.
    """
    subset = convergence_df[convergence_df["target"] == target_name]
    if subset.empty:
        return None

    n_range_str = "_".join([str(n) for n in n_values])
    hash_str = "-".join(file_hashes)

    apply_house_style()

    # Wider figure to accommodate legend on the right (like SHAP)
    fig, ax = plt.subplots(figsize=(10, 5))

    for param_name, group in subset.groupby("parameter"):
        group = group.sort_values("N")
        ax.errorbar(
            group["N"],
            group["ST"],
            yerr=group["ST_conf"],
            marker="o",
            capsize=2,
            label=param_name,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Base sample size N (log scale)")
    ax.set_ylabel("$S_T$ (total-order index)")
    ax.set_title(f"Sobol convergence — {target_name}")
    
    # Legend outside to the right (like SHAP)
    ax.legend(frameon=False, fontsize=8, ncol=1, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)  # Make room for legend (like SHAP)

    stem = f"sobol_convergence_{target_name}_{figure_label}_K{k_voyages}_N{n_range_str}_{hash_str}"
    pdf_path = os.path.join(figures_dir, f"{stem}.pdf")
    png_path = os.path.join(figures_dir, f"{stem}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)

    return pdf_path, png_path

# ================================================================
# CORE EXECUTION FOR ONE N VALUE
# ================================================================

def _execute_for_n(args, param_names, target_names, problem, base_settings, results_dir, figures_dir,
                    samples, pool):
    """
    Run (or resume) the full Sobol evaluation and analysis for one
    base sample size N, writing its own indices/raw/interactions CSVs
    and figures. Returns the indices DataFrame for this N.
    """
    settings = dict(base_settings)
    signature = _run_signature(settings["k_voyages"], samples, args.seeds, settings["global_seed"],
                                args.second_order, param_names)
    checkpoint_path = _checkpoint_path(results_dir, signature)

    if args.fresh and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"--fresh specified: removed existing checkpoint {checkpoint_path}")

    print("-" * 70)
    print(f"N = {samples}  |  signature = {signature}")
    print("-" * 70)

    param_matrix = saltelli.sample(problem, samples, calc_second_order=args.second_order)
    num_rows = param_matrix.shape[0]
    total_evaluations = num_rows * args.seeds
    print(f"{num_rows:,} parameter vectors x {args.seeds} replicate seeds = "
          f"{total_evaluations:,} total model evaluations.")

    all_results = _load_checkpoint(checkpoint_path)
    if all_results:
        print(f"Found existing checkpoint with {len(all_results):,} completed evaluations. Resuming.")

    evaluation_specs = [
        (row_index, replicate_id, param_matrix[row_index])
        for row_index in range(num_rows)
        for replicate_id in range(args.seeds)
        if (row_index, replicate_id) not in all_results
    ]

    if evaluation_specs:
        print(f"Remaining evaluations: {len(evaluation_specs):,} / {total_evaluations:,}")
        start_time = time.time()

        try:
            for batch_start in range(0, len(evaluation_specs), args.batch):
                batch_specs = evaluation_specs[batch_start: batch_start + args.batch]
                batch_results = _run_batch_with_retry(batch_specs, pool, settings["max_retries"])

                _append_checkpoint(checkpoint_path, batch_results)
                all_results.update(batch_results)

                completed = len(all_results)
                elapsed = time.time() - start_time
                session_done = completed - (total_evaluations - len(evaluation_specs))
                rate = session_done / elapsed if elapsed > 0 else 0.0
                print(f"  {completed:,}/{total_evaluations:,} complete ({rate:.1f} evals/s this session)")

        except KeyboardInterrupt:
            print()
            print("=" * 70)
            print("SOBOL RUN INTERRUPTED BY USER")
            print("=" * 70)
            print(f"Completed evaluations saved to checkpoint: {len(all_results):,}/{total_evaluations:,}")
            print(f"Checkpoint file: {checkpoint_path}")
            print("Rerun the exact same command to resume automatically.")
            print("=" * 70)
            raise
    else:
        print("All evaluations already completed in the checkpoint.")

    # ------------------------------------------------------------
    # Average replicate outcomes, compute indices
    # ------------------------------------------------------------
    target_outcomes = {name: np.empty(num_rows, dtype=float) for name in target_names}
    raw_rows = []

    for row_index in range(num_rows):
        replicate_results = [all_results[(row_index, r)] for r in range(args.seeds)]
        row_record = {"row_index": row_index}
        row_record.update(dict(zip(param_names, param_matrix[row_index])))

        for target_name in target_names:
            values = [float(result.get(target_name, np.nan)) for result in replicate_results]
            mean_value = float(np.nanmean(values))
            target_outcomes[target_name][row_index] = mean_value
            row_record[f"{target_name}_mean"] = mean_value
            row_record[f"{target_name}_std"] = float(np.nanstd(values))

        raw_rows.append(row_record)

    indices_rows = []
    interactions_rows = []

    for target_name in target_names:
        y = target_outcomes[target_name]
        if np.isnan(y).any():
            print(f"WARNING: target '{target_name}' contains NaN outcomes; skipping.")
            continue

        analysis = sobol_analyze.analyze(problem, y, calc_second_order=args.second_order, print_to_console=False)

        for i, param_name in enumerate(param_names):
            indices_rows.append({
                "target": target_name, "parameter": param_name,
                "S1": analysis["S1"][i], "S1_conf": analysis["S1_conf"][i],
                "ST": analysis["ST"][i], "ST_conf": analysis["ST_conf"][i],
            })

        if args.second_order and "S2" in analysis:
            for i, j in combinations(range(len(param_names)), 2):
                s2 = analysis["S2"][i][j]
                if np.isnan(s2):
                    continue
                interactions_rows.append({
                    "target": target_name, "param_a": param_names[i], "param_b": param_names[j],
                    "S2": s2, "S2_conf": analysis["S2_conf"][i][j],
                })

    indices_df = pd.DataFrame(indices_rows)
    raw_df = pd.DataFrame(raw_rows)
    interactions_df = pd.DataFrame(interactions_rows)

    indices_path, raw_path, interactions_path = _generate_csv_paths(results_dir, signature, args.output)
    indices_df.to_csv(indices_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    print(f"Indices written to: {indices_path}")
    print(f"Raw evaluations written to: {raw_path}")

    if not interactions_df.empty:
        interactions_df.to_csv(interactions_path, index=False)
        print(f"Interactions written to: {interactions_path}")

    if not args.no_figures:
        file_hash = signature.rsplit("_", 1)[-1]
        figure_label = args.label if args.label else "_".join(sorted(param_names))

        for target_name in target_names:
            result = _plot_sobol_indices(indices_df, target_name, figures_dir, settings["k_voyages"], param_names, samples, file_hash, figure_label)
            if result:
                print(f"Figure written to: {result[0]}")
                print(f"Figure written to: {result[1]}")

        if not interactions_df.empty:
            for target_name in target_names:
                result = _plot_interaction_heatmap(interactions_df, target_name, param_names, figures_dir, settings["k_voyages"], samples, file_hash, figure_label)
                if result:
                    print(f"Figure written to: {result[0]}")
                    print(f"Figure written to: {result[1]}")

    for target_name in target_names:
        subset = indices_df[indices_df["target"] == target_name].sort_values("ST", ascending=False)
        if subset.empty:
            continue
        print(f"Target: {target_name}  (ranked by ST)")
        for _, row in subset.iterrows():
            print(f"  {row['parameter']:<28} S1={row['S1']:.4f}  ST={row['ST']:.4f}")

    indices_df["N"] = samples
    return indices_df


# ================================================================
# MAIN EXECUTION
# ================================================================

def run_sobol(args):
    """
    Execute a Sobol global sensitivity analysis run: either a single N
    (--samples) or a multi-N convergence sweep (--convergence-n).

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    None
    """
    # ================================================================
    # FIGURES-ONLY MODE
    # ================================================================
    if args.figures_only:
        print("=" * 70)
        print("FIGURES-ONLY MODE: Regenerating figures from existing CSV")
        print("=" * 70)

        if not args.output or not os.path.exists(args.output):
            sys.exit(
                f"ERROR: --figures-only requires --output pointing to an existing indices CSV.\n"
                f"Got: {args.output or 'not provided'}\n"
                f"Expected: a Sobol indices CSV (e.g. sobol_*_indices.csv)"
            )

        print(f"--figures-only: loading {args.output}")
        indices_df = pd.read_csv(args.output)

        # Validate the CSV has required columns
        required_cols = {"target", "parameter", "S1", "S1_conf", "ST", "ST_conf"}
        missing_cols = required_cols - set(indices_df.columns)
        if missing_cols:
            sys.exit(
                f"ERROR: {args.output} does not appear to be a valid Sobol indices CSV.\n"
                f"Missing columns: {missing_cols}\n"
                f"Expected columns: {required_cols}"
            )

        # Extract metadata from filename
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(this_dir))
        figures_dir = _figures_dir(project_root)

        basename = os.path.basename(args.output)
        import re

        # Extract N from filename - MUST exist
        n_match = re.search(r'_N(\d+)_', basename)
        if not n_match:
            sys.exit(
                f"ERROR: Could not extract N from filename: {basename}\n"
                f"Expected format: sobol_*_N<digits>_*.csv"
            )
        samples = int(n_match.group(1))

        # Extract K from filename - MUST exist
        k_match = re.search(r'K(\d+)_', basename)
        if not k_match:
            sys.exit(
                f"ERROR: Could not extract K from filename: {basename}\n"
                f"Expected format: sobol_K<digits>_*.csv"
            )
        k_voyages = int(k_match.group(1))

        # Extract hash from filename - MUST exist, no fallback to "figuresonly"
        hash_match = re.search(r'_([a-f0-9]{10})_(?:indices|raw|interactions)\.csv$', basename)
        if not hash_match:
            sys.exit(
                f"ERROR: Could not extract 10-character hash from filename: {basename}\n"
                f"Expected format: sobol_*_<10charhash>_indices.csv\n"
                f"Example: sobol_K30_N256_seeds5_1st_443a133853_indices.csv"
            )
        file_hash = hash_match.group(1)

        # Extract param_names from the CSV
        param_names = indices_df["parameter"].unique().tolist()
        targets = indices_df["target"].unique().tolist()

        figure_label = args.label if args.label else "_".join(sorted(param_names))

        print(f"Detected: {len(param_names)} parameters, {len(targets)} targets, N={samples}, K={k_voyages}, hash={file_hash}")

        if not args.no_figures:
            for target_name in targets:
                result = _plot_sobol_indices(
                    indices_df, target_name, figures_dir, k_voyages,
                    param_names, samples, file_hash, figure_label
                )
                if result:
                    print(f"Figure written to: {result[0]}")
                    print(f"Figure written to: {result[1]}")

            # Check for interaction CSV in the same directory
            interactions_path = args.output.replace("_indices.csv", "_interactions.csv")
            if os.path.exists(interactions_path):
                interactions_df = pd.read_csv(interactions_path)
                for target_name in targets:
                    result = _plot_interaction_heatmap(
                        interactions_df, target_name, param_names, figures_dir,
                        k_voyages, samples, file_hash, figure_label
                    )
                    if result:
                        print(f"Figure written to: {result[0]}")
                        print(f"Figure written to: {result[1]}")
            else:
                print("No interactions CSV found (skipping heatmap).")

        print("=" * 70)
        print("FIGURES-ONLY MODE COMPLETE")
        print("=" * 70)
        return

    # ================================================================
    # NORMAL EXECUTION (existing code continues below)
    # ================================================================

    param_names = [name.strip() for name in args.vars.split(",") if name.strip()]
    target_names = [name.strip() for name in args.targets.split(",") if name.strip()]

    _validate_args(args, param_names)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(this_dir))
    convicts_path, drought_path = default_data_paths(project_root)
    results_dir = _sobol_results_dir(project_root)
    figures_dir = _figures_dir(project_root)

    base_settings = {
        "k_voyages": args.k,
        "global_seed": args.seed,
        "model_version": args.version,
        "max_retries": args.max_retries,
        "param_names": param_names,
    }

    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": [list(PARAM_BOUNDS[name]) for name in param_names],
    }

    n_values = [int(x.strip()) for x in args.convergence_n.split(",")] if args.convergence_n else [args.samples]

    worker_count = _resolve_worker_count(args)
    ctx = multiprocessing.get_context("fork" if os.name == "posix" else "spawn")

    print("=" * 70)
    print("ABM Sobol Sensitivity Analysis")
    print("=" * 70)
    print(f"PARAMETERS:      {param_names}")
    print(f"TARGETS:         {target_names}")
    print(f"N VALUES:        {n_values}")
    print(f"REPLICATE SEEDS: {args.seeds}")
    print(f"SECOND-ORDER:    {args.second_order}")
    print(f"K_VOYAGES:       {base_settings['k_voyages']}")
    print(f"WORKERS:         {worker_count} "
          f"(reserve_cores={args.reserve_cores}, total cores={multiprocessing.cpu_count()})")
    print("=" * 70)

    print()
    inputs = load_simulation_inputs(convicts_path, drought_path)

    print()
    print(f"Starting persistent worker pool ({worker_count} processes)...")
    pool = ctx.Pool(processes=worker_count, initializer=_pool_initializer, initargs=(inputs, base_settings))

    all_indices = []
    try:
        for samples in n_values:
            indices_df = _execute_for_n(
                args, param_names, target_names, problem, base_settings,
                results_dir, figures_dir, samples, pool,
            )
            all_indices.append(indices_df)
    finally:
        pool.close()
        pool.join()

    # ------------------------------------------------------------
    # Convergence summary (only meaningful with more than one N)
    # ------------------------------------------------------------
    if len(n_values) > 1:
        convergence_df = pd.concat(all_indices, ignore_index=True)
        convergence_path = os.path.join(
            results_dir, f"sobol_convergence_K{base_settings['k_voyages']}_seeds{args.seeds}.csv"
        )
        convergence_df.to_csv(convergence_path, index=False)
        print()
        print(f"Convergence table written to: {convergence_path}")

        if not args.no_figures:
            file_hashes = [
                _run_signature(base_settings["k_voyages"], n, args.seeds, base_settings["global_seed"],
                            args.second_order, param_names).rsplit("_", 1)[-1]
                for n in n_values
            ]
            figure_label = args.label if args.label else "_".join(sorted(param_names))
            for target_name in target_names:
                result = _plot_convergence(convergence_df, target_name, figures_dir, base_settings["k_voyages"], param_names, n_values, file_hashes, figure_label)
                if result:
                    print(f"Figure written to: {result[0]}")
                    print(f"Figure written to: {result[1]}")

    print()
    print("=" * 70)
    print("SOBOL ANALYSIS COMPLETE")
    print("=" * 70)