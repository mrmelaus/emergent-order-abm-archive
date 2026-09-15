"""
K-Value Structural Sensitivity Runner for the ABM Engine.

Tests whether the paper's key conclusions depend on a modeling choice
that is NOT part of the model's stochastic parameters (those are
covered by 'robustness' and 'sobol'): the number of voyages sampled
per universe (K_VOYAGES).

Re-runs the standard robustness simulation at several K values (e.g.
K=10, K=30, K=50), each at the SAME sample size (default N=5000), so
the comparison is not confounded by sample size. K=30 is included as
the baseline that matches the paper's main 200,000-universe
robustness run, so this is a like-for-like structural comparison, not
a comparison against a differently-sized dataset.

This module is deliberately independent from diagnostic_runner.py
(the Q13 collapse-timing check): that check is a post-hoc examination
of an EXISTING results file, requires no new simulation, and answers
a different kind of question ("does an apparent data artifact affect
the conclusion?" rather than "does a structural modeling choice affect
the conclusion?"). Keeping them in separate modules with separate
output directories keeps each one independently reusable and keeps
the paper's Methods (K comparison) and Discussion/Limitations (Q13)
sections backed by clearly separated artifacts.

Worker-count handling mirrors sobol_runner.py: every core is used by
default (correct for an unattended NAS/server run), and --reserve-cores lets
an interactive user (e.g. running on a personal machine) leave cores
free for the rest of the system.

Outputs:
    analysis/sensitivity_results/  -> per-K raw robustness CSVs,
                                       K-comparison summary CSV
    analysis/figures/              -> K-comparison bar chart
                                       (PDF, vector, for direct
                                       manuscript inclusion, and a
                                       300 DPI PNG for previews)

Usage examples:
    # Run full K-value sensitivity comparison (simulation + figures)
    python main.py sensitivity --n 5000 --k-values 10,30,50

    # Regenerate comparison figure from existing summary CSV (no simulation)
    python main.py sensitivity --figures-only --output analysis/sensitivity_results/k_sensitivity_summary_K10-30-50_n5000_seed7777.csv

    # Skip figures entirely (simulation only)
    python main.py sensitivity --no-figures --n 5000 --k-values 10,30,50
"""

import multiprocessing
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_style import apply_house_style, FIGSIZE_STANDARD, FIGSIZE_WIDE, save_figure, COLORS


# ================================================================
# DEFAULT SETTINGS
# ================================================================

DEFAULT_K_VALUES = (10, 30, 50)
DEFAULT_N = 5000
DEFAULT_BATCH = 100
DEFAULT_GLOBAL_SEED = 7777

# ================================================================
# ARGUMENT REGISTRATION
# ================================================================

def build_sensitivity_parser(subparsers):
    """
    Register the 'sensitivity' subcommand (K-value comparison only).

    Parameters
    ----------
    subparsers : argparse._SubParsersAction

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = subparsers.add_parser(
        "sensitivity",
        help="K-value structural sensitivity comparison (K=10/30/50 by default).",
    )

    parser.add_argument("--n", type=int, default=DEFAULT_N,
                         help=f"Number of universes for each K-value run (default: {DEFAULT_N}).")
    parser.add_argument("--k-values", type=str, default=",".join(str(k) for k in DEFAULT_K_VALUES),
                         help=f"Comma-separated K values to compare, should include the paper's "
                              f"baseline (default: {','.join(str(k) for k in DEFAULT_K_VALUES)}).")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                         help=f"Batch size passed through to each robustness run (default: {DEFAULT_BATCH}).")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of multiprocessing workers passed through to each robustness "
                              "run. Default: all available cores minus --reserve-cores.")
    parser.add_argument("--reserve-cores", type=int, default=0,
                         help="Number of CPU cores to leave free for the rest of the system. "
                              "Use e.g. --reserve-cores 1 on a machine you are using interactively. "
                              "Ignored if --workers is explicitly set. Default 0 (use every core), "
                              "correct for an unattended NAS/server run.")
    parser.add_argument("--seed", type=int, default=DEFAULT_GLOBAL_SEED,
                         help=f"Global random seed for each K-value run (default: {DEFAULT_GLOBAL_SEED}).")
    parser.add_argument("--fresh", action="store_true",
                         help="Force fresh starts for K-value runs, ignoring any existing output files.")
    parser.add_argument("--no-figures", action="store_true",
                         help="Skip figure generation and only write the CSVs.")
    parser.add_argument("--figures-only", action="store_true",
                         help="Skip simulation entirely and (re)generate figures from an existing CSV. "
                              "Requires --k-values to be a single value (e.g. --k-values 30) pointing to "
                              "an existing results file. For multi-K comparison figures, run without --figures-only.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path to use for --figures-only. "
                             "For a single K, point to the raw CSV (e.g. k30_*.csv). "
                             "For the comparison figure, point to the summary CSV "
                             "(e.g. k_sensitivity_summary_*.csv). Default: auto-generated.")

    return parser


# ================================================================
# PATHS
# ================================================================

def _project_root():
    """src/runners/sensitivity_runner.py -> src/runners -> src -> project root."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(this_dir))


def _sensitivity_results_dir(project_root):
    results_dir = os.path.join(project_root, "analysis", "sensitivity_results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _figures_dir(project_root):
    figures_dir = os.path.join(project_root, "analysis", "figures", "sensitivity")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def _resolve_worker_count(args):
    """Every core by default; --reserve-cores leaves cores free for an interactive machine."""
    if args.workers is not None:
        return max(1, args.workers)
    return max(1, multiprocessing.cpu_count() - args.reserve_cores)


# ================================================================
# K-VALUE RUNS
# ================================================================

def _run_robustness_for_k(project_root, k_value, args, output_dir, worker_count):
    """
    Run (or resume) a robustness simulation at one K value via
    subprocess, reusing the 'robustness' subcommand's own resume
    logic so an interrupted K-value run can simply be rerun.
    """
    output_file = os.path.join(output_dir, f"k{k_value}_n{args.n}_seed{args.seed}.csv")
    complete_file = output_file + ".complete"

    if os.path.exists(complete_file):
        print(f"K={k_value} already complete. Skipping.")
        return output_file

    if args.fresh and os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed existing incomplete file for K={k_value}: {output_file}")

    print()
    print("=" * 70)
    print(f"Running K={k_value} ({args.n:,} universes)")
    print(f"Output: {output_file}")
    print("=" * 70)

    cmd = [
        "python3", "main.py", "robustness",
        "--k", str(k_value),
        "--n", str(args.n),
        "--batch", str(args.batch),
        "--workers", str(worker_count),
        "--seed", str(args.seed),
        "--output", output_file,
    ]

    if args.fresh:
        cmd.append("--fresh")

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode != 0:
        print(f"WARNING: K={k_value} run exited with code {result.returncode}. "
              f"Rerun the same 'sensitivity' command to resume this K value.")

    return output_file


def _summarize_k_run(output_file, k_value, n):
    """Compute the comparison metrics used to judge structural sensitivity to K."""
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return {"k": k_value, "n_requested": n, "status": "missing"}

    df = pd.read_csv(output_file)
    summary = {
        "k": k_value,
        "n_requested": n,
        "n_completed": len(df),
        "status": "ok",
        "collapse_rate_pct": (df["collapse_quarter"] > 0).mean() * 100,
    }

    collapsed = df[df["collapse_quarter"] > 0]
    summary["median_collapse_quarter"] = collapsed["collapse_quarter"].median() if not collapsed.empty else np.nan

    if "intervention_type" in df.columns:
        carrot = df[df["intervention_type"] == 1]
        stick = df[df["intervention_type"] == 0]
        carrot_rate = (carrot["collapse_quarter"] > 0).mean() * 100 if len(carrot) > 0 else np.nan
        stick_rate = (stick["collapse_quarter"] > 0).mean() * 100 if len(stick) > 0 else np.nan
        summary["carrot_collapse_rate_pct"] = carrot_rate
        summary["stick_collapse_rate_pct"] = stick_rate
        summary["carrot_stick_gap_pct"] = stick_rate - carrot_rate

    return summary

def _load_existing_k_run(results_dir, k_value, n, seed):
    """
    Load a single existing K-value CSV and return the data and summary.

    Used by --figures-only mode.

    Parameters
    ----------
    results_dir : str
        Directory containing the CSV files.
    k_value : int
        K value to load.
    n : int
        Number of universes (used for filename).
    seed : int
        Random seed (used for filename).

    Returns
    -------
    tuple
        (pd.DataFrame, dict) - The loaded data and its summary.

    Raises
    ------
    SystemExit
        If the CSV file does not exist.
    """
    csv_path = os.path.join(results_dir, f"k{k_value}_n{n}_seed{seed}.csv")

    if not os.path.exists(csv_path):
        sys.exit(
            f"ERROR: --figures-only requires CSV file to exist.\n"
            f"Missing: {csv_path}\n"
            f"(Run sensitivity without --figures-only to generate this file.)"
        )

    print(f"--figures-only: loading {csv_path}")
    df = pd.read_csv(csv_path)
    summary = _summarize_k_run(csv_path, k_value, n)

    return df, summary


# ================================================================
# FIGURE (PUBLICATION QUALITY)
# ================================================================

def _plot_k_comparison(comparison_df, figures_dir, run_tag):
    """
    Grouped bar chart: collapse rate and Carrot-Stick gap across K
    values. Saved as both a vector PDF (for direct manuscript
    inclusion) and a 300 DPI PNG (for previews/slides).
    """
    if comparison_df.empty or "carrot_stick_gap_pct" not in comparison_df.columns:
        return None

    apply_house_style()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    x = np.arange(len(comparison_df))
    labels = [f"K={int(k)}" for k in comparison_df["k"]]

    axes[0].bar(x, comparison_df["collapse_rate_pct"], color=COLORS["blue"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Collapse rate (%)")
    axes[0].set_title("Overall collapse rate")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].bar(x, comparison_df["carrot_stick_gap_pct"], color=COLORS["orange"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Stick − Carrot collapse rate (pp)")
    axes[1].set_title("Carrot vs. Stick gap")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    pdf_path = os.path.join(figures_dir, f"k_sensitivity_comparison_{run_tag}.pdf")
    png_path = os.path.join(figures_dir, f"k_sensitivity_comparison_{run_tag}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)

    return pdf_path, png_path


# ================================================================
# MAIN EXECUTION
# ================================================================

def run_sensitivity(args):
    """
    Execute the K-value structural comparison: run (or resume) a
    robustness simulation at each requested K value, summarize the
    key comparison metrics, and produce a citation-ready figure.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    None
    """
    project_root = _project_root()
    results_dir = _sensitivity_results_dir(project_root)
    figures_dir = _figures_dir(project_root)
    worker_count = _resolve_worker_count(args)
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]

    kvals_tag = "-".join(str(k) for k in k_values)
    run_tag = f"K{kvals_tag}_n{args.n}_seed{args.seed}"

    # ================================================================
    # FIGURES-ONLY MODE
    # ================================================================
    if args.figures_only:
        print("=" * 70)
        print("FIGURES-ONLY MODE: Regenerating comparison figure from existing summary CSV")
        print("=" * 70)

        if not args.output or not os.path.exists(args.output):
            sys.exit(
                f"ERROR: --figures-only requires --output pointing to an existing summary CSV.\n"
                f"Got: {args.output or 'not provided'}\n"
                f"Expected: k_sensitivity_summary_*.csv (generated during normal sensitivity runs)"
            )

        # Enforce that only summary CSV is allowed
        basename = os.path.basename(args.output)
        if not basename.startswith("k_sensitivity_summary_"):
            sys.exit(
                f"ERROR: --figures-only for sensitivity requires the summary CSV.\n"
                f"Got: {basename}\n"
                f"Expected: k_sensitivity_summary_*.csv\n"
                f"This file is generated automatically during normal sensitivity runs.\n"
                f"Hint: Run 'python main.py sensitivity --n 5000 --k-values 10,30,50' first."
            )

        print(f"--figures-only: loading {args.output}")
        comparison_df = pd.read_csv(args.output)

        # Validate summary has required columns
        required_cols = ["k", "collapse_rate_pct", "carrot_stick_gap_pct"]
        missing_cols = [c for c in required_cols if c not in comparison_df.columns]
        if missing_cols:
            sys.exit(
                f"ERROR: {args.output} does not appear to be a valid summary CSV.\n"
                f"Missing columns: {missing_cols}\n"
                f"Expected columns: {required_cols}"
            )

        print(f"Loaded summary with {len(comparison_df)} K-values: {comparison_df['k'].tolist()}")

        if not args.no_figures:
            result = _plot_k_comparison(comparison_df, figures_dir, run_tag)
            if result:
                print(f"Figure written to: {result[0]}")
                print(f"Figure written to: {result[1]}")

        print("=" * 70)
        print("FIGURES-ONLY MODE COMPLETE")
        print("=" * 70)
        return

    # ================================================================
    # NORMAL EXECUTION (Run simulations, then generate figures)
    # ================================================================

    print("=" * 70)
    print("K-VALUE STRUCTURAL SENSITIVITY COMPARISON")
    print("=" * 70)
    print(f"K values: {k_values}")
    print(f"N per K:  {args.n:,}")
    print(f"Run tag:  {run_tag}")
    print(f"Workers:  {worker_count} (reserve_cores={args.reserve_cores}, "
          f"total cores={multiprocessing.cpu_count()})")
    print(f"Results directory: {results_dir}")
    print("=" * 70)

    summaries = []
    for k_value in k_values:
        output_file = _run_robustness_for_k(project_root, k_value, args, results_dir, worker_count)
        summaries.append(_summarize_k_run(output_file, k_value, args.n))

    comparison_df = pd.DataFrame(summaries)
    comparison_path = os.path.join(results_dir, f"k_sensitivity_summary_{run_tag}.csv")
    comparison_df.to_csv(comparison_path, index=False)

    print()
    print(f"K-comparison summary written to: {comparison_path}")
    print(comparison_df.to_string(index=False))

    if not args.no_figures:
        result = _plot_k_comparison(comparison_df, figures_dir, run_tag)
        if result:
            print(f"Figure written to: {result[0]}")
            print(f"Figure written to: {result[1]}")

    print()
    print("=" * 70)
    print("K-VALUE SENSITIVITY COMPARISON COMPLETE")
    print("=" * 70)