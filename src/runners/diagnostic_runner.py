"""
Q13 Collapse-Timing Diagnostic Runner for the ABM Engine.

This is a diagnostic / outlier-sensitivity check, NOT a model
ablation and NOT a structural sensitivity run (see
sensitivity_runner.py for the K-value comparison, which IS a
structural check and DOES re-run new simulations). This module runs
no simulation at all; it re-examines an EXISTING robustness results
file to answer one specific question:

    "If collapses concentrated at quarter 13 (a potential artifact
     of model timing) are excluded, does the paper's core conclusion
     — Carrot > Stick — still hold?"

A prior ad hoc check only compared the overall collapse distribution
before and after exclusion (median collapse quarter, collapse count).
That is necessary but not sufficient: it does not tell you whether the
paper's actual conclusion survives. This module additionally computes
the Carrot-vs-Stick collapse-rate gap both with and without quarter-13
collapses, and reports whether the direction of that gap is preserved.

Kept as a separate module and separate output directory from
sensitivity_runner.py because the two checks differ in every relevant
way: this one requires no new simulation and no multiprocessing, reads
an existing CSV in seconds, and belongs in a different part of the
paper (Discussion / Limitations / diagnostic appendix) than the K-value
structural comparison (Methods / Robustness Checks).

Outputs:
    analysis/diagnostic_results/  -> Q13 diagnostic summary CSV,
                                      filtered (Q13-excluded) dataset
    analysis/figures/             -> collapse-timing distribution
                                      histogram (Q13 highlighted),
                                      Carrot-Stick gap comparison
                                      (PDF, vector, for direct
                                      manuscript inclusion, and a
                                      300 DPI PNG for previews)

Usage examples:
    # Run full diagnostic analysis on a robustness results CSV
    python main.py diagnostic --input analysis/robustness_results/run.csv

    # Regenerate gap comparison figure from an existing summary CSV (no analysis)
    python main.py diagnostic --figures-only --output analysis/diagnostic_results/q13_diagnostic_summary_*.csv

    # Skip figures entirely (only write CSVs)
    python main.py diagnostic --input analysis/robustness_results/run.csv --no-figures

    # Test a different quarter (e.g., Q12)
    python main.py diagnostic --input analysis/robustness_results/run.csv --quarter 12
"""

import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_style import apply_house_style, FIGSIZE_STANDARD, save_figure, COLORS


# ================================================================
# DEFAULT SETTINGS
# ================================================================

Q13_QUARTER = 13

# Publication figure settings, consistent with sobol_runner.py and
# sensitivity_runner.py.


# ================================================================
# ARGUMENT REGISTRATION
# ================================================================

def build_diagnostic_parser(subparsers):
    """
    Register the 'diagnostic' subcommand.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = subparsers.add_parser(
        "diagnostic",
        help="Post-hoc diagnostic checks on an existing robustness results file "
             "(currently: the Q13 collapse-timing check).",
    )

    parser.add_argument("--input", type=str, default=None,
                         help="Path to an existing robustness results CSV. If not provided, "
                              "uses the most recently modified file in analysis/robustness_results/.")
    parser.add_argument("--quarter", type=int, default=Q13_QUARTER,
                         help=f"The collapse quarter to test for anomalous clustering "
                              f"(default: {Q13_QUARTER}).")
    parser.add_argument("--output", type=str, default=None,
                         help="Path to an existing diagnostic summary CSV for --figures-only mode. "
                              "Must point to a valid summary file from a previous diagnostic run "
                              "(e.g. q13_diagnostic_summary_*.csv).")
    parser.add_argument("--figures-only", action="store_true",
                         help="Skip analysis and (re)generate figures from an existing --output summary CSV. "
                              "Requires --output to point to an existing diagnostic summary CSV.")
    parser.add_argument("--no-figures", action="store_true",
                         help="Skip figure generation and only write the summary CSV.")

    return parser

# ================================================================
# PATHS
# ================================================================

def _project_root():
    """src/runners/diagnostic_runner.py -> src/runners -> src -> project root."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(this_dir))


def _diagnostic_results_dir(project_root):
    results_dir = os.path.join(project_root, "analysis", "diagnostic_results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _figures_dir(project_root):
    figures_dir = os.path.join(project_root, "analysis", "figures", "diagnostic")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def _find_latest_robustness_file(project_root):
    """Locate the most recently modified robustness results CSV."""
    results_dir = os.path.join(project_root, "analysis", "robustness_results")
    if not os.path.exists(results_dir):
        return None

    csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv") and "simulation_results" in f]
    if not csv_files:
        return None

    csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)
    return os.path.join(results_dir, csv_files[0])

def _extract_run_tag(input_path):
    """
    Extract a run tag from the input filename, matching the naming
    convention used by robustness_runner.py's own output figures and
    shap_runner.py (e.g. "K30_v1_20260803_024351" from
    "simulation_results_K30_v1_20260803_024351.csv"). Falls back to
    the input file's stem if the expected pattern is not found.
    """
    basename = os.path.basename(input_path)
    match = re.match(r"simulation_results_(.+)\.csv$", basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]


# ================================================================
# ANALYSIS
# ================================================================

def _compute_carrot_stick_gap(df):
    """Return (carrot_rate, stick_rate, gap) collapse-rate percentages for a dataframe."""
    if "intervention_type" not in df.columns or len(df) == 0:
        return np.nan, np.nan, np.nan

    carrot = df[df["intervention_type"] == 1]
    stick = df[df["intervention_type"] == 0]
    carrot_rate = (carrot["collapse_quarter"] > 0).mean() * 100 if len(carrot) > 0 else np.nan
    stick_rate = (stick["collapse_quarter"] > 0).mean() * 100 if len(stick) > 0 else np.nan
    return carrot_rate, stick_rate, stick_rate - carrot_rate


# ================================================================
# FIGURES (PUBLICATION QUALITY)
# ================================================================

def _plot_distribution(df, quarter, figures_dir, run_tag):
    """Histogram of collapse_quarter, highlighting the tested quarter, to check for anomalous clustering."""
    collapsed = df[df["collapse_quarter"] > 0]
    if collapsed.empty:
        return None

    max_quarter = int(collapsed["collapse_quarter"].max())
    counts = collapsed["collapse_quarter"].value_counts().reindex(range(1, max_quarter + 1), fill_value=0)

    apply_house_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    colors = [COLORS["red"] if q == quarter else COLORS["blue"] for q in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_xlabel("Collapse quarter")
    ax.set_ylabel("Number of universes")
    ax.set_title(f"Distribution of collapse timing (quarter {quarter} highlighted)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    pdf_path = os.path.join(figures_dir, f"q{quarter}_distribution_{run_tag}.pdf")
    png_path = os.path.join(figures_dir, f"q{quarter}_distribution_{run_tag}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)

    return pdf_path, png_path

def _plot_gap_comparison(gap_with, gap_without, quarter, figures_dir, run_tag):
    """Bar chart: Carrot-Stick gap before vs. after exclusion — the key diagnostic figure."""
    apply_house_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    labels = [f"All collapses\n(including Q{quarter})", f"Excluding Q{quarter}"]
    values = [gap_with, gap_without]

    x_pos = [0, 0.65]  
    ax.bar(x_pos, values, color=["#4C72B0", "#DD8452"], width=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.45, 1.1)  

    ax.set_ylabel("Stick − Carrot collapse rate (pp)")
    ax.set_title(f"Carrot vs. Stick gap: with vs. without Q{quarter}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    pdf_path = os.path.join(figures_dir, f"q{quarter}_carrot_stick_gap_{run_tag}.pdf")
    png_path = os.path.join(figures_dir, f"q{quarter}_carrot_stick_gap_{run_tag}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)

    return pdf_path, png_path

# ================================================================
# MAIN EXECUTION
# ================================================================

def run_diagnostic(args):
    """
    Execute the Q13 (or --quarter N) collapse-timing diagnostic check
    on an existing robustness results file: no new simulation is run.

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
        print("FIGURES-ONLY MODE: Regenerating figures from existing summary CSV")
        print("=" * 70)

        if not args.output or not os.path.exists(args.output):
            sys.exit(
                f"ERROR: --figures-only requires --output pointing to an existing diagnostic summary CSV.\n"
                f"Got: {args.output or 'not provided'}\n"
                f"Expected: q{args.quarter}_diagnostic_summary_*.csv"
            )

        print(f"--figures-only: loading {args.output}")
        summary_df = pd.read_csv(args.output)

        # Extract required values from the summary
        required_cols = ["quarter_tested", "carrot_stick_gap_all_pp", "carrot_stick_gap_excluded_pp"]
        missing_cols = [c for c in required_cols if c not in summary_df.columns]
        if missing_cols:
            sys.exit(
                f"ERROR: {args.output} does not appear to be a valid diagnostic summary CSV.\n"
                f"Missing columns: {missing_cols}\n"
                f"Expected columns: {required_cols}"
            )

        # Get the first row (summary is one row)
        row = summary_df.iloc[0]
        quarter = int(row["quarter_tested"])
        gap_with = float(row["carrot_stick_gap_all_pp"])
        gap_without = float(row["carrot_stick_gap_excluded_pp"])
        run_tag = row.get("run_tag", "diagnostic")

        project_root = _project_root()
        figures_dir = _figures_dir(project_root)

        print(f"Quarter tested: Q{quarter}")
        print(f"Gap (all): {gap_with:.2f} pp")
        print(f"Gap (excluded): {gap_without:.2f} pp")
        print(f"Conclusion survives: {(gap_with > 0) == (gap_without > 0)}")

        if not args.no_figures:
            # Regenerate gap comparison figure
            result = _plot_gap_comparison(gap_with, gap_without, quarter, figures_dir, run_tag)
            if result:
                print(f"Figure written to: {result[0]}")
                print(f"Figure written to: {result[1]}")

            # For distribution histogram, we need the original data
            # The summary CSV doesn't contain the full distribution data
            # Check if we can infer from the summary or warn the user
            print("NOTE: Distribution histogram requires the original robustness CSV.")
            print("      Run without --figures-only to generate the distribution histogram.")
            print("      Or use --input to point to the original CSV.")

        print("=" * 70)
        print("FIGURES-ONLY MODE COMPLETE")
        print("=" * 70)
        return

    # ================================================================
    # NORMAL EXECUTION (existing code continues below)
    # ================================================================
    project_root = _project_root()
    results_dir = _diagnostic_results_dir(project_root)
    figures_dir = _figures_dir(project_root)
    quarter = args.quarter

    input_file = args.input or _find_latest_robustness_file(project_root)

    print("=" * 70)
    print(f"Q{quarter} COLLAPSE-TIMING DIAGNOSTIC CHECK")
    print(f"Question: does the Carrot > Stick conclusion still hold if quarter-{quarter}")
    print("collapses (a potential artifact of model timing) are excluded?")
    print("=" * 70)

    if not input_file or not os.path.exists(input_file):
        print("WARNING: no robustness results file found. Use --input to specify one. Aborting.")
        return

    run_tag = _extract_run_tag(input_file)
    print(f"Input: {input_file}")
    print(f"Run tag: {run_tag}")
    df = pd.read_csv(input_file)

    collapsed_all = df[df["collapse_quarter"] > 0]
    q_collapse = df[df["collapse_quarter"] == quarter]
    df_excluded = df[df["collapse_quarter"] != quarter]

    carrot_rate_with, stick_rate_with, gap_with = _compute_carrot_stick_gap(df)
    carrot_rate_without, stick_rate_without, gap_without = _compute_carrot_stick_gap(df_excluded)

    summary = {
        "quarter_tested": quarter,
        "run_tag": run_tag,
        "total_universes": len(df),
        "q_collapse": len(q_collapse),
        "q_pct_of_total": len(q_collapse) / len(df) * 100 if len(df) > 0 else np.nan,
        "q_pct_of_collapses": len(q_collapse) / len(collapsed_all) * 100 if len(collapsed_all) > 0 else np.nan,
        "collapse_rate_all_pct": (df["collapse_quarter"] > 0).mean() * 100,
        "collapse_rate_excluded_pct": (df_excluded["collapse_quarter"] > 0).mean() * 100,
        "median_collapse_quarter_all": collapsed_all["collapse_quarter"].median() if not collapsed_all.empty else np.nan,
        "median_collapse_quarter_excluded": (
            df_excluded.loc[df_excluded["collapse_quarter"] > 0, "collapse_quarter"].median()
            if (df_excluded["collapse_quarter"] > 0).any() else np.nan
        ),
        "carrot_collapse_rate_all_pct": carrot_rate_with,
        "stick_collapse_rate_all_pct": stick_rate_with,
        "carrot_stick_gap_all_pp": gap_with,
        "carrot_collapse_rate_excluded_pct": carrot_rate_without,
        "stick_collapse_rate_excluded_pct": stick_rate_without,
        "carrot_stick_gap_excluded_pp": gap_without,
        "conclusion_direction_survives": bool(
            np.isfinite(gap_with) and np.isfinite(gap_without)
            and (gap_with > 0) == (gap_without > 0)
        ),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(results_dir, f"q{quarter}_diagnostic_summary_{run_tag}_{timestamp}.csv")
    filtered_path = os.path.join(results_dir, f"q{quarter}_diagnostic_excluded_{run_tag}_{timestamp}.csv")

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    df_excluded.to_csv(filtered_path, index=False)

    print()
    print(f"Total universes:                    {summary['total_universes']:,}")
    print(f"Q{quarter} collapses:                      {summary['q_collapse']:,} "
          f"({summary['q_pct_of_total']:.2f}% of all universes, "
          f"{summary['q_pct_of_collapses']:.2f}% of all collapses)")
    print(f"Collapse rate (all / excluding Q{quarter}): "
          f"{summary['collapse_rate_all_pct']:.2f}% / {summary['collapse_rate_excluded_pct']:.2f}%")
    print(f"Median collapse quarter (all / excluding Q{quarter}): "
          f"{summary['median_collapse_quarter_all']} / {summary['median_collapse_quarter_excluded']}")
    print(f"Carrot-Stick gap (all / excluding Q{quarter}, pp): {gap_with:.2f} / {gap_without:.2f}")
    print(f"Conclusion direction (Carrot > Stick) survives exclusion: {summary['conclusion_direction_survives']}")
    print()
    print(f"Summary written to: {summary_path}")
    print(f"Filtered dataset written to: {filtered_path}")

    if not args.no_figures:
        result = _plot_distribution(df, quarter, figures_dir, run_tag)
        if result:
            print(f"Figure written to: {result[0]}")
            print(f"Figure written to: {result[1]}")

        result = _plot_gap_comparison(gap_with, gap_without, quarter, figures_dir, run_tag)
        if result:
            print(f"Figure written to: {result[0]}")
            print(f"Figure written to: {result[1]}")

    print()
    print("=" * 70)
    print("Q13 DIAGNOSTIC CHECK COMPLETE")
    print("=" * 70)