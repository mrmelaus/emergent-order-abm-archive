"""
Two-Proportion Z-Test Runner for the ABM Engine.

Answers: "is this difference in collapse rates between two groups
large enough to be unlikely under pure chance, or could it plausibly
be random noise given the sample sizes involved?"

This is a post-hoc statistical check on an EXISTING robustness
results CSV — no new simulation is run, and only ONE input file is
required. All four comparison groups (Carrot, Stick, Drought On,
Drought Off, and their four combinations) already exist as columns
(intervention_type, drought_enabled) within that single file; no
second data source is needed.

For each comparison, this computes the standard two-proportion
z-test (pooled variance, two-tailed):

    p1 = x1 / n1                      (collapse rate of group 1)
    p2 = x2 / n2                      (collapse rate of group 2)
    p_pool = (x1 + x2) / (n1 + n2)    (pooled collapse rate)
    se = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    p_value = 2 * (1 - Phi(|z|))      (two-tailed)

A small p-value (conventionally < 0.05) means the observed difference
is unlikely to have arisen purely by chance, given the sample sizes
involved. This is a standard, uncontroversial test for comparing two
proportions.

Performance note: this reads one CSV and computes simple boolean
sums/means over it — no multiprocessing, no training, no simulation.
Runs in a few seconds even on the full 200,000-row main results file;
no --reserve-cores / --workers / caffeinate needed.

Outputs:
    analysis/ztest_results/  -> comparison results CSV
"""

import os
import re
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm
except ImportError:
    norm = None


# ================================================================
# ARGUMENT REGISTRATION
# ================================================================

def build_ztest_parser(subparsers):
    """
    Register the 'ztest' subcommand.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = subparsers.add_parser(
        "ztest",
        help="Two-proportion z-tests for collapse-rate comparisons on an existing robustness results CSV.",
    )

    parser.add_argument("--input", type=str, default=None,
                         help="Path to an existing robustness results CSV. If not provided, uses the "
                              "most recently modified file in analysis/robustness_results/.")

    return parser


def _validate_args(args):
    if norm is None:
        sys.exit("ERROR: scipy is required. Install it with: pip install scipy --break-system-packages")
    if args.input and not os.path.exists(args.input):
        sys.exit(f"ERROR: --input file not found: {args.input}")


# ================================================================
# PATHS / RUN TAG
# ================================================================

def _project_root():
    """src/runners/ztest_runner.py -> src/runners -> src -> project root."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(this_dir))


def _ztest_results_dir(project_root):
    results_dir = os.path.join(project_root, "analysis", "ztest_results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


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
    """Same convention used by diagnostic_runner.py / shap_runner.py."""
    basename = os.path.basename(input_path)
    match = re.match(r"simulation_results_(.+)\.csv$", basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]


# ================================================================
# STATISTICAL TEST
# ================================================================

def _two_proportion_ztest(x1, n1, x2, n2):
    """
    Standard two-proportion z-test (pooled variance, two-tailed).

    Parameters
    ----------
    x1, n1 : int
        Successes (collapses) and total trials for group 1.
    x2, n2 : int
        Successes (collapses) and total trials for group 2.

    Returns
    -------
    dict
        p1, p2 (as percentages), diff_pp (percentage points, group1 -
        group2), z, p_value, significant_at_0.05.
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se == 0:
        z = np.nan
        p_value = np.nan
    else:
        z = (p1 - p2) / se
        p_value = 2 * (1 - norm.cdf(abs(z)))

    return {
        "p1_pct": p1 * 100, "n1": n1,
        "p2_pct": p2 * 100, "n2": n2,
        "diff_pp": (p1 - p2) * 100,
        "z": z, "p_value": p_value,
        "significant_at_0.05": bool(np.isfinite(p_value) and p_value < 0.05),
    }


def _collapse_count_and_n(df):
    """Return (collapse_count, total_n) for a subgroup dataframe."""
    n = len(df)
    x = int((df["collapse_quarter"] > 0).sum())
    return x, n


# ================================================================
# MAIN EXECUTION
# ================================================================

def run_ztest(args):
    """
    Execute the two-proportion z-test battery on an existing
    robustness results CSV: no new simulation is run.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    None
    """
    _validate_args(args)

    project_root = _project_root()
    results_dir = _ztest_results_dir(project_root)

    input_file = args.input or _find_latest_robustness_file(project_root)

    print("=" * 70)
    print("TWO-PROPORTION Z-TESTS: COLLAPSE RATE COMPARISONS")
    print("=" * 70)

    if not input_file or not os.path.exists(input_file):
        print("WARNING: no robustness results file found. Use --input to specify one. Aborting.")
        return

    run_tag = _extract_run_tag(input_file)
    print(f"Input: {input_file}")
    print(f"Run tag: {run_tag}")
    print("=" * 70)

    df = pd.read_csv(input_file)
    required_cols = {"collapse_quarter", "intervention_type", "drought_enabled"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"ERROR: input CSV is missing required columns: {missing}")

    carrot = df[df["intervention_type"] == 1]
    stick = df[df["intervention_type"] == 0]
    drought_off = df[df["drought_enabled"] == 0]
    drought_on = df[df["drought_enabled"] == 1]

    carrot_off = df[(df["intervention_type"] == 1) & (df["drought_enabled"] == 0)]
    carrot_on = df[(df["intervention_type"] == 1) & (df["drought_enabled"] == 1)]
    stick_off = df[(df["intervention_type"] == 0) & (df["drought_enabled"] == 0)]
    stick_on = df[(df["intervention_type"] == 0) & (df["drought_enabled"] == 1)]

    comparisons = {
        "Carrot vs Stick (overall, Layer 3 main effect)": (carrot, stick),
        "Drought Off vs On (overall, Layer 2 main effect)": (drought_off, drought_on),
        "Carrot+DroughtOn vs Stick+DroughtOn (interaction cell)": (carrot_on, stick_on),
        "Carrot+DroughtOff vs Stick+DroughtOff (interaction cell)": (carrot_off, stick_off),
        "Carrot: DroughtOff vs DroughtOn (within-Carrot drought effect)": (carrot_off, carrot_on),
        "Stick: DroughtOff vs DroughtOn (within-Stick drought effect)": (stick_off, stick_on),
    }

    rows = []
    for label, (group1, group2) in comparisons.items():
        x1, n1 = _collapse_count_and_n(group1)
        x2, n2 = _collapse_count_and_n(group2)
        result = _two_proportion_ztest(x1, n1, x2, n2)
        result["comparison"] = label
        rows.append(result)

        print()
        print(label)
        print(f"  Group 1: {result['p1_pct']:.2f}% (n={n1:,})")
        print(f"  Group 2: {result['p2_pct']:.2f}% (n={n2:,})")
        print(f"  Difference: {result['diff_pp']:.2f} percentage points")
        if np.isfinite(result["z"]):
            print(f"  z = {result['z']:.3f}, p = {result['p_value']:.2e}, "
                  f"significant at 0.05: {result['significant_at_0.05']}")
        else:
            print("  z-test undefined (zero variance in pooled proportion; "
                  "both groups identical and at 0% or 100%).")

    results_df = pd.DataFrame(rows)[
        ["comparison", "p1_pct", "n1", "p2_pct", "n2", "diff_pp", "z", "p_value", "significant_at_0.05"]
    ]

    output_path = os.path.join(results_dir, f"ztest_{run_tag}.csv")
    results_df.to_csv(output_path, index=False)

    print()
    print("=" * 70)
    print(f"Results written to: {output_path}")
    print("=" * 70)