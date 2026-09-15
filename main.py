"""
Agent-Based Model — Unified Analysis Entry Point.

This is the single entry point for every analysis mode used in this
project: large-scale robustness testing, K-value structural
sensitivity comparison, Q13 collapse-timing diagnostic checks, Sobol
global sensitivity analysis, SHAP-based surrogate explanation, and
single-variable ablation studies.

Usage
-----
    python3 main.py robustness --n 200000 --batch 200
    python3 main.py sensitivity --n 5000 --reserve-cores 1
    python3 main.py diagnostic --input analysis/robustness_results/run.csv
    python3 main.py sobol --samples 2048
    python3 main.py ztest --input analysis/robustness_results/run.csv
    python3 main.py shap --input analysis/robustness_results/run.csv
    python3 main.py ablation --fix w3=0.0 --n 50000

Design
------
Each analysis mode owns its own command-line arguments and execution
logic in a dedicated runner module under src/runners/. main.py only
registers each runner's subcommand and dispatches to it; it does not
duplicate any argument definitions or orchestration logic. This same
file is used both for local smoke testing and for large unattended
runs on remote compute (e.g. an unattended NAS/server run), so it must
remain the single canonical entry point for both environments.

Note on 'sensitivity' vs 'diagnostic': these are deliberately separate
subcommands backed by separate runner modules. 'sensitivity' re-runs
new simulations to test a structural modeling choice (K_VOYAGES) and
belongs with the paper's Methods / Robustness Checks. 'diagnostic'
runs no simulation at all — it re-examines an existing robustness
results file to check whether the paper's core conclusion survives
excluding a specific, potentially anomalous collapse quarter (Q13),
and belongs with the paper's Discussion / Limitations. They write to
separate output directories (analysis/sensitivity_results/ and
analysis/diagnostic_results/, respectively) for the same reason.

Note on 'ablation': it reuses build_robustness_parser (with a
different subcommand name) instead of maintaining a second, separately
hand-written copy of every robustness argument. This was previously a
duplicated argument list that silently drifted out of sync with
robustness_runner.py's own parser (e.g. --reserve-cores was added to
'robustness' but not to the old, separately-defined 'ablation' parser,
causing an AttributeError at runtime). Reusing the same parser
function means any future argument added to build_robustness_parser is
automatically available under 'ablation' as well, with no separate
edit required here. This requires build_robustness_parser (in
robustness_runner.py) to accept optional `name` and `help_text`
overrides — see that module for the corresponding small signature
change.
"""

import argparse
import sys

from src.runners.robustness_runner import build_robustness_parser, run_robustness
from src.runners.sensitivity_runner import build_sensitivity_parser, run_sensitivity
from src.runners.diagnostic_runner import build_diagnostic_parser, run_diagnostic
from src.runners.sobol_runner import build_sobol_parser, run_sobol
from src.runners.ztest_runner import build_ztest_parser, run_ztest
from src.runners.shap_runner import build_shap_parser, run_shap


# ================================================================
# ABLATION SUBCOMMAND
#
# Ablation reuses the robustness runner's orchestration logic
# unchanged, and now also reuses its argument parser (via the `name`
# override on build_robustness_parser) instead of maintaining a
# separate, hand-written copy of the same arguments. The only
# additions are --fix (required, to specify which parameters to hold
# constant) and a lighter default --n, since ablation runs are
# typically smaller-scale than the main robustness sweep.
# ================================================================

def build_ablation_parser(subparsers):
    """
    Register the 'ablation' subcommand by reusing
    build_robustness_parser's argument set, then adding the --fix
    argument that is unique to ablation runs.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = build_robustness_parser(
        subparsers,
        name="ablation",
        help_text="Robustness run with one or more parameters fixed to a constant value.",
    )

    # Ablation runs are typically exploratory and smaller-scale than
    # the main 200,000-universe robustness sweep; override just the
    # default for --n while keeping every other inherited default.
    parser.set_defaults(n=50_000)

    parser.add_argument(
        "--fix", type=str, action="append", required=True,
        help="A 'parameter=value' pair to fix for every universe. "
             "May be supplied multiple times to fix several parameters at once, "
             "e.g. --fix w3=0.0 --fix despair_rate=0.01",
    )

    return parser


def _parse_fix_arguments(fix_list):
    """
    Convert a list of 'name=value' strings into a param_overrides dict.

    Parameters
    ----------
    fix_list : list of str

    Returns
    -------
    dict
        Mapping of parameter name -> float value.
    """
    overrides = {}

    for item in fix_list:
        if "=" not in item:
            sys.exit(f"ERROR: --fix argument must be in the form name=value, got: {item}")

        name, _, raw_value = item.partition("=")
        name = name.strip()

        if not name:
            sys.exit(f"ERROR: --fix argument is missing a parameter name, got: {item}")

        try:
            value = float(raw_value.strip())
        except ValueError:
            sys.exit(f"ERROR: could not parse numeric value for --fix {item}")

        overrides[name] = value

    return overrides


# ================================================================
# TOP-LEVEL PARSER
# ================================================================

def build_parser():
    """Construct the top-level parser with every registered subcommand."""
    parser = argparse.ArgumentParser(
        description="Agent-Based Model — Unified Analysis Entry Point.",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    build_robustness_parser(subparsers)
    build_sensitivity_parser(subparsers)
    build_diagnostic_parser(subparsers)
    build_sobol_parser(subparsers)
    build_ztest_parser(subparsers)
    build_shap_parser(subparsers)
    build_ablation_parser(subparsers)

    return parser


# ================================================================
# ENTRY POINT
# ================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "robustness":
        run_robustness(args)

    elif args.mode == "sensitivity":
        run_sensitivity(args)

    elif args.mode == "diagnostic":
        run_diagnostic(args)

    elif args.mode == "sobol":
        run_sobol(args)

    elif args.mode == "ztest":
        run_ztest(args)

    elif args.mode == "shap":
        run_shap(args)

    elif args.mode == "ablation":
        param_overrides = _parse_fix_arguments(args.fix)
        run_robustness(args, param_overrides=param_overrides)

    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()