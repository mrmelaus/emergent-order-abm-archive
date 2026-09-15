"""
Robustness Test Runner for the ABM Engine.

Orchestrates large-scale random-parameter robustness simulations.

Responsibilities
----------------
1. Load and validate model inputs (delegated to src.data_loader).
2. Determine the multiprocessing context according to the operating
   system.
3. Generate deterministic universe-level random seeds.
4. Execute universes in parallel batches.
5. Save completed universes incrementally to CSV.
6. Resume incomplete simulations using universe_id.
7. Retry failed universes without changing their assigned random seed.
8. Validate output structure before appending.
9. Create a completion marker only after all requested universes are
   verified.
10. Generate publication-quality summary figures (Carrot vs. Stick
    collapse rate, drought-condition collapse rate, and the
    collapse-timing distribution) once the run completes.

The model version is defined here and passed explicitly to worker.py;
worker.py does not define its own model version.

This module exposes two entry points used by main.py:

    build_robustness_parser(subparsers, name="robustness", help_text=...)
        Registers a subcommand and all of its command-line arguments.
        The `name`/`help_text` overrides let other subcommands (e.g.
        'ablation' in main.py) reuse this exact argument set under a
        different subcommand name, instead of maintaining a second,
        separately hand-written copy that can silently drift out of
        sync (this previously caused a --reserve-cores AttributeError
        under 'ablation' after it was added only to 'robustness').

    run_robustness(args, param_overrides=None)
        Executes the robustness run described by the parsed
        arguments. `param_overrides` allows individual sampled
        parameters to be fixed to a constant value for every universe,
        which supports single-variable ablation studies without
        duplicating any simulation or orchestration logic.
"""

import csv
import glob
import multiprocessing
import os
import platform
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import re

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from .plot_style import apply_house_style, FIGSIZE_STANDARD, save_figure, COLORS

from src.data_loader import load_simulation_inputs, default_data_paths
from src.worker import execute_universe_worker
from src.config import K_VOYAGES as DEFAULT_K


# ================================================================
# DEFAULT SETTINGS
# ================================================================

DEFAULT_TARGET_UNIVERSES = 200_000
DEFAULT_BATCH_SIZE = 200
DEFAULT_GLOBAL_SEED = 7777
DEFAULT_MAX_RETRIES = 3
DEFAULT_MODEL_VERSION = "v1"

# ================================================================
# ARGUMENT REGISTRATION
# ================================================================

def build_robustness_parser(subparsers, name="robustness",
                             help_text="Large-scale random-parameter robustness simulation."):
    """
    Register a robustness-style subcommand on the shared argparse
    subparsers object owned by main.py.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object created by the top-level parser.

    name : str, optional
        The subcommand name to register (default: "robustness").
        Overridden by main.py's 'ablation' subcommand to reuse this
        entire argument set under a different name, instead of
        maintaining a separate, hand-written copy.

    help_text : str, optional
        The help string shown for this subcommand.

    Returns
    -------
    argparse.ArgumentParser
        The registered subparser, in case further customization is
        needed by the caller (e.g. main.py's 'ablation' subcommand
        overrides the --n default and adds --fix).
    """
    parser = subparsers.add_parser(
        name,
        help=help_text,
    )

    parser.add_argument("--k", type=int, default=DEFAULT_K,
                         help=f"Number of voyages sampled per universe (default: {DEFAULT_K}).")
    parser.add_argument("--n", type=int, default=DEFAULT_TARGET_UNIVERSES,
                         help=f"Target number of universes (default: {DEFAULT_TARGET_UNIVERSES}).")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                         help=f"Number of universes submitted per batch (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of multiprocessing workers. Default: all available cores "
                              "minus --reserve-cores.")
    parser.add_argument("--reserve-cores", type=int, default=0,
                         help="Number of CPU cores to leave free for the rest of the system. "
                              "Use e.g. --reserve-cores 1 on a machine you are using interactively. "
                              "Ignored if --workers is explicitly set. Default 0 (use every core), "
                              "correct for an unattended NAS/server run.")
    parser.add_argument("--seed", type=int, default=DEFAULT_GLOBAL_SEED,
                         help=f"Global random seed (default: {DEFAULT_GLOBAL_SEED}).")
    parser.add_argument("--output", type=str, default=None,
                         help="Output CSV path. Default: automatically generated.")
    parser.add_argument("--version", type=str, default=DEFAULT_MODEL_VERSION,
                         help=f"Model version tag (default: {DEFAULT_MODEL_VERSION}).")
    parser.add_argument("--mp-context", choices=("auto", "fork", "spawn", "forkserver"), default="auto",
                         help="Multiprocessing start method. Default 'auto': fork on Linux, spawn otherwise.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                         help=f"Maximum attempts for a failed universe (default: {DEFAULT_MAX_RETRIES}).")
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete run.")
    parser.add_argument("--fresh", action="store_true", help="Force a new run and ignore incomplete files.")
    parser.add_argument("--no-figures", action="store_true",
                         help="Skip figure generation and only write the CSV.")
    parser.add_argument("--figures-only", action="store_true",
                     help="Skip the simulation entirely and (re)generate figures from an existing "
                          "--output CSV. Requires --output to point to an existing results file.")

    return parser


def _validate_args(args):
    """Validate resolved robustness settings; exits with an error message on invalid input."""
    if args.k <= 0:
        sys.exit("ERROR: --k must be greater than zero.")
    if args.n <= 0:
        sys.exit("ERROR: --n must be greater than zero.")
    if args.batch <= 0:
        sys.exit("ERROR: --batch must be greater than zero.")
    if args.workers is not None and args.workers <= 0:
        sys.exit("ERROR: --workers must be greater than zero.")
    if args.reserve_cores < 0:
        sys.exit("ERROR: --reserve-cores cannot be negative.")
    if args.max_retries <= 0:
        sys.exit("ERROR: --max-retries must be greater than zero.")
    if args.resume and args.fresh:
        sys.exit("ERROR: --resume and --fresh cannot be used together.")


# ================================================================
# MULTIPROCESSING CONTEXT
# ================================================================

def _determine_mp_context(requested_context):
    """
    Determine the multiprocessing start method.

    Automatic policy:
        Linux   -> fork
        macOS   -> spawn
        Windows -> spawn

    The context can be explicitly overridden via --mp-context.
    """
    if requested_context != "auto":
        return requested_context

    system = platform.system().lower()

    if system == "linux":
        return "fork"

    if system in ("darwin", "windows"):
        return "spawn"

    # Conservative fallback for unrecognized operating systems.
    return "spawn"


# ================================================================
# OUTPUT FILE MANAGEMENT
# ================================================================

def _generate_new_filename(output_dir, k_voyages, model_version):
    """Generate a unique, timestamped output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"simulation_results_K{k_voyages}_{model_version}_{timestamp}.csv")


def _find_incomplete_file(output_dir, k_voyages, model_version, target_path=None):
    """
    Locate an incomplete output file.

    If target_path is supplied, only that file is considered.
    Otherwise, search for files matching the current K and model version.
    """
    if target_path:
        if os.path.exists(target_path):
            if not os.path.exists(target_path + ".complete"):
                return target_path
            print(f"File already has a completion marker: {target_path}")
            return None
        return None

    pattern = os.path.join(output_dir, f"simulation_results_K{k_voyages}_{model_version}_*.csv")
    candidates = [p for p in glob.glob(pattern) if not os.path.exists(p + ".complete")]

    if len(candidates) == 0:
        return None

    if len(candidates) == 1:
        return candidates[0]

    print(f"ERROR: Found {len(candidates)} incomplete files matching K={k_voyages}, model version={model_version}.")
    for candidate in candidates:
        print(f"  {candidate}")
    print("Specify the desired file explicitly using --output, or use --fresh to start a new simulation.")
    sys.exit(1)


def _get_output_path(args, output_dir, k_voyages, model_version):
    """Determine the output file path according to the requested run mode."""
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Fresh run
    # ------------------------------------------------------------
    if args.fresh:
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            return args.output
        return _generate_new_filename(output_dir, k_voyages, model_version)

    # ------------------------------------------------------------
    # Explicit resume
    # ------------------------------------------------------------
    if args.resume:
        incomplete = _find_incomplete_file(output_dir, k_voyages, model_version, args.output)
        if incomplete:
            return incomplete
        sys.exit("ERROR: --resume was specified but no incomplete output file was found.")

    # ------------------------------------------------------------
    # Explicit output path
    # ------------------------------------------------------------
    if args.output:
        if os.path.exists(args.output):
            if not os.path.exists(args.output + ".complete"):
                print(f"Found incomplete output file: {args.output}")
                return args.output
            sys.exit(f"Output file is already complete: {args.output}. Use --fresh to start a new run.")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        return args.output

    # ------------------------------------------------------------
    # Automatic resume
    # ------------------------------------------------------------
    incomplete = _find_incomplete_file(output_dir, k_voyages, model_version)
    if incomplete:
        print(f"Automatically resuming from: {os.path.basename(incomplete)}")
        return incomplete

    # ------------------------------------------------------------
    # New output
    # ------------------------------------------------------------
    return _generate_new_filename(output_dir, k_voyages, model_version)


def _backup_corrupt_file(path):
    """Move an unreadable output file to a timestamped backup."""
    backup_path = f"{path}.corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.move(path, backup_path)
    print(f"Unreadable output moved to: {backup_path}")
    return backup_path


def _validate_header_match(output_path, expected_columns):
    """Ensure an existing CSV has the same column structure as the current simulation output."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return

    try:
        with open(output_path, "r", newline="", encoding="utf-8") as file:
            existing_columns = next(csv.reader(file))
    except Exception as exc:
        sys.exit(f"ERROR: Unable to read existing CSV header: {exc}")

    if existing_columns == expected_columns:
        return

    print()
    print("=" * 70)
    print("ERROR: OUTPUT HEADER MISMATCH")
    print("=" * 70)
    print(f"File: {output_path}")
    print(f"Existing columns ({len(existing_columns)}): {existing_columns}")
    print(f"Current columns ({len(expected_columns)}): {expected_columns}")
    print("Appending these results would create an invalid CSV.")
    print("Archive or rename the existing file before rerunning.")
    print("=" * 70)
    sys.exit(1)


def _get_completed_universe_ids(output_path):
    """Read the output file and return the set of completed universe IDs."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return set()

    try:
        df = pd.read_csv(output_path, usecols=["universe_id"])
    except ValueError as exc:
        raise ValueError("Existing output does not contain a 'universe_id' column.") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not read existing universe IDs: {exc}") from exc

    if df.empty:
        return set()

    ids = pd.to_numeric(df["universe_id"], errors="coerce").dropna()
    return set(ids.astype(np.int64).tolist())


# ================================================================
# DETERMINISTIC UNIVERSE SEEDS
# ================================================================

def _generate_universe_seed(global_seed, universe_id):
    """
    Generate a deterministic seed for a specific universe.

    The seed depends only on (global_seed, universe_id), so a failed
    universe can be retried without changing its assigned random
    stream.
    """
    return np.random.SeedSequence([int(global_seed), int(universe_id)])


# ================================================================
# DISPLAY HELPERS
# ================================================================

def _print_header(settings, worker_count, output_path):
    print("=" * 70)
    print("ABM Robustness Test Runner")
    print("=" * 70)
    print(f"MODEL VERSION:       {settings['model_version']}")
    print(f"K_VOYAGES:           {settings['k_voyages']}")
    print(f"TARGET UNIVERSES:    {settings['target_universes']:,}")
    print(f"BATCH SIZE:          {settings['batch_size']}")
    print(f"WORKERS:             {worker_count}")
    print(f"GLOBAL SEED:         {settings['global_seed']}")
    print(f"MULTIPROCESSING:     {settings['mp_context']}")
    print(f"OPERATING SYSTEM:    {platform.system()}")
    print(f"MAX RETRIES:         {settings['max_retries']}")
    print(f"OUTPUT FILE:         {output_path}")
    if settings["param_overrides"]:
        print(f"PARAMETER OVERRIDES: {settings['param_overrides']}")
    print("=" * 70)


def _print_progress(batch_number, successful_count, total_completed, target, elapsed,
                     df_batch=None, total_carrot=None, total_stick=None):
    percentage = total_completed / target * 100 if target > 0 else 0.0
    rate = successful_count / elapsed if elapsed > 0 else 0.0
    remaining = target - total_completed

    # Batch Carrot/Stick
    if df_batch is not None and "intervention_type" in df_batch.columns:
        batch_carrot = (df_batch["intervention_type"] == 1).sum()
        batch_stick = len(df_batch) - batch_carrot
        batch_info = f" (batch carrot={batch_carrot}, stick={batch_stick})"
    else:
        batch_info = ""

    # Total Carrot/Stick (if provided)
    if total_carrot is not None and total_stick is not None:
        total_info = f", total carrot={total_carrot:,}, stick={total_stick:,}"
    else:
        total_info = ""

    print(
        f"Batch {batch_number}: {successful_count} completed, "
        f"rate={rate:.2f} universes/s, "
        f"progress={total_completed:,}/{target:,} ({percentage:.1f}%), "
        f"remaining={remaining:,}"
        f"{batch_info}"
        f"{total_info}"
    )


def _print_resume_status(completed_ids, output_path, target_universes):
    print()
    print(f"Resuming from checkpoint: {os.path.basename(output_path)}")
    print(f"Completed universe IDs: {len(completed_ids):,}")
    remaining = max(0, target_universes - len({uid for uid in completed_ids if 0 <= uid < target_universes}))
    print(f"Remaining target universes: {remaining:,}")
    print()


def _extract_identifier(filepath):
    """
    Extract a clean identifier from a file path.
    
    Priority:
    1. Match pattern K{number}_v{number}_{timestamp} (e.g., K30_v1_20260803_024351)
    2. Fallback to basename without extension
    """
    if not filepath:
        return None
    
    basename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Pattern: K followed by digits, _v, digits, _, 8 digits, _, 6 digits
    pattern = r"K\d+_v\d+_\d{8}_\d{6}"
    match = re.search(pattern, name_without_ext)
    
    if match:
        return match.group(0)
    
    # Fallback: use the full basename without extension
    return name_without_ext

def _extract_k_from_filename(filepath):
    """
    Extract K value from filename like 'k10_n5000_seed7777.csv'.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    
    Returns
    -------
    int or None
        K value if found, else None.
    """
    if not filepath:
        return None
    basename = os.path.basename(filepath)
    match = re.search(r'k(\d+)_', basename)
    if match:
        return int(match.group(1))
    return None

# ================================================================
# FIGURES (PUBLICATION QUALITY)
# ================================================================

def _figures_dir(project_root):
    figures_dir = os.path.join(project_root, "analysis", "figures", "robustness")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir

def _wilson_ci_95(p, n):
    """Approximate 95% CI half-width (normal approximation) for a proportion, as a percentage."""
    if n <= 0:
        return 0.0
    se = np.sqrt(p * (1 - p) / n)
    return 1.96 * se * 100


def _plot_carrot_stick(df, figures_dir, identifier=None):
    """Bar chart: collapse rate by intervention type (Carrot vs. Stick)."""
    if "intervention_type" not in df.columns or "collapse_quarter" not in df.columns:
        return None

    apply_house_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    groups = {"Carrot": df[df["intervention_type"] == 1], "Stick": df[df["intervention_type"] == 0]}
    rates, errs = [], []
    for group_df in groups.values():
        n = len(group_df)
        p = (group_df["collapse_quarter"] > 0).mean() if n > 0 else 0.0
        rates.append(p * 100)
        errs.append(_wilson_ci_95(p, n))

    ax.bar(list(groups.keys()), rates, yerr=errs, capsize=4, color=[COLORS["blue"], COLORS["orange"]])
    ax.set_ylabel("Collapse rate (%)")
    ax.set_title("Collapse rate by intervention type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    suffix = f"_{identifier}" if identifier else ""
    pdf_path = os.path.join(figures_dir, f"robustness_carrot_stick{suffix}.pdf")
    png_path = os.path.join(figures_dir, f"robustness_carrot_stick{suffix}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)
    return pdf_path, png_path

def _plot_drought_effect(df, figures_dir, identifier=None):
    """Bar chart: collapse rate by drought condition (on vs. off)."""
    if "drought_enabled" not in df.columns or "collapse_quarter" not in df.columns:
        return None

    apply_house_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    groups = {"Drought Off": df[df["drought_enabled"] == 0], "Drought On": df[df["drought_enabled"] == 1]}
    rates, errs = [], []
    for group_df in groups.values():
        n = len(group_df)
        p = (group_df["collapse_quarter"] > 0).mean() if n > 0 else 0.0
        rates.append(p * 100)
        errs.append(_wilson_ci_95(p, n))

    ax.bar(list(groups.keys()), rates, yerr=errs, capsize=4, color=[COLORS["blue"], COLORS["orange"]])
    ax.set_ylabel("Collapse rate (%)")
    ax.set_title("Collapse rate by drought condition")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    suffix = f"_{identifier}" if identifier else ""
    pdf_path = os.path.join(figures_dir, f"robustness_drought_effect{suffix}.pdf")
    png_path = os.path.join(figures_dir, f"robustness_drought_effect{suffix}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)
    return pdf_path, png_path


def _plot_collapse_distribution(df, figures_dir, run_tag, k_value=None):
    """
    Histogram of collapse_quarter, all bars in blue.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing collapse_quarter column.
    figures_dir : str
        Directory to save figures.
    run_tag : str
        Run tag for filename.
    k_value : int, optional
        K value to include in the title for reader clarity.
    """
    collapsed = df[df["collapse_quarter"] > 0]
    if collapsed.empty:
        return None

    max_quarter = int(collapsed["collapse_quarter"].max())
    counts = collapsed["collapse_quarter"].value_counts().reindex(range(1, max_quarter + 1), fill_value=0)

    apply_house_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    # All bars in blue (no Q13 highlight)
    colors = [COLORS["blue"] for q in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_xlabel("Collapse quarter")
    ax.set_ylabel("Number of universes")
    
    # Update title to include K if provided
    if k_value is not None:
        ax.set_title(f"Distribution of collapse timing — K={k_value}")
    else:
        ax.set_title("Distribution of collapse timing")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    pdf_path = os.path.join(figures_dir, f"robustness_collapse_distribution_{run_tag}.pdf")
    png_path = os.path.join(figures_dir, f"robustness_collapse_distribution_{run_tag}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)

    return pdf_path, png_path

def _generate_figures(df, figures_dir, input_path=None, k_value=None):
    """Generate every robustness summary figure and print their paths."""
    identifier = _extract_identifier(input_path) if input_path else None
    run_tag = identifier if identifier else "output"
    
    # For carrot_stick and drought_effect (use identifier)
    for plot_fn in (_plot_carrot_stick, _plot_drought_effect):
        result = plot_fn(df, figures_dir, identifier=identifier)
        if result:
            print(f"Figure written to: {result[0]}")
            print(f"Figure written to: {result[1]}")
    
    # For collapse distribution, pass k_value
    result = _plot_collapse_distribution(df, figures_dir, run_tag, k_value=k_value)
    if result:
        print(f"Figure written to: {result[0]}")
        print(f"Figure written to: {result[1]}")

def _print_footer(total_time, output_path, target_universes, project_root, generate_figures=True, k_value=None):
    """Print the final run summary and, unless disabled, generate summary figures."""
    print("=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)
    print(f"Target universes: {target_universes:,}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Output file: {output_path}")

    try:
        df = pd.read_csv(output_path)
        print()
        print("Output summary:")
        print(f"  Total records: {len(df):,}")

        if "collapse_quarter" in df.columns:
            collapse_rate = (df["collapse_quarter"] > 0).mean() * 100
            print(f"  Collapse rate: {collapse_rate:.2f}%")

        if "peak_mutiny" in df.columns:
            print(f"  Peak mutiny range: {df['peak_mutiny'].min():.2f} - {df['peak_mutiny'].max():.2f}")

        if "rebellion_sentence_weight" in df.columns:
            print(
                f"  Rebellion sentence weight range: "
                f"{df['rebellion_sentence_weight'].min():.4f} - "
                f"{df['rebellion_sentence_weight'].max():.4f}"
            )

        if "drought_enabled" in df.columns:
            print(f"  Drought enabled: {(df['drought_enabled'] == 1).sum():,}")
            print(f"  Drought disabled: {(df['drought_enabled'] == 0).sum():,}")

        if "intervention_type" in df.columns:
            print(f"  Carrot universes: {(df['intervention_type'] == 1).sum():,}")
            print(f"  Stick universes: {(df['intervention_type'] == 0).sum():,}")

        if generate_figures:
            print()
            figures_dir = _figures_dir(project_root)
            _generate_figures(df, figures_dir, input_path=output_path, k_value=k_value)

    except Exception as exc:
        print(f"Could not generate final summary: {exc}")

    print("=" * 70)


# ================================================================
# FINAL VALIDATION AND COMPLETION MARKER
# ================================================================

def _verify_completion(output_path, target_universes):
    """Verify that every requested universe ID (0 .. target_universes-1) exists in the output."""
    completed_ids = _get_completed_universe_ids(output_path)
    missing_ids = set(range(target_universes)) - completed_ids

    if missing_ids:
        print("ERROR: Simulation cannot be marked complete.")
        print(f"Missing universes: {len(missing_ids):,}")
        return False

    return True


def _create_completion_marker(output_path, settings):
    """Create a completion marker only after full verification."""
    complete_path = output_path + ".complete"
    with open(complete_path, "w", encoding="utf-8") as file:
        file.write(f"Completed at: {datetime.now().isoformat()}\n")
        file.write(f"Target universes: {settings['target_universes']}\n")
        file.write(f"Global seed: {settings['global_seed']}\n")
        file.write(f"Model version: {settings['model_version']}\n")
        file.write(f"K_VOYAGES: {settings['k_voyages']}\n")
        file.write(f"Multiprocessing context: {settings['mp_context']}\n")
        if settings["param_overrides"]:
            file.write(f"Parameter overrides: {settings['param_overrides']}\n")

    print(f"Completion marker created: {os.path.basename(complete_path)}")


# ================================================================
# MAIN EXECUTION
# ================================================================

def run_robustness(args, param_overrides=None):
    """
    Execute a robustness simulation run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments, as produced by the parser
        registered in build_robustness_parser.

    param_overrides : dict, optional
        Mapping of parameter name -> fixed value. When supplied, the
        named parameters are held constant for every universe instead
        of being randomly sampled inside worker.execute_universe_worker.
        This supports single-variable ablation studies while reusing
        the same simulation and orchestration code as a standard
        robustness run.

    Returns
    -------
    None
    """
    _validate_args(args)

    # src/runners/robustness_runner.py -> src/runners -> src -> project root
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(this_dir))
  
    # ============================================================
    # FIGURES-ONLY MODE: Skip simulation, just generate figures
    # ============================================================
    if hasattr(args, 'figures_only') and args.figures_only:
        if not args.output or not os.path.exists(args.output):
            sys.exit("ERROR: --figures-only requires --output pointing to an existing results CSV.")
        print(f"--figures-only: loading {args.output}")
        try:
            # Auto-detect K from filename, fallback to args.k
            k_from_filename = _extract_k_from_filename(args.output)
            if k_from_filename is not None:
                print(f"Auto-detected K={k_from_filename} from filename")
                k_value = k_from_filename
            else:
                k_value = args.k
                print(f"Using K={k_value} from command line (--k)")
            
            df = pd.read_csv(args.output)
            figures_dir = _figures_dir(project_root)
            _generate_figures(df, figures_dir, input_path=args.output, k_value=k_value)
            print(f"Figures saved to: {figures_dir}")
            return
        except Exception as exc:
            sys.exit(f"ERROR: Could not load or generate figures from {args.output}: {exc}")
    convicts_path, drought_path = default_data_paths(project_root)
    output_dir = os.path.join(project_root, "analysis", "robustness_results")

    settings = {
        "k_voyages": args.k,
        "target_universes": args.n,
        "batch_size": args.batch,
        "global_seed": args.seed,
        "model_version": args.version,
        "max_retries": args.max_retries,
        "mp_context": _determine_mp_context(args.mp_context),
        "param_overrides": param_overrides,
    }

    cpu_count = multiprocessing.cpu_count()
    if args.workers is None:
        available = max(1, cpu_count - args.reserve_cores)
        worker_count = min(available, settings["batch_size"])
    else:
        worker_count = min(args.workers, settings["batch_size"])

    os.makedirs(output_dir, exist_ok=True)
    output_path = _get_output_path(args, output_dir, settings["k_voyages"], settings["model_version"])
    _print_header(settings, worker_count, output_path)

    print()
    inputs = load_simulation_inputs(convicts_path, drought_path)
    grouped_voyages = inputs["grouped_voyages"]
    voyage_ids = inputs["voyage_ids"]
    drought_history_source = inputs["drought_history_source"]
    year_stress_map = inputs["year_stress_map"]

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    existing_universe_ids = set()

    if file_exists:
        try:
            existing_universe_ids = _get_completed_universe_ids(output_path)
        except Exception as exc:
            print(f"ERROR: Existing output could not be read: {exc}")
            _backup_corrupt_file(output_path)
            existing_universe_ids = set()
            file_exists = False

    if existing_universe_ids:
        out_of_range_ids = {uid for uid in existing_universe_ids if uid < 0 or uid >= settings["target_universes"]}
        if out_of_range_ids:
            print("WARNING: Existing file contains universe IDs outside the current target range.")
            print(f"Out-of-range IDs: {len(out_of_range_ids):,}")

        valid_completed_ids = {uid for uid in existing_universe_ids if 0 <= uid < settings["target_universes"]}
        _print_resume_status(valid_completed_ids, output_path, settings["target_universes"])

        if len(valid_completed_ids) >= settings["target_universes"]:
            print("Target universe set is already complete. No additional simulation was executed.")
            if not args.no_figures:
                try:
                    df = pd.read_csv(output_path)
                    figures_dir = _figures_dir(project_root)
                    _generate_figures(df, figures_dir, input_path=output_path, k_value=args.k)
                except Exception as exc:
                    print(f"Could not generate figures for the already-complete run: {exc}")
            return
    else:
        valid_completed_ids = set()

    pending_ids = [uid for uid in range(settings["target_universes"]) if uid not in valid_completed_ids]
    total_carrot = 0
    total_stick = 0

    if not pending_ids:
        print("All requested universes are already complete.")
        return

    print(f"Universes remaining: {len(pending_ids):,}")

    ctx = multiprocessing.get_context(settings["mp_context"])

    batch_number = 0
    total_successful_this_run = 0
    retry_counts = {uid: 0 for uid in pending_ids}
    simulation_start = time.time()
    header_validated = False

    try:
        while pending_ids:
            batch_number += 1
            batch_ids = pending_ids[: settings["batch_size"]]
            batch_start_time = time.time()

            # --------------------------------------------------------
            # Build deterministic worker arguments
            # --------------------------------------------------------
            worker_args = []
            for universe_id in batch_ids:
                child_seed = _generate_universe_seed(settings["global_seed"], universe_id)
                worker_args.append((
                    universe_id,
                    grouped_voyages,
                    voyage_ids,
                    drought_history_source,
                    year_stress_map,
                    settings["k_voyages"],
                    child_seed,
                    settings["model_version"],
                    param_overrides,
                ))

            # --------------------------------------------------------
            # Execute batch
            # --------------------------------------------------------
            with ctx.Pool(processes=worker_count) as pool:
                raw_results = pool.map(execute_universe_worker, worker_args)

            # --------------------------------------------------------
            # Process results
            # --------------------------------------------------------
            successful_results = []
            successful_ids = set()

            for universe_id, result in zip(batch_ids, raw_results):
                if result is None:
                    retry_counts[universe_id] += 1
                    attempts = retry_counts[universe_id]
                    print(f"Universe {universe_id} failed (attempt {attempts}/{settings['max_retries']}).")

                    if attempts >= settings["max_retries"]:
                        raise RuntimeError(
                            f"Universe {universe_id} failed {settings['max_retries']} times. "
                            "Simulation aborted."
                        )
                    continue

                result_universe_id = result.get("universe_id")
                if result_universe_id is None:
                    raise RuntimeError(
                        f"Worker returned a result without universe_id for submitted universe {universe_id}."
                    )

                if int(result_universe_id) != int(universe_id):
                    raise RuntimeError(
                        f"Universe ID mismatch: submitted {universe_id}, received {result_universe_id}."
                    )

                # Ensure model version provenance is preserved.
                returned_version = result.get("model_version")
                if returned_version != settings["model_version"]:
                    raise RuntimeError(
                        f"Model version mismatch for universe {universe_id}: "
                        f"expected {settings['model_version']}, received {returned_version}."
                    )

                successful_results.append(result)
                successful_ids.add(universe_id)

            # --------------------------------------------------------
            # Write successful results
            # --------------------------------------------------------
            if successful_results:
                df_batch = pd.DataFrame(successful_results)

                if "universe_id" not in df_batch.columns:
                    raise RuntimeError("Worker output does not contain 'universe_id'.")
                if "model_version" not in df_batch.columns:
                    raise RuntimeError("Worker output does not contain 'model_version'.")

                if not header_validated:
                    _validate_header_match(output_path, list(df_batch.columns))
                    header_validated = True

                df_batch.to_csv(output_path, mode="a", index=False, header=not file_exists)
                file_exists = True

                pending_ids = [uid for uid in pending_ids if uid not in successful_ids]
                total_successful_this_run += len(successful_results)

                # Update cumulative Carrot / Stick counts
                batch_carrot = (df_batch["intervention_type"] == 1).sum()
                batch_stick = len(df_batch) - batch_carrot
                total_carrot += batch_carrot
                total_stick += batch_stick

                elapsed_batch = time.time() - batch_start_time
                total_completed = settings["target_universes"] - len(pending_ids)

                if batch_number % 10 == 0 or not pending_ids:
                    _print_progress(
                        batch_number,
                        len(successful_results),
                        total_completed,
                        settings["target_universes"],
                        elapsed_batch,
                        df_batch,
                        total_carrot,
                        total_stick,
                    )
            else:
                print(f"Batch {batch_number}: no successful universes. Failed universes will be retried.")

            time.sleep(0.05)

        # ------------------------------------------------------------
        # Final verification
        # ------------------------------------------------------------
        print()
        print("Verifying final output...")

        if not _verify_completion(output_path, settings["target_universes"]):
            raise RuntimeError("Final verification failed.")

        total_time = time.time() - simulation_start
        _print_footer(total_time, output_path, settings["target_universes"], project_root,
                    generate_figures=not args.no_figures, k_value=settings["k_voyages"])
        _create_completion_marker(output_path, settings)

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("SIMULATION INTERRUPTED BY USER")
        print("=" * 70)
        completed_now = settings["target_universes"] - len(pending_ids)
        print(f"Completed universes in current run: {total_successful_this_run:,}")
        print(f"Total completed universe IDs: {completed_now:,}")
        print(f"Output file: {output_path}")
        print()
        print("The output remains an incomplete checkpoint.")
        print("Rerun the same command to resume automatically.")
        print("Use --fresh to explicitly start a new simulation.")
        print("=" * 70)
        sys.exit(0)