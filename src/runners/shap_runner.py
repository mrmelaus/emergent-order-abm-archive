"""
SHAP-based surrogate explanation for the ABM Engine.

Trains an XGBoost surrogate model on an existing robustness results
CSV (no new simulations are run) to predict a chosen outcome from the
sampled input parameters, then computes SHAP (SHapley Additive
exPlanations) values to explain feature importance and effect
direction. Complements the Sobol analysis in two ways:

1. Categorical factors (intervention_type, drought_enabled), which
   Sobol's variance-decomposition framework cannot handle, are
   included here alongside the continuous parameters, on the same
   importance scale.

2. SHAP interaction values for a chosen feature pair (default
   w1_energy x w4_policy) provide an independent, game-theoretic
   cross-check against the corresponding Sobol second-order index S2.
   The two methods use different units (SHAP interaction magnitude is
   in outcome units, e.g. quarters or probability; Sobol S2 is a
   proportion of variance), so they should be compared qualitatively
   (does this pair show up as an important interaction under both
   methods?), not treated as numerically equivalent.

Design notes
------------
1. File naming / cache-collision prevention. Every output filename
   embeds a config signature built from every argument that actually
   changes the underlying computation: --features (hashed, since the
   full list can be long), --seed (training / split), --shap-seed
   (explanation subsampling), and --sample-size (main SHAP output
   only, where relevant). This mirrors the fix already applied to
   sobol_runner.py's --vars-aware filenames. Without this, changing
   e.g. --sample-size between runs would silently reuse a stale
   cached SHAP-value array sized for the OLD sample size (the same
   class of bug that previously required manually remembering to pass
   --fresh whenever --interaction-pair changed) — this version makes
   that class of bug structurally impossible: a different
   --sample-size (or --seed, or --features) always produces a
   different filename, so stale results are never silently reused
   under a name that looks current.

2. Two independent seeds. --seed controls the train/test split and
   model training. --shap-seed independently controls which rows are
   subsampled for SHAP value computation and the stability sweep.
   They default to the same value if --shap-seed is not given, but
   can be varied independently to check whether an explanation
   subsample was representative, without retraining the model.

3. Sample-size stability sweep (--stability-sizes). Recomputes
   mean(|SHAP value|) per feature at several subsample sizes (reusing
   the already-trained model and explainer, no retraining) and plots
   how each feature's importance changes as the subsample size grows.
   This is a stability check on the SIZE OF THE EXPLANATION SUBSAMPLE,
   not on the amount of TRAINING data used to fit the surrogate — the
   surrogate is always trained on the full input dataset.

4. Performance / CPU handling. Unlike robustness/sobol/sensitivity,
   this runner does not launch its own multiprocessing.Pool: XGBoost
   is single-process with its own internal multi-threading. CPU
   control is exposed the same way as elsewhere (--reserve-cores,
   --workers) but is applied to XGBoost's n_jobs parameter.

5. Checkpointing / resume. Trained models (native XGBoost format) and
   the main SHAP value array are saved to
   analysis/shap_results/checkpoints/, keyed by the config signature
   described above. Rerunning the exact same command automatically
   loads them instead of retraining / recomputing. Use --fresh to
   force retraining and recomputation regardless. The stability sweep
   always recomputes (it is cheap relative to training).

Outputs:
    analysis/shap_results/               -> metrics CSV, importance
                                             CSV, interaction CSV,
                                             stability CSV
    analysis/shap_results/checkpoints/   -> trained models, cached
                                             main SHAP value array
    analysis/figures/shap/               -> importance bar charts,
                                             beeswarm summary plots,
                                             stability sweep plot
                                             
Usage examples:
    # Run SHAP analysis on a robustness results CSV
    python main.py shap --input analysis/robustness_results/run.csv

    # Regenerate figures from existing SHAP importance CSV (no training)
    python main.py shap --figures-only --output analysis/shap_results/shap_importance_*.csv

    # Regenerate stability plot from existing stability CSV (no training)
    python main.py shap --figures-only --output analysis/shap_results/shap_stability_*.csv
"""

import hashlib
import multiprocessing
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_style import apply_house_style, FIGSIZE_STANDARD, save_figure, COLORS

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import shap as shap_lib
except ImportError:
    shap_lib = None

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
except ImportError:
    train_test_split = None


# ================================================================
# DEFAULT SETTINGS
# ================================================================

DEFAULT_FEATURES = (
    "w1_energy,w2_drought,w3_rebellion,w4_policy,w5_social,despair_rate,threshold,"
    "rebellion_sentence_weight,natural_death_rate_quarterly,intervention_type,drought_enabled"
)
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SHAP_SAMPLE_SIZE = 5000
DEFAULT_SEED = 7777
DEFAULT_INTERACTION_PAIR = ("w1_energy", "w4_policy")

# ================================================================
# ARGUMENT REGISTRATION
# ================================================================

def build_shap_parser(subparsers):
    """
    Register the 'shap' subcommand.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = subparsers.add_parser(
        "shap",
        help="Train an XGBoost surrogate on an existing robustness results CSV and compute SHAP values.",
    )

    # Change required=True to required=False
    parser.add_argument("--input", type=str, required=False,
                         help="Path to an existing robustness results CSV to train the surrogate on. "
                              "Required for normal mode (without --figures-only).")
    parser.add_argument("--target", choices=("regression", "classification", "both"), default="both",
                         help="Which surrogate(s) to train: regression on collapse_quarter, "
                              "classification on collapse_binary (collapse_quarter > 0), or both "
                              "(default: both).")
    parser.add_argument("--features", type=str, default=DEFAULT_FEATURES,
                         help=f"Comma-separated list of candidate feature columns "
                              f"(default matches the actual robustness output schema). Any not "
                              f"present in --input are skipped with a warning, not an error.")
    parser.add_argument("--interaction-pair", type=str, default=",".join(DEFAULT_INTERACTION_PAIR),
                         help=f"Comma-separated pair of feature names to compute SHAP interaction "
                              f"values for (default: {','.join(DEFAULT_INTERACTION_PAIR)}). "
                              f"Both must be present in --features.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                         help=f"Fraction of data held out for evaluation (default: {DEFAULT_TEST_SIZE}).")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SHAP_SAMPLE_SIZE,
                         help=f"Number of test-set rows to sample for the main SHAP value computation "
                              f"and beeswarm plot (default: {DEFAULT_SHAP_SAMPLE_SIZE}). Training always "
                              f"uses the full dataset; only SHAP value computation is subsampled.")
    parser.add_argument("--stability-sizes", type=str, default=None,
                         help="Comma-separated list of subsample sizes to sweep for a SHAP-value "
                              "stability check, e.g. '1000,2000,4000,8000,16000'. Reuses the already "
                              "-trained model (no retraining). When set, produces a stability CSV and "
                              "a convergence-style line plot showing how mean(|SHAP value|) for each "
                              "feature changes with subsample size. This is separate from --sample-size, "
                              "which controls the main (single-size) SHAP output.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help=f"Random seed for the train/test split and model training "
                              f"(default: {DEFAULT_SEED}).")
    parser.add_argument("--shap-seed", type=int, default=None,
                         help="Independent random seed for SHAP value subsampling and the stability "
                              "sweep, separate from --seed (which controls training). Default: same "
                              "value as --seed. Use a different --shap-seed with the same --seed to "
                              "check whether the explanation subsample was representative, without "
                              "retraining the model.")
    parser.add_argument("--workers", type=int, default=None,
                         help="XGBoost n_jobs (thread count). Default: all available cores minus "
                              "--reserve-cores.")
    parser.add_argument("--reserve-cores", type=int, default=0,
                         help="Number of CPU cores to leave free for the rest of the system. "
                              "Use e.g. --reserve-cores 1 on a machine you are using interactively. "
                              "Ignored if --workers is explicitly set. Default 0 (use every core).")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from existing checkpoints (default behaviour already resumes "
                              "automatically if a matching checkpoint exists).")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore existing model/SHAP checkpoints and retrain / recompute from scratch. "
                              "Not required for correctness (every config now gets its own filename), "
                              "but still useful to force a genuinely new computation under an identical "
                              "configuration.")
    parser.add_argument("--output", type=str, default=None,
                         help="Path to an existing SHAP results CSV for --figures-only mode. "
                              "Must point to a valid output file from a previous SHAP run "
                              "(e.g. shap_importance_*.csv or shap_stability_*.csv).")
    parser.add_argument("--figures-only", action="store_true",
                         help="Skip training/SHAP computation entirely and (re)generate figures "
                              "from an existing --output CSV. Requires --output to point to an "
                              "existing SHAP results file.")
    parser.add_argument("--no-figures", action="store_true",
                         help="Skip figure generation and only write the CSVs.")

    return parser

def _validate_args(args):
    """Validate resolved SHAP settings; exits with an error message on invalid input."""
    if xgb is None:
        sys.exit("ERROR: xgboost is required. Install it with: pip install xgboost --break-system-packages")
    if shap_lib is None:
        sys.exit("ERROR: shap is required. Install it with: pip install shap --break-system-packages")
    if train_test_split is None:
        sys.exit("ERROR: scikit-learn is required. Install it with: pip install scikit-learn --break-system-packages")

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: --input file not found: {args.input}")
    if not (0.0 < args.test_size < 1.0):
        sys.exit("ERROR: --test-size must be between 0 and 1.")
    if args.sample_size <= 0:
        sys.exit("ERROR: --sample-size must be greater than zero.")
    if args.workers is not None and args.workers <= 0:
        sys.exit("ERROR: --workers must be greater than zero.")
    if args.reserve_cores < 0:
        sys.exit("ERROR: --reserve-cores cannot be negative.")
    if args.resume and args.fresh:
        sys.exit("ERROR: --resume and --fresh cannot be used together.")

    pair = [p.strip() for p in args.interaction_pair.split(",") if p.strip()]
    if len(pair) != 2:
        sys.exit("ERROR: --interaction-pair must contain exactly two comma-separated feature names.")

    if args.stability_sizes:
        try:
            sizes = [int(s.strip()) for s in args.stability_sizes.split(",") if s.strip()]
        except ValueError:
            sys.exit("ERROR: --stability-sizes must be a comma-separated list of integers.")
        if any(s <= 0 for s in sizes):
            sys.exit("ERROR: --stability-sizes values must be greater than zero.")


def _resolve_worker_count(args):
    """Every core by default; --reserve-cores leaves cores free for an interactive machine."""
    if args.workers is not None:
        return max(1, args.workers)
    return max(1, multiprocessing.cpu_count() - args.reserve_cores)


def _resolve_shap_seed(args):
    """--shap-seed defaults to --seed when not explicitly supplied."""
    return args.shap_seed if args.shap_seed is not None else args.seed


# ================================================================
# PATHS / RUN TAG / CONFIG SIGNATURE
# ================================================================

def _project_root():
    """src/runners/shap_runner.py -> src/runners -> src -> project root."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(this_dir))


def _shap_results_dir(project_root):
    results_dir = os.path.join(project_root, "analysis", "shap_results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _checkpoint_dir(results_dir):
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def _figures_dir(project_root):
    figures_dir = os.path.join(project_root, "analysis", "figures", "shap")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def _extract_run_tag(input_path):
    """
    Extract a run tag from the input filename, matching the naming
    convention used by robustness_runner.py's own output figures
    (e.g. "K30_v1_20260803_024351" from
    "simulation_results_K30_v1_20260803_024351.csv"). Falls back to
    the input file's stem if the expected pattern is not found.
    """
    basename = os.path.basename(input_path)
    match = re.match(r"simulation_results_(.+)\.csv$", basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]


def _feature_hash(features):
    """Short, deterministic hash of the feature set, order-independent."""
    raw = ",".join(sorted(features))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:6]


def _model_tag(features, seed):
    """
    Config tag for anything that depends only on training: the
    feature set and the training/split seed. Used for the model
    checkpoint and its metrics, which are unaffected by --shap-seed
    or --sample-size.
    """
    return f"feat{_feature_hash(features)}_seed{seed}"


def _shap_tag(features, seed, shap_seed, sample_size):
    """
    Config tag for anything that depends on the SHAP explanation step:
    adds shap_seed and sample_size on top of the model tag. Used for
    the main SHAP value cache, the importance table/figure, the
    beeswarm figure, and the interaction result.
    """
    return f"{_model_tag(features, seed)}_shapseed{shap_seed}_n{sample_size}"


def _stability_tag(features, seed, shap_seed, sizes):
    """Config tag for the stability sweep, adding a hash of the tested sizes."""
    sizes_hash = hashlib.md5(",".join(str(s) for s in sizes).encode("utf-8")).hexdigest()[:6]
    return f"{_model_tag(features, seed)}_shapseed{shap_seed}_sizes{sizes_hash}"


# ================================================================
# DATA PREPARATION
# ================================================================

def _prepare_data(df, requested_features, target_col, source_name):
    """
    Intersect requested features with what is actually present in the
    dataframe, warn about anything missing, and return a cleaned
    (X, y) pair with rows containing NaNs in the used columns dropped.
    """
    available = [f for f in requested_features if f in df.columns]
    missing = [f for f in requested_features if f not in df.columns]

    if missing:
        print(f"WARNING: requested features not found in {source_name} and skipped: {missing}")

    if not available:
        sys.exit("ERROR: none of the requested --features are present in the input CSV.")

    if target_col not in df.columns:
        sys.exit(f"ERROR: target column '{target_col}' not present in the input CSV.")

    subset = df[available + [target_col]].dropna()
    X = subset[available]
    y = subset[target_col]

    return X, y, available


# ================================================================
# MODEL TRAINING (WITH CHECKPOINTING)
# ================================================================

def _model_path(checkpoint_dir, run_tag, target_kind, model_tag):
    return os.path.join(checkpoint_dir, f"shap_model_{target_kind}_{run_tag}_{model_tag}.json")


def _main_shap_path(checkpoint_dir, run_tag, target_kind, shap_tag):
    return os.path.join(checkpoint_dir, f"shap_values_{target_kind}_{run_tag}_{shap_tag}.npz")


def _train_regressor(X_train, y_train, X_test, y_test, n_jobs, seed):
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=n_jobs, random_state=seed,
        early_stopping_rounds=20, eval_metric="rmse",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, {"metric": "R2", "value": r2}


def _train_classifier(X_train, y_train, X_test, y_test, n_jobs, seed):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=n_jobs, random_state=seed,
        objective="binary:logistic", eval_metric="aucpr",
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    return model, {"roc_auc": roc_auc, "pr_auc": pr_auc}


# ================================================================
# FIGURES (PUBLICATION QUALITY)
# ================================================================

def _plot_importance_bar(mean_abs_shap, feature_names, title, figures_dir, filename_stem):
    apply_house_style()
    order = np.argsort(mean_abs_shap)
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.barh(np.array(feature_names)[order], np.array(mean_abs_shap)[order], color=COLORS["blue"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    pdf_path = os.path.join(figures_dir, f"{filename_stem}.pdf")
    png_path = os.path.join(figures_dir, f"{filename_stem}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)
    return pdf_path, png_path


def _plot_beeswarm(shap_values, X_sample, title, figures_dir, filename_stem):
    apply_house_style()
    shap_lib.summary_plot(shap_values, X_sample, show=False, plot_size=None)
    fig = plt.gcf()
    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    pdf_path = os.path.join(figures_dir, f"{filename_stem}.pdf")
    png_path = os.path.join(figures_dir, f"{filename_stem}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)
    return pdf_path, png_path


def _plot_stability(stability_df, title, figures_dir, filename_stem):
    """Line plot: mean(|SHAP value|) vs. subsample size, one line per feature."""
    apply_house_style()
    
    # Wider figure to accommodate legend on the right
    fig, ax = plt.subplots(figsize=(10, 5))

    for feature_name, group in stability_df.groupby("feature"):
        group = group.sort_values("sample_size")
        ax.plot(group["sample_size"], group["mean_abs_shap"], marker="o", label=feature_name)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("SHAP explanation subsample size (log scale)")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(title)
    
    # Remove top and right spines (JASSS style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legend outside to the right
    ax.legend(frameon=False, fontsize=8, ncol=1, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)  # Make room for legend

    pdf_path = os.path.join(figures_dir, f"{filename_stem}.pdf")
    png_path = os.path.join(figures_dir, f"{filename_stem}.png")
    save_figure(fig, pdf_path)
    save_figure(fig, png_path)
    plt.close(fig)
    return pdf_path, png_path

# ================================================================
# ONE SURROGATE PIPELINE (TRAIN -> SHAP -> STABILITY -> OUTPUTS)
# ================================================================

def _run_surrogate(target_kind, X, y, feature_names, args, run_tag, results_dir, checkpoint_dir,
                    figures_dir, n_jobs, shap_seed, interaction_pair):
    """Train (or load) one surrogate, compute (or load) SHAP values, and write all outputs."""
    print()
    print("-" * 60)
    print(f"Surrogate: {target_kind}")
    print("-" * 60)

    model_tag = _model_tag(feature_names, args.seed)
    shap_tag = _shap_tag(feature_names, args.seed, shap_seed, args.sample_size)

    model_path = _model_path(checkpoint_dir, run_tag, target_kind, model_tag)
    main_shap_path = _main_shap_path(checkpoint_dir, run_tag, target_kind, shap_tag)

    print(f"Model config tag: {model_tag}")
    print(f"SHAP config tag:  {shap_tag}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    if args.fresh and os.path.exists(model_path):
        os.remove(model_path)

    # ------------------------------------------------------------
    # Model (train or load)
    # ------------------------------------------------------------
    if os.path.exists(model_path) and not args.fresh:
        print(f"Found existing model checkpoint: {model_path}. Loading (skip training).")
        model = xgb.XGBRegressor() if target_kind == "regression" else xgb.XGBClassifier()
        model.load_model(model_path)
        if target_kind == "regression":
            metrics = {"metric": "R2", "value": r2_score(y_test, model.predict(X_test))}
        else:
            proba = model.predict_proba(X_test)[:, 1]
            metrics = {"roc_auc": roc_auc_score(y_test, proba),
                       "pr_auc": average_precision_score(y_test, proba)}
    else:
        print(f"Training XGBoost {target_kind} surrogate (n_jobs={n_jobs})...")
        if target_kind == "regression":
            model, metrics = _train_regressor(X_train, y_train, X_test, y_test, n_jobs, args.seed)
        else:
            model, metrics = _train_classifier(X_train, y_train, X_test, y_test, n_jobs, args.seed)
        model.save_model(model_path)
        print(f"Model checkpoint written to: {model_path}")

    print(f"Held-out performance: {metrics}")

    metrics_row = {"target": target_kind, "run_tag": run_tag, "model_tag": model_tag,
                   "n_features": len(feature_names)}
    metrics_row.update(metrics)
    metrics_path = os.path.join(results_dir, f"shap_metrics_{target_kind}_{run_tag}_{model_tag}.csv")
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)
    print(f"Metrics written to: {metrics_path}")

    explainer = shap_lib.TreeExplainer(model)

    # ------------------------------------------------------------
    # Main SHAP values (checkpointed, keyed by the full SHAP config tag)
    # ------------------------------------------------------------
    main_sample_n = min(args.sample_size, len(X_test))
    X_main_sample = X_test.sample(n=main_sample_n, random_state=shap_seed)

    interaction_values = None
    if os.path.exists(main_shap_path) and not args.fresh:
        print(f"Found existing SHAP value checkpoint: {main_shap_path}. Loading (skip recomputation).")
        cached = np.load(main_shap_path, allow_pickle=True)
        shap_values = cached["shap_values"]
        if "interaction_values" in cached:
            interaction_values = cached["interaction_values"]
    else:
        print(f"Computing SHAP values (n={main_sample_n}, shap_seed={shap_seed})...")
        shap_values = explainer.shap_values(X_main_sample)

        if interaction_pair[0] in feature_names and interaction_pair[1] in feature_names:
            print(f"Computing SHAP interaction values for {interaction_pair[0]} x {interaction_pair[1]}...")
            interaction_values = explainer.shap_interaction_values(X_main_sample)

        save_kwargs = {"shap_values": shap_values}
        if interaction_values is not None:
            save_kwargs["interaction_values"] = interaction_values
        np.savez(main_shap_path, **save_kwargs)
        print(f"SHAP values written to: {main_shap_path}")

    # ------------------------------------------------------------
    # Importance table + figure
    # ------------------------------------------------------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap}) \
        .sort_values("mean_abs_shap", ascending=False)
    importance_path = os.path.join(results_dir, f"shap_importance_{target_kind}_{run_tag}_{shap_tag}.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Importance table written to: {importance_path}")
    print(importance_df.to_string(index=False))

    if not args.no_figures:
        result = _plot_importance_bar(
            mean_abs_shap, feature_names,
            f"SHAP feature importance — {target_kind}",
            figures_dir, f"shap_importance_{target_kind}_{run_tag}_{shap_tag}",
        )
        print(f"Figure written to: {result[0]}")
        print(f"Figure written to: {result[1]}")

        result = _plot_beeswarm(
            shap_values, X_main_sample,
            f"SHAP summary — {target_kind}",
            figures_dir, f"shap_beeswarm_{target_kind}_{run_tag}_{shap_tag}",
        )
        print(f"Figure written to: {result[0]}")
        print(f"Figure written to: {result[1]}")

    # ------------------------------------------------------------
    # Interaction pair (cross-check against Sobol S2)
    # ------------------------------------------------------------
    if interaction_values is not None:
        i = feature_names.index(interaction_pair[0])
        j = feature_names.index(interaction_pair[1])
        interaction_strength = (
            np.abs(interaction_values[:, i, j]).mean() + np.abs(interaction_values[:, j, i]).mean()
        )
        interaction_row = {
            "target": target_kind, "run_tag": run_tag, "shap_tag": shap_tag,
            "param_a": interaction_pair[0], "param_b": interaction_pair[1],
            "mean_abs_shap_interaction": interaction_strength,
            "note": "SHAP interaction magnitude is in outcome units, not a variance proportion; "
                    "compare qualitatively against the corresponding Sobol S2, not numerically.",
        }
        interaction_path = os.path.join(
            results_dir,
            f"shap_interaction_{interaction_pair[0]}_{interaction_pair[1]}_{target_kind}_{run_tag}_{shap_tag}.csv",
        )
        pd.DataFrame([interaction_row]).to_csv(interaction_path, index=False)
        print(f"{interaction_pair[0]} x {interaction_pair[1]} interaction written to: {interaction_path}")
        print(f"  mean |SHAP interaction|({interaction_pair[0]},{interaction_pair[1]}) = {interaction_strength:.4f}")

    # ------------------------------------------------------------
    # Stability sweep (always recomputed; cheap relative to training)
    # ------------------------------------------------------------
    if args.stability_sizes:
        sizes = [int(s.strip()) for s in args.stability_sizes.split(",") if s.strip()]
        stability_tag = _stability_tag(feature_names, args.seed, shap_seed, sizes)

        print()
        print(f"Running SHAP subsample-size stability sweep: {sizes}")

        stability_rows = []
        for size in sizes:
            n = min(size, len(X_test))
            X_sweep_sample = X_test.sample(n=n, random_state=shap_seed)
            sweep_values = explainer.shap_values(X_sweep_sample)
            sweep_mean_abs = np.abs(sweep_values).mean(axis=0)

            for feature_name, value in zip(feature_names, sweep_mean_abs):
                stability_rows.append({
                    "target": target_kind, "sample_size": n, "feature": feature_name,
                    "mean_abs_shap": value,
                })
            print(f"  size={n:,}: done")

        stability_df = pd.DataFrame(stability_rows)
        stability_path = os.path.join(
            results_dir, f"shap_stability_{target_kind}_{run_tag}_{stability_tag}.csv"
        )
        stability_df.to_csv(stability_path, index=False)
        print(f"Stability table written to: {stability_path}")

        if not args.no_figures:
            result = _plot_stability(
                stability_df, f"SHAP stability sweep — {target_kind}",
                figures_dir, f"shap_stability_{target_kind}_{run_tag}_{stability_tag}",
            )
            print(f"Figure written to: {result[0]}")
            print(f"Figure written to: {result[1]}")


# ================================================================
# MAIN EXECUTION
# ================================================================

def run_shap(args):
    """
    Execute the SHAP surrogate-model explanation pipeline on an
    existing robustness results CSV.

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
                f"ERROR: --figures-only requires --output pointing to an existing SHAP results CSV.\n"
                f"Got: {args.output or 'not provided'}\n"
                f"Expected: a SHAP importance CSV (e.g. shap_importance_*.csv) or "
                f"stability CSV (e.g. shap_stability_*.csv)"
            )

        print(f"--figures-only: loading {args.output}")
        df = pd.read_csv(args.output)

        project_root = _project_root()
        figures_dir = _figures_dir(project_root)

        basename = os.path.basename(args.output)
        stem = os.path.splitext(basename)[0]
        import re

        # Detect file type from columns
        if "mean_abs_shap" in df.columns and "feature" in df.columns and "sample_size" not in df.columns:
            # This is an importance CSV → Regenerate importance bar chart
            print(f"Detected: SHAP importance table with {len(df)} features")
            feature_names = df["feature"].tolist()
            mean_abs_shap = df["mean_abs_shap"].values

            # Try to extract target from filename
            target_match = re.search(r'_(\w+?)_(?:run|K)', basename)
            target_kind = target_match.group(1) if target_match else "unknown"

            if not args.no_figures:
                result = _plot_importance_bar(
                    mean_abs_shap, feature_names,
                    f"SHAP feature importance — {target_kind}",
                    figures_dir, stem,
                )
                if result:
                    print(f"Figure written to: {result[0]}")
                    print(f"Figure written to: {result[1]}")

            print("=" * 70)
            print("FIGURES-ONLY MODE COMPLETE")
            print("=" * 70)
            return

        elif "sample_size" in df.columns and "feature" in df.columns and "mean_abs_shap" in df.columns:
            # This is a stability CSV → Regenerate stability plot
            print(f"Detected: SHAP stability sweep with {len(df)} rows")
            target_match = re.search(r'_(\w+?)_(?:run|K)', basename)
            target_kind = target_match.group(1) if target_match else "unknown"

            if not args.no_figures:
                result = _plot_stability(
                    df, f"SHAP stability sweep — {target_kind}",
                    figures_dir, stem,
                )
                if result:
                    print(f"Figure written to: {result[0]}")
                    print(f"Figure written to: {result[1]}")

            print("=" * 70)
            print("FIGURES-ONLY MODE COMPLETE")
            print("=" * 70)
            return

        else:
            sys.exit(
                f"ERROR: {args.output} does not appear to be a recognized SHAP results file.\n"
                f"Expected: shap_importance_*.csv or shap_stability_*.csv\n"
                f"Found columns: {df.columns.tolist()}"
            )

    # ================================================================
    # NORMAL EXECUTION (existing code continues below)
    # ================================================================
    _validate_args(args)

    project_root = _project_root()
    results_dir = _shap_results_dir(project_root)
    checkpoint_dir = _checkpoint_dir(results_dir)
    figures_dir = _figures_dir(project_root)
    run_tag = _extract_run_tag(args.input)
    n_jobs = _resolve_worker_count(args)
    shap_seed = _resolve_shap_seed(args)
    interaction_pair = tuple(p.strip() for p in args.interaction_pair.split(","))

    print("=" * 70)
    print("SHAP SURROGATE-MODEL EXPLANATION")
    print("=" * 70)
    print(f"Input:      {args.input}")
    print(f"Run tag:    {run_tag}")
    print(f"Target:     {args.target}")
    print(f"n_jobs:     {n_jobs} (reserve_cores={args.reserve_cores})")
    print(f"seed:       {args.seed} (training / train-test split)")
    print(f"shap_seed:  {shap_seed} (SHAP subsampling / stability sweep)")
    print(f"sample_size: {args.sample_size}")
    print("=" * 70)

    df = pd.read_csv(args.input)
    requested_features = [f.strip() for f in args.features.split(",") if f.strip()]

    if "collapse_quarter" not in df.columns:
        sys.exit("ERROR: input CSV does not contain a 'collapse_quarter' column.")

    df["collapse_binary"] = (df["collapse_quarter"] > 0).astype(int)

    targets_to_run = []
    if args.target in ("regression", "both"):
        targets_to_run.append(("regression", "collapse_quarter"))
    if args.target in ("classification", "both"):
        targets_to_run.append(("classification", "collapse_binary"))

    for target_kind, target_col in targets_to_run:
        X, y, available_features = _prepare_data(df, requested_features, target_col, os.path.basename(args.input))
        _run_surrogate(
            target_kind, X, y, available_features, args, run_tag,
            results_dir, checkpoint_dir, figures_dir, n_jobs, shap_seed, interaction_pair,
        )

    print()
    print("=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)