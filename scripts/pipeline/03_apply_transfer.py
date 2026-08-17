"""Apply a trained PCC pipeline to a new target cohort (transfer-validation).

Loads a ``final_model.pkl`` produced by ``scripts/pipeline/02_run_pipeline.py`` and
scores it on a new analytic sample (e.g. the NAKO non-MRI cohort) without
re-training.  Reports metrics, bootstrap CIs, calibration, modality SHAP,
Spearman ranking correlation against the source run, DCA, and a
``transfer_summary.json`` whose verdict is a TOST equivalence test on
ROC-AUC, calibration slope and calibration intercept.  The per-modality SHAP
shares travel with it as a descriptive interval, not as a pass/fail gate.

Usage
-----
    uv run python scripts/pipeline/03_apply_transfer.py \\
        --from-run results/pcc_pipeline_mri_lean_bahmer_<ts> \\
        --target-cohort non_mri
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from scipy.stats import ks_2samp, spearmanr

from pcc_analysis._gpu_utils import xgb_device
from pcc_analysis._provenance import package_versions, write_provenance
from pcc_analysis.config import get_processed_dir, load_config
from pcc_analysis.data_manager import load_pipeline_data
from pcc_analysis.evaluation import evaluate_model_comprehensive
from pcc_analysis.shap_analysis import compute_meta_learner_shap

if TYPE_CHECKING:
    from pcc_analysis._types import FinalModelResult
    from pcc_analysis.data_manager import CohortName, StackVariant, TargetName

console = Console()
logger = logging.getLogger(__name__)


# `args` comes from the command line and `source_config` from the source run's
# config.csv, so both arrive as plain strings. These check them against the
# domains `load_pipeline_data` accepts, which turns a stale or hand-edited
# value into a named error here rather than a confusing failure downstream.


def _as_target(value: str) -> TargetName:
    if value not in ("bahmer", "neurocog"):
        raise SystemExit(f"Unknown target in source config: {value!r}")
    return value


def _as_cohort(value: str) -> CohortName:
    if value not in ("mri", "non_mri", "mri_plus_non_mri", "all"):
        raise SystemExit(f"Unknown cohort: {value!r}")
    return value


def _as_stack_variant(value: str) -> StackVariant:
    if value not in ("full", "lean"):
        raise SystemExit(f"Unknown stack variant in source config: {value!r}")
    return value


BASELINE_MH_MODALITY = "mental_health"
SHAP_BOOTSTRAP_ITERATIONS = 1000

# Primary transfer success criterion — TOST-style performance
# equivalence on the three literature-standard transportability metrics
# (Steyerberg 2019; Van Calster 2019; TRIPOD-AI).  The target-cohort
# 95 % CI must lie entirely inside the margin band for a PASS.
TOST_DELTA_ROC_AUC_MARGIN = 0.03
"""Two-sided margin for |target ROC-AUC − source ROC-AUC|.

0.03 is the common non-inferiority/equivalence margin for clinical
prediction-model discrimination transportability.
"""

TOST_CALIBRATION_SLOPE_BAND = (0.85, 1.15)
"""Equivalence band for the target calibration slope (ideal = 1.0).

Van Calster (2019) considers (0.85, 1.15) "acceptable calibration" for
external validation of a binary clinical prediction model.
"""

TOST_CALIBRATION_INTERCEPT_BAND = (-0.05, 0.05)
"""Equivalence band for the target calibration-in-the-large intercept
(ideal = 0.0).  On the logit scale; practitioner-facing equivalent to
~±1 percentage point of mean-predicted vs. observed prevalence."""

CALIB_BOOTSTRAP_ITERATIONS = 1000
"""Bootstrap resamples for target calibration-slope/intercept 95 % CIs."""
"""Number of resamples for the SHAP-share 95 % CI bootstrap.

Used for both the source (MRI) and target (non-MRI) CI computations,
so the overlap test is symmetric.  The meta-learner is already fitted;
TreeExplainer is reused per-call, so the cost is ~2x`this`-many tree-SHAP
evaluations on an (n_rows × n_modalities) matrix — seconds on Lean-Stack.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a trained PCC model to a target cohort without refit "
            "(within-study transfer validation)."
        ),
    )
    parser.add_argument(
        "--from-run",
        type=Path,
        required=True,
        help="Source run directory with final_model.pkl and config.csv.",
    )
    parser.add_argument(
        "--target-cohort",
        choices=["mri", "non_mri", "all"],
        required=True,
        help=(
            "Cohort filter for the target data (relative to MRI "
            "availability).  Must be compatible with the source run's "
            "stack variant (non_mri + full is not allowed)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override automatic timestamped output directory; default is "
            "results/pcc_transfer_<source_name>_to_<target>_<ts>/."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: auto-detected).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap iterations for metric CIs (default: 1000).",
    )
    return parser.parse_args()


def _load_source_artifacts(
    from_run: Path,
) -> tuple[FinalModelResult, dict[str, str]]:
    pkl = from_run / "final_model.pkl"
    cfg = from_run / "config.csv"
    if not pkl.exists():
        raise FileNotFoundError(
            f"Source run {from_run} does not contain final_model.pkl.  "
            "Re-run 02_run_pipeline.py on the updated codebase to get a "
            "transfer-ready pickle."
        )
    if not cfg.exists():
        raise FileNotFoundError(f"Source run {from_run} is missing config.csv.")
    with pkl.open("rb") as f:
        final_model = pickle.load(f)
    for required in (
        "meta_learner",
        "modality_names",
        "modality_pipelines",
        "oof_predictions",
        "confounders_train",
    ):
        if required not in final_model:
            raise ValueError(
                f"Source final_model.pkl is missing '{required}' — it was "
                "likely produced by an older pipeline version.  Re-run "
                "the source training on the current codebase."
            )
    cfg_df = pd.read_csv(cfg)
    if cfg_df.empty:
        raise ValueError(f"Source config.csv in {from_run} is empty.")
    source_config = {str(k): str(v) for k, v in cfg_df.iloc[0].items()}
    _check_source_stack(from_run, source_config)
    return final_model, source_config


def _check_source_stack(from_run: Path, source_config: dict[str, str]) -> None:
    """Refuse a source run that was fitted against a different library stack.

    This step is the one place in the analysis where a fitted model crosses
    from one run to another — and, in a rerun split over two machines, from
    one machine to the other. It arrives as a pickle of sklearn and XGBoost
    estimators, which is a format with no version negotiation: a stack that
    does not match either fails to unpickle with an opaque attribute error
    or, worse, unpickles into an estimator whose semantics moved.

    The numbers make the same point without the pickle. DECISIONS §2.20
    measured 13 of 16 output files moving between two XGBoost releases, so
    a transfer whose source and target stacks differ measures the release
    as much as the transport it is meant to test.

    A source run from before the stamp existed (DECISIONS §2.21) cannot be
    checked. That is reported and allowed: the guard is here to catch a
    drifting environment, not to invalidate the runs that predate it.
    """
    source_stack = source_config.get("package_versions", "").strip()
    if not source_stack or source_stack == "nan":
        console.print(
            f"[yellow]Source run {from_run} predates the package_versions "
            "stamp; the stack it was fitted under cannot be verified."
        )
        return

    running_stack = package_versions()
    if source_stack == running_stack:
        _warn_on_source_device(from_run, source_config)
        return

    raise ValueError(
        f"Source run {from_run} was fitted under a different library stack "
        f"than this process is running:\n"
        f"  source:  {source_stack}\n"
        f"  running: {running_stack}\n"
        "final_model.pkl carries fitted sklearn and XGBoost estimators, "
        "which do not survive a stack change with their behaviour intact, "
        "and the transfer metrics would mix the release difference into the "
        "transport result. Run the transfer in the environment that produced "
        "the source run (`uv sync --frozen` from the same lockfile), or "
        "re-fit the source run here."
    )


def _warn_on_source_device(from_run: Path, source_config: dict[str, str]) -> None:
    """Say early when the source was fitted on a different XGBoost backend.

    Not fatal, and not part of the stack check above: the pickle loads and
    predicts across backends, and this run's own numbers are computed here,
    so recording the local device is correct. The transport panel, however,
    subtracts this run's metrics from the source run's against a
    pre-specified equivalence margin, and refuses that subtraction across
    devices — which it reaches a minute of bootstrap later.
    """
    source_device = source_config.get("device", "").strip()
    running_device = xgb_device()
    if source_device and source_device not in {"nan", running_device}:
        console.print(
            f"[yellow]Source run {from_run} was fitted on device "
            f"'{source_device}', this process runs on '{running_device}'. The "
            "transfer will complete, but the transport panel compares the two "
            "run directories and will refuse a delta across backends."
        )


def _score_modalities(
    final_model: FinalModelResult,
    X_target: dict[str, pd.DataFrame],
    confounders_target: pd.DataFrame,
) -> np.ndarray:
    """Apply each persisted per-modality pipeline to the target cohort.

    Columns are reindexed to the exact layout captured at source-fit
    time (``modality_column_layout`` in ``final_model.pkl``); columns
    absent from the target fixture are filled with NaN so the KNN
    imputer inside the pipeline can handle them exactly as it does for
    systematically missing source features.
    """
    n_samples = len(confounders_target)
    modality_names = final_model["modality_names"]
    pipelines = final_model["modality_pipelines"]
    col_layout = final_model["modality_column_layout"]
    oof = np.full((n_samples, len(modality_names)), np.nan)

    for i, mod_name in enumerate(modality_names):
        expected_cols = col_layout.get(mod_name, [])
        if not expected_cols:
            logger.warning(
                "Source pipeline for '%s' has no column layout; skipping.",
                mod_name,
            )
            continue
        if mod_name not in X_target:
            logger.warning(
                "Target sample is missing modality '%s' entirely; leaving "
                "column as NaN.  Meta-learner will impute at predict time.",
                mod_name,
            )
            continue
        # Align index to confounder frame so predict_proba rows line up,
        # and align columns to the source-fit layout.  Missing columns
        # become all-NaN, extra columns are dropped.
        aligned = (
            X_target[mod_name]
            .reindex(confounders_target.index)
            .reindex(columns=expected_cols)
        )
        dropped = set(X_target[mod_name].columns) - set(expected_cols)
        if dropped:
            logger.info(
                "Target modality '%s' had %d extra columns not seen at "
                "source-fit time; dropping them: %s",
                mod_name,
                len(dropped),
                sorted(dropped),
            )
        missing = set(expected_cols) - set(X_target[mod_name].columns)
        if missing:
            logger.info(
                "Target modality '%s' is missing %d columns vs. source-fit "
                "layout; imputing as NaN: %s",
                mod_name,
                len(missing),
                sorted(missing),
            )

        all_missing = np.asarray(pd.isna(aligned).all(axis=1).to_numpy())
        has_data = ~all_missing
        if has_data.sum() == 0:
            logger.warning(
                "All target rows for modality '%s' are NaN; skipping.", mod_name
            )
            continue

        pipe = pipelines[mod_name]
        conf_slice = confounders_target.loc[has_data]
        X_slice = aligned.loc[has_data]
        proba = pipe.predict_proba(X_slice, confounders=conf_slice)
        # Binary classifier: take class-1 probability.
        proba_pos = np.asarray(proba)[:, 1]
        oof[has_data, i] = proba_pos

    return oof


def _spearman_ranking(
    source_importance: pd.DataFrame, target_importance: pd.DataFrame
) -> dict[str, Any]:
    """Spearman correlation of modality SHAP rankings source vs. target."""
    merged = source_importance.merge(
        target_importance,
        on="modality",
        how="inner",
        suffixes=("_source", "_target"),
    )
    if len(merged) < 2:
        return {
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n_modalities": len(merged),
            "warning": "Need >=2 shared modalities to compute rank correlation.",
        }
    rho, p = spearmanr(merged["mean_abs_shap_source"], merged["mean_abs_shap_target"])
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "n_modalities": len(merged),
    }


def _bootstrap_shap_shares(
    meta_learner: Any,
    oof_matrix: np.ndarray,
    modality_names: list[str],
    n_bootstrap: int = SHAP_BOOTSTRAP_ITERATIONS,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Bootstrap the per-modality SHAP share on ``oof_matrix``.

    The fitted ``meta_learner`` is reused across resamples (SHAP values
    depend on the model and inputs, not on labels), so this costs one
    TreeExplainer instantiation plus ``n_bootstrap`` SHAP calls.

    Returns one row per modality with the bootstrap-mean share and
    non-parametric 95 % CI (percentile method).
    """
    import shap  # lazy import — keep script import-time light

    rng = np.random.default_rng(random_state)
    X_meta = meta_learner.transform_meta_features(oof_matrix)
    feature_names = list(meta_learner.feature_names_)
    # Fixed lookup for aggregating feature-level importance to modality
    # level — mirror of shap_analysis._aggregate_to_modalities.
    mod_set = set(modality_names)

    explainer = shap.TreeExplainer(meta_learner.meta_model_)

    n = X_meta.shape[0]
    boot_shares: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        shap_values = np.asarray(explainer.shap_values(X_meta[idx]))
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        total = float(mean_abs.sum())
        if total <= 0:
            continue
        mod_imp: dict[str, float] = dict.fromkeys(modality_names, 0.0)
        for fname, imp in zip(feature_names, mean_abs, strict=True):
            if " x " in fname:
                parts = fname.split(" x ")
                n_parts = sum(1 for p in parts if p in mod_set)
                if n_parts > 0:
                    for p in parts:
                        if p in mod_imp:
                            mod_imp[p] += float(imp) / n_parts
            elif fname in mod_imp:
                mod_imp[fname] += float(imp)
        boot_shares.append({k: v / total for k, v in mod_imp.items()})

    boot_df = pd.DataFrame(boot_shares)
    rows: list[dict[str, Any]] = []
    for mod in modality_names:
        if mod not in boot_df.columns:
            rows.append(
                {
                    "modality": mod,
                    "share_mean": float("nan"),
                    "share_ci_lo": float("nan"),
                    "share_ci_hi": float("nan"),
                    "n_bootstrap": len(boot_df),
                }
            )
            continue
        values = boot_df[mod].dropna().to_numpy()
        if len(values) == 0:
            rows.append(
                {
                    "modality": mod,
                    "share_mean": float("nan"),
                    "share_ci_lo": float("nan"),
                    "share_ci_hi": float("nan"),
                    "n_bootstrap": 0,
                }
            )
            continue
        rows.append(
            {
                "modality": mod,
                "share_mean": float(values.mean()),
                "share_ci_lo": float(np.percentile(values, 2.5)),
                "share_ci_hi": float(np.percentile(values, 97.5)),
                "n_bootstrap": len(values),
            }
        )
    return pd.DataFrame(rows)


def _ci_overlap_check(
    source_ci: pd.DataFrame, target_ci: pd.DataFrame
) -> dict[str, Any]:
    """Per-modality 95 % CI-overlap between source and target SHAP shares.

    Transfer success criterion: the Baseline-MH share intervals must
    overlap; a pass means there is no statistically credible deviation
    in the dominant modality's contribution between cohorts.
    """
    merged = source_ci.merge(
        target_ci, on="modality", suffixes=("_source", "_target"), how="inner"
    )
    per_modality: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        s_lo = float(r["share_ci_lo_source"])
        s_hi = float(r["share_ci_hi_source"])
        t_lo = float(r["share_ci_lo_target"])
        t_hi = float(r["share_ci_hi_target"])
        overlaps = (
            bool(np.isfinite([s_lo, s_hi, t_lo, t_hi]).all())
            and s_hi >= t_lo
            and t_hi >= s_lo
        )
        per_modality.append(
            {
                "modality": r["modality"],
                "source_share_mean": float(r["share_mean_source"]),
                "source_ci": [s_lo, s_hi],
                "target_share_mean": float(r["share_mean_target"]),
                "target_ci": [t_lo, t_hi],
                "cis_overlap": overlaps,
            }
        )

    mh_entry = next(
        (p for p in per_modality if p["modality"] == BASELINE_MH_MODALITY), None
    )
    return {
        "baseline_mh_cis_overlap": (
            mh_entry["cis_overlap"] if mh_entry is not None else None
        ),
        "baseline_mh_source_ci": (
            mh_entry["source_ci"] if mh_entry is not None else None
        ),
        "baseline_mh_target_ci": (
            mh_entry["target_ci"] if mh_entry is not None else None
        ),
        "all_modalities_cis_overlap": bool(all(p["cis_overlap"] for p in per_modality)),
        "per_modality": per_modality,
    }


def _bootstrap_target_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = CALIB_BOOTSTRAP_ITERATIONS,
    random_state: int | None = None,
) -> dict[str, tuple[float, float]]:
    """Stratified bootstrap 95 % CIs for calibration slope and intercept.

    Mirrors the logistic-regression calibration formulation used in
    ``EvaluationMetrics._calibration_metrics`` so the point estimates
    are comparable.  Stratified resampling keeps class balance stable
    per resample.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(np.int8)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    slopes: list[float] = []
    intercepts: list[float] = []
    for _ in range(n_bootstrap):
        pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pos_sample, neg_sample])
        proba = np.clip(y_pred[idx], 1e-7, 1 - 1e-7)
        logits = np.log(proba / (1 - proba)).reshape(-1, 1)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        lr = LogisticRegression(C=1e10, max_iter=1000)
        lr.fit(logits, y_b)
        slopes.append(float(lr.coef_[0][0]))
        intercepts.append(float(lr.intercept_[0]))

    slope_arr = np.asarray(slopes)
    intercept_arr = np.asarray(intercepts)
    return {
        "calibration_slope_ci": (
            float(np.percentile(slope_arr, 2.5)),
            float(np.percentile(slope_arr, 97.5)),
        ),
        "calibration_intercept_ci": (
            float(np.percentile(intercept_arr, 2.5)),
            float(np.percentile(intercept_arr, 97.5)),
        ),
    }


def _performance_equivalence_check(
    source_roc_auc: float,
    target_roc_auc: float,
    target_roc_auc_ci: tuple[float, float],
    target_calibration_slope: float,
    target_calibration_slope_ci: tuple[float, float],
    target_calibration_intercept: float,
    target_calibration_intercept_ci: tuple[float, float],
) -> dict[str, Any]:
    """Primary transfer success criterion — TOST on three metrics.

    Each sub-test passes when the target-cohort 95 % CI lies *entirely
    inside* the pre-specified equivalence band.  The overall PASS
    requires all three sub-tests to pass (conservative conjunction).
    """
    # ROC-AUC: target CI ⊂ [source − margin, source + margin]
    roc_band_lo = source_roc_auc - TOST_DELTA_ROC_AUC_MARGIN
    roc_band_hi = source_roc_auc + TOST_DELTA_ROC_AUC_MARGIN
    roc_pass = bool(
        np.isfinite(target_roc_auc_ci).all()
        and target_roc_auc_ci[0] >= roc_band_lo
        and target_roc_auc_ci[1] <= roc_band_hi
    )
    # Calibration slope: CI ⊂ band
    slope_lo_band, slope_hi_band = TOST_CALIBRATION_SLOPE_BAND
    slope_pass = bool(
        np.isfinite(target_calibration_slope_ci).all()
        and target_calibration_slope_ci[0] >= slope_lo_band
        and target_calibration_slope_ci[1] <= slope_hi_band
    )
    # Calibration intercept: CI ⊂ band
    int_lo_band, int_hi_band = TOST_CALIBRATION_INTERCEPT_BAND
    intercept_pass = bool(
        np.isfinite(target_calibration_intercept_ci).all()
        and target_calibration_intercept_ci[0] >= int_lo_band
        and target_calibration_intercept_ci[1] <= int_hi_band
    )

    all_pass = roc_pass and slope_pass and intercept_pass
    return {
        "all_pass": all_pass,
        "delta_roc_auc": {
            "source": float(source_roc_auc),
            "target": float(target_roc_auc),
            "delta_point": float(target_roc_auc - source_roc_auc),
            "target_ci": [float(target_roc_auc_ci[0]), float(target_roc_auc_ci[1])],
            "equivalence_band": [roc_band_lo, roc_band_hi],
            "margin": TOST_DELTA_ROC_AUC_MARGIN,
            "pass": roc_pass,
        },
        "calibration_slope": {
            "target": float(target_calibration_slope),
            "target_ci": [
                float(target_calibration_slope_ci[0]),
                float(target_calibration_slope_ci[1]),
            ],
            "equivalence_band": list(TOST_CALIBRATION_SLOPE_BAND),
            "pass": slope_pass,
        },
        "calibration_intercept": {
            "target": float(target_calibration_intercept),
            "target_ci": [
                float(target_calibration_intercept_ci[0]),
                float(target_calibration_intercept_ci[1]),
            ],
            "equivalence_band": list(TOST_CALIBRATION_INTERCEPT_BAND),
            "pass": intercept_pass,
        },
    }


def _ks_confounder_drift(
    source_conf: pd.DataFrame, target_conf: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Per-column Kolmogorov-Smirnov distance between source and target."""
    drift: dict[str, dict[str, float]] = {}
    common = [c for c in source_conf.columns if c in target_conf.columns]
    for col in common:
        src = source_conf[col].dropna().to_numpy()
        tgt = target_conf[col].dropna().to_numpy()
        if len(src) == 0 or len(tgt) == 0:
            drift[col] = {"ks_statistic": float("nan"), "ks_p": float("nan")}
            continue
        stat, pval = ks_2samp(src, tgt)
        drift[col] = {"ks_statistic": float(stat), "ks_p": float(pval)}
    return drift


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    console.rule("[bold]PCC Transfer-Validation Pipeline")

    # --- Source artifacts ---
    from_run = args.from_run.resolve()
    if not from_run.is_dir():
        console.print(f"[red]Source run directory not found: {from_run}")
        sys.exit(1)
    console.print(f"[bold]Loading source artifacts from {from_run}...")
    final_model, source_config = _load_source_artifacts(from_run)
    source_cohort = source_config.get("cohort", "unknown")
    source_stack = source_config.get("stack_variant", "unknown")
    console.print(f"  Source cohort: {source_cohort}")
    console.print(f"  Source stack:  {source_stack}")
    console.print(f"  Modalities:    {final_model['modality_names']}")

    if source_cohort == "non_mri":
        console.print(
            "[red]Refusing to transfer FROM a non_mri source run — that is "
            "not a supported pathway.  Train on MRI (or 'all') and transfer "
            "to non-MRI, not the other way around."
        )
        sys.exit(2)

    # --- Target data ---
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        console.print(f"[red]{e}")
        sys.exit(1)
    processed_dir = get_processed_dir(config)
    console.print(
        f"[bold]Loading target data (cohort={args.target_cohort}, "
        f"stack={source_stack}) from {processed_dir}..."
    )
    try:
        y_target, X_target, conf_target = load_pipeline_data(
            processed_dir=processed_dir,
            target=_as_target(source_config.get("target", "bahmer")),
            clean_controls=source_config.get("clean_controls", "True") != "False",
            cohort=_as_cohort(args.target_cohort),
            stack_variant=_as_stack_variant(source_stack),
        )
    except ValueError as e:
        console.print(f"[red]Target data load failed: {e}")
        sys.exit(1)

    console.print(f"  Target sample: N={len(y_target):,}")
    console.print(f"  Prevalence:    {float(y_target.mean()) * 100:.1f}%")

    # Align confounder columns to what the source expected.
    source_conf_train: pd.DataFrame = final_model["confounders_train"]
    missing_conf = [
        c for c in source_conf_train.columns if c not in conf_target.columns
    ]
    if missing_conf:
        console.print(
            f"[red]Target confounders missing columns {missing_conf} that "
            "source training relied on.  Aborting."
        )
        sys.exit(1)
    conf_target_aligned = conf_target[source_conf_train.columns.tolist()]

    # --- Inference ---
    console.print("\n[bold]Scoring per-modality pipelines on target...")
    target_oof = _score_modalities(final_model, X_target, conf_target_aligned)

    meta = final_model["meta_learner"]
    console.print("[bold]Applying meta-learner to target modality predictions...")
    y_pred_proba = np.asarray(meta.predict_proba(target_oof))[:, 1]
    y_true = y_target.to_numpy(dtype=np.int8)

    # --- Output directory ---
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        from pcc_analysis.config import get_results_dir

        results_dir = get_results_dir(config)
        source_name = from_run.name
        ts = f"{datetime.now().astimezone():%Y%m%d_%H%M%S}"
        out_dir = results_dir / (
            f"pcc_transfer_{source_name}_to_{args.target_cohort}_{ts}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Writing results to {out_dir}")
    # The same record every pipeline run leaves, for the same reason: this
    # directory is read side by side with the source run in the transport
    # panel, and "which machine produced it" is part of reading a delta.
    write_provenance(out_dir)

    # --- Evaluation (metrics + bootstrap CIs + DCA + plots) ---
    console.print("[bold]Computing metrics + bootstrap CIs + DCA...")
    evaluation = evaluate_model_comprehensive(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        model_name=f"Transfer ({source_cohort}→{args.target_cohort})",
        n_bootstrap=args.n_bootstrap,
        random_state=int(source_config.get("random_state", 42)),
        output_dir=out_dir,
        save_latex=True,
        n_jobs=1,
    )
    pd.DataFrame([evaluation["metrics"]]).to_csv(out_dir / "metrics.csv", index=False)
    ci = evaluation.get("ci_results", {})
    if ci:
        ci_out = {k: [float(v[0]), float(v[1])] for k, v in ci.items()}
        with (out_dir / "confidence_intervals.json").open("w") as f:
            json.dump(ci_out, f, indent=2)
    pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred_proba": y_pred_proba,
            "id": conf_target_aligned.index,
        }
    ).to_csv(out_dir / "predictions.csv", index=False)

    # --- SHAP on target ---
    console.print("[bold]Computing modality SHAP on target predictions...")
    shap_results = compute_meta_learner_shap(
        meta, target_oof, final_model["modality_names"]
    )
    shap_results["modality_importance"].to_csv(
        out_dir / "shap_modality_importance_transfer.csv", index=False
    )
    shap_results["feature_importance"].to_csv(
        out_dir / "shap_feature_importance_transfer.csv", index=False
    )

    # --- Ranking correlation ---
    source_shap_path = from_run / "shap_modality_importance.csv"
    if source_shap_path.exists():
        source_importance = pd.read_csv(source_shap_path)
        ranking = _spearman_ranking(
            source_importance, shap_results["modality_importance"]
        )
    else:
        logger.warning(
            "Source run does not have shap_modality_importance.csv — skipping "
            "ranking correlation."
        )
        ranking = {"warning": "source SHAP csv not found"}

    # --- Primary success criterion: TOST performance equivalence ---
    # Target-cohort 95 % CIs for ROC-AUC (from evaluate_model_comprehensive)
    # plus a fresh stratified bootstrap for calibration slope/intercept.
    # All three intervals must lie entirely inside the pre-specified
    # equivalence bands (margins defined at module top) for the overall
    # transfer PASS.  This is the literature-standard transportability
    # criterion (Steyerberg 2019; Van Calster 2019); the earlier
    # SHAP-share CI-overlap check is demoted to descriptive output
    # below because it becomes structurally conservative at the large
    # N available in the non-MRI arm.
    console.print(
        f"[bold]Bootstrapping target calibration CIs "
        f"({CALIB_BOOTSTRAP_ITERATIONS} resamples)..."
    )
    bootstrap_seed = int(source_config.get("random_state", 42))
    calib_ci = _bootstrap_target_calibration(
        y_true=y_true,
        y_pred=y_pred_proba,
        n_bootstrap=CALIB_BOOTSTRAP_ITERATIONS,
        random_state=bootstrap_seed + 2,
    )
    source_metrics_path = from_run / "metrics.csv"
    if not source_metrics_path.exists():
        console.print(
            f"[red]Source metrics.csv not found at {source_metrics_path}; "
            "cannot run TOST equivalence check."
        )
        sys.exit(1)
    source_metrics = pd.read_csv(source_metrics_path).iloc[0].to_dict()
    target_roc_ci = ci.get("roc_auc_ci", (float("nan"), float("nan")))
    performance_eq = _performance_equivalence_check(
        source_roc_auc=float(source_metrics["roc_auc"]),
        target_roc_auc=float(evaluation["metrics"]["roc_auc"]),
        target_roc_auc_ci=(float(target_roc_ci[0]), float(target_roc_ci[1])),
        target_calibration_slope=float(evaluation["metrics"]["calibration_slope"]),
        target_calibration_slope_ci=calib_ci["calibration_slope_ci"],
        target_calibration_intercept=float(
            evaluation["metrics"]["calibration_intercept"]
        ),
        target_calibration_intercept_ci=calib_ci["calibration_intercept_ci"],
    )

    # --- Descriptive SHAP-share CI profile ---
    # Qualitative only ("MH remains dominant modality with stable
    # contribution across cohorts"), not a pass/fail gate: the large-N
    # CI-overlap test is too strict to be informative here, and the TOST
    # check above carries the formal success criterion.
    console.print(
        f"[bold]Bootstrapping SHAP-share CIs (descriptive, "
        f"{SHAP_BOOTSTRAP_ITERATIONS} resamples each side)..."
    )
    source_oof = np.asarray(final_model["oof_predictions"])
    source_ci = _bootstrap_shap_shares(
        meta,
        source_oof,
        final_model["modality_names"],
        n_bootstrap=SHAP_BOOTSTRAP_ITERATIONS,
        random_state=bootstrap_seed,
    )
    target_ci = _bootstrap_shap_shares(
        meta,
        target_oof,
        final_model["modality_names"],
        n_bootstrap=SHAP_BOOTSTRAP_ITERATIONS,
        random_state=bootstrap_seed + 1,
    )
    source_ci.to_csv(out_dir / "shap_share_ci_source.csv", index=False)
    target_ci.to_csv(out_dir / "shap_share_ci_target.csv", index=False)
    overlap = _ci_overlap_check(source_ci, target_ci)

    # --- Confounder-drift KS distances ---
    ks = _ks_confounder_drift(source_conf_train, conf_target_aligned)

    # --- Provenance + summary ---
    (out_dir / "source_run.txt").write_text(str(from_run) + "\n")

    transfer_config = {
        "mode": "transfer",
        "source_run": str(from_run),
        "source_cohort": source_cohort,
        "source_stack_variant": source_stack,
        "target_cohort": args.target_cohort,
        "target_stack_variant": source_stack,
        "n_target": len(y_target),
        "prevalence_target": float(y_target.mean()),
        "timestamp": datetime.now(UTC).isoformat(),
        "package_versions": package_versions(),
        # Recorded under the same name the pipeline runs use, so the
        # comparability check reads this directory with the same two fields
        # it reads theirs by rather than reporting a gap it cannot judge.
        "device": xgb_device(),
    }
    pd.DataFrame([transfer_config]).to_csv(out_dir / "config.csv", index=False)

    summary: dict[str, Any] = {
        **transfer_config,
        "n_bootstrap_shap": SHAP_BOOTSTRAP_ITERATIONS,
        "n_bootstrap_calibration": CALIB_BOOTSTRAP_ITERATIONS,
        "metrics": {k: _to_json(v) for k, v in evaluation["metrics"].items()},
        "confidence_intervals": {k: [float(v[0]), float(v[1])] for k, v in ci.items()}
        if ci
        else {},
        "performance_equivalence": performance_eq,
        "ranking_correlation_source_vs_target": ranking,
        "shap_share_ci_descriptive": overlap,
        "confounder_ks_drift": ks,
    }
    with (out_dir / "transfer_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=_to_json)

    console.rule("[bold green]Transfer Complete")
    roc = evaluation["metrics"].get("roc_auc", float("nan"))
    pr = evaluation["metrics"].get("pr_auc", float("nan"))
    console.print(f"  ROC-AUC:  {roc:.3f}")
    console.print(f"  PR-AUC:   {pr:.3f}")

    # Primary transfer success criterion
    eq = performance_eq
    overall = "[green]PASS" if eq["all_pass"] else "[red]FAIL"
    console.print(f"  Performance-Equivalence: {overall}")
    roc_info = eq["delta_roc_auc"]
    console.print(
        "    ΔROC-AUC target vs source: "
        f"{roc_info['delta_point']:+.4f}   "
        f"target CI [{roc_info['target_ci'][0]:.3f}, "
        f"{roc_info['target_ci'][1]:.3f}]   "
        f"band [{roc_info['equivalence_band'][0]:.3f}, "
        f"{roc_info['equivalence_band'][1]:.3f}]   "
        f"{'[green]PASS' if roc_info['pass'] else '[red]FAIL'}"
    )
    slope_info = eq["calibration_slope"]
    console.print(
        f"    Calibration slope:   {slope_info['target']:.3f}   "
        f"CI [{slope_info['target_ci'][0]:.3f}, "
        f"{slope_info['target_ci'][1]:.3f}]   "
        f"band [{slope_info['equivalence_band'][0]:.2f}, "
        f"{slope_info['equivalence_band'][1]:.2f}]   "
        f"{'[green]PASS' if slope_info['pass'] else '[red]FAIL'}"
    )
    int_info = eq["calibration_intercept"]
    console.print(
        f"    Calibration intercept: {int_info['target']:+.4f}   "
        f"CI [{int_info['target_ci'][0]:+.3f}, "
        f"{int_info['target_ci'][1]:+.3f}]   "
        f"band [{int_info['equivalence_band'][0]:+.2f}, "
        f"{int_info['equivalence_band'][1]:+.2f}]   "
        f"{'[green]PASS' if int_info['pass'] else '[red]FAIL'}"
    )
    # SHAP-share CI overlap is now descriptive-only; keep a one-line
    # note for the operator but do NOT treat it as a gate.
    mh_overlap = overlap.get("baseline_mh_cis_overlap")
    if mh_overlap is not None:
        src_ci_mh = overlap.get("baseline_mh_source_ci") or [float("nan")] * 2
        tgt_ci_mh = overlap.get("baseline_mh_target_ci") or [float("nan")] * 2
        console.print(
            f"  (descriptive) MH share CI source/target: "
            f"[{src_ci_mh[0]:.3f}, {src_ci_mh[1]:.3f}] / "
            f"[{tgt_ci_mh[0]:.3f}, {tgt_ci_mh[1]:.3f}]"
        )
    console.print(f"  Output:   {out_dir}")


def _to_json(value: Any) -> Any:
    """Serialize numpy scalars, tuples, and NaN into JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == "__main__":
    main()
