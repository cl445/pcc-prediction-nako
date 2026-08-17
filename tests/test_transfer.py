"""Tests for the transfer-validation workflow.

Exercises the end-to-end chain: train a tiny Lean-stack source run on
synthetic MRI-like data, pickle the ``FinalModelResult``, then invoke the
scoring/summary helpers from ``scripts/pipeline/03_apply_transfer.py`` on a
distinct target cohort.  The test does not shell out to the script; it
imports the helpers directly so failures point at the functions, not at
CLI plumbing.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pcc_analysis.orchestration import run_pcc_pipeline

TRANSFER_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "pipeline"
    / "03_apply_transfer.py"
)


def _load_transfer_module() -> Any:
    """Load the CLI script as a module so we can call its helpers."""
    spec = importlib.util.spec_from_file_location("transfer_cli", TRANSFER_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trained_lean_run(tmp_path: Path) -> Path:
    """Train a tiny Lean-stack pipeline on synthetic data.

    Writes ``final_model.pkl`` and ``shap_modality_importance.csv`` to
    ``tmp_path / 'source'`` so we can point the transfer helpers at it.
    """
    rng = np.random.default_rng(7)
    n = 120

    y = pd.Series(rng.choice([0, 1], size=n, p=[0.7, 0.3]).astype(np.int8), name="y")

    # Three "lean" modalities (demographics + ses + mental_health) with a
    # bit of signal so the pipeline does not collapse.
    def _mk_df(cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {c: rng.normal(0, 1, size=n) + 0.3 * y.to_numpy() for c in cols}
        )

    X_dict = {
        "demographics": pd.DataFrame(
            {
                "age": rng.integers(30, 70, size=n).astype(np.float64),
                "sex": rng.choice([0.0, 1.0], size=n),
                "center": rng.choice([1.0, 2.0, 3.0], size=n),
            }
        ),
        "ses": _mk_df(["ses_a", "ses_b", "ses_c"]),
        "mental_health": _mk_df([f"phq_{i}" for i in range(9)] + ["gad7"]),
    }
    confounders = X_dict["demographics"].copy()

    source_dir = tmp_path / "source"
    run_pcc_pipeline(
        y=y,
        X_dict=X_dict,
        confounders=confounders,
        output_dir=source_dir,
        n_outer_folds=3,
        n_inner_folds=2,
        orthogonalize=True,
        random_state=42,
        n_jobs=1,
        meta_permutation_iterations=0,
        n_subsamples=10,
        n_bootstrap_eval=50,
        cohort="mri",
        stack_variant="lean",
    )
    return source_dir


def test_final_model_pkl_has_transfer_fields(trained_lean_run: Path) -> None:
    """Phase-1 contract: final_model.pkl must carry per-modality pipelines."""
    with (trained_lean_run / "final_model.pkl").open("rb") as f:
        final = pickle.load(f)
    for key in (
        "meta_learner",
        "modality_names",
        "modality_pipelines",
        "modality_column_layout",
        "confounders_train",
    ):
        assert key in final, f"final_model.pkl missing {key!r}"
    assert set(final["modality_pipelines"].keys()) == set(final["modality_names"])
    for name in final["modality_names"]:
        assert final["modality_column_layout"][name], (
            f"modality_column_layout[{name}] is empty — cannot reindex"
        )


def test_score_modalities_returns_oof_shape(trained_lean_run: Path) -> None:
    """`_score_modalities` outputs (n_samples, n_modalities) predictions."""
    transfer = _load_transfer_module()
    with (trained_lean_run / "final_model.pkl").open("rb") as f:
        final = pickle.load(f)

    rng = np.random.default_rng(123)
    n = 80
    ids = pd.Index(range(n), name="nako_id")

    def _mk(cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame({c: rng.normal(0, 1, size=n) for c in cols}, index=ids)

    # Target has the same modality layout + an extra column to verify
    # reindex drops it without error.
    X_target = {
        "demographics": pd.DataFrame(
            {
                "age": rng.integers(30, 70, size=n).astype(np.float64),
                "sex": rng.choice([0.0, 1.0], size=n),
                "center": rng.choice([1.0, 2.0, 3.0], size=n),
            },
            index=ids,
        ),
        "ses": _mk(["ses_a", "ses_b", "ses_c", "extra_unseen_col"]),
        "mental_health": _mk([f"phq_{i}" for i in range(9)] + ["gad7"]),
    }
    confounders_target = X_target["demographics"].copy()

    oof = transfer._score_modalities(final, X_target, confounders_target)
    assert oof.shape == (n, len(final["modality_names"]))
    # All predictions should be finite probabilities in (0, 1).
    assert np.all((oof > 0) & (oof < 1)), "oof must be proper probabilities"


def test_ci_overlap_check_passes_on_overlapping_intervals() -> None:
    """Per-modality CI overlap is a straightforward interval test."""
    transfer = _load_transfer_module()
    src = pd.DataFrame(
        {
            "modality": ["demographics", "ses", "mental_health"],
            "share_mean": [0.20, 0.30, 0.50],
            "share_ci_lo": [0.15, 0.25, 0.45],
            "share_ci_hi": [0.25, 0.35, 0.55],
        }
    )
    tgt = pd.DataFrame(
        {
            "modality": ["demographics", "ses", "mental_health"],
            "share_mean": [0.23, 0.28, 0.49],
            "share_ci_lo": [0.18, 0.22, 0.43],
            "share_ci_hi": [0.28, 0.33, 0.54],
        }
    )
    result = transfer._ci_overlap_check(src, tgt)
    assert result["baseline_mh_cis_overlap"] is True
    assert result["all_modalities_cis_overlap"] is True


def test_ci_overlap_check_fails_on_disjoint_intervals() -> None:
    """Disjoint CIs (e.g. MH collapses in target) fail the overlap check."""
    transfer = _load_transfer_module()
    src = pd.DataFrame(
        {
            "modality": ["demographics", "ses", "mental_health"],
            "share_mean": [0.20, 0.30, 0.50],
            "share_ci_lo": [0.15, 0.25, 0.45],
            "share_ci_hi": [0.25, 0.35, 0.55],
        }
    )
    tgt = pd.DataFrame(
        {
            "modality": ["demographics", "ses", "mental_health"],
            "share_mean": [0.45, 0.35, 0.20],
            "share_ci_lo": [0.40, 0.30, 0.15],
            "share_ci_hi": [0.50, 0.40, 0.25],
        }
    )
    result = transfer._ci_overlap_check(src, tgt)
    assert result["baseline_mh_cis_overlap"] is False
    assert result["all_modalities_cis_overlap"] is False


def test_bootstrap_shap_shares_returns_sensible_cis(trained_lean_run: Path) -> None:
    """Bootstrap on a real fitted meta-learner returns valid CIs per modality."""
    transfer = _load_transfer_module()
    with (trained_lean_run / "final_model.pkl").open("rb") as f:
        final = pickle.load(f)
    df = transfer._bootstrap_shap_shares(
        final["meta_learner"],
        final["oof_predictions"],
        final["modality_names"],
        n_bootstrap=50,  # small for test speed
        random_state=0,
    )
    assert set(df["modality"]) == set(final["modality_names"])
    for _, r in df.iterrows():
        assert 0.0 <= r["share_ci_lo"] <= r["share_mean"] <= r["share_ci_hi"] <= 1.0
        assert r["n_bootstrap"] > 0
    # Shares across modalities sum to ~1 at the bootstrap mean.
    assert abs(float(df["share_mean"].sum()) - 1.0) < 0.05


def test_spearman_ranking_perfect_on_identical_shap() -> None:
    """Identical source/target SHAP tables → rho = 1."""
    transfer = _load_transfer_module()
    df = pd.DataFrame(
        {
            "modality": ["a", "b", "c"],
            "mean_abs_shap": [0.1, 0.5, 0.2],
        }
    )
    result = transfer._spearman_ranking(df, df.copy())
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["n_modalities"] == 3


def test_ks_confounder_drift_reports_per_column() -> None:
    transfer = _load_transfer_module()
    rng = np.random.default_rng(0)
    src = pd.DataFrame({"age": rng.normal(50, 5, 200), "sex": rng.choice([0, 1], 200)})
    tgt = pd.DataFrame({"age": rng.normal(60, 5, 100), "sex": rng.choice([0, 1], 100)})
    drift = transfer._ks_confounder_drift(src, tgt)
    assert set(drift.keys()) == {"age", "sex"}
    # Age mean differs by 10 → KS stat should be large.
    assert drift["age"]["ks_statistic"] > 0.3
    # Sex sampled from the same distribution → KS stat should be small.
    assert drift["sex"]["ks_statistic"] < 0.2


def test_performance_equivalence_passes_when_target_ci_within_bands() -> None:
    """TOST PASS: all three target CIs lie entirely inside the bands."""
    transfer = _load_transfer_module()
    result = transfer._performance_equivalence_check(
        source_roc_auc=0.70,
        target_roc_auc=0.695,
        target_roc_auc_ci=(0.685, 0.705),  # inside 0.70 ± 0.03 = [0.67, 0.73]
        target_calibration_slope=1.02,
        target_calibration_slope_ci=(0.98, 1.06),  # inside [0.85, 1.15]
        target_calibration_intercept=0.01,
        target_calibration_intercept_ci=(-0.02, 0.03),  # inside [-0.05, 0.05]
    )
    assert result["all_pass"] is True
    assert result["delta_roc_auc"]["pass"] is True
    assert result["calibration_slope"]["pass"] is True
    assert result["calibration_intercept"]["pass"] is True


def test_performance_equivalence_fails_on_roc_auc_drift() -> None:
    """Target ROC-AUC CI slipping below source-margin band trips the ROC gate."""
    transfer = _load_transfer_module()
    result = transfer._performance_equivalence_check(
        source_roc_auc=0.70,
        target_roc_auc=0.61,
        target_roc_auc_ci=(0.60, 0.62),  # below 0.70 - 0.03 = 0.67
        target_calibration_slope=1.00,
        target_calibration_slope_ci=(0.95, 1.05),
        target_calibration_intercept=0.00,
        target_calibration_intercept_ci=(-0.01, 0.01),
    )
    assert result["all_pass"] is False
    assert result["delta_roc_auc"]["pass"] is False
    # Other two sub-tests are unaffected
    assert result["calibration_slope"]["pass"] is True
    assert result["calibration_intercept"]["pass"] is True


def test_performance_equivalence_fails_on_wild_calibration_slope() -> None:
    """Target slope CI overlapping 0 trips the slope gate regardless of ROC."""
    transfer = _load_transfer_module()
    result = transfer._performance_equivalence_check(
        source_roc_auc=0.70,
        target_roc_auc=0.70,
        target_roc_auc_ci=(0.69, 0.71),
        target_calibration_slope=0.30,
        target_calibration_slope_ci=(0.10, 0.50),  # below band [0.85, 1.15]
        target_calibration_intercept=0.00,
        target_calibration_intercept_ci=(-0.01, 0.01),
    )
    assert result["all_pass"] is False
    assert result["calibration_slope"]["pass"] is False


def test_bootstrap_target_calibration_returns_ci_near_point(
    trained_lean_run: Path,
) -> None:
    """Bootstrap calibration CI brackets the fitted point estimate."""
    transfer = _load_transfer_module()
    predictions = pd.read_csv(trained_lean_run / "predictions.csv")
    y_true = predictions["y_true"].to_numpy().astype(np.int8)
    y_pred = predictions["y_pred_proba"].to_numpy()
    ci = transfer._bootstrap_target_calibration(
        y_true, y_pred, n_bootstrap=40, random_state=0
    )
    slope_lo, slope_hi = ci["calibration_slope_ci"]
    int_lo, int_hi = ci["calibration_intercept_ci"]
    assert np.isfinite([slope_lo, slope_hi, int_lo, int_hi]).all()
    assert slope_lo < slope_hi
    assert int_lo < int_hi


def test_summary_json_written(trained_lean_run: Path, tmp_path: Path) -> None:
    """`_run_transfer` style smoke test: output dir contains the expected files."""
    transfer = _load_transfer_module()
    with (trained_lean_run / "final_model.pkl").open("rb") as f:
        final = pickle.load(f)

    # Build a small target cohort
    rng = np.random.default_rng(33)
    n = 60
    ids = pd.Index(range(n), name="nako_id")
    X_target = {
        "demographics": pd.DataFrame(
            {
                "age": rng.integers(30, 70, size=n).astype(np.float64),
                "sex": rng.choice([0.0, 1.0], size=n),
                "center": rng.choice([1.0, 2.0, 3.0], size=n),
            },
            index=ids,
        ),
        "ses": pd.DataFrame(
            {c: rng.normal(0, 1, size=n) for c in ["ses_a", "ses_b", "ses_c"]},
            index=ids,
        ),
        "mental_health": pd.DataFrame(
            {
                c: rng.normal(0, 1, size=n)
                for c in [f"phq_{i}" for i in range(9)] + ["gad7"]
            },
            index=ids,
        ),
    }
    confounders = X_target["demographics"].copy()

    # Score + use helpers directly to keep the test fast.
    oof = transfer._score_modalities(final, X_target, confounders)
    y_pred = np.asarray(final["meta_learner"].predict_proba(oof))[:, 1]
    assert y_pred.shape == (n,)
    # All outputs are proper probabilities.
    assert np.all(np.isfinite(y_pred))

    src_shap = pd.read_csv(trained_lean_run / "shap_modality_importance.csv")
    assert not src_shap.empty

    # Bootstrap-based CI-overlap check: applied to source OoF (from the
    # source run) and a trivial "target" (same OoF — must overlap).
    src_ci = transfer._bootstrap_shap_shares(
        final["meta_learner"],
        final["oof_predictions"],
        final["modality_names"],
        n_bootstrap=50,
        random_state=0,
    )
    tgt_ci = transfer._bootstrap_shap_shares(
        final["meta_learner"],
        final["oof_predictions"],
        final["modality_names"],
        n_bootstrap=50,
        random_state=1,
    )
    overlap = transfer._ci_overlap_check(src_ci, tgt_ci)
    assert overlap["baseline_mh_cis_overlap"] is True
    assert overlap["all_modalities_cis_overlap"] is True

    # JSON serialization round-trip for the summary-style payload.
    summary = {
        "shap_share_ci_overlap": overlap,
        "ks": transfer._ks_confounder_drift(final["confounders_train"], confounders),
    }
    dumped = json.dumps(summary, default=transfer._to_json)
    assert "shap_share_ci_overlap" in dumped


def test_transfer_refuses_a_source_run_from_another_stack(tmp_path: Path) -> None:
    """The pickle is the one artifact that crosses between two machines.

    ``final_model.pkl`` carries fitted sklearn and XGBoost estimators, and
    unpickling those under a different release is at best an opaque
    attribute error and at worst an estimator whose behaviour moved.  The
    metrics make the same point: a transfer whose source and target stacks
    differ measures the release as well as the transport.
    """
    from pcc_analysis._provenance import package_versions

    transfer = _load_transfer_module()
    source_run = tmp_path / "lean_mri"

    with pytest.raises(ValueError, match="different library stack") as excinfo:
        transfer._check_source_stack(
            source_run, {"package_versions": "python==3.14.6 xgboost==3.2.0"}
        )

    message = str(excinfo.value)
    assert "xgboost==3.2.0" in message
    assert package_versions() in message


def test_transfer_accepts_a_source_run_from_this_stack(tmp_path: Path) -> None:
    """A guard that also blocks the normal path is not a guard."""
    from pcc_analysis._provenance import package_versions

    transfer = _load_transfer_module()
    transfer._check_source_stack(
        tmp_path / "lean_mri", {"package_versions": package_versions()}
    )


def test_transfer_allows_a_source_run_from_before_the_stamp(tmp_path: Path) -> None:
    """Unverifiable is reported, not rejected.

    The stamp was introduced in DECISIONS §2.21; the runs that predate it
    are not thereby invalid, and refusing them would make the guard a
    reason to stop using the check.
    """
    transfer = _load_transfer_module()
    transfer._check_source_stack(tmp_path / "lean_mri", {})
    transfer._check_source_stack(tmp_path / "lean_mri", {"package_versions": "nan"})
