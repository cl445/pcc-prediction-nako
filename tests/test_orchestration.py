"""Tests for orchestration permutation test on CV OOF predictions."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pcc_analysis.meta_learner import META_HYPERPARAM_ITERATIONS, MetaLearner
from pcc_analysis.orchestration import (
    PCCMultimodalPipeline,
    merge_fold_results,
    run_pcc_pipeline,
)
from pcc_analysis.run_comparability import fingerprint_analysis_data

if TYPE_CHECKING:
    from pcc_analysis._types import FoldArtifact, PerFoldOoS


@pytest.fixture
def pipeline(tmp_path: Path) -> PCCMultimodalPipeline:
    """Minimal pipeline instance with small permutation count."""
    return PCCMultimodalPipeline(
        output_dir=tmp_path,
        meta_permutation_iterations=200,
        random_state=42,
    )


def test_permutation_test_cv_perfect_predictions(
    pipeline: PCCMultimodalPipeline,
) -> None:
    """Perfect predictions should yield a small p-value."""
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 50 + [1] * 50, dtype=np.float64)
    # Predictions that perfectly separate the classes
    y_pred = np.where(
        y_true == 1, rng.uniform(0.8, 1.0, 100), rng.uniform(0.0, 0.2, 100)
    )

    result = pipeline._permutation_test_cv(y_true, y_pred)

    assert result["p_value"] < 0.05
    assert result["score"] > 0.9


def test_permutation_test_cv_random_predictions(
    pipeline: PCCMultimodalPipeline,
) -> None:
    """Random predictions should yield a large p-value."""
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 50 + [1] * 50, dtype=np.float64)
    # Predictions uncorrelated with labels
    y_pred = rng.uniform(0, 1, 100)

    result = pipeline._permutation_test_cv(y_true, y_pred)

    assert result["p_value"] > 0.01


def test_permutation_test_cv_structure(
    pipeline: PCCMultimodalPipeline,
) -> None:
    """Return dict must contain expected keys and output file must exist."""
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 50 + [1] * 50, dtype=np.float64)
    y_pred = rng.uniform(0, 1, 100)

    result = pipeline._permutation_test_cv(y_true, y_pred)

    assert set(result.keys()) == {"score", "p_value", "perm_mean", "n_permutations"}
    assert result["n_permutations"] == 200
    assert 0 <= result["p_value"] <= 1
    assert isinstance(result["score"], float)
    assert isinstance(result["perm_mean"], float)

    # Check JSON output file
    json_path = pipeline.output_dir / "meta_permutation_test.json"
    assert json_path.exists()
    saved = json.loads(json_path.read_text())
    assert saved["score"] == result["score"]
    assert saved["p_value"] == result["p_value"]


# ------------------------------------------------------------------
# Permutation Dropout Tests
# ------------------------------------------------------------------


@pytest.fixture
def fitted_meta_and_oof() -> tuple[MetaLearner, np.ndarray, np.ndarray, list[str]]:
    """Fitted MetaLearner with OoF predictions for dropout/incremental tests."""
    rng = np.random.default_rng(42)
    n = 120
    y = np.zeros(n, dtype=np.float64)
    y[rng.choice(n, size=40, replace=False)] = 1.0

    modality_names = ["demographics", "lab_values", "cognitive"]
    oof = np.column_stack(
        [
            # demographics: weak signal
            y * 0.2 + rng.uniform(0, 0.8, n),
            # lab_values: strong signal
            y * 0.6 + rng.uniform(0, 0.4, n),
            # cognitive: medium signal
            y * 0.4 + rng.uniform(0, 0.6, n),
        ]
    )

    meta = MetaLearner(
        add_interactions=False,
        hyperparam_iterations=0,
        random_state=42,
    )
    meta.fit(oof, y, feature_names=modality_names)
    return meta, oof, y, modality_names


def test_modality_ablation_shape(
    pipeline: PCCMultimodalPipeline,
    fitted_meta_and_oof: tuple[MetaLearner, np.ndarray, np.ndarray, list[str]],
) -> None:
    """Should return one row per modality."""
    meta, oof, y, names = fitted_meta_and_oof
    result = pipeline._modality_ablation(meta, oof, y, names)

    assert result is not None
    assert len(result) == len(names)
    assert "modality" in result.columns
    assert "delta_pr_auc" in result.columns


def test_modality_ablation_signal_modality_largest_delta(
    pipeline: PCCMultimodalPipeline,
    fitted_meta_and_oof: tuple[MetaLearner, np.ndarray, np.ndarray, list[str]],
) -> None:
    """The strongest-signal modality should have the largest ablation delta."""
    meta, oof, y, names = fitted_meta_and_oof
    result = pipeline._modality_ablation(meta, oof, y, names)

    assert result is not None
    # lab_values has strongest signal (0.6 coefficient); it should rank above
    # pure-noise modalities, though stochastic tie-breaking may put demographics
    # (medium signal) on top occasionally
    top_modality = result.iloc[0]["modality"]
    assert top_modality in ("lab_values", "demographics")


def test_modality_ablation_saves_csv(
    pipeline: PCCMultimodalPipeline,
    fitted_meta_and_oof: tuple[MetaLearner, np.ndarray, np.ndarray, list[str]],
) -> None:
    """Should save a CSV file."""
    meta, oof, y, names = fitted_meta_and_oof
    pipeline._modality_ablation(meta, oof, y, names)

    csv_path = pipeline.output_dir / "modality_ablation.csv"
    assert csv_path.exists()


# ------------------------------------------------------------------
# Incremental Performance Tests
# ------------------------------------------------------------------


def test_incremental_performance_shape(
    pipeline: PCCMultimodalPipeline,
    fitted_meta_and_oof: tuple[MetaLearner, np.ndarray, np.ndarray, list[str]],
) -> None:
    """Should return k-1 rows (baseline not included)."""
    _, oof, y, names = fitted_meta_and_oof
    result = pipeline._incremental_performance(oof, y, names)

    assert result is not None
    # 3 modalities - 1 baseline = 2 rows
    assert len(result) == 2
    assert "demographics" not in result["modality"].values


def test_incremental_performance_columns(
    pipeline: PCCMultimodalPipeline,
    fitted_meta_and_oof: tuple[MetaLearner, np.ndarray, np.ndarray, list[str]],
) -> None:
    """Should have expected columns."""
    _, oof, y, names = fitted_meta_and_oof
    result = pipeline._incremental_performance(oof, y, names)

    assert result is not None
    expected_cols = {"modality", "pr_auc_baseline", "pr_auc_augmented", "delta_pr_auc"}
    assert expected_cols.issubset(set(result.columns))


def test_incremental_performance_saves_csv(
    pipeline: PCCMultimodalPipeline,
    fitted_meta_and_oof: tuple[MetaLearner, np.ndarray, np.ndarray, list[str]],
) -> None:
    """Should save a CSV file."""
    _, oof, y, names = fitted_meta_and_oof
    pipeline._incremental_performance(oof, y, names)

    csv_path = pipeline.output_dir / "incremental_performance.csv"
    assert csv_path.exists()


# ------------------------------------------------------------------
# Out-of-Sample Ablation / Incremental Tests
# ------------------------------------------------------------------


@pytest.fixture
def per_fold_oos_data() -> list[PerFoldOoS]:
    """Three synthetic folds with train-OoF, test-preds, and fitted meta-learners."""
    rng = np.random.default_rng(42)
    n = 300
    y = np.zeros(n, dtype=np.float64)
    y[rng.choice(n, size=120, replace=False)] = 1.0
    modality_names = ["demographics", "lab_values", "cognitive"]
    # Full stacking matrix — per-fold meta sees train slice, tests on held-out slice
    oof_full = np.column_stack(
        [
            y * 0.2 + rng.uniform(0, 0.8, n),
            y * 0.6 + rng.uniform(0, 0.4, n),
            y * 0.4 + rng.uniform(0, 0.6, n),
        ]
    )

    # 3 folds; test = fold's slice, train = the rest
    folds: list[PerFoldOoS] = []
    fold_size = n // 3
    for k in range(3):
        test_idx = np.arange(k * fold_size, (k + 1) * fold_size)
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        meta = MetaLearner(
            add_interactions=False,
            hyperparam_iterations=0,
            random_state=42,
        )
        meta.fit(oof_full[train_idx], y[train_idx], feature_names=modality_names)
        folds.append(
            {
                "fold_idx": k,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "y_tr": y[train_idx],
                "y_te": y[test_idx],
                "oof_train": oof_full[train_idx],
                "test_preds": oof_full[test_idx],
                "modality_names": modality_names,
                "meta": meta,
            }
        )
    return folds


def test_modality_ablation_oos_shape(
    pipeline: PCCMultimodalPipeline,
    per_fold_oos_data: list[PerFoldOoS],
) -> None:
    """Should return one row per modality and save CSV."""
    result = pipeline._modality_ablation_oos(per_fold_oos_data)

    assert result is not None
    assert len(result) == 3  # one row per modality
    assert {"modality", "delta_pr_auc", "pr_auc_full", "pr_auc_dropped"}.issubset(
        result.columns
    )
    csv_path = pipeline.output_dir / "modality_ablation_oos.csv"
    assert csv_path.exists()


def test_modality_ablation_oos_empty_folds(
    pipeline: PCCMultimodalPipeline,
) -> None:
    """Empty fold list returns None silently (distributed runs)."""
    assert pipeline._modality_ablation_oos([]) is None


def test_incremental_performance_oos_shape(
    pipeline: PCCMultimodalPipeline,
    per_fold_oos_data: list[PerFoldOoS],
) -> None:
    """Should return k-1 rows (baseline excluded) and save CSV."""
    result = pipeline._incremental_performance_oos(per_fold_oos_data)

    assert result is not None
    assert len(result) == 2  # 3 modalities - 1 baseline
    assert "demographics" not in result["modality"].values
    assert {"modality", "pr_auc_baseline", "pr_auc_augmented", "delta_pr_auc"}.issubset(
        result.columns
    )
    csv_path = pipeline.output_dir / "incremental_performance_oos.csv"
    assert csv_path.exists()


def test_incremental_performance_oos_empty_folds(
    pipeline: PCCMultimodalPipeline,
) -> None:
    """Empty fold list returns None silently."""
    assert pipeline._incremental_performance_oos([]) is None


# ------------------------------------------------------------------
# Distributed Fold Computation Tests
# ------------------------------------------------------------------


def test_config_hash_deterministic(pipeline: PCCMultimodalPipeline) -> None:
    """Config hash should be deterministic across calls."""
    h1 = pipeline._config_hash()
    h2 = pipeline._config_hash()
    assert h1 == h2
    assert len(h1) == 16


def test_config_records_package_versions(pipeline: PCCMultimodalPipeline) -> None:
    """Every run must be attributable to the stack that produced it."""
    stamp = pipeline._get_config()["package_versions"]
    for package in ("python", "numpy", "scikit-learn", "xgboost"):
        assert f"{package}==" in stamp, f"{package} missing from {stamp!r}"
    assert "absent" not in stamp, "the pinned stack must be fully installed"


def test_config_hash_differs_across_package_versions(
    pipeline: PCCMultimodalPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Folds computed against different stacks must not merge silently."""
    baseline = pipeline._config_hash()
    monkeypatch.setattr(
        "pcc_analysis.orchestration.package_versions", lambda: "xgboost==0.0.0"
    )
    assert pipeline._config_hash() != baseline


def test_config_hash_differs_across_xgboost_devices(
    pipeline: PCCMultimodalPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GPU machine and a CPU machine do not compute the same folds.

    The device is detected, not configured, so it is the one thing about a
    distributed run that differs without anyone choosing it — and without
    the stamp, nothing in the run directory would say which backend built
    the trees.
    """
    baseline = pipeline._config_hash()
    monkeypatch.setattr("pcc_analysis.orchestration.xgb_device", lambda: "cuda")
    on_gpu = pipeline._config_hash()
    monkeypatch.setattr("pcc_analysis.orchestration.xgb_device", lambda: "cpu")
    on_cpu = pipeline._config_hash()

    assert on_gpu != on_cpu
    assert baseline in (on_gpu, on_cpu)


def test_config_records_the_device_that_fitted_the_model(
    pipeline: PCCMultimodalPipeline,
) -> None:
    """``config.csv`` has to name the backend, not just imply it."""
    assert pipeline._get_config()["device"] in {"cpu", "cuda"}


def test_config_hash_differs_for_different_config(tmp_path: Path) -> None:
    """Different pipeline configs should produce different hashes."""
    p1 = PCCMultimodalPipeline(
        output_dir=tmp_path / "a",
        n_outer_folds=5,
        random_state=42,
    )
    p2 = PCCMultimodalPipeline(
        output_dir=tmp_path / "b",
        n_outer_folds=10,
        random_state=42,
    )
    assert p1._config_hash() != p2._config_hash()


def test_config_records_the_load_time_analysis_configuration(
    tmp_path: Path,
) -> None:
    """``config.csv`` has to name what makes one run differ from another.

    ``target`` and ``clean_controls`` are applied upstream in
    ``load_pipeline_data``, so unless they are carried through as
    provenance tags the primary run and the two outcome-sensitivity runs
    write identical configuration records.
    """
    pipeline = PCCMultimodalPipeline(
        output_dir=tmp_path / "run",
        target="neurocog",
        clean_controls=False,
        split_mh_submodalities=True,
    )

    cfg = pipeline._get_config()

    assert cfg["target"] == "neurocog"
    assert cfg["clean_controls"] is False
    assert cfg["split_mh_submodalities"] is True


def test_hyperparameter_search_width_reaches_the_meta_learner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike the other tags this one is applied, not just recorded.

    The earlier run searches 10 draws where the runs after it search 50,
    and a different search refits the meta-learner on a different model. A
    control run reproducing the earlier width therefore needs the count
    threaded all the way through rather than read off the module constant.
    """
    import pandas as pd

    from pcc_analysis import orchestration
    from pcc_analysis.meta_learner import train_meta_learner

    seen: list[int] = []

    def record(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["hyperparam_iterations"])
        return train_meta_learner(*args, **kwargs)

    monkeypatch.setattr(orchestration, "train_meta_learner", record)

    rng = np.random.default_rng(0)
    n = 80
    y = pd.Series(rng.choice([0, 1], size=n, p=[0.7, 0.3]).astype(np.int8))
    X_dict = {
        "demographics": pd.DataFrame(
            {"age": rng.integers(18, 80, n).astype(np.float64)}
        ),
        "lab_values": pd.DataFrame({"lab": rng.normal(5, 1, n)}),
    }

    run_pcc_pipeline(
        y=y,
        X_dict=X_dict,
        output_dir=tmp_path / "control",
        n_outer_folds=2,
        n_inner_folds=2,
        orthogonalize=False,
        n_jobs=1,
        meta_permutation_iterations=0,
        n_subsamples=5,
        n_bootstrap_eval=10,
        hyperparam_iterations=0,
    )

    assert seen
    assert set(seen) == {0}


def test_recorded_search_width_is_the_width_that_is_searched(
    tmp_path: Path,
) -> None:
    """``config.csv`` must not claim draws the meta-learner never took.

    The randomised search caps itself at ``META_HYPERPARAM_ITERATIONS``, so a
    request for 200 fits exactly the model a request for 50 fits. Recording
    the raw request would put a search that never ran into the run directory
    and split the config hash between fold sets holding identical models,
    refusing a merge that is in fact sound.
    """
    requested_high = PCCMultimodalPipeline(
        output_dir=tmp_path / "requested_high",
        hyperparam_iterations=4 * META_HYPERPARAM_ITERATIONS,
    )
    at_the_cap = PCCMultimodalPipeline(
        output_dir=tmp_path / "at_the_cap",
        hyperparam_iterations=META_HYPERPARAM_ITERATIONS,
    )

    assert requested_high.hyperparam_iterations == META_HYPERPARAM_ITERATIONS
    assert (
        requested_high._get_config()["hyperparam_iterations"]
        == META_HYPERPARAM_ITERATIONS
    )
    assert requested_high._config_hash() == at_the_cap._config_hash()


def test_untuned_search_width_stays_addressable(tmp_path: Path) -> None:
    """Zero is a setting, not an absent one: it selects XGBoost defaults."""
    untuned = PCCMultimodalPipeline(
        output_dir=tmp_path / "untuned",
        hyperparam_iterations=0,
    )

    assert untuned.hyperparam_iterations == 0
    assert untuned._get_config()["hyperparam_iterations"] == 0


def test_negative_search_width_is_rejected(tmp_path: Path) -> None:
    """A negative count behaves like zero; accepting it records a lie."""
    with pytest.raises(ValueError, match="hyperparam_iterations"):
        PCCMultimodalPipeline(
            output_dir=tmp_path / "negative",
            hyperparam_iterations=-1,
        )


def test_config_hash_separates_the_outcome_and_sample_definitions(
    tmp_path: Path,
) -> None:
    """Fold artifacts from two outcome definitions must not merge silently."""
    primary = PCCMultimodalPipeline(output_dir=tmp_path / "primary")
    neurocog = PCCMultimodalPipeline(
        output_dir=tmp_path / "neurocog",
        target="neurocog",
    )
    mixed = PCCMultimodalPipeline(
        output_dir=tmp_path / "mixed",
        clean_controls=False,
    )
    split_mh = PCCMultimodalPipeline(
        output_dir=tmp_path / "split_mh",
        split_mh_submodalities=True,
    )

    hashes = {
        pipeline._config_hash() for pipeline in (primary, neurocog, mixed, split_mh)
    }
    assert len(hashes) == 4


def test_save_and_load_fold_artifact(pipeline: PCCMultimodalPipeline) -> None:
    """Fold artifact should round-trip through pickle correctly."""
    artifact: FoldArtifact = {
        "fold_idx": 2,
        "test_idx": np.array([10, 20, 30]),
        "y_pred_fold": np.array([0.1, 0.9, 0.5]),
        "y_te": np.array([0.0, 1.0, 1.0]),
        "fold_result": {"fold": 3, "roc_auc": 0.85, "pr_auc": 0.72},
        "modality_scores_fold": [
            {"fold": 3, "modality": "demo", "roc_auc": 0.8, "pr_auc": 0.65}
        ],
        "fold_perm_importance": None,
        "config_hash": pipeline._config_hash(),
    }

    pipeline._save_fold_artifact(2, artifact)

    pkl_path = pipeline.output_dir / "fold_2.pkl"
    assert pkl_path.exists()

    with pkl_path.open("rb") as f:
        loaded = pickle.load(f)

    assert loaded["fold_idx"] == 2
    np.testing.assert_array_equal(loaded["test_idx"], artifact["test_idx"])
    np.testing.assert_array_equal(loaded["y_pred_fold"], artifact["y_pred_fold"])
    np.testing.assert_array_equal(loaded["y_te"], artifact["y_te"])
    assert loaded["fold_result"] == artifact["fold_result"]
    assert loaded["config_hash"] == artifact["config_hash"]


def test_fold_indices_validation(tmp_path: Path) -> None:
    """Out-of-range fold indices should raise ValueError."""
    pipeline = PCCMultimodalPipeline(
        output_dir=tmp_path,
        n_outer_folds=5,
        random_state=42,
    )
    import pandas as pd

    # Minimal data just to trigger validation before any computation
    dummy_y = pd.Series([0, 1], name="y")
    dummy_X = {"demo": pd.DataFrame({"f1": [0.0, 1.0]})}

    with pytest.raises(ValueError, match="out of range"):
        pipeline.run(
            dummy_y,
            dummy_X,
            fold_subset=[0, 7],
        )

    with pytest.raises(ValueError, match="out of range"):
        pipeline.run(
            dummy_y,
            dummy_X,
            fold_subset=[-1],
        )


def test_merge_missing_folds(tmp_path: Path) -> None:
    """Merging with missing folds should raise ValueError listing what's missing."""
    import pandas as pd

    fold_dir = tmp_path / "folds"
    fold_dir.mkdir()

    # Create only fold 0 and 2 (missing 1, 3, 4)
    for idx in [0, 2]:
        artifact: FoldArtifact = {
            "fold_idx": idx,
            "test_idx": np.array([idx]),
            "y_pred_fold": np.array([0.5]),
            "y_te": np.array([1.0]),
            "fold_result": {"fold": idx + 1, "roc_auc": 0.5, "pr_auc": 0.5},
            "modality_scores_fold": [],
            "fold_perm_importance": None,
            "config_hash": "abc123",
        }
        with (fold_dir / f"fold_{idx}.pkl").open("wb") as f:
            pickle.dump(artifact, f)

    dummy_y = pd.Series([0, 1], name="y")
    dummy_X = {"demo": pd.DataFrame({"f1": [0.0, 1.0]})}

    with pytest.raises(ValueError, match=r"Missing folds.*1.*3.*4"):
        merge_fold_results(
            fold_dir=fold_dir,
            y=dummy_y,
            X_dict=dummy_X,
            n_outer_folds=5,
        )


def test_merge_config_mismatch(tmp_path: Path) -> None:
    """Merging folds with different config hashes should raise ValueError."""
    import pandas as pd

    fold_dir = tmp_path / "folds"
    fold_dir.mkdir()

    for idx in range(5):
        artifact: FoldArtifact = {
            "fold_idx": idx,
            "test_idx": np.array([idx]),
            "y_pred_fold": np.array([0.5]),
            "y_te": np.array([1.0]),
            "fold_result": {"fold": idx + 1, "roc_auc": 0.5, "pr_auc": 0.5},
            "modality_scores_fold": [],
            "fold_perm_importance": None,
            # Different hash for fold 3
            "config_hash": "different_hash" if idx == 3 else "same_hash",
        }
        with (fold_dir / f"fold_{idx}.pkl").open("wb") as f:
            pickle.dump(artifact, f)

    dummy_y = pd.Series([0, 1], name="y")
    dummy_X = {"demo": pd.DataFrame({"f1": [0.0, 1.0]})}

    with pytest.raises(ValueError, match="Config mismatch"):
        merge_fold_results(
            fold_dir=fold_dir,
            y=dummy_y,
            X_dict=dummy_X,
            n_outer_folds=5,
        )


def test_merge_no_fold_files(tmp_path: Path) -> None:
    """Merging from empty directory should raise FileNotFoundError."""
    import pandas as pd

    fold_dir = tmp_path / "empty"
    fold_dir.mkdir()

    dummy_y = pd.Series([0, 1], name="y")
    dummy_X = {"demo": pd.DataFrame({"f1": [0.0, 1.0]})}

    with pytest.raises(FileNotFoundError, match=r"No fold_.*pkl"):
        merge_fold_results(
            fold_dir=fold_dir,
            y=dummy_y,
            X_dict=dummy_X,
        )


def _skip_post_cv_steps(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Stand in for the half of a merge that fits models.

    The configuration and provenance checks happen before it, so tests aimed
    at them do not need minutes of SHAP and bootstrap work behind them.
    """
    return {}


def _write_mergeable_folds(fold_dir: Path, config_hash: str, n_folds: int = 5) -> None:
    """Write *n_folds* artifacts that agree on *config_hash*.

    One participant per fold, labelled ``1`` throughout, so the merge's
    ``y_te`` reassembly check passes and the configuration checks are what
    the merge is left to decide on.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(n_folds):
        artifact: FoldArtifact = {
            "fold_idx": idx,
            "test_idx": np.array([idx]),
            "y_pred_fold": np.array([0.5]),
            "y_te": np.array([1.0]),
            "fold_result": {"fold": idx + 1, "roc_auc": 0.5, "pr_auc": 0.5},
            "modality_scores_fold": [],
            "fold_perm_importance": None,
            "config_hash": config_hash,
        }
        with (fold_dir / f"fold_{idx}.pkl").open("wb") as f:
            pickle.dump(artifact, f)


def test_merge_rejects_a_configuration_the_folds_never_saw(tmp_path: Path) -> None:
    """The merge fits the final model, so its config has to be the folds'.

    Fold hashes agreeing with each other says nothing about the merge.
    ``--merge-folds`` invoked without the flags the fold runs were given
    passes that check and then fits the meta-learner, and writes
    ``config.csv``, under the defaults — the unattributable run the config
    hash exists to prevent.
    """
    import pandas as pd

    fold_run = PCCMultimodalPipeline(
        output_dir=tmp_path / "fold_run",
        n_outer_folds=5,
        hyperparam_iterations=10,
        amendment_features=False,
    )
    fold_dir = tmp_path / "folds"
    _write_mergeable_folds(fold_dir, fold_run._config_hash())

    y = pd.Series([1, 1, 1, 1, 1], name="y")
    X_dict = {"demographics": pd.DataFrame({"age": [40.0, 50.0, 60.0, 30.0, 70.0]})}

    merged_dir = tmp_path / "merged"
    with pytest.raises(
        ValueError, match="Config mismatch between the folds and this merge"
    ) as excinfo:
        merge_fold_results(
            fold_dir=fold_dir,
            y=y,
            X_dict=X_dict,
            n_outer_folds=5,
            output_dir=merged_dir,
        )

    message = str(excinfo.value)
    # The artifacts carry only the hash, so the message has to name the
    # fields and the values the merge holds them at.
    assert "hyperparam_iterations" in message
    assert "amendment_features" in message
    # Aborting counts only if nothing was published first: the merged run
    # directory must hold no results assembled under a configuration the
    # folds were not computed under.
    assert not (merged_dir / "cv_fold_results.csv").exists()
    assert not (merged_dir / "config.csv").exists()


def test_merge_accepts_the_configuration_the_folds_were_computed_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A merge given the folds' own arguments has to go through.

    A check that also stops the workflow it protects is not a check. The
    distributed run is only usable if the same comparison that rejects a
    merge invoked without the fold-subset runs' flags passes one invoked
    with them.
    """
    import pandas as pd

    monkeypatch.setattr(
        PCCMultimodalPipeline, "_run_post_cv_steps", _skip_post_cv_steps
    )

    y = pd.Series([1, 1, 1, 1, 1], name="y")
    X_dict = {"demographics": pd.DataFrame({"age": [40.0, 50.0, 60.0, 30.0, 70.0]})}

    fold_run = PCCMultimodalPipeline(
        output_dir=tmp_path / "fold_run",
        n_outer_folds=5,
        hyperparam_iterations=10,
        amendment_features=False,
        data_fingerprint=fingerprint_analysis_data(y, X_dict),
    )
    fold_dir = tmp_path / "folds"
    _write_mergeable_folds(fold_dir, fold_run._config_hash())
    merged_dir = tmp_path / "merged"

    merge_fold_results(
        fold_dir=fold_dir,
        y=y,
        X_dict=X_dict,
        n_outer_folds=5,
        output_dir=merged_dir,
        hyperparam_iterations=10,
        amendment_features=False,
    )

    assert (merged_dir / "cv_fold_results.csv").exists()


def test_merge_rejects_folds_computed_against_other_data(tmp_path: Path) -> None:
    """The per-fold ``y_te`` check sees the labels, not the features.

    Two machines that each processed the raw export themselves can end up
    with parquets of different vintages. As long as the outcome is
    unchanged, every fold passes the label check and the merge would fit
    the final model on a feature matrix no fold was scored under. The
    fingerprint in the config hash is what makes that visible.
    """
    import pandas as pd

    y = pd.Series([1, 1, 1, 1, 1], name="y")
    as_computed = {
        "demographics": pd.DataFrame({"age": [40.0, 50.0, 60.0, 30.0, 70.0]})
    }
    # Same shape, same labels, one feature value moved — the change a label
    # check cannot see.
    as_merged = {"demographics": pd.DataFrame({"age": [40.0, 50.0, 60.0, 30.0, 71.0]})}

    fold_run = PCCMultimodalPipeline(
        output_dir=tmp_path / "fold_run",
        n_outer_folds=5,
        data_fingerprint=fingerprint_analysis_data(y, as_computed),
    )
    fold_dir = tmp_path / "folds"
    _write_mergeable_folds(fold_dir, fold_run._config_hash())

    with pytest.raises(ValueError, match="Config mismatch") as excinfo:
        merge_fold_results(
            fold_dir=fold_dir,
            y=y,
            X_dict=as_merged,
            n_outer_folds=5,
            output_dir=tmp_path / "merged",
        )
    assert "data_fingerprint" in str(excinfo.value)


def test_merge_keeps_the_provenance_of_the_run_that_computed_the_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two records, because they answer two questions.

    ``output_dir`` defaults to the fold directory, so a merge writing
    ``provenance.json`` would claim its own checkout and platform produced a
    cross-validation that ran somewhere else entirely.
    """
    import pandas as pd

    monkeypatch.setattr(
        PCCMultimodalPipeline, "_run_post_cv_steps", _skip_post_cv_steps
    )

    y = pd.Series([1, 1, 1, 1, 1], name="y")
    X_dict = {"demographics": pd.DataFrame({"age": [40.0, 50.0, 60.0, 30.0, 70.0]})}

    fold_run = PCCMultimodalPipeline(
        output_dir=tmp_path / "fold_run",
        n_outer_folds=5,
        data_fingerprint=fingerprint_analysis_data(y, X_dict),
    )
    fold_dir = tmp_path / "folds"
    _write_mergeable_folds(fold_dir, fold_run._config_hash())
    (fold_dir / "provenance.json").write_text(
        json.dumps({"git": {"commit": "the-checkout-that-computed-the-folds"}})
    )

    merge_fold_results(
        fold_dir=fold_dir,
        y=y,
        X_dict=X_dict,
        n_outer_folds=5,
    )

    fold_provenance = json.loads((fold_dir / "provenance.json").read_text())
    assert fold_provenance["git"]["commit"] == "the-checkout-that-computed-the-folds"

    merge_provenance = json.loads((fold_dir / "provenance_merge.json").read_text())
    assert "platform" in merge_provenance
    assert "packages" in merge_provenance


# ------------------------------------------------------------------
# Full Roundtrip Smoke Test: Single-Run vs. Fold-by-Fold + Merge
# ------------------------------------------------------------------


@pytest.mark.slow
def test_distributed_fold_roundtrip(tmp_path: Path) -> None:
    """Full run must produce identical y_oof/y_true as fold-by-fold + merge."""
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 120
    n_outer = 3
    n_inner = 2

    # Synthetic binary PCC labels with a realistic prevalence
    y = pd.Series(
        rng.choice([0, 1], size=n, p=[0.78, 0.22]).astype(np.int8),
        name="y",
    )

    # Two modalities
    X_dict = {
        "demographics": pd.DataFrame(
            {
                "age": rng.integers(18, 80, size=n).astype(np.float64),
                "sex": rng.choice([0.0, 1.0], size=n),
                "center": rng.choice([1.0, 2.0, 3.0], size=n),
            }
        ),
        "lab_values": pd.DataFrame(
            {f"lab_{i}": rng.normal(5, 1, size=n) for i in range(4)}
        ),
    }

    confounders = X_dict["demographics"].copy()

    common_kwargs: dict[str, Any] = {
        "y": y,
        "X_dict": X_dict,
        "confounders": confounders,
        "n_outer_folds": n_outer,
        "n_inner_folds": n_inner,
        "orthogonalize": True,
        "random_state": 42,
        "n_jobs": 1,
        "meta_permutation_iterations": 0,
        "n_subsamples": 10,
    }

    # --- A: Full single-machine run ---
    full_dir = tmp_path / "full"
    full_results = run_pcc_pipeline(output_dir=full_dir, **common_kwargs)

    # --- B: Fold-by-fold ---
    dist_dir = tmp_path / "distributed"

    # Simulate two machines: folds [0,1] and fold [2]
    run_pcc_pipeline(output_dir=dist_dir, fold_subset=[0, 1], **common_kwargs)
    run_pcc_pipeline(output_dir=dist_dir, fold_subset=[2], **common_kwargs)

    # Verify fold artifacts exist
    for i in range(n_outer):
        assert (dist_dir / f"fold_{i}.pkl").exists(), f"fold_{i}.pkl missing"

    # --- C: Merge ---
    merge_dir = tmp_path / "merged"
    merged_results = merge_fold_results(
        fold_dir=dist_dir,
        output_dir=merge_dir,
        **common_kwargs,
    )

    # --- D: Compare ---
    y_true_full = full_results["cv_results"]["y_true"]
    y_oof_full = full_results["cv_results"]["y_pred_proba"]
    y_true_merged = merged_results["cv_results"]["y_true"]
    y_oof_merged = merged_results["cv_results"]["y_pred_proba"]

    np.testing.assert_array_equal(
        y_true_full, y_true_merged, err_msg="y_true differs between full and merged"
    )
    np.testing.assert_array_almost_equal(
        y_oof_full,
        y_oof_merged,
        decimal=10,
        err_msg="y_oof differs between full and merged",
    )

    # Also check fold-level results match
    full_fold_results = full_results["cv_results"]["fold_results"]
    merged_fold_results = merged_results["cv_results"]["fold_results"]
    assert len(full_fold_results) == len(merged_fold_results)
    for fr_full, fr_merged in zip(full_fold_results, merged_fold_results, strict=True):
        assert fr_full["fold"] == fr_merged["fold"]
        if np.isnan(fr_full["roc_auc"]):
            assert np.isnan(fr_merged["roc_auc"])
        else:
            assert abs(fr_full["roc_auc"] - fr_merged["roc_auc"]) < 1e-10
        if np.isnan(fr_full["pr_auc"]):
            assert np.isnan(fr_merged["pr_auc"])
        else:
            assert abs(fr_full["pr_auc"] - fr_merged["pr_auc"]) < 1e-10

    # Final evaluation metrics should match
    full_roc = full_results["evaluation"]["metrics"]["roc_auc"]
    merged_roc = merged_results["evaluation"]["metrics"]["roc_auc"]
    if np.isnan(full_roc):
        assert np.isnan(merged_roc)
    else:
        assert abs(full_roc - merged_roc) < 1e-10

    full_pr = full_results["evaluation"]["metrics"]["pr_auc"]
    merged_pr = merged_results["evaluation"]["metrics"]["pr_auc"]
    if np.isnan(full_pr):
        assert np.isnan(merged_pr)
    else:
        assert abs(full_pr - merged_pr) < 1e-10

    # --- E: final_model.pkl is transfer-ready ---
    # The pickle must carry per-modality pipelines trained on the full
    # source cohort plus the source confounder frame so the transfer runs
    # can score a target cohort without refit.
    import pandas as pd

    for run_dir in (full_dir, merge_dir):
        pkl = run_dir / "final_model.pkl"
        assert pkl.exists(), f"final_model.pkl missing in {run_dir}"
        with pkl.open("rb") as f:
            final = pickle.load(f)
        assert "modality_pipelines" in final, (
            f"final_model.pkl in {run_dir} is missing modality_pipelines — "
            "transfer runs would fail to load."
        )
        assert set(final["modality_pipelines"].keys()) == set(
            final["modality_names"]
        ), "modality_pipelines keys must match modality_names"
        assert "confounders_train" in final
        assert isinstance(final["confounders_train"], pd.DataFrame)
        # Each persisted pipeline must be able to score at least one row
        # without refit (verifies the pipelines were fitted, not just
        # cloned empty).
        for mod_name, pipe in final["modality_pipelines"].items():
            X_mod = X_dict[mod_name]
            n_score = min(5, len(X_mod))
            proba = pipe.predict_proba(
                X_mod.iloc[:n_score], confounders=confounders.iloc[:n_score]
            )
            assert proba.shape == (n_score, 2), (
                f"{mod_name}: predict_proba shape {proba.shape} unexpected"
            )


# ---------------------------------------------------------------------------
# _create_pipelines dispatch
# ---------------------------------------------------------------------------
# The mh_* sub-modality keys that load_pipeline_data produces under
# split_mh_submodalities=True each need a branch of the dispatch. A name that
# matches none of them and only warns leaves the model trained without that
# modality at all, and mental_health is the dominant one: it carries ~40% of
# the SHAP mass, and a split-MH run that loses it scores several points of
# ROC-AUC below the same run keeping it. An unmatched name therefore raises.


def test_create_pipelines_handles_mh_submodalities(
    pipeline: PCCMultimodalPipeline,
) -> None:
    """All five mh_* sub-modality names must produce a pipeline."""
    from pcc_analysis.modality_pipelines import ModalityPipelineFactory

    factory = ModalityPipelineFactory(random_state=42)
    pipes = pipeline._create_pipelines(
        factory,
        ["demographics", "mh_phq9", "mh_gad7", "mh_mini", "mh_panic", "mh_stress"],
    )
    assert set(pipes.keys()) == {
        "demographics",
        "mh_phq9",
        "mh_gad7",
        "mh_mini",
        "mh_panic",
        "mh_stress",
    }, "mh_* sub-modalities must each receive a pipeline; silent drops are forbidden"


def test_create_pipelines_raises_on_unmatched_modality(
    pipeline: PCCMultimodalPipeline,
) -> None:
    """Unknown modality names must raise loudly, not silently drop."""
    from pcc_analysis.modality_pipelines import ModalityPipelineFactory

    factory = ModalityPipelineFactory(random_state=42)
    with pytest.raises(ValueError, match="could not match modality name"):
        pipeline._create_pipelines(
            factory,
            ["demographics", "totally_unknown_modality"],
        )
