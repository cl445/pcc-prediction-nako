"""Smoke tests for MetaLearner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcc_analysis.meta_learner import MetaLearner


def _make_meta_data(
    rng: np.random.Generator, n: int, n_modalities: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.random((n, n_modalities))
    y = np.zeros(n, dtype=np.float64)
    y[rng.choice(n, size=int(n * 0.35), replace=False)] = 1.0
    return X, y


def _meta(*, add_interactions: bool = True, n_jobs: int = 1) -> MetaLearner:
    return MetaLearner(
        hyperparam_iterations=0,
        random_state=42,
        n_jobs=n_jobs,
        add_interactions=add_interactions,
    )


def test_fit_predict_shapes(rng: np.random.Generator) -> None:
    X, y = _make_meta_data(rng, 120)
    meta = _meta()
    meta.fit(X, y)
    preds = meta.predict(X)
    proba = meta.predict_proba(X)
    assert preds.shape == (120,)
    assert proba.shape == (120, 2)


def test_probabilities_sum_to_one(rng: np.random.Generator) -> None:
    X, y = _make_meta_data(rng, 120)
    meta = _meta()
    meta.fit(X, y)
    proba = meta.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_feature_importance(rng: np.random.Generator) -> None:
    X, y = _make_meta_data(rng, 120)
    names = ["blood", "imaging", "questionnaire"]
    meta = _meta(add_interactions=True)
    meta.fit(X, y, feature_names=names)
    imp = meta.get_feature_importance()
    assert isinstance(imp, pd.DataFrame)
    assert "feature" in imp.columns
    assert "importance" in imp.columns


def test_interactions_added(rng: np.random.Generator) -> None:
    X, y = _make_meta_data(rng, 120, n_modalities=3)
    meta = _meta(add_interactions=True)
    meta.fit(X, y)
    # 3 base + 3 interaction terms = 6
    assert len(meta.feature_names_) > X.shape[1]


def test_no_interactions(rng: np.random.Generator) -> None:
    X, y = _make_meta_data(rng, 120, n_modalities=3)
    meta = _meta(add_interactions=False)
    meta.fit(X, y)
    assert len(meta.feature_names_) == X.shape[1]


def test_skips_calibration_small_sample(rng: np.random.Generator) -> None:
    X, y = _make_meta_data(rng, 50, n_modalities=2)
    meta = _meta()
    meta.fit(X, y)
    # Small sample → calibrated_model_ should be meta_model_
    assert meta.calibrated_model_ is meta.meta_model_


def test_feature_names_default(rng: np.random.Generator) -> None:
    X, y = _make_meta_data(rng, 120, n_modalities=4)
    meta = _meta(add_interactions=False)
    meta.fit(X, y)
    assert meta.base_feature_names_ == [
        "modality_0",
        "modality_1",
        "modality_2",
        "modality_3",
    ]


def test_nan_modality_predictions(rng: np.random.Generator) -> None:
    """MetaLearner must handle a column of all-NaN (missing modality)."""
    X, y = _make_meta_data(rng, 120, n_modalities=3)
    X[:, 1] = np.nan  # entire modality missing
    meta = _meta(add_interactions=False)
    meta.fit(X, y)
    proba = meta.predict_proba(X)
    assert proba.shape == (120, 2)
    # Probabilities should still be valid (no NaN in output)
    assert not np.isnan(proba).any()
