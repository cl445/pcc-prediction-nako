"""Tests for SHAP meta-learner analysis."""

from __future__ import annotations

import numpy as np
import pytest

from pcc_analysis.meta_learner import MetaLearner
from pcc_analysis.shap_analysis import compute_meta_learner_shap


@pytest.fixture
def fitted_meta_no_interactions(
    large_sample_y: np.ndarray,
) -> tuple[MetaLearner, np.ndarray, list[str]]:
    """Fitted MetaLearner without interactions."""
    rng = np.random.default_rng(42)
    n = len(large_sample_y)
    modality_names = ["demographics", "lab_values", "cognitive"]
    oof = np.column_stack(
        [
            # demographics: weak signal
            large_sample_y * 0.3 + rng.uniform(0, 0.7, n),
            # lab_values: strong signal
            large_sample_y * 0.6 + rng.uniform(0, 0.4, n),
            # cognitive: medium signal
            large_sample_y * 0.4 + rng.uniform(0, 0.6, n),
        ]
    )
    meta = MetaLearner(
        add_interactions=False,
        hyperparam_iterations=0,
        random_state=42,
    )
    meta.fit(oof, large_sample_y, feature_names=modality_names)
    return meta, oof, modality_names


@pytest.fixture
def fitted_meta_with_interactions(
    large_sample_y: np.ndarray,
) -> tuple[MetaLearner, np.ndarray, list[str]]:
    """Fitted MetaLearner with interactions."""
    rng = np.random.default_rng(42)
    n = len(large_sample_y)
    modality_names = ["demographics", "lab_values", "cognitive"]
    oof = np.column_stack(
        [
            large_sample_y * 0.3 + rng.uniform(0, 0.7, n),
            large_sample_y * 0.6 + rng.uniform(0, 0.4, n),
            large_sample_y * 0.4 + rng.uniform(0, 0.6, n),
        ]
    )
    meta = MetaLearner(
        add_interactions=True,
        hyperparam_iterations=0,
        random_state=42,
    )
    meta.fit(oof, large_sample_y, feature_names=modality_names)
    return meta, oof, modality_names


def test_shap_output_structure(
    fitted_meta_no_interactions: tuple[MetaLearner, np.ndarray, list[str]],
) -> None:
    """Output dict should have expected keys."""
    meta, oof, names = fitted_meta_no_interactions
    result = compute_meta_learner_shap(meta, oof, names)

    expected_keys = {
        "shap_values",
        "expected_value",
        "feature_names",
        "feature_importance",
        "modality_importance",
    }
    assert set(result.keys()) == expected_keys


def test_shap_values_shape(
    fitted_meta_no_interactions: tuple[MetaLearner, np.ndarray, list[str]],
) -> None:
    """SHAP values should have shape (n_samples, n_features)."""
    meta, oof, names = fitted_meta_no_interactions
    result = compute_meta_learner_shap(meta, oof, names)

    assert result["shap_values"].shape[0] == oof.shape[0]
    assert result["shap_values"].shape[1] == len(result["feature_names"])


def test_shap_feature_importance_positive(
    fitted_meta_no_interactions: tuple[MetaLearner, np.ndarray, list[str]],
) -> None:
    """Feature importances (mean |SHAP|) should be non-negative."""
    meta, oof, names = fitted_meta_no_interactions
    result = compute_meta_learner_shap(meta, oof, names)

    assert (result["feature_importance"]["mean_abs_shap"] >= 0).all()


def test_shap_modality_importance_positive(
    fitted_meta_no_interactions: tuple[MetaLearner, np.ndarray, list[str]],
) -> None:
    """Modality importances should be non-negative."""
    meta, oof, names = fitted_meta_no_interactions
    result = compute_meta_learner_shap(meta, oof, names)

    assert (result["modality_importance"]["mean_abs_shap"] >= 0).all()
    assert len(result["modality_importance"]) == len(names)


def test_shap_with_interactions(
    fitted_meta_with_interactions: tuple[MetaLearner, np.ndarray, list[str]],
) -> None:
    """SHAP should work with interaction features and aggregate correctly."""
    meta, oof, names = fitted_meta_with_interactions
    result = compute_meta_learner_shap(meta, oof, names)

    # With 3 modalities + 3 interactions = 6 features
    assert result["shap_values"].shape[1] == 6
    # But modality importance should still have 3 entries
    assert len(result["modality_importance"]) == 3
