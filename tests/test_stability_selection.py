"""Smoke tests for StabilitySelector.

The class is no longer part of any pipeline (DECISIONS.md §2.33). These tests keep
it loadable and correct, because the fitted instances inside every
``final_model.pkl`` written before the removal are unpickled through it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pcc_analysis.stability_selection import StabilitySelector

_LAMBDA_GRID = np.logspace(-1, 0, 3)


def _fit_selector(X: pd.DataFrame | np.ndarray, y: np.ndarray) -> StabilitySelector:
    sel = StabilitySelector(
        n_subsamples=5, lambda_grid=_LAMBDA_GRID, threshold=0.3, random_state=42
    )
    sel.fit(X, y)
    return sel


def test_fit_sets_attributes(sample_X_df: pd.DataFrame, sample_y: np.ndarray) -> None:
    sel = _fit_selector(sample_X_df, sample_y)
    assert hasattr(sel, "selected_features_")
    assert hasattr(sel, "selection_probabilities_")
    assert hasattr(sel, "stability_path_")
    assert len(sel.selection_probabilities_) == sample_X_df.shape[1]


def test_transform_emits_exactly_the_selected_features(
    sample_X_df: pd.DataFrame, sample_y: np.ndarray
) -> None:
    """``transform`` returns the selected columns and nothing else.

    Width alone says very little — ``<= n_features`` holds for any selector,
    including one that selects everything — so the check is against
    ``selected_features_`` rather than against a bound.
    """
    sel = _fit_selector(sample_X_df, sample_y)
    X_out = sel.transform(sample_X_df)
    assert isinstance(X_out, pd.DataFrame)
    assert X_out.shape[1] == len(sel.selected_features_)
    assert list(X_out.columns) == [
        sample_X_df.columns[int(i)] for i in sel.selected_features_
    ]


def test_dataframe_roundtrip(sample_X_df: pd.DataFrame, sample_y: np.ndarray) -> None:
    sel = _fit_selector(sample_X_df, sample_y)
    X_out = sel.transform(sample_X_df)
    assert isinstance(X_out, pd.DataFrame)
    assert (X_out.index == sample_X_df.index).all()


def test_ndarray_roundtrip(sample_X_array: np.ndarray, sample_y: np.ndarray) -> None:
    sel = _fit_selector(sample_X_array, sample_y)
    X_out = sel.transform(sample_X_array)
    assert isinstance(X_out, np.ndarray)


def test_get_support_mask_and_indices(
    sample_X_df: pd.DataFrame, sample_y: np.ndarray
) -> None:
    sel = _fit_selector(sample_X_df, sample_y)
    mask = sel.get_support(indices=False)
    indices = sel.get_support(indices=True)
    assert mask.dtype == bool
    assert len(mask) == sample_X_df.shape[1]
    assert np.array_equal(np.where(mask)[0], indices)


def test_get_feature_names_out(sample_X_df: pd.DataFrame, sample_y: np.ndarray) -> None:
    sel = _fit_selector(sample_X_df, sample_y)
    names_in = sample_X_df.columns.tolist()
    names_out = sel.get_feature_names_out(names_in)
    assert all(n in names_in for n in names_out)
    assert len(names_out) == len(sel.selected_features_)


def test_default_estimator_uses_l1_penalty(
    sample_X_df: pd.DataFrame, sample_y: np.ndarray
) -> None:
    """Default base estimator must use L1 penalty for sparsity."""
    sel = StabilitySelector(n_subsamples=5, lambda_grid=_LAMBDA_GRID, random_state=42)
    sel.fit(sample_X_df, sample_y)
    est = sel.base_estimator_
    assert isinstance(est, LogisticRegression)
    # sklearn >= 1.8: l1_ratio=1.0 is the new way to specify L1 penalty
    assert est.get_params()["l1_ratio"] == 1.0


def _signal_and_noise(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two features that drive the outcome, eight that do not."""
    n = 500
    X_signal = rng.standard_normal((n, 2))
    X_noise = rng.standard_normal((n, 8))
    X = np.hstack([X_signal, X_noise])
    y = (X_signal[:, 0] * 3.0 + X_signal[:, 1] * 3.0 > 0).astype(float)
    return X, y


def test_noise_features_excluded(rng: np.random.Generator) -> None:
    """The selector finds the signal and drops most of the noise.

    At threshold 0.8. At 0.6 this assertion does not hold: on an outcome of
    pure coin flips the selector keeps all ten features (DECISIONS.md §2.33).
    """
    X, y = _signal_and_noise(rng)
    sel = StabilitySelector(
        n_subsamples=100,
        lambda_grid=np.array([0.01, 0.1, 1.0]),
        threshold=0.8,
        random_state=42,
    )
    sel.fit(X, y)
    selected = sel.selected_features_
    assert any(i < 2 for i in selected)
    assert len(selected) < 10
