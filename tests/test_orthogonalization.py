"""Smoke tests for Orthogonalizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcc_analysis.orthogonalization import Orthogonalizer


def test_fit_requires_confounders(
    sample_X_df: pd.DataFrame, sample_y: np.ndarray
) -> None:
    ortho = Orthogonalizer(cv=3)
    with pytest.raises(ValueError, match="confounders must be provided"):
        ortho.fit(sample_X_df, sample_y)


def test_fit_sets_attributes(
    sample_X_df: pd.DataFrame,
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
) -> None:
    ortho = Orthogonalizer(cv=3, random_state=42)
    ortho.fit(sample_X_df, sample_y, confounders=sample_confounders)
    assert ortho.is_fitted_ is True
    assert hasattr(ortho, "nuisance_model_X_")
    assert ortho._confounders_fit is not None
    assert hasattr(ortho, "fitted_models_X_")
    assert len(ortho.fitted_models_X_) == sample_X_df.shape[1]


def test_transform_X_only(
    sample_X_df: pd.DataFrame,
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
) -> None:
    ortho = Orthogonalizer(cv=3, random_state=42)
    ortho.fit(sample_X_df, sample_y, confounders=sample_confounders)
    result = ortho.transform(sample_X_df, confounders=sample_confounders)
    assert not isinstance(result, tuple)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == sample_X_df.shape


def test_dataframe_preserved(
    sample_X_df: pd.DataFrame,
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
) -> None:
    ortho = Orthogonalizer(cv=3, random_state=42)
    ortho.fit(sample_X_df, sample_y, confounders=sample_confounders)
    X_ortho = ortho.transform(sample_X_df, confounders=sample_confounders)
    assert isinstance(X_ortho, pd.DataFrame)
    assert (X_ortho.index == sample_X_df.index).all()


def test_residuals_shape(
    sample_X_df: pd.DataFrame,
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
) -> None:
    ortho = Orthogonalizer(cv=3, random_state=42)
    ortho.fit(sample_X_df, sample_y, confounders=sample_confounders)
    X_ortho = ortho.transform(sample_X_df, confounders=sample_confounders)
    assert X_ortho.shape == sample_X_df.shape


def test_fit_transform_returns_oof_residuals(
    sample_X_df: pd.DataFrame,
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
) -> None:
    ortho = Orthogonalizer(cv=3, random_state=42)
    X_ortho = ortho.fit_transform(sample_X_df, sample_y, confounders=sample_confounders)
    assert isinstance(X_ortho, pd.DataFrame)
    assert X_ortho.shape == sample_X_df.shape
    # Cache should be cleared after fit_transform
    assert ortho._cached_X_residuals_ is None


def test_fit_transform_x_only(
    sample_X_df: pd.DataFrame,
    sample_confounders: pd.DataFrame,
) -> None:
    ortho = Orthogonalizer(cv=3, random_state=42)
    result = ortho.fit_transform(sample_X_df, y=None, confounders=sample_confounders)
    assert not isinstance(result, tuple)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == sample_X_df.shape


def test_transform_on_new_data(
    sample_X_df: pd.DataFrame,
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    ortho = Orthogonalizer(cv=3, random_state=42)
    ortho.fit(sample_X_df, sample_y, confounders=sample_confounders)

    # Create a small subset of new data (10 samples)
    new_X = pd.DataFrame(
        rng.standard_normal((10, 10)),
        columns=[f"feat_{i}" for i in range(10)],
    )
    new_confounders = pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=10).astype(np.float64),
            "sex": rng.choice([0.0, 1.0], size=10),
            "center": rng.choice([1.0, 2.0, 3.0], size=10),
        }
    )

    result = ortho.transform(new_X, confounders=new_confounders)
    assert not isinstance(result, tuple)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (10, 10)


def test_fit_without_y(
    sample_X_df: pd.DataFrame,
    sample_confounders: pd.DataFrame,
) -> None:
    ortho = Orthogonalizer(cv=3, random_state=42)
    ortho.fit(sample_X_df, y=None, confounders=sample_confounders)

    assert ortho.is_fitted_ is True
    # X models should still be fitted
    assert len(ortho.fitted_models_X_) == sample_X_df.shape[1]
    assert all(m is not None for m in ortho.fitted_models_X_)


def test_ndarray_input(
    sample_X_array: np.ndarray,
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
) -> None:
    ortho = Orthogonalizer(cv=3, random_state=42)
    ortho.fit(sample_X_array, sample_y, confounders=sample_confounders)
    X_ortho = ortho.transform(sample_X_array, confounders=sample_confounders)
    assert isinstance(X_ortho, np.ndarray)
    assert X_ortho.shape == sample_X_array.shape


def test_residuals_uncorrelated_with_confounders(
    rng: np.random.Generator,
) -> None:
    """After orthogonalization, residuals should be uncorrelated with confounders."""
    n = 300
    confounders = pd.DataFrame(
        {
            "age": rng.standard_normal(n),
            "sex": rng.choice([0.0, 1.0], size=n),
        }
    )
    # Features that are correlated with confounders
    X = pd.DataFrame(
        {
            "feat_0": confounders["age"] * 2.0 + rng.standard_normal(n) * 0.5,
            "feat_1": confounders["sex"] * 3.0 + rng.standard_normal(n) * 0.5,
            "feat_2": rng.standard_normal(n),
        }
    )

    ortho = Orthogonalizer(cv=3, random_state=42)
    X_ortho = ortho.fit_transform(X, y=None, confounders=confounders)

    conf_arr = confounders.to_numpy()
    assert isinstance(X_ortho, pd.DataFrame)
    X_orth_arr = X_ortho.to_numpy()
    for i in range(X_orth_arr.shape[1]):
        for j in range(conf_arr.shape[1]):
            corr = np.abs(np.corrcoef(X_orth_arr[:, i], conf_arr[:, j])[0, 1])
            assert corr < 0.15, (
                f"Feature {i} still correlated with confounder {j}: r={corr:.3f}"
            )
