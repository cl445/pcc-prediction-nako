"""Smoke tests for ConfoundedPipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pcc_analysis.custom_pipeline import ConfoundedPipeline
from pcc_analysis.orthogonalization import Orthogonalizer
from pcc_analysis.preprocessing import FeatureToConfounder


def test_fit_predict_with_confounders(
    sample_X_df: pd.DataFrame,
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
) -> None:
    pipe = ConfoundedPipeline(
        steps=[
            ("ortho", Orthogonalizer(cv=3, random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    pipe.fit(sample_X_df, sample_y, confounders=sample_confounders)
    preds = pipe.predict(sample_X_df, confounders=sample_confounders)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_fit_predict_without_confounders(
    sample_X_df: pd.DataFrame, sample_y: np.ndarray
) -> None:
    pipe = ConfoundedPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    pipe.fit(sample_X_df, sample_y, confounders=None)
    preds = pipe.predict(sample_X_df)
    assert preds.shape == (100,)


def test_predict_proba_shape(sample_X_df: pd.DataFrame, sample_y: np.ndarray) -> None:
    pipe = ConfoundedPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    pipe.fit(sample_X_df, sample_y)
    proba = pipe.predict_proba(sample_X_df)
    assert proba.shape == (100, 2)


def test_confounders_routed_to_orthogonalizer(
    sample_X_df: pd.DataFrame,
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
) -> None:
    ortho = Orthogonalizer(cv=3, random_state=42)
    pipe = ConfoundedPipeline(
        steps=[
            ("ortho", ortho),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    pipe.fit(sample_X_df, sample_y, confounders=sample_confounders)
    # After fit, the orthogonalizer should have been fitted
    assert pipe.steps[0][1].is_fitted_ is True


def test_pipeline_predict_on_subset(
    sample_X_df: pd.DataFrame,
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    pipe = ConfoundedPipeline(
        steps=[
            ("ortho", Orthogonalizer(cv=3, random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    pipe.fit(sample_X_df, sample_y, confounders=sample_confounders)

    # Predict on a small subset (10 samples)
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

    preds = pipe.predict(new_X, confounders=new_confounders)
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_extra_confounders_reach_orthogonalizer(
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
) -> None:
    """FeatureToConfounder should augment confounders passed to Orthogonalizer."""
    n = len(sample_y)
    rng = np.random.default_rng(42)
    # Create features with an extra 'etiv' column
    X = pd.DataFrame(
        rng.standard_normal((n, 5)),
        columns=[f"feat_{i}" for i in range(5)],
    )
    X["etiv"] = rng.standard_normal(n) * 100 + 1500

    pipe = ConfoundedPipeline(
        steps=[
            ("etiv_confounder", FeatureToConfounder(columns="etiv")),
            ("ortho", Orthogonalizer(cv=3, random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    pipe.fit(X, sample_y, confounders=sample_confounders)

    # Orthogonalizer should have been fitted with 4 confounders (age, sex, center + etiv)
    ortho = pipe.steps[1][1]
    assert ortho.is_fitted_
    # The fitted confounders array should have 4 columns
    assert ortho._confounders_fit.shape[1] == sample_confounders.shape[1] + 1

    # Predictions should also work (transform path)
    proba = pipe.predict_proba(X.copy(), confounders=sample_confounders)
    assert proba.shape == (n, 2)


def test_pipeline_without_extra_confounders_unchanged(
    sample_X_df: pd.DataFrame,
    sample_y: np.ndarray,
    sample_confounders: pd.DataFrame,
) -> None:
    """Pipelines without FeatureToConfounder should behave identically."""
    pipe = ConfoundedPipeline(
        steps=[
            ("ortho", Orthogonalizer(cv=3, random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    pipe.fit(sample_X_df, sample_y, confounders=sample_confounders)
    ortho = pipe.steps[0][1]
    # Should have exactly 3 confounders (age, sex, center) — no extras
    assert ortho._confounders_fit.shape[1] == sample_confounders.shape[1]
