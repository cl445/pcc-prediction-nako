"""Shared fixtures for smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def sample_X_df(rng: np.random.Generator) -> pd.DataFrame:
    data = rng.standard_normal((100, 10))
    columns = [f"feat_{i}" for i in range(10)]
    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def sample_X_array(sample_X_df: pd.DataFrame) -> np.ndarray:
    return np.asarray(sample_X_df.to_numpy())


@pytest.fixture
def sample_y(rng: np.random.Generator) -> np.ndarray:
    y = np.zeros(100, dtype=np.float64)
    y[rng.choice(100, size=30, replace=False)] = 1.0
    return y


@pytest.fixture
def sample_confounders(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=100).astype(np.float64),
            "sex": rng.choice([0.0, 1.0], size=100),
            "center": rng.choice([1.0, 2.0, 3.0], size=100),
        }
    )


@pytest.fixture
def large_sample_y(rng: np.random.Generator) -> np.ndarray:
    y = np.zeros(120, dtype=np.float64)
    y[rng.choice(120, size=40, replace=False)] = 1.0
    return y
