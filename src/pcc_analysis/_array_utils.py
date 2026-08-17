"""Shared array conversion utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_float_array(X: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    """Convert pandas or numpy input to a float64 numpy array."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return np.asarray(X.to_numpy(dtype="float64", na_value=np.nan))
    return np.asarray(X, dtype="float64")
