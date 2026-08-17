"""Structural typing protocols for sklearn-compatible interfaces.

These protocols provide type-safe alternatives to using ``Any`` or
untyped ``BaseEstimator`` references from sklearn (which has no type
stubs).  They are runtime-checkable so they can also serve as guards.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Estimator(Protocol):
    """Protocol for sklearn-compatible estimators (fit / get_params)."""

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | None = ..., **kwargs: Any
    ) -> Estimator: ...

    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> Estimator: ...


@runtime_checkable
class Transformer(Protocol):
    """Protocol for sklearn-compatible transformers (fit + transform)."""

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | None = ..., **kwargs: Any
    ) -> Transformer: ...

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame: ...


@runtime_checkable
class Regressor(Protocol):
    """Protocol for sklearn-compatible regressors (fit + continuous predict)."""

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray, **kwargs: Any
    ) -> Regressor: ...

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...


@runtime_checkable
class Classifier(Protocol):
    """Protocol for sklearn-compatible classifiers."""

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray, **kwargs: Any
    ) -> Classifier: ...

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...

    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> Classifier: ...


@runtime_checkable
class ConfoundedEstimator(Protocol):
    """Protocol for pipelines that route confounders to orthogonalization."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = ...,
        confounders: pd.DataFrame | None = ...,
        **fit_params: Any,
    ) -> ConfoundedEstimator: ...

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        confounders: pd.DataFrame | None = ...,
    ) -> np.ndarray: ...

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
        confounders: pd.DataFrame | None = ...,
    ) -> np.ndarray: ...
