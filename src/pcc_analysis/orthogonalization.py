"""Orthogonalization (Double ML) for confounder control.

Removes confounding effects of demographics (age, sex, center)
from predictors using X-residualization. The classifier trains on
the original binary target — in a prediction-focused pipeline,
y-residualization is not needed because the demographics modality
captures the direct confounder-to-outcome relationship.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast, override

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor

from ._array_utils import to_float_array
from ._sklearn_typing import clone

if TYPE_CHECKING:
    from .protocols import Regressor

logger = logging.getLogger(__name__)


def _xgb_params(
    *, n_jobs: int = 1, random_state: int | None = None
) -> dict[str, int | float | str | None]:
    """Return default XGBoost parameters for orthogonalization.

    Always uses CPU: confounders are only ~3 features (age, sex, center),
    so GPU overhead exceeds any benefit.  ``n_jobs`` defaults to 1 because
    the feature loop is parallelised at a higher level.
    """
    return {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": n_jobs,
        "random_state": random_state,
        "verbosity": 0,
    }


class Orthogonalizer(BaseEstimator, TransformerMixin):
    """Orthogonalize features with respect to confounders.

    Removes the confounder-predictable component from each feature
    via out-of-fold predictions (DML-style X-residualization).

    Parameters
    ----------
    nuisance_model_X : estimator, optional
        Model to predict X from confounders. Defaults to XGBRegressor.
    cv : int
        Folds for out-of-fold predictions.
    random_state : int, optional
        Random seed.
    n_jobs : int
        Parallel jobs for per-feature residualization.
    """

    def __init__(
        self,
        nuisance_model_X: Regressor | None = None,
        cv: int = 5,
        random_state: int | None = None,
        n_jobs: int = -1,
    ) -> None:
        self.nuisance_model_X = nuisance_model_X
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        confounders: pd.DataFrame | np.ndarray | None = None,
    ) -> Orthogonalizer:
        """Fit nuisance models predicting each feature from confounders.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix to orthogonalize.
        y : ndarray, optional
            Ignored (sklearn API compatibility).
        confounders : DataFrame or ndarray
            Confounder matrix (e.g. age, sex, center). Required.
        """
        if confounders is None:
            raise ValueError("confounders must be provided")

        self._input_is_dataframe = isinstance(X, pd.DataFrame)
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
            self._index = X.index
            numeric_cols = X.select_dtypes(include=np.number).columns
            self._numeric_cols: list[str] | None = numeric_cols.tolist()
        else:
            self._numeric_cols = None

        confounders_array = to_float_array(confounders)

        # Create template model
        if self.nuisance_model_X is None:
            template_X: Regressor = XGBRegressor(
                **_xgb_params(random_state=self.random_state)
            )
        else:
            template_X = clone(self.nuisance_model_X)
        self.nuisance_model_X_ = template_X

        # Prepare X array
        if isinstance(X, pd.DataFrame):
            numeric_cols_idx = X.select_dtypes(include=np.number).columns
            X_array = X[numeric_cols_idx].to_numpy()
        else:
            X_array = to_float_array(X)

        # Compute OOF residuals and fit full models for X
        self._cached_X_residuals_: np.ndarray | None = self._compute_oof_residuals_X(
            X_array, confounders_array
        )
        self.fitted_models_X_ = self._fit_full_models_X(X_array, confounders_array)

        self._confounders_fit = confounders_array
        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        confounders: pd.DataFrame | np.ndarray | None = None,
    ) -> pd.DataFrame | np.ndarray:
        """Compute residuals X - E[X|confounders] using fitted nuisance models.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix to orthogonalize.
        y : ndarray, optional
            Ignored.
        confounders : DataFrame or ndarray, optional
            If None, uses the confounders from ``fit()``.
        """
        if confounders is None:
            confounders_array = self._confounders_fit
        else:
            confounders_array = to_float_array(confounders)

        if isinstance(X, pd.DataFrame):
            original_index = X.index
            numeric_cols = X.select_dtypes(include=np.number).columns
            if (
                self._numeric_cols is not None
                and numeric_cols.tolist() != self._numeric_cols
            ):
                raise ValueError(
                    f"Feature mismatch between fit and transform: "
                    f"fit had {self._numeric_cols}, "
                    f"transform got {numeric_cols.tolist()}"
                )
            X_array = X[numeric_cols].to_numpy()
            feature_names: list[str] | None = numeric_cols.tolist()
        else:
            X_array = to_float_array(X)
            original_index = None
            feature_names = None

        X_ortho_array = self._predict_residuals_X(X_array, confounders_array)

        if feature_names is not None and original_index is not None:
            return pd.DataFrame(
                X_ortho_array, columns=feature_names, index=original_index
            )
        return X_ortho_array

    @override
    def fit_transform(  # pyrefly: ignore[bad-override]
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        confounders: pd.DataFrame | np.ndarray | None = None,
    ) -> pd.DataFrame | np.ndarray:
        self.fit(X, y, confounders)

        # Retrieve cached OOF residuals. Raised rather than asserted: an
        # empty cache here means ``fit`` returned without producing residuals,
        # and `python -O` would strip an assert and hand the None onward.
        X_residuals = self._cached_X_residuals_
        if X_residuals is None:
            raise RuntimeError(
                "fit() left no cached out-of-fold residuals; "
                "fit_transform cannot return residualised features"
            )

        # Free cache
        self._cached_X_residuals_ = None

        # Wrap as DataFrame if input was DataFrame
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=np.number).columns
            return pd.DataFrame(
                X_residuals, columns=numeric_cols.tolist(), index=X.index
            )
        return X_residuals

    # --- Private helpers: OOF residuals (training) ---

    def _compute_oof_residuals_X(
        self, X: np.ndarray, confounders: np.ndarray
    ) -> np.ndarray:
        n_features = X.shape[1]

        def _residualize_feature(i: int) -> np.ndarray:
            X_i = X[:, i]
            try:
                if np.all(np.isnan(X_i)):
                    return X_i
            except (TypeError, ValueError):
                return X_i
            try:
                model = clone(self.nuisance_model_X_)
                # sklearn takes estimators nominally; the nuisance model is
                # described structurally so any regressor can be injected.
                X_i_pred = cross_val_predict(
                    cast("BaseEstimator", model), confounders, X_i, cv=self.cv, n_jobs=1
                )
                return np.asarray(X_i - X_i_pred)
            except (ValueError, RuntimeError):
                logger.warning("Failed to residualize feature %d, keeping original", i)
                return X_i

        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(_residualize_feature)(i) for i in range(n_features)
        )
        return np.column_stack(results)

    # --- Private helpers: fit full models (for transform on new data) ---

    def _fit_full_models_X(
        self, X: np.ndarray, confounders: np.ndarray
    ) -> list[Regressor | None]:
        n_features = X.shape[1]

        def _fit_one(i: int) -> Regressor | None:
            X_i = X[:, i]
            try:
                if np.all(np.isnan(X_i)):
                    return None
            except (TypeError, ValueError):
                return None
            try:
                model = clone(self.nuisance_model_X_)
                model.fit(confounders, X_i)
            except (ValueError, RuntimeError):
                logger.warning("Failed to fit full model for feature %d", i)
                return None
            else:
                return model

        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(_fit_one)(i) for i in range(n_features)
        )
        return list(results)

    # --- Private helpers: predict residuals (new data) ---

    def _predict_residuals_X(
        self, X: np.ndarray, confounders: np.ndarray
    ) -> np.ndarray:
        n_features = X.shape[1]
        X_ortho = np.zeros_like(X)

        for i in range(n_features):
            X_i = X[:, i]
            model = self.fitted_models_X_[i]
            if model is None:
                X_ortho[:, i] = X_i
            else:
                X_i_pred = model.predict(confounders)
                X_ortho[:, i] = X_i - X_i_pred

        return X_ortho

    @override
    def set_output(self, *, transform: str | None = None) -> Orthogonalizer:
        return self
