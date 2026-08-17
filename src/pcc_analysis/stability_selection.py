"""Stability selection via subsampled L1 regularization."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, override

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from ._array_utils import to_float_array
from ._sklearn_typing import clone

if TYPE_CHECKING:
    from .protocols import Classifier

logger = logging.getLogger(__name__)


class StabilitySelector(BaseEstimator, TransformerMixin):
    """Feature selection via stability selection.

    Parameters
    ----------
    base_estimator : estimator, optional
        L1-penalised estimator. Defaults to LogisticRegression(l1).
    lambda_grid : array, optional
        Grid of regularization strengths.
    n_subsamples : int
        Number of subsampling iterations (without replacement).
    sample_fraction : float
        Fraction of samples per subsample.
    threshold : float
        Selection probability threshold.
    random_state : int, optional
        Random seed.
    name : str, optional
        Label for logging.
    """

    def __init__(
        self,
        base_estimator: Classifier | None = None,
        lambda_grid: np.ndarray | None = None,
        n_subsamples: int = 100,
        sample_fraction: float = 0.5,
        threshold: float = 0.6,
        random_state: int | None = None,
        name: str | None = None,
        n_jobs: int = -1,
    ) -> None:
        self.base_estimator = base_estimator
        self.lambda_grid = lambda_grid
        self.n_subsamples = n_subsamples
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.random_state = random_state
        self.name = name
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> StabilitySelector:
        """Compute selection probabilities via bootstrap L1 fits.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Binary target.
        """
        self._input_is_dataframe = isinstance(X, pd.DataFrame)
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
            self._index = X.index

        X_array = to_float_array(X)
        y_array = to_float_array(y).ravel()
        n_samples, n_features = X_array.shape

        if self.base_estimator is None:
            self.base_estimator_ = LogisticRegression(
                # l1_ratio=1.0 => pure L1 penalty (sklearn >= 1.8 API;
                # penalty="l1" is deprecated since 1.8)
                l1_ratio=1.0,
                solver="liblinear",
                max_iter=10000,
                tol=1e-5,
                random_state=self.random_state,
                class_weight="balanced",
            )
        else:
            self.base_estimator_ = clone(self.base_estimator)

        if self.lambda_grid is None:
            self.lambda_grid_ = np.logspace(-3, 1, 20)
        else:
            self.lambda_grid_ = self.lambda_grid

        rng = np.random.default_rng(self.random_state)
        n_subsample = int(n_samples * self.sample_fraction)
        subsamples = [
            rng.choice(n_samples, size=n_subsample, replace=False)
            for _ in range(self.n_subsamples)
        ]

        def _fit_one(subsample_idx: int, lambda_val: float) -> np.ndarray | None:
            try:
                X_sub = X_array[subsamples[subsample_idx]]
                y_sub = y_array[subsamples[subsample_idx]]
                est = clone(self.base_estimator_)
                if hasattr(est, "C"):
                    est.set_params(C=1.0 / lambda_val)
                elif hasattr(est, "alpha"):
                    est.set_params(alpha=lambda_val)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    est.fit(X_sub, y_sub)

                if hasattr(est, "coef_"):
                    coef: np.ndarray = np.asarray(est.coef_)
                    if coef.ndim > 1:
                        coef = coef.ravel()
                    selected: np.ndarray = (np.abs(coef) > 1e-10).astype(int)
                    return selected
                # Not moved into an `else` block: this runs in a joblib worker,
                # and the guard exists so that no single subsample can kill the
                # run. Narrowing what it covers would defeat that.
                return None  # noqa: TRY300
            except Exception as e:
                logger.debug(
                    "Subsample fit failed for s=%d, lambda=%.3f: %s",
                    subsample_idx,
                    lambda_val,
                    e,
                )
                return None

        all_tasks = [
            (b, lam) for b in range(self.n_subsamples) for lam in self.lambda_grid_
        ]
        n_total = len(all_tasks)
        n_chunks = 10
        chunk_size = max(1, n_total // n_chunks)
        results: list[np.ndarray | None] = []
        logger.info(
            "%sStability selection: %d fits (%d subsamples x %d lambda)",
            self._log_prefix(),
            n_total,
            self.n_subsamples,
            len(self.lambda_grid_),
        )
        for chunk_start in range(0, n_total, chunk_size):
            chunk = all_tasks[chunk_start : chunk_start + chunk_size]
            chunk_results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_fit_one)(b, lam) for b, lam in chunk
            )
            results.extend(chunk_results)
            done = min(chunk_start + chunk_size, n_total)
            logger.info(
                "%sStability selection progress: %d/%d fits (%.0f%%)",
                self._log_prefix(),
                done,
                n_total,
                100.0 * done / n_total,
            )

        n_runs = self.n_subsamples * len(self.lambda_grid_)
        stability_matrix = np.zeros((n_runs, n_features))
        for run_idx, result in enumerate(results):
            if result is not None:
                stability_matrix[run_idx] = result

        self.selection_probabilities_ = stability_matrix.mean(axis=0)
        self.selected_features_ = np.where(
            self.selection_probabilities_ >= self.threshold
        )[0]
        self.stability_path_ = stability_matrix

        logger.info(
            "%sStability selection: %d / %d features selected",
            self._log_prefix(),
            len(self.selected_features_),
            n_features,
        )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Select features whose selection probability exceeds the threshold."""
        if isinstance(X, pd.DataFrame):
            original_index = X.index
            feature_names: list[str] | None = X.columns.tolist()
            X_array = to_float_array(X)
        else:
            X_array = to_float_array(X)
            original_index = None
            feature_names = None

        if len(self.selected_features_) == 0:
            logger.warning("%sNo features selected! Returning all.", self._log_prefix())
            return X

        X_selected = X_array[:, self.selected_features_]

        if feature_names is not None:
            selected_names = [feature_names[i] for i in self.selected_features_]
            return pd.DataFrame(
                X_selected, columns=selected_names, index=original_index
            )
        return X_selected

    @override
    def fit_transform(  # pyrefly: ignore[bad-override]
        self, X: pd.DataFrame | np.ndarray, y: np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        if indices:
            return self.selected_features_
        mask = np.zeros(len(self.selection_probabilities_), dtype=bool)
        mask[self.selected_features_] = True
        return mask

    def get_feature_names_out(
        self, feature_names: list[str] | None = None
    ) -> list[str]:
        if feature_names is None:
            return [f"feature_{i}" for i in self.selected_features_]
        return [feature_names[i] for i in self.selected_features_]

    @override
    def set_output(self, *, transform: str | None = None) -> StabilitySelector:
        return self

    def _log_prefix(self) -> str:
        return f"[{self.name}] " if self.name else ""
