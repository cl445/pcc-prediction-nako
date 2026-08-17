"""Meta-learner for multi-modal stacking.

Combines out-of-fold predictions from modality base learners via
XGBoost GBDT with hyperparameter tuning and calibration.
"""

from __future__ import annotations

import logging
import time
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from ._array_utils import to_float_array
from ._gpu_utils import xgb_device
from ._sklearn_typing import clone

if TYPE_CHECKING:
    from .protocols import Classifier, ConfoundedEstimator

logger = logging.getLogger(__name__)

META_HYPERPARAM_ITERATIONS: int = 50
"""Randomized-search draws for the meta-learner's XGBoost stage.

The search space below spans seven hyperparameters, so a ten-draw search
covers a vanishingly small part of it and leaves the meta-learner closer to a
default XGBoost than to a tuned one. Fifty is also the ceiling: the search
below clamps to this constant, and ``PCCMultimodalPipeline`` clamps the
number it records in ``config.csv`` to the same one, so the configured
count and the count of draws actually taken cannot drift apart.

This is the default only. The incremental-performance comparison in
``orchestration.py`` deliberately passes ``0`` so that every pairwise
[baseline + modality] contrast is evaluated under identical, untuned
meta-learner settings.
"""


class MetaLearner(BaseEstimator):
    """Meta-learner combining modality predictions via XGBoost GBDT.

    Parameters
    ----------
    add_interactions : bool
        Add pairwise interaction terms.
    calibration_method : {'sigmoid', 'isotonic'}
        Calibration method passed to ``CalibratedClassifierCV``.
    cv : int
        CV folds for calibration.
    random_state : int, optional
        Random seed.
    n_jobs : int
        Parallel jobs.
    hyperparam_iterations : int
        RandomizedSearchCV draws; ``0`` disables the search and uses
        default XGBoost parameters.  See ``META_HYPERPARAM_ITERATIONS``.
    """

    def __init__(
        self,
        add_interactions: bool = False,
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
        cv: int = 5,
        random_state: int | None = None,
        n_jobs: int = -1,
        hyperparam_iterations: int = META_HYPERPARAM_ITERATIONS,
    ) -> None:
        self.add_interactions = add_interactions
        self.calibration_method = calibration_method
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.hyperparam_iterations = hyperparam_iterations

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> MetaLearner:
        """Fit XGBoost GBDT with optional hyperparameter search, then calibrate.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_modalities)
            Out-of-fold probability predictions from base learners.
        y : ndarray of shape (n_samples,)
            Binary target.
        feature_names : list[str], optional
            Modality names. Auto-generated if not provided.
        """
        X = to_float_array(X)
        y = to_float_array(y).ravel()
        n_modalities = X.shape[1]

        if feature_names is None:
            feature_names = [f"modality_{i}" for i in range(n_modalities)]
        self.base_feature_names_ = feature_names

        if self.add_interactions:
            X_meta, self.feature_names_ = self._add_interactions(X, feature_names)
        else:
            X_meta = X
            self.feature_names_ = feature_names

        if np.any(np.isnan(X_meta)):
            n_nan_cols = int(np.any(np.isnan(X_meta), axis=0).sum())
            logger.info(
                "Meta-learner input has NaN in %d feature(s) (systematic missingness)",
                n_nan_cols,
            )

        # XGBoost GBDT with hyperparameter search and automatic GPU detection
        device = xgb_device()

        # Balanced class weight: scale_pos_weight compensates for class
        # imbalance by weighting the positive class proportionally.
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        base_params = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "device": device,
            # XGBoost n_jobs=1: RandomizedSearchCV handles outer parallelism
            # via self.n_jobs; letting XGBoost also spawn threads causes
            # oversubscription.  On GPU, parallelism is handled by CUDA.
            "n_jobs": 1,
            "random_state": self.random_state,
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": spw,
        }
        base_model = XGBClassifier(**base_params)

        if self.hyperparam_iterations > 0:
            param_distributions = {
                "learning_rate": [0.01, 0.03, 0.1, 0.2],
                "max_depth": [2, 3, 4, 5],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
                "gamma": [0.0, 0.05, 0.1, 0.2],
                "n_estimators": [100, 200, 400],
            }
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=min(self.hyperparam_iterations, META_HYPERPARAM_ITERATIONS),
                scoring="average_precision",
                cv=max(3, self.cv),
                random_state=self.random_state,
                n_jobs=1 if device == "cuda" else self.n_jobs,
            )
            search.fit(X_meta, y)
            logger.info("Best GBDT params: %s", search.best_params_)
            # refit=True (default) ensures best_estimator_ is already
            # fitted on the full dataset — no need to clone + refit. The cast
            # restores what the search itself guarantees: best_estimator_ is
            # the ``estimator`` argument's class, here XGBClassifier.
            self.meta_model_: Classifier = cast("Classifier", search.best_estimator_)
        else:
            self.meta_model_ = base_model
            self.meta_model_.fit(X_meta, y)

        # Calibrate
        n_positive = int(np.sum(y == 1))
        if X_meta.shape[0] >= 100 and n_positive >= 30:
            self.calibrated_model_: Classifier = CalibratedClassifierCV(
                # sklearn takes the estimator nominally; the meta-model is
                # described structurally, which is what its wrapper needs.
                cast("BaseEstimator", self.meta_model_),
                method=self.calibration_method,
                cv=min(self.cv, 3),
            )
            self.calibrated_model_.fit(X_meta, y)
        else:
            logger.warning(
                "Skipping calibration: n=%d, n_positive=%d",
                X_meta.shape[0],
                n_positive,
            )
            self.calibrated_model_ = self.meta_model_

        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return calibrated class probabilities of shape (n_samples, 2)."""
        X = to_float_array(X)
        if self.add_interactions:
            X_meta, _ = self._add_interactions(X, self.base_feature_names_)
        else:
            X_meta = X
        return np.asarray(self.calibrated_model_.predict_proba(X_meta))

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return class labels (0 or 1) based on argmax of predicted probabilities."""
        return np.asarray(self.predict_proba(X).argmax(axis=1))

    def transform_meta_features(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Transform raw OoF predictions into meta-features (add interactions if enabled)."""
        X = to_float_array(X)
        if self.add_interactions:
            X_meta, _ = self._add_interactions(X, self.base_feature_names_)
        else:
            X_meta = X
        return X_meta

    def _add_interactions(
        self, X: np.ndarray, feature_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        interactions = []
        interaction_names = []
        for i, j in combinations(range(X.shape[1]), 2):
            interactions.append(X[:, i] * X[:, j])
            interaction_names.append(f"{feature_names[i]} x {feature_names[j]}")
        if interactions:
            X_out = np.column_stack([X, *interactions])
            return X_out, feature_names + interaction_names
        return X, feature_names

    def get_feature_importance(self) -> pd.DataFrame:
        if hasattr(self.meta_model_, "feature_importances_"):
            importance = self.meta_model_.feature_importances_
        elif hasattr(self.meta_model_, "coef_"):
            importance = np.abs(self.meta_model_.coef_).ravel()
        else:
            importance = np.zeros(len(self.feature_names_))
        return pd.DataFrame(
            {"feature": self.feature_names_, "importance": importance}
        ).sort_values("importance", ascending=False)


def _process_one_modality(
    modality_name: str,
    pipeline: ConfoundedEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splitter: StratifiedKFold,
    confounders: pd.DataFrame | None,
    return_fitted: bool,
) -> tuple[np.ndarray, ConfoundedEstimator | None]:
    """Process a single modality: compute OOF predictions and optionally refit."""
    n_samples = len(y)
    all_missing = np.asarray(pd.isna(X).all(axis=1).to_numpy())

    try:
        oof_pred = np.full(n_samples, np.nan)
        n_folds = cv_splitter.get_n_splits()
        n_features = X.shape[1]
        logger.info(
            "[%s] Starting OoF (%d folds, %d features, %d samples)",
            modality_name,
            n_folds,
            n_features,
            (~all_missing).sum(),
        )
        for fold_i, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y), 1):
            train_has = ~all_missing[train_idx]
            test_has = ~all_missing[test_idx]
            train_idx_d = train_idx[train_has]
            test_idx_d = test_idx[test_has]
            if len(train_idx_d) == 0 or len(test_idx_d) == 0:
                continue

            pipe_fold = clone(pipeline)
            X_tr = X.iloc[train_idx_d]
            X_te = X.iloc[test_idx_d]
            y_tr = y[train_idx_d]
            conf_tr = confounders.iloc[train_idx_d] if confounders is not None else None
            conf_te = confounders.iloc[test_idx_d] if confounders is not None else None

            fit_kw: dict[str, Any] = {"confounders": conf_tr}
            _t = time.monotonic()
            pipe_fold.fit(X_tr, y_tr, **fit_kw)
            oof_pred[test_idx_d] = pipe_fold.predict_proba(X_te, confounders=conf_te)[
                :, 1
            ]
            logger.info(
                "[%s] Inner fold %d/%d done (%.1fs)",
                modality_name,
                fold_i,
                n_folds,
                time.monotonic() - _t,
            )

        # Optionally fit on full data to avoid a redundant refit later
        fitted_pipeline: ConfoundedEstimator | None = None
        if return_fitted:
            has_data = ~all_missing
            if has_data.sum() > 0:
                full_pipe = clone(pipeline)
                conf_full = confounders[has_data] if confounders is not None else None
                fit_kw_full: dict[str, Any] = {"confounders": conf_full}
                logger.info("[%s] Fitting final model on full data", modality_name)
                full_pipe.fit(X[has_data], y[has_data], **fit_kw_full)
                fitted_pipeline = full_pipe

        logger.info("[%s] Complete", modality_name)
    except Exception:
        # exception() over error(): the traceback is the only way to tell a
        # convergence failure from a bug once a modality has silently
        # degraded to all-NaN predictions.
        logger.exception("OoF failed for %s", modality_name)
        return np.full(n_samples, np.nan), None
    else:
        return oof_pred, fitted_pipeline


def create_oof_predictions(
    pipelines: dict[str, ConfoundedEstimator],
    X_dict: dict[str, pd.DataFrame],
    y: np.ndarray,
    cv: int = 5,
    random_state: int | None = None,
    confounders: pd.DataFrame | None = None,
    *,
    n_jobs: int = 1,
    return_fitted: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, ConfoundedEstimator] | None]:
    """Create out-of-fold predictions from modality pipelines.

    Handles systematic missingness by passing NaN to the meta-learner
    for participants without data in a given modality.

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs for modality-level parallelism.
    return_fitted : bool
        If True, also fit each pipeline on the full dataset and return
        them as a dict, avoiding a redundant refit later.

    Returns
    -------
    oof_predictions : ndarray of shape (n_samples, n_modalities)
    modality_names : list of str
    fitted_pipelines : dict or None
        Only returned (non-None) when ``return_fitted=True``.
    """
    modality_names = list(pipelines.keys())
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    results = Parallel(n_jobs=n_jobs, verbose=0, prefer="threads")(
        delayed(_process_one_modality)(
            mod_name,
            pipelines[mod_name],
            X_dict[mod_name],
            y,
            cv_splitter,
            confounders,
            return_fitted,
        )
        for mod_name in modality_names
    )

    n_samples = len(y)
    oof_predictions = np.full((n_samples, len(modality_names)), np.nan)
    fitted_pipelines: dict[str, ConfoundedEstimator] | None = (
        {} if return_fitted else None
    )

    for i, (oof_pred, fitted_pipe) in enumerate(results):
        oof_predictions[:, i] = oof_pred
        if fitted_pipelines is not None and fitted_pipe is not None:
            fitted_pipelines[modality_names[i]] = fitted_pipe

    return oof_predictions, modality_names, fitted_pipelines


def train_meta_learner(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    modality_names: list[str],
    add_interactions: bool = False,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
    random_state: int | None = None,
    n_jobs: int = -1,
    hyperparam_iterations: int = META_HYPERPARAM_ITERATIONS,
) -> MetaLearner:
    """Train meta-learner on out-of-fold predictions."""
    meta = MetaLearner(
        add_interactions=add_interactions,
        calibration_method=calibration_method,
        random_state=random_state,
        n_jobs=n_jobs,
        hyperparam_iterations=hyperparam_iterations,
    )
    meta.fit(oof_predictions, y, feature_names=modality_names)
    return meta
