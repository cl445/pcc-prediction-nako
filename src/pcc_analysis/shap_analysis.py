"""SHAP-based feature and modality importance for the meta-learner.

Uses TreeExplainer on the raw XGBoost model (before calibration) to
compute SHAP values, then aggregates to modality-level importance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import shap

if TYPE_CHECKING:
    from ._types import ShapResults
    from .meta_learner import MetaLearner

logger = logging.getLogger(__name__)


def compute_meta_learner_shap(
    meta_learner: MetaLearner,
    oof_predictions: np.ndarray,
    modality_names: list[str],
) -> ShapResults:
    """Compute SHAP values for the meta-learner.

    Parameters
    ----------
    meta_learner : MetaLearner
        Fitted meta-learner with ``meta_model_`` attribute.
    oof_predictions : ndarray, shape (n_samples, n_modalities)
        Out-of-fold predictions used as meta-learner input.
    modality_names : list of str
        Names of the base-learner modalities.

    Returns
    -------
    ShapResults
        Typed dictionary with SHAP values, expected value, feature names,
        feature-level and modality-level importance DataFrames.
    """
    # Transform to meta-feature space (includes interactions if enabled)
    X_meta = meta_learner.transform_meta_features(oof_predictions)
    feature_names = meta_learner.feature_names_

    # Use TreeExplainer on the raw XGBoost model (not calibrated)
    explainer = shap.TreeExplainer(meta_learner.meta_model_)
    shap_values = np.asarray(explainer.shap_values(X_meta))

    # The pinned shap returns (n_samples, n_features) for a binary XGBoost
    # model, but has historically returned one array per class. When it does,
    # the class axis is the last one, so the positive class is taken from
    # there — indexing the first axis would silently return one sample's
    # matrix instead, in the right rank and the wrong meaning.
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 1]
    if shap_values.shape != X_meta.shape:
        raise ValueError(
            f"SHAP values have shape {shap_values.shape}, expected "
            f"{X_meta.shape} (one value per meta-feature per sample)"
        )

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = (
            float(expected_value[1])
            if len(expected_value) > 1
            else float(expected_value[0])
        )
    else:
        expected_value = float(expected_value)

    # Feature-level importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Modality-level importance: aggregate features to their parent modality
    modality_importance = _aggregate_to_modalities(
        feature_names, mean_abs_shap, modality_names
    )

    logger.info(
        "SHAP analysis: %d features, %d modalities, top feature: %s (%.4f)",
        len(feature_names),
        len(modality_names),
        feature_importance.iloc[0]["feature"],
        feature_importance.iloc[0]["mean_abs_shap"],
    )

    return {
        "shap_values": shap_values,
        "expected_value": expected_value,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "modality_importance": modality_importance,
    }


def _aggregate_to_modalities(
    feature_names: list[str],
    importances: np.ndarray,
    modality_names: list[str],
) -> pd.DataFrame:
    """Aggregate feature importances to modality level.

    Interaction terms (e.g. "modA x modB") are split equally between
    their parent modalities.
    """
    modality_imp: dict[str, float] = dict.fromkeys(modality_names, 0.0)

    for fname, imp in zip(feature_names, importances, strict=True):
        if " x " in fname:
            # Interaction term: split between parents
            parts = fname.split(" x ")
            n_parts = sum(1 for p in parts if p in modality_imp)
            if n_parts > 0:
                share = float(imp) / n_parts
                for p in parts:
                    if p in modality_imp:
                        modality_imp[p] += share
        elif fname in modality_imp:
            modality_imp[fname] += float(imp)
        else:
            logger.debug("Feature '%s' not mapped to any modality", fname)

    return (
        pd.DataFrame(
            [{"modality": m, "mean_abs_shap": v} for m, v in modality_imp.items()]
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
