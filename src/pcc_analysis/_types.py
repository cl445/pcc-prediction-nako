"""Shared TypedDict definitions for structured return types.

Functions across the codebase return these rather than ``dict[str, Any]``,
so that callers and the type checker see the exact key set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .meta_learner import MetaLearner
    from .protocols import ConfoundedEstimator


# -- SHAP analysis ----------------------------------------------------------


class ShapResults(TypedDict):
    """Return type of :func:`shap_analysis.compute_meta_learner_shap`."""

    shap_values: np.ndarray
    expected_value: float
    feature_names: list[str]
    feature_importance: pd.DataFrame
    modality_importance: pd.DataFrame


# -- Evaluation --------------------------------------------------------------


class EvaluationResult(TypedDict):
    """Return type of :func:`evaluation.evaluate_model_comprehensive`."""

    model_name: str
    metrics: dict[str, float]
    ci_results: dict[str, tuple[float, float]]
    dca: pd.DataFrame
    fig_curves: Figure
    fig_dca: Figure
    threshold_summary: dict[str, float]
    threshold_detail: pd.DataFrame


# -- Statistical tests -------------------------------------------------------


class NadeauBengioResult(TypedDict):
    """Return type of :func:`statistical_tests.nadeau_bengio_corrected_t_test`."""

    t_statistic: float
    p_value: float
    df: float
    mean_diff: float
    corrected_std: float


# -- Orchestration -----------------------------------------------------------


class FoldResult(TypedDict):
    """Per-fold metric scores from nested CV."""

    fold: int
    roc_auc: float
    pr_auc: float


class ModalityScore(TypedDict):
    """Per-fold per-modality metric scores from nested CV."""

    fold: int
    modality: str
    roc_auc: float
    pr_auc: float


class CVResults(TypedDict):
    """Nested CV predictions and per-fold metrics."""

    y_true: np.ndarray
    y_pred_proba: np.ndarray
    fold_results: list[FoldResult]
    modality_scores: list[ModalityScore]
    feature_importance_permutation: pd.DataFrame | None


class PartialCVResults(CVResults):
    """Partial CV results from a fold-subset run."""

    partial: bool
    computed_folds: list[int]


class FinalModelResult(TypedDict):
    """Return type of :meth:`PCCMultimodalPipeline._train_final_model`.

    The ``modality_pipelines`` and ``confounders_train`` fields are
    required for the transfer-validation runs
    (``scripts/pipeline/03_apply_transfer.py``): per-modality pipelines trained
    on the whole source cohort are applied to the target cohort without
    refit, with a privacy-safe source-cohort confounder snapshot kept
    around for KS-distance reporting of confounder-distribution drift.

    The ``confounders_train`` snapshot is column-wise shuffled with a
    :class:`pandas.RangeIndex` so that per-row tuples (age, sex, centre)
    cannot be reconstructed from a leaked pickle; KS-distance is a
    per-column statistic, so this preserves the transfer-report's
    drift signal exactly.
    """

    meta_learner: MetaLearner
    modality_names: list[str]
    feature_importance: pd.DataFrame
    oof_predictions: np.ndarray
    modality_pipelines: dict[str, ConfoundedEstimator]
    modality_column_layout: dict[str, list[str]]
    confounders_train: pd.DataFrame


class PermutationTestResult(TypedDict):
    """Return type of :meth:`PCCMultimodalPipeline._permutation_test_cv`."""

    score: float
    p_value: float
    perm_mean: float
    n_permutations: int


class FoldPermImportance(TypedDict):
    """Per-fold permutation importance from the meta-learner."""

    fold: int
    feature_names: list[str]
    importances_mean: np.ndarray
    importances_std: np.ndarray


class FoldArtifact(TypedDict):
    """Schema for pickled per-fold artifacts in distributed runs."""

    fold_idx: int
    test_idx: np.ndarray
    y_pred_fold: np.ndarray
    y_te: np.ndarray
    fold_result: FoldResult
    modality_scores_fold: list[ModalityScore]
    fold_perm_importance: FoldPermImportance | None
    config_hash: str


class PerFoldOoS(TypedDict):
    """In-memory per-fold artifacts for out-of-sample meta-learner analyses.

    Carries the data needed by ``_modality_ablation_oos`` and
    ``_incremental_performance_oos``: the train-fold stacking matrix the
    fold-specific meta-learner was trained on, the test-fold base-learner
    predictions the meta-learner has never seen, and the fitted meta-learner
    itself.
    """

    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray
    oof_train: np.ndarray
    test_preds: np.ndarray
    modality_names: list[str]
    meta: MetaLearner


class PipelineConfig(TypedDict):
    """Return type of :meth:`PCCMultimodalPipeline._get_config`.

    Every field except ``timestamp`` feeds the config hash that
    ``merge_fold_results`` uses to reject fold artifacts computed under
    different settings, so this is the record of what makes one run
    different from another. It has to name every such setting, including
    the ones the pipeline never sees itself: ``target`` and
    ``clean_controls`` are applied upstream in ``load_pipeline_data``, so
    leaving them out of this dict makes the primary and the mixed-controls
    configurations — two runs that disagree about the outcome definition —
    produce byte-identical configuration records, distinguishable only by
    the path of their run directory.

    Environment provenance (commit, library versions) deliberately lives
    outside this dict — see :mod:`pcc_analysis._provenance`.
    """

    n_outer_folds: int
    n_inner_folds: int
    meta_model: str
    orthogonalize: bool
    random_state: int
    timestamp: str
    cohort: str
    stack_variant: str
    package_versions: str
    device: str
    data_fingerprint: str
    target: str
    clean_controls: bool
    split_mh_submodalities: bool
    amendment_features: bool
    hyperparam_iterations: int
    n_subsamples: int


class PipelineResult(TypedDict):
    """Full pipeline result returned by :meth:`PCCMultimodalPipeline.run`."""

    cv_results: CVResults
    permutation_test: PermutationTestResult | None
    modality_comparisons: pd.DataFrame | None
    final_model: FinalModelResult
    shap_results: ShapResults | None
    modality_ablation: pd.DataFrame | None
    modality_ablation_oos: pd.DataFrame | None
    incremental_performance: pd.DataFrame | None
    incremental_performance_oos: pd.DataFrame | None
    evaluation: EvaluationResult
    config: PipelineConfig
