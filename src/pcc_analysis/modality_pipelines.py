"""Modality-specific pipelines for PCC prediction.

Each pipeline includes preprocessing, orthogonalization, imputation,
feature selection, and a base learner (LogisticRegressionCV).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from .custom_pipeline import ConfoundedPipeline
from .orthogonalization import Orthogonalizer
from .preprocessing import (
    FeatureToConfounder,
    LogTransformer,
    MissingnessThreshold,
    create_cognitive_preprocessor,
    create_demographics_preprocessor,
    create_lab_values_preprocessor,
    create_lung_function_preprocessor,
    create_medical_history_preprocessor,
    create_mental_health_preprocessor,
    create_physical_activity_preprocessor,
    create_ses_preprocessor,
)
from .stability_selection import StabilitySelector

# Per-modality log1p targets — heavily right-skewed measurements that
# approximate log-normal distributions. Applied upstream of the DML
# orthogonaliser so the linear classifier sees near-Gaussian features.
LAB_LOG_COLUMNS: list[str] = [
    "crp",
    "alt",
    "ast",
    "ggt",
    "triglycerides",
    "glucose",
]
SES_LOG_COLUMNS: list[str] = [
    "income_category",
    "income_position",
    "income_weighted",
    "needs_weighted",
    "income_adequacy",
    "employment_duration_years",
    "employment_duration_total",
]
COGNITIVE_LOG_COLUMNS: list[str] = [
    "stroop_colors_time",
    "stroop_interference_time",
]
PHYSICAL_ACTIVITY_LOG_COLUMNS: list[str] = [
    "household_minutes_week",
    "active_transport_summer",
    "active_transport_winter",
    "walking_summer",
    "walking_winter",
    "cycling_summer",
    "cycling_winter",
    "sports_spring",
    "sports_summer",
    "sports_autumn",
    "sports_winter",
    "sports_combined",
    "met_sports_combined",
    "met_sports_spring",
    "met_sports_summer",
    "met_sports_autumn",
    "met_sports_winter",
    "sitting_weekday",
    "sitting_saturday",
    "sitting_sunday",
    "pa_total",
    "pa_summer",
    "pa_winter",
    "met_total",
    "met_summer",
    "met_winter",
    # Second activity instrument, same heavy right skew as the QUAP MET
    # totals (median 3,360, max 80,600 MET-minutes/week). Left untransformed
    # it would enter the linear base learner on a different scale from the
    # log1p-transformed met_total it sits beside.
    "gpaq_met_total",
]

logger = logging.getLogger(__name__)


class ModalityPipelineFactory:
    """Factory for creating modality-specific pipelines.

    Parameters
    ----------
    random_state : int, optional
        Random seed.
    orthogonalize : bool
        Whether to orthogonalize (except demographics).
    cv : int
        CV folds for orthogonalization / stability selection.
    """

    def __init__(
        self,
        random_state: int | None = None,
        orthogonalize: bool = True,
        cv: int = 5,
        n_jobs: int = -1,
        n_subsamples: int = 100,
    ) -> None:
        self.random_state = random_state
        self.orthogonalize = orthogonalize
        self.cv = cv
        self.n_jobs = n_jobs
        self.n_subsamples = n_subsamples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lr_cv(
        self,
        l1_ratios: list[float] | None = None,
        cs: int | list[float] = 10,
        max_iter: int = 10_000,
        solver: Literal[
            "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
        ] = "saga",
    ) -> LogisticRegressionCV:
        """Logistic regression with penalty type and C selected by 3-fold CV."""
        if l1_ratios is None:
            l1_ratios = [0.0, 0.5, 1.0]
        return LogisticRegressionCV(
            l1_ratios=l1_ratios,
            Cs=cs,
            scoring="average_precision",
            cv=3,
            solver=solver,
            max_iter=max_iter,
            tol=1e-4,
            random_state=self.random_state,
            class_weight="balanced",
            # Opt into the simplified fitted-attribute layout of sklearn 1.10
            # (coefs_paths_, Cs_, l1_ratios_ drop redundant dimensions);
            # without it every .fit() call emits a FutureWarning. The
            # parameter exists in the installed sklearn 1.9. Pyrefly does not
            # see it because it type-checks sklearn against the stubs bundled
            # in its own binary rather than against the installed package,
            # and those describe an older release.
            use_legacy_attributes=False,  # pyrefly: ignore[unexpected-keyword]
        )

    def _ortho_step(self) -> tuple[str, Orthogonalizer]:
        return (
            "orthogonalizer",
            Orthogonalizer(
                cv=self.cv, random_state=self.random_state, n_jobs=self.n_jobs
            ),
        )

    def _stability_step(
        self, name: str, lambda_grid: np.ndarray | None = None
    ) -> tuple[str, StabilitySelector]:
        return (
            "stability_selector",
            StabilitySelector(
                lambda_grid=lambda_grid,
                threshold=0.6,
                n_subsamples=self.n_subsamples,
                random_state=self.random_state,
                name=name,
                n_jobs=self.n_jobs,
            ),
        )

    # ------------------------------------------------------------------
    # Pipeline builders
    # ------------------------------------------------------------------

    def create_demographics_pipeline(self) -> ConfoundedPipeline:
        """Demographics (baseline, no orthogonalization). 3 raw → ~20 after one-hot."""
        steps = [
            ("preprocessor", create_demographics_preprocessor()),
            ("scaler", StandardScaler()),
            (
                "classifier",
                self._lr_cv(
                    l1_ratios=[0.0],
                    cs=np.logspace(-2, 2, 5).tolist(),
                    solver="lbfgs",
                ),
            ),
        ]
        return ConfoundedPipeline(steps)

    def create_ses_pipeline(self) -> ConfoundedPipeline:
        """Socioeconomic status. 43 → 27 raw, ~75 after encoding."""
        steps: list[tuple[str, BaseEstimator]] = [
            ("preprocessor", create_ses_preprocessor()),
            ("log1p", LogTransformer(columns=SES_LOG_COLUMNS)),
            ("missingness", MissingnessThreshold()),
            ("imputer", KNNImputer(n_neighbors=5)),
        ]
        if self.orthogonalize:
            steps.append(self._ortho_step())
        steps += [
            ("scaler", StandardScaler()),
            self._stability_step(
                "ses",
                lambda_grid=np.array([0.001, 0.01, 0.1, 1.0, 10.0]),
            ),
            (
                "classifier",
                self._lr_cv(
                    l1_ratios=[0.0, 1.0],
                    cs=np.logspace(-3, 2, 5).tolist(),
                    max_iter=20_000,
                ),
            ),
        ]
        return ConfoundedPipeline(steps)

    def create_cognitive_pipeline(self) -> ConfoundedPipeline:
        """Cognitive tests. 12 → 8 features after dropping difference scores."""
        steps: list[tuple[str, BaseEstimator]] = [
            ("preprocessor", create_cognitive_preprocessor()),
            ("log1p", LogTransformer(columns=COGNITIVE_LOG_COLUMNS)),
            ("missingness", MissingnessThreshold()),
            ("imputer", KNNImputer(n_neighbors=5)),
        ]
        if self.orthogonalize:
            steps.append(self._ortho_step())
        steps += [
            ("scaler", StandardScaler()),
            self._stability_step(
                "cognitive", lambda_grid=np.array([0.001, 0.01, 0.1, 1.0])
            ),
            (
                "classifier",
                self._lr_cv(
                    l1_ratios=[0.5, 1.0],
                    cs=np.logspace(-3, 2, 5).tolist(),
                ),
            ),
        ]
        return ConfoundedPipeline(steps)

    def create_physical_activity_pipeline(self) -> ConfoundedPipeline:
        """Physical activity. 17 features."""
        steps: list[tuple[str, BaseEstimator]] = [
            ("preprocessor", create_physical_activity_preprocessor()),
            ("log1p", LogTransformer(columns=PHYSICAL_ACTIVITY_LOG_COLUMNS)),
            ("missingness", MissingnessThreshold()),
            ("imputer", KNNImputer(n_neighbors=5)),
        ]
        if self.orthogonalize:
            steps.append(self._ortho_step())
        steps += [
            ("scaler", StandardScaler()),
            self._stability_step(
                "physical_activity",
                lambda_grid=np.array([0.001, 0.01, 0.1, 1.0, 10.0]),
            ),
            (
                "classifier",
                self._lr_cv(
                    l1_ratios=[0.5, 1.0],
                    cs=np.logspace(-3, 2, 5).tolist(),
                ),
            ),
        ]
        return ConfoundedPipeline(steps)

    def create_medical_history_pipeline(self) -> ConfoundedPipeline:
        """Medical history. ~35 → ~29 features after dropping derived columns."""
        steps: list[tuple[str, BaseEstimator]] = [
            ("preprocessor", create_medical_history_preprocessor()),
            ("missingness", MissingnessThreshold()),
            ("imputer", KNNImputer(n_neighbors=5)),
        ]
        if self.orthogonalize:
            steps.append(self._ortho_step())
        steps += [
            ("scaler", StandardScaler()),
            self._stability_step(
                "medical_history",
                lambda_grid=np.array([0.001, 0.01, 0.1, 1.0, 10.0]),
            ),
            (
                "classifier",
                self._lr_cv(
                    l1_ratios=[0.5, 1.0],
                    cs=np.logspace(-3, 2, 5).tolist(),
                    max_iter=20_000,
                ),
            ),
        ]
        return ConfoundedPipeline(steps)

    def create_lab_values_pipeline(self) -> ConfoundedPipeline:
        """Lab values. 22 → 20 biomarkers after dropping cholesterol/creatinine."""
        steps: list[tuple[str, BaseEstimator]] = [
            ("preprocessor", create_lab_values_preprocessor()),
            ("log1p", LogTransformer(columns=LAB_LOG_COLUMNS)),
            ("missingness", MissingnessThreshold()),
            ("imputer", KNNImputer(n_neighbors=5)),
        ]
        if self.orthogonalize:
            steps.append(self._ortho_step())
        steps += [
            ("scaler", StandardScaler()),
            self._stability_step(
                "lab_values", lambda_grid=np.array([0.01, 0.1, 1.0, 10.0])
            ),
            (
                "classifier",
                self._lr_cv(
                    l1_ratios=[0.5, 1.0],
                    cs=np.logspace(-3, 2, 5).tolist(),
                ),
            ),
        ]
        return ConfoundedPipeline(steps)

    def create_cardiovascular_pipeline(self) -> ConfoundedPipeline:
        """Cardiovascular. ~7 features."""
        steps: list[tuple[str, BaseEstimator]] = [
            ("missingness", MissingnessThreshold()),
            ("imputer", KNNImputer(n_neighbors=5)),
        ]
        if self.orthogonalize:
            steps.append(self._ortho_step())
        steps += [
            ("scaler", StandardScaler()),
            (
                "classifier",
                self._lr_cv(
                    l1_ratios=[0.0],
                    cs=np.logspace(-1, 3, 5).tolist(),
                    solver="lbfgs",
                ),
            ),
        ]
        return ConfoundedPipeline(steps)

    def create_lung_function_pipeline(self) -> ConfoundedPipeline:
        """Lung function. 7 → 4 features after dropping absolute volumes."""
        steps: list[tuple[str, BaseEstimator]] = [
            ("preprocessor", create_lung_function_preprocessor()),
            ("missingness", MissingnessThreshold()),
            ("imputer", KNNImputer(n_neighbors=5)),
        ]
        if self.orthogonalize:
            steps.append(self._ortho_step())
        steps += [
            ("scaler", StandardScaler()),
            (
                "classifier",
                self._lr_cv(
                    l1_ratios=[0.0],
                    cs=np.logspace(-1, 2, 5).tolist(),
                    solver="lbfgs",
                ),
            ),
        ]
        return ConfoundedPipeline(steps)

    def create_mental_health_pipeline(self) -> ConfoundedPipeline:
        """Baseline mental health. 19 -> ~14 features after dropping redundant cutoffs."""
        steps: list[tuple[str, BaseEstimator]] = [
            ("preprocessor", create_mental_health_preprocessor()),
            ("missingness", MissingnessThreshold()),
            ("imputer", KNNImputer(n_neighbors=5)),
        ]
        if self.orthogonalize:
            steps.append(self._ortho_step())
        steps += [
            ("scaler", StandardScaler()),
            self._stability_step(
                "mental_health",
                lambda_grid=np.array([0.001, 0.01, 0.1, 1.0]),
            ),
            (
                "classifier",
                self._lr_cv(
                    l1_ratios=[0.5, 1.0],
                    cs=np.logspace(-3, 2, 5).tolist(),
                ),
            ),
        ]
        return ConfoundedPipeline(steps)

    def create_mri_pipeline(self, atlas_name: str = "desikan") -> ConfoundedPipeline:
        """MRI atlas-specific pipeline.

        Uses the residual method for ICV correction: eTIV is extracted from
        the feature matrix and passed to the Orthogonalizer as an additional
        covariate (via ``FeatureToConfounder``), rather than dividing all
        volumes by eTIV (proportion method).  This avoids double-correcting
        sex effects (eTIV correlates with sex) and handles non-proportional
        region-ICV relationships correctly.

        PCA (95% variance) is applied to all atlases with >20 features.
        Stability selection is not used — elastic net regularisation in the
        classifier handles both feature selection and correlated-feature
        grouping, which is the standard approach in the neuroimaging
        literature for FreeSurfer ROI-based classification.
        """
        steps: list[tuple[str, BaseEstimator]] = [
            ("etiv_confounder", FeatureToConfounder(columns="etiv")),
            ("missingness", MissingnessThreshold()),
            ("imputer", KNNImputer(n_neighbors=5)),
        ]
        if self.orthogonalize:
            steps.append(self._ortho_step())

        # PCA for atlases with >20 features
        if atlas_name in ("desikan", "destrieux", "julich", "subcortical", "yeo"):
            steps += [
                ("pca_scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=self.random_state)),
            ]
        elif atlas_name != "cerebellar":
            raise ValueError(f"Unknown atlas: {atlas_name}")

        steps += [
            ("scaler", StandardScaler()),
            (
                "classifier",
                self._lr_cv(
                    l1_ratios=[0.5, 1.0],
                    cs=np.logspace(-3, 1, 5).tolist(),
                ),
            ),
        ]

        return ConfoundedPipeline(steps)

    def create_all_pipelines(self) -> dict[str, ConfoundedPipeline]:
        """Create all 15 modality pipelines (no psychometric)."""
        pipelines = {
            "demographics": self.create_demographics_pipeline(),
            "ses": self.create_ses_pipeline(),
            "cognitive": self.create_cognitive_pipeline(),
            "physical_activity": self.create_physical_activity_pipeline(),
            "medical_history": self.create_medical_history_pipeline(),
            "lab_values": self.create_lab_values_pipeline(),
            "cardiovascular": self.create_cardiovascular_pipeline(),
            "lung_function": self.create_lung_function_pipeline(),
            "mental_health": self.create_mental_health_pipeline(),
        }
        for atlas in [
            "desikan",
            "destrieux",
            "julich",
            "yeo",
            "subcortical",
            "cerebellar",
        ]:
            pipelines[f"mri_{atlas}"] = self.create_mri_pipeline(atlas)

        logger.info("Created %d modality pipelines", len(pipelines))
        return pipelines
