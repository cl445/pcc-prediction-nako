"""Preprocessing transformers for modality-specific data preparation.

Handles non-numeric data types (categorical, ordinal, dates) before
feeding data into ML pipelines.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Literal, override

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)


class FrameTransformer(ABC, BaseEstimator, TransformerMixin):
    """Base for the transformers in this module: a DataFrame in, one out.

    Three things every one of them needs and sklearn's mixin cannot give them.

    ``set_output`` is a no-op returning ``self``: these transformers already
    hand back a DataFrame, so sklearn's output-wrapping machinery has nothing
    left to do and would only re-wrap what is already in the requested form.

    ``fit_transform`` is declared rather than inherited, because
    ``TransformerMixin.fit_transform`` is typed as returning an ndarray. Right
    for sklearn's own transformers, wrong for these, and it left every caller
    here — the tests above all — reasoning about an array that never existed.
    The body is what the mixin does.

    ``fit`` and ``transform`` are abstract so the DataFrame contract is stated
    once instead of re-derived from each subclass.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> FrameTransformer: ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    @override
    def set_output(self, *, transform: str | None = None) -> FrameTransformer:
        return self

    @override
    def fit_transform(  # pyrefly: ignore[bad-override]
        self, X: pd.DataFrame, y: np.ndarray | None = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class MissingnessThreshold(FrameTransformer):
    """Drop columns whose missing fraction exceeds *threshold* (fit on train).

    Learns which columns to drop from the training data and applies the same
    column mask to unseen data.  Placed before imputation so the imputer
    only sees columns with enough observed values.

    Parameters
    ----------
    threshold : float
        Maximum allowed fraction of missing values per column (default 0.5).
    """

    def __init__(self, threshold: float = 0.50) -> None:
        self.threshold = threshold

    @override
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> MissingnessThreshold:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MissingnessThreshold requires a pandas DataFrame")
        frac = X.isna().mean()
        self.columns_to_drop_: list[str] = frac[frac > self.threshold].index.tolist()
        if self.columns_to_drop_:
            logger.info(
                "MissingnessThreshold: dropping %d columns with >%.0f%% missing",
                len(self.columns_to_drop_),
                self.threshold * 100,
            )
        return self

    @override
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MissingnessThreshold requires a pandas DataFrame")
        cols = [c for c in self.columns_to_drop_ if c in X.columns]
        return X.drop(columns=cols) if cols else X

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        if input_features is None:
            raise ValueError("input_features required")
        return np.array([f for f in input_features if f not in self.columns_to_drop_])


class LogTransformer(FrameTransformer):
    """Apply ``np.log1p`` to a configured list of non-negative columns.

    Some baseline measurements (CRP, transaminases, MET-minutes,
    monetary income, stroop reaction times) are heavily right-skewed
    and approximately log-normal. Applying ``log1p`` upstream of the
    DML orthogonaliser and the linear classifier restores
    near-linearity, which improves both convergence and SHAP
    interpretation. Columns not present in the input frame are
    silently skipped, so the same instance can be reused after
    upstream filters dropped some columns.

    Parameters
    ----------
    columns : list[str]
        Column names to log-transform. Values are first coerced to
        floats, then run through ``np.log1p``. Negative values are
        clipped to ``0`` before transformation (the canonical value
        for ratio-scale measurements with a small floor of noise);
        a warning is emitted when clipping occurs.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns or []

    @override
    def fit(
        self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None
    ) -> LogTransformer:
        return self

    @override
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            return X
        present = [c for c in self.columns if c in X.columns]
        if not present:
            return X
        X_t = X.copy()
        for col in present:
            s = pd.to_numeric(X_t[col], errors="coerce").astype("float64")
            n_neg = int((s < 0).sum())
            if n_neg > 0:
                logger.warning(
                    "LogTransformer: clipped %d negative value(s) in '%s' to 0 "
                    "before log1p",
                    n_neg,
                    col,
                )
                s = s.clip(lower=0.0)
            X_t[col] = np.log1p(s)
        return X_t

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        if input_features is None:
            raise ValueError("input_features required")
        return np.array(input_features)


class FeatureToConfounder(FrameTransformer):
    """Extract column(s) from the feature matrix and route them as confounders.

    Removes the named column(s) from X and stores them so that
    ``ConfoundedPipeline`` can pass them to the ``Orthogonalizer``
    as additional covariates.  This implements the *residual method*
    for nuisance correction (e.g. regressing out eTIV instead of
    dividing by it).

    Parameters
    ----------
    columns : str or list[str]
        Column name(s) to extract from X and reroute as confounders.
    """

    def __init__(self, columns: str | list[str] = "etiv") -> None:
        self.columns = columns

    @override
    def fit(
        self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None
    ) -> FeatureToConfounder:
        return self

    def _column_list(self) -> list[str]:
        return [self.columns] if isinstance(self.columns, str) else list(self.columns)

    @override
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove confounder column(s) from X and store them.

        **Side effect:** sets ``self.extra_confounders_`` so that
        ``ConfoundedPipeline`` can retrieve them via
        ``get_extra_confounders()`` immediately after this call.
        """
        if not isinstance(X, pd.DataFrame):
            return X
        present = [c for c in self._column_list() if c in X.columns]
        if not present:
            self.extra_confounders_: pd.DataFrame | None = None
            return X
        self.extra_confounders_ = pd.DataFrame(X[present])
        X_out = X.drop(columns=present)
        logger.info("FeatureToConfounder: extracted %s as confounder(s)", present)
        return X_out

    def get_extra_confounders(self) -> pd.DataFrame | None:
        """Return extracted confounder columns, or None if unavailable."""
        return getattr(self, "extra_confounders_", None)


class ModalityPreprocessor(FrameTransformer):
    """Preprocess modality-specific data by handling non-numeric columns.

    Parameters
    ----------
    ordinal_columns : dict, optional
        Mapping of column names to ordered categories.
    nominal_columns : list, optional
        Columns to one-hot encode.
    date_columns : list, optional
        Date columns to extract year/month features.
    drop_columns : list, optional
        Columns to drop.
    """

    def __init__(
        self,
        ordinal_columns: dict[str, list[str]] | None = None,
        nominal_columns: list[str] | None = None,
        date_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
        handle_unknown: Literal["error", "use_encoded_value"] = "use_encoded_value",
        unknown_value: int = -1,
    ) -> None:
        self.ordinal_columns = ordinal_columns
        self.nominal_columns = nominal_columns
        self.date_columns = date_columns
        self.drop_columns = drop_columns
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    @override
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> ModalityPreprocessor:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        self.ordinal_encoders_: dict[str, OrdinalEncoder] = {}
        self.nominal_encoder_: OneHotEncoder | None = None
        self.feature_names_out_: list[str] | None = None

        if self.ordinal_columns:
            for col, categories in self.ordinal_columns.items():
                if col in X.columns:
                    encoder = OrdinalEncoder(
                        categories=[categories],
                        handle_unknown=self.handle_unknown,
                        unknown_value=self.unknown_value,
                    )
                    encoder.fit(X[[col]])
                    self.ordinal_encoders_[col] = encoder

        if self.nominal_columns:
            nom_cols = [c for c in self.nominal_columns if c in X.columns]
            if nom_cols:
                # Ensure uniform string dtype for nominal cols (handles mixed NA/int)
                X_nom = (
                    X[nom_cols]
                    .astype(str)
                    .replace({"<NA>": "_missing", "nan": "_missing"})
                )
                self.nominal_encoder_ = OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="infrequent_if_exist",
                )
                self.nominal_encoder_.fit(X_nom)

        self._compute_feature_names(X)
        return self

    @override
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        X_t = X.copy()
        original_index = X_t.index

        if self.drop_columns:
            for col in self.drop_columns:
                if col in X_t.columns:
                    X_t = X_t.drop(columns=[col])

        for col, encoder in self.ordinal_encoders_.items():
            if col in X_t.columns:
                encoded = encoder.transform(X_t[[col]])
                X_t[col] = encoded.ravel()

        if self.date_columns:
            for col in self.date_columns:
                if col in X_t.columns:
                    X_t = self._extract_date_features(X_t, col)

        if self.nominal_encoder_ is not None and self.nominal_columns:
            nom_cols = [c for c in self.nominal_columns if c in X_t.columns]
            if nom_cols:
                X_nom = (
                    X_t[nom_cols]
                    .astype(str)
                    .replace({"<NA>": "_missing", "nan": "_missing"})
                )
                # The encoder is built with ``sparse_output=False``, so this is
                # already dense; asarray states that for the DataFrame call,
                # whose sparse overload would otherwise apply.
                encoded = np.asarray(self.nominal_encoder_.transform(X_nom))
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=self.nominal_encoder_.get_feature_names_out(nom_cols),
                    index=X_t.index,
                )
                X_t = X_t.drop(columns=nom_cols)
                X_t = pd.concat([X_t, encoded_df], axis=1)

        # Convert remaining object columns to numeric or drop. Selected one
        # dtype at a time and concatenated: `append` preserves column order,
        # where a set union would not.
        object_cols = X_t.select_dtypes(include="object").columns.append(
            X_t.select_dtypes(include="string").columns
        )
        for col in object_cols:
            if X_t[col].isna().all():
                X_t = X_t.drop(columns=[col])
            else:
                try:
                    X_t[col] = pd.to_numeric(X_t[col], errors="coerce")
                except Exception as e:
                    logger.debug("Failed to convert column %s to numeric: %s", col, e)
                    X_t = X_t.drop(columns=[col])

        # Convert pandas nullable types to standard float64
        for col in X_t.columns:
            if hasattr(X_t[col].dtype, "numpy_dtype"):
                X_t[col] = X_t[col].astype("float64")

        # Convert categorical columns to numeric codes
        categorical_cols = X_t.select_dtypes(include="category").columns
        for col in categorical_cols:
            X_t[col] = X_t[col].cat.codes.astype("float64")
            X_t[col] = X_t[col].replace(-1, np.nan)

        X_t.index = original_index
        return X_t

    def _extract_date_features(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        try:
            dates = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}_year"] = dates.dt.year.astype("float64")
            X[f"{col}_month"] = dates.dt.month.astype("float64")
            X = X.drop(columns=[col])
        except Exception as e:
            logger.debug("Failed to extract date features from column %s: %s", col, e)
            X = X.drop(columns=[col])
        return X

    def _compute_feature_names(self, X: pd.DataFrame) -> None:
        feature_names: list[str] = []
        drop_cols = self.drop_columns or []
        date_cols = self.date_columns or []
        nominal_cols = self.nominal_columns or []
        for col in X.columns:
            if col in drop_cols:
                continue
            if col in date_cols:
                feature_names.extend([f"{col}_year", f"{col}_month"])
            elif col in nominal_cols:
                if self.nominal_encoder_ is not None:
                    enc_names = self.nominal_encoder_.get_feature_names_out()
                    feature_names.extend(
                        [n for n in enc_names if n.startswith(f"{col}_")]
                    )
                else:
                    unique_vals = X[col].dropna().unique()
                    feature_names.extend([f"{col}_{val}" for val in unique_vals[1:]])
            else:
                feature_names.append(col)
        self.feature_names_out_ = feature_names

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        if self.feature_names_out_ is None:
            raise ValueError("Preprocessor has not been fitted yet")
        return np.array(self.feature_names_out_)


def create_ses_preprocessor() -> ModalityPreprocessor:
    return ModalityPreprocessor(
        nominal_columns=["isco_major", "isco_submajor"],
        date_columns=["employment_date"],
        drop_columns=[
            # Reference categories
            "retired",
            "widowed",
            "has_partner",
            # Components of income_adequacy
            "income_weighted",
            "needs_weighted",
            # Nominal originals (dummies exist)
            "marital_status",
            "employment_status",
            # ISCO/KLDB hierarchy - keep isco_submajor, kldb_skill_level, siops
            "occupation_status",
            "isco_code",
            "isco_minor",
            "isco_skill_level",
            "kldb_code",
            "kldb_major",
            "kldb_leadership",
            "kldb_segment",
            "kldb_sector",
        ],
    )


def create_physical_activity_preprocessor() -> ModalityPreprocessor:
    return ModalityPreprocessor(
        drop_columns=[
            # Raw minutes where MET exists (near-perfect collinearity)
            "sports_spring",
            "sports_summer",
            "sports_autumn",
            "sports_winter",
            "sports_combined",
            "pa_total",
            "pa_summer",
            "pa_winter",
            # Participation flags (missingness indicators, not PA)
            "household_participation",
            "active_transport_participation",
            "sports_participation",
            "sitting_participation",
            "stairs_participation",
            "leisure_participation",
            # Seasonal sports MET (aggregate met_sports_combined suffices)
            "met_sports_spring",
            "met_sports_summer",
            "met_sports_autumn",
            "met_sports_winter",
        ],
    )


def create_medical_history_preprocessor() -> ModalityPreprocessor:
    """Medical-history preprocessor.

    Drops two families of columns:

    - **Redundant aggregates**: ``bmi_self_reported`` (kept:
      ``body_fat_percentage``), ``bmi_category`` (ordinal recode of
      the same), ``number_medications`` and ``polypharmacy`` (zero-
      variance / heavy-skew aggregates of the medication binaries),
      ``has_surgery_history`` (perfectly collinear with
      ``number_surgeries`` per the 2026-04-27 audit profile),
      ``has_cancer_history`` (highly collinear with
      ``number_cancers``).
    - **Near-constant medication binaries**: medications endorsed by
      <1 % of the analytic sample (``medication_antiparkinson``
      0.4 %, ``has_psoriasis_arthritis`` 0.4 %). With <40 positives in
      N≈8 461, DML residualisation is pure noise and the column
      cannot stratify across CV folds reliably; the EN penalty would
      shrink them to zero anyway.

    One derived column is deliberately kept against that first rule.
    ``current_smoker`` is ``smoking_status == 3``, so it is redundant in the
    sense the surgery and cancer flags were — but the redundancy is the
    point, and it is not the same kind. This modality ends in an
    elastic-net ``LogisticRegressionCV``, so ``smoking_status`` enters as an
    unencoded 1/2/3 numeric and carries a never < former < current spacing
    that nothing guarantees. The binary supplies that contrast without the
    assumption, and the two are linearly independent, unlike the pairs above.
    The price is that stability selection scores them separately and splits
    the tobacco signal between them; the tobacco *effect* is therefore read
    from ``scripts/supplementary/smoking_adjustment.py``, which expands the
    status into never/former/current indicators for exactly this reason, and
    not from the per-feature selection frequencies here. See DECISIONS.md
    §1.12 for the alternative a rerun could adopt.
    """
    return ModalityPreprocessor(
        drop_columns=[
            "bmi_self_reported",
            "bmi_category",
            "number_medications",
            "polypharmacy",
            "has_surgery_history",
            "has_cancer_history",
            # Sparsity drops (<1 % positive in analytic sample).
            "medication_antiparkinson",
            "has_psoriasis_arthritis",
        ],
    )


def create_cognitive_preprocessor() -> ModalityPreprocessor:
    return ModalityPreprocessor(
        drop_columns=[
            "word_list_learning",
            "word_list_forgetting",
            "stroop_interference_effect",
            "cognitive_composite_score",
        ],
    )


def create_lung_function_preprocessor() -> ModalityPreprocessor:
    """Lung function preprocessor.

    Drops absolute volumes (``fev1``, ``fvc``) in favour of the
    height/age/sex-standardised percent-predicted versions.
    Also drops the GLI-2012 reference values (``fev1_predicted_l``,
    ``fvc_predicted_l``) — those are deterministic functions of
    age × sex × height (correlation ~0.99 with each other) and after
    DML orthogonalisation against age/sex/centre they reduce to a
    pure body-stature proxy that does not belong to lung physiology.
    The ratio (``fev1_fvc_ratio``) is dropped as well; the absolute
    volumes it would be reconstructed from are not part of the
    feature set, and percent-predicted FEV1 / FVC carry the relevant
    obstruction signal.
    """
    return ModalityPreprocessor(
        drop_columns=[
            "fev1",
            "fvc",
            "fev1_fvc_ratio",
            "fev1_predicted_l",
            "fvc_predicted_l",
        ],
    )


def create_lab_values_preprocessor() -> ModalityPreprocessor:
    return ModalityPreprocessor(
        drop_columns=["cholesterol", "creatinine"],
    )


def create_demographics_preprocessor() -> ModalityPreprocessor:
    return ModalityPreprocessor(
        nominal_columns=["basis_sex", "basis_uort"],
    )


def create_mental_health_preprocessor() -> ModalityPreprocessor:
    return ModalityPreprocessor(
        drop_columns=[
            # Binary cutoffs redundant with continuous sum scores
            "phq9_moderate_depression",
            "gad7_moderate_anxiety",
            "phq_stress_moderate",
            # Ordinal categories redundant with the matching sum scores
            # (gad7_diagnosis correlates 0.91 with gad7_sum, mirroring the
            # phq9 case).
            "phq9_severity_category",
            "gad7_diagnosis",
            # Imputed version of MINI diagnosis (keep original)
            "mini_major_depression_imputed",
            # Constant (value 8 for all participants) — zero variance
            "depression_incidence",
            # Reconstructed from the nine baseline PHQ-9 items and identical
            # to phq9_sum wherever both exist (r = 1.0000): a literal
            # duplicate, not merely a collinear one.
            "phq9_sum_items",
            # Completeness flag for those items, true for ~97 % of
            # participants — a missingness marker, not a predictor.
            "phq9_items_complete",
        ],
    )


def create_preprocessor_for_modality(
    modality_name: str,
) -> ModalityPreprocessor | None:
    preprocessor_map = {
        "ses": create_ses_preprocessor,
        "physical_activity": create_physical_activity_preprocessor,
        "medical_history": create_medical_history_preprocessor,
        "lab_values": create_lab_values_preprocessor,
        "demographics": create_demographics_preprocessor,
        "cognitive": create_cognitive_preprocessor,
        "lung_function": create_lung_function_preprocessor,
        "mental_health": create_mental_health_preprocessor,
    }
    if modality_name in preprocessor_map:
        return preprocessor_map[modality_name]()
    return None
