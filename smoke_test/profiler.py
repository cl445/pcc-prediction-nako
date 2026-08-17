"""Extract aggregate statistics from real parquet files for smoke-test generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd

from smoke_test._profile_types import (
    BooleanColumnProfile,
    CategoricalColumnProfile,
    ColumnProfile,
    DerivationGroupProfile,
    IntegerColumnProfile,
    ModalityProfile,
    NumericColumnProfile,
    Profile,
    ProfileMeta,
    TextColumnProfile,
    UnknownColumnProfile,
)

# Modalities the analysis pipeline actually uses (skip kvad and
# baseline_sex_age — they are either too large or auxiliary).
MODALITIES = [
    "demographics",
    "corona2_pcc",
    "socioeconomic_status",
    "cognitive_tests",
    "physical_activity",
    "medical_history",
    "lab_values",
    "cardiovascular",
    "lung_function",
    "mental_health",
    # Joined onto an existing modality rather than loaded as one of their
    # own (see load_pipeline_data), but they still need a synthetic
    # counterpart or the smoke run silently exercises the
    # amendment-absent path instead of the wiring it is meant to check.
    "phq9_items",
    "smoking",
    # Not a model modality at all: read only by the supplementary analyses
    # (reporting_style_robustness needs the current Corona-2 PHQ/GAD scores).
    # Without it that step cannot run under --smoke, so the check would
    # silently cover one script less than it appears to.
    "psychometric_scores",
    # Same reason, and in this case never a model modality by design: the
    # acute course is post-exposure, so severity_interaction is its only
    # consumer.
    "acute_infection",
    "mri_cortical_desikan_killiany",
    "mri_cortical_destrieux",
    "mri_cortical_julich",
    "mri_cortical_yeo_networks",
    "mri_subcortical",
    "mri_cerebellar",
    "mri_etiv",
    "followup1_olfactometry",
    "followup1_visit_meta",
    "followup1_lab_values",
    "followup1_sleep_objective",
    "followup1_feno",
    "followup1_markers_misc",
    "mh_longitudinal",
]

# Mutually exclusive boolean-dummy groups that must be sampled together.
# Keys = modality, values = dict of group_name → list of column names.
DERIVATION_GROUPS: dict[str, dict[str, list[str]]] = {
    "socioeconomic_status": {
        "marital_status": ["married", "single", "divorced_separated", "widowed"],
        "employment_status": ["employed", "unemployed", "retired"],
    },
}

# Columns derived deterministically from other columns — profiled but
# regenerated from source columns during generation.
DERIVED_COLUMNS: dict[str, list[str]] = {
    "socioeconomic_status": [
        "living_alone",
        "has_children",
        "income_adequacy",
    ],
    "medical_history": [
        "bmi_category",
        "overweight",
        "obese",
        "has_hypertension",
        "has_cancer_history",
        "number_cancers",
        "has_infection_history",
        "number_surgeries",
        "has_surgery_history",
        "number_medications",
        "polypharmacy",
        "has_neurological_disease",
        "cv_risk_factor_count",
        "high_cv_risk",
    ],
    "corona2_pcc": [
        "n_symptoms",
        "any_pcc",
        "severe_pcc",
        "pcc_severity_score",
        "pcc_severity_weighted",
        "n_domains_affected",
        "bahmer_pcs_score",
        "bahmer_severity",
        "bahmer_any_pcs",
        "bahmer_severe_pcs",
    ],
    "mental_health": [
        # Threshold flags and DSM / severity summaries are derived from the
        # sum scores; re-derive them on generation so the flags stay
        # internally consistent with phq9_sum / gad7_sum / phq_stress.
        "phq9_moderate_depression",
        "phq9_severity_category",
        "phq9_dsm_criteria",
        "gad7_moderate_anxiety",
        "gad7_diagnosis",
        "phq_stress_moderate",
    ],
    "acute_infection": [
        # All four are functions of acute_care_level: the flags are its
        # thresholds (hospital ward at 2, intensive care at 3), and each day
        # count is a structural zero below the level that would make it a
        # measurement. Sampled independently they contradict the level
        # within the row, and severity_interaction — the only consumer of
        # this modality — then sees a severity axis on which no recode can
        # be wrong.
        "acute_hospitalised",
        "acute_icu",
        "hospital_days",
        "icu_days",
    ],
}


# How much precision the stored statistics carry. The generator needs a
# plausible shape, not a faithful one, and three significant figures give it
# that. Coarser than any descriptive table a paper prints, which is the point:
# this file ships with the repository, so it should not be the most precise
# description of the cohort in existence.
_FRACTION_DIGITS = 3
_SIGNIFICANT_DIGITS = 3


def _round_fraction(value: float) -> float:
    """Round a proportion, which is always in [0, 1]."""
    return round(float(value), _FRACTION_DIGITS)


def _round_stat(value: float) -> float:
    """Round to significant figures, so precision does not depend on scale.

    Fixed decimals would erase a lab value measured in thousandths while
    leaving a brain volume in cubic millimetres essentially untouched.
    """
    number = float(value)
    if number == 0.0 or not math.isfinite(number):
        return number
    exponent = math.floor(math.log10(abs(number)))
    return round(number, -(exponent - (_SIGNIFICANT_DIGITS - 1)))


def _profile_column(series: pd.Series) -> ColumnProfile:
    """Extract aggregate statistics for a single column."""
    missing_frac = _round_fraction(series.isna().mean())
    dtype_str = str(series.dtype)

    # Category columns
    if dtype_str == "category":
        counts = series.dropna().value_counts(normalize=True)
        return CategoricalColumnProfile(
            dtype="categorical",
            categories={str(k): _round_fraction(v) for k, v in counts.items()},
            missing_frac=missing_frac,
        )

    # String columns
    if dtype_str in ("string", "object", "string[python]"):
        return TextColumnProfile(
            dtype="datetime" if _looks_like_date(series) else "string",
            missing_frac=missing_frac,
        )

    # Boolean columns (native boolean dtype or 0/1 integer)
    s_clean = series.dropna()
    if dtype_str in ("boolean", "bool"):
        true_frac = float(s_clean.mean()) if len(s_clean) > 0 else 0.0
        return BooleanColumnProfile(
            dtype="boolean",
            true_frac=_round_fraction(true_frac),
            missing_frac=missing_frac,
        )
    unique_vals: set[int | float] = set()
    if pd.api.types.is_integer_dtype(series) and len(s_clean) > 0:
        unique_vals = {int(v) for v in s_clean.unique()}
    if unique_vals <= {0, 1} and pd.api.types.is_integer_dtype(series):
        true_frac = float(s_clean.mean()) if len(s_clean) > 0 else 0.0
        return BooleanColumnProfile(
            dtype="boolean",
            true_frac=_round_fraction(true_frac),
            missing_frac=missing_frac,
        )

    # Numeric columns (int / float variants)
    if pd.api.types.is_numeric_dtype(series):
        desc = s_clean.describe()
        # The generator needs a plausible range to clip its draws to, not the
        # true extremes. Storing raw min/max would put one real participant's
        # measured value per column into a file that ships with the repository
        # — for ~830 numeric columns that is ~1660 individual data points.
        # The 1st/99th percentiles bound the synthetic data just as well and
        # describe no single participant.
        mean = _round_stat(desc["mean"])
        std = _round_stat(desc["std"]) if desc["std"] == desc["std"] else 0.0
        q01 = _round_stat(s_clean.quantile(0.01))
        q25 = _round_stat(desc["25%"])
        q50 = _round_stat(desc["50%"])
        q75 = _round_stat(desc["75%"])
        q99 = _round_stat(s_clean.quantile(0.99))
        # Preserve integer domain: the generator can then draw discrete
        # values and keep the target dtype so the quality harness still
        # passes on synthetic smoke data.
        if pd.api.types.is_integer_dtype(series):
            return IntegerColumnProfile(
                dtype="integer",
                pandas_dtype=dtype_str,
                mean=mean,
                std=std,
                q01=q01,
                q25=q25,
                q50=q50,
                q75=q75,
                q99=q99,
                missing_frac=missing_frac,
            )
        return NumericColumnProfile(
            dtype="numeric",
            mean=mean,
            std=std,
            q01=q01,
            q25=q25,
            q50=q50,
            q75=q75,
            q99=q99,
            missing_frac=missing_frac,
        )

    # Fallback
    return UnknownColumnProfile(dtype="unknown", missing_frac=missing_frac)


def _looks_like_date(series: pd.Series) -> bool:
    """Heuristic: does a string column look like dates?"""
    sample = series.dropna().head(20)
    if len(sample) == 0:
        return False
    try:
        pd.to_datetime(sample)
    except (ValueError, TypeError):
        return False
    return True


def _compute_systematic_missing_frac(df: pd.DataFrame) -> float:
    """Fraction of rows where ALL feature columns are NaN."""
    feat_cols = [c for c in df.columns if c != "ID"]
    if not feat_cols:
        return 0.0
    return _round_fraction(df[feat_cols].isna().all(axis=1).mean())


def _profile_derivation_groups(
    df: pd.DataFrame, groups: dict[str, list[str]]
) -> dict[str, DerivationGroupProfile]:
    """Compute proportions for each derivation group."""
    result: dict[str, DerivationGroupProfile] = {}
    for group_name, columns in groups.items():
        present = [c for c in columns if c in df.columns]
        if not present:
            continue
        # Each row should have exactly one 1 — compute proportions
        proportions: dict[str, float] = {}
        total = 0.0
        for c in present:
            frac = float(df[c].dropna().mean()) if c in df.columns else 0.0
            proportions[c] = _round_fraction(frac)
            total += frac
        # Normalize if total != 1 (due to missingness etc.)
        if total > 0:
            proportions = {
                k: _round_fraction(v / total) for k, v in proportions.items()
            }
        result[group_name] = DerivationGroupProfile(
            columns=present,
            proportions=proportions,
        )
    return result


def extract_profile(input_dir: Path) -> Profile:
    """Read all modality parquets and extract aggregate statistics.

    Returns a JSON-serialisable dict with metadata and per-modality profiles.
    """
    modalities: dict[str, ModalityProfile] = {}

    for name in MODALITIES:
        path = input_dir / f"{name}.parquet"
        if not path.exists():
            print(f"  SKIP {name} (not found)")
            continue

        df = pd.read_parquet(path)
        derived = set(DERIVED_COLUMNS.get(name, []))
        groups = DERIVATION_GROUPS.get(name, {})
        group_cols = {c for cols in groups.values() for c in cols}

        columns: dict[str, ColumnProfile] = {}
        for col in df.columns:
            if col == "ID":
                continue
            # Mark derived and group membership
            profile = _profile_column(df[col])
            if col in derived:
                profile["derived"] = True
            if col in group_cols:
                profile["in_derivation_group"] = True
            columns[col] = profile

        mod_profile = ModalityProfile(
            systematic_missing_frac=_compute_systematic_missing_frac(df),
            columns=columns,
        )

        if groups:
            mod_profile["derivation_groups"] = _profile_derivation_groups(df, groups)

        modalities[name] = mod_profile
        print(f"  {name}: {len(df)} rows x {len(df.columns)} cols")

    return Profile(
        meta=ProfileMeta(n_modalities=len(modalities)),
        modalities=modalities,
    )


def save_profile(profile: Profile, output: Path) -> None:
    """Write profile dict to JSON file."""
    output.parent.mkdir(parents=True, exist_ok=True)

    class _NumpyEncoder(json.JSONEncoder):
        @override
        def default(self, o: object) -> object:
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    output.write_text(
        json.dumps(profile, indent=2, cls=_NumpyEncoder) + "\n",
        encoding="utf-8",
    )
    print(f"Profile saved to {output}")
