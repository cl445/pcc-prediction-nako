"""Medical history: anthropometry, comorbidities, medications, surgeries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis._plausibility import MEDICAL_HISTORY_ANTHROPOMETRIC_RANGES
from pcc_analysis.data_processing._common import (
    _enforce_plausibility,
    _load_nako_csv,
    _replace_nako_missing,
    _save_parquet,
)

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


def _extract_medical_history(df: DataFrame) -> DataFrame:
    """Extract medical history variables from baseline data."""
    r = pd.DataFrame()
    r["ID"] = df["ID"]

    # Anthropometry (self-reported)
    r["bmi_self_reported"] = df["a_anthro_bmi_eig"]
    r["height_cm_self_reported"] = df["a_anthro_groe_eig"]
    r["weight_kg_self_reported"] = df["a_anthro_gew_eig"]

    # Anthropometry (measured)
    r["body_fat_percentage"] = df.get("anthro_fettmasse")

    # Cardiovascular diseases (NAKO coding: 1=yes, 2=no → boolean)
    r["has_hypertension"] = (df["a_hte_ad_cur"] == 1).astype("boolean")
    r["hypertension_age_onset"] = df["a_hte_ad_onset"]
    r["hypertension_age_category"] = df["a_hte_ad_onset_cat"].replace([8, 9], pd.NA)
    r["has_psoriasis"] = (df["a_hte_pso_cur"] == 1).astype("boolean")
    r["psoriasis_age_onset"] = df["a_hte_pso_onset"]
    r["has_psoriasis_arthritis"] = (df["a_hte_psa_cur"] == 1).astype("boolean")

    # Medications (0/1 → boolean)
    r["medication_antihypertensive"] = (df["a_antihypertens"] == 1).astype("boolean")
    r["medication_antiparkinson"] = (df["a_antiparkinson"] == 1).astype("boolean")
    r["medication_antiepileptic"] = (df["a_antiepilept"] == 1).astype("boolean")
    r["medication_antidiabetic"] = (df["a_antidiabetika"] == 1).astype("boolean")
    r["medication_betablocker"] = (df["a_betablock"] == 1).astype("boolean")
    r["medication_lipid_lowering"] = (df["a_lipidsenkend"] == 1).astype("boolean")
    r["medication_anticoagulant"] = (df["a_gerinnung"] == 1).astype("boolean")

    # Infections (age at diagnosis)
    r["infection_shingles_age"] = df["a_inf_age_zos"]
    r["infection_hiv_age"] = df["a_inf_age_hiv"]
    r["infection_hepatitis_b_age"] = df["a_inf_age_hepb"]
    r["infection_hepatitis_c_age"] = df["a_inf_age_hepc"]
    r["infection_sepsis_age"] = df["a_inf_age_seps"]
    r["infection_tuberculosis_age"] = df["a_inf_age_tub"]

    # Cancer history
    r["cancer_1_age"] = df["a_kre_1_alter"]
    r["cancer_2_age"] = df["a_kre_2_alter"]
    r["cancer_3_age"] = df["a_kre_3_alter"]
    r["cancer_4_age"] = df["a_kre_4_alter"]
    r["cancer_1_type"] = df["a_kre_1_nc"]

    # Surgeries (age at surgery)
    r["surgery_general_anesthesia_1_age"] = df["a_op_alg1a_age"]
    r["surgery_general_anesthesia_2_age"] = df["a_op_alg1b_age"]
    r["surgery_general_anesthesia_3_age"] = df["a_op_alg1c_age"]
    r["surgery_general_anesthesia_4_age"] = df["a_op_alg1d_age"]
    r["surgery_general_anesthesia_5_age"] = df["a_op_alg1e_age"]
    r["surgery_general_anesthesia_6_age"] = df["a_op_alg1f_age"]
    r["surgery_general_anesthesia_7_age"] = df["a_op_alg1g_age"]
    r["surgery_removal_age"] = df["a_op_entf1g_age"]

    # Neurological conditions
    r["stroke_age"] = df.get("a_apo_yearsfirst")
    r["epilepsy_age"] = df.get("a_epi_yearsfirst")
    r["parkinsons_age"] = df.get("a_park_yearsfirst")

    return _replace_nako_missing(r)


def _derive_medical_metrics(df: DataFrame) -> DataFrame:
    """Compute derived medical risk indicators."""
    # BMI categories (WHO)
    df["bmi_category"] = pd.cut(
        df["bmi_self_reported"],
        bins=[0, 18.5, 25, 30, 35, 40, 100],
        labels=["underweight", "normal", "overweight", "obese_1", "obese_2", "obese_3"],
        include_lowest=True,
    )

    # Cancer history
    cancer_cols = ["cancer_1_age", "cancer_2_age", "cancer_3_age", "cancer_4_age"]
    df["has_cancer_history"] = df[cancer_cols].notna().any(axis=1).astype("boolean")
    df["number_cancers"] = df[cancer_cols].notna().sum(axis=1).astype("Int8")

    # Infection history
    infection_cols = [
        "infection_shingles_age",
        "infection_hiv_age",
        "infection_hepatitis_b_age",
        "infection_hepatitis_c_age",
        "infection_sepsis_age",
        "infection_tuberculosis_age",
    ]
    df["has_infection_history"] = (
        df[infection_cols].notna().any(axis=1).astype("boolean")
    )

    # Surgery count
    surgery_cols = [f"surgery_general_anesthesia_{i}_age" for i in range(1, 8)]
    df["number_surgeries"] = df[surgery_cols].notna().sum(axis=1).astype("Int8")
    df["has_surgery_history"] = (df["number_surgeries"] > 0).astype("boolean")

    # Medication count
    med_cols = [
        "medication_antihypertensive",
        "medication_antiparkinson",
        "medication_antiepileptic",
        "medication_antidiabetic",
        "medication_betablocker",
        "medication_lipid_lowering",
        "medication_anticoagulant",
    ]
    df["number_medications"] = sum(
        (df[c] == 1).astype("Int8") for c in med_cols if c in df.columns
    )
    df["polypharmacy"] = (df["number_medications"] >= 5).astype("boolean")

    # Neurological disease flag
    neuro_cols = ["stroke_age", "epilepsy_age", "parkinsons_age"]
    df["has_neurological_disease"] = (
        df[neuro_cols].notna().any(axis=1).astype("boolean")
    )

    return df


def process_medical_history(
    paths: NAKOPaths, output_dir: Path, *, baseline_df: DataFrame | None = None
) -> Path:
    """Process medical history -> ``medical_history.parquet``."""
    logger.info("Processing medical history ...")
    if baseline_df is not None:
        df = baseline_df
    else:
        csv_path = paths.baseline_dir / "export_baseline.csv"
        df = _load_nako_csv(csv_path)
    result = _extract_medical_history(df)
    # Plausibility on anthropometrics runs before derived metrics so that
    # BMI categories use only plausible heights/weights.
    result = _enforce_plausibility(
        result,
        MEDICAL_HISTORY_ANTHROPOMETRIC_RANGES,
        module="medical_history",
    )
    result = _derive_medical_metrics(result)
    return _save_parquet(result, "medical_history", output_dir)
