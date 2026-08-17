"""Baseline mental health (PHQ-9, GAD-7, MINI) and Corona-2 psychometrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis.data_processing._common import (
    NAKO_SENTINEL_CODES,
    _load_nako_csv,
    _save_parquet,
)
from pcc_analysis.data_processing._likert import _score_likert_questionnaire

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


# Corona-2 PHQ-9 items (9 items, coded 1-4 in NAKO → recode to 0-3)
_PHQ9_CO2_ITEMS = [f"d_co2_phq9_d{i}" for i in range(1, 10)]

# Corona-2 GAD-7 items (7 items, coded 1-4 in NAKO → recode to 0-3)
_GAD7_CO2_ITEMS = [f"d_co2_gad7_d{i}" for i in range(10, 17)]


_MENTAL_HEALTH_COLUMNS: dict[str, str] = {
    "a_emo_phq9_sum": "phq9_sum",
    "a_emo_phq9_cut10": "phq9_moderate_depression",
    "a_emo_phq9_kat": "phq9_severity_category",
    "a_emo_phq9_dsm": "phq9_dsm_criteria",
    "a_emo_gad7_sum": "gad7_sum",
    "a_emo_gad7_cut10": "gad7_moderate_anxiety",
    "a_emo_gad7_dia": "gad7_diagnosis",
    "a_emo_mini_dxmdd": "mini_major_depression",
    "a_emo_mini_dxmdd_imp": "mini_major_depression_imputed",
    "a_emo_miniscr_dp": "mini_screen_depression",
    "a_emo_inz_dp": "depression_incidence",
    "a_emo_age_dp": "age_depression_onset",
    "a_emo_age_angst": "age_anxiety_onset",
    "a_emo_dauer_dp": "depression_duration",
    "a_emo_dauer_dp20": "depression_duration_past20y",
    "a_emo_dauer_dperiode": "depression_episode_duration",
    "a_emo_phq_panik": "phq_panic",
    "a_emo_phq_stress": "phq_stress",
    "a_emo_phq_str_cut10": "phq_stress_moderate",
}


def _extract_mental_health(df: DataFrame) -> DataFrame:
    """Extract baseline mental health variables from supplementary delivery."""
    r = pd.DataFrame()
    r["ID"] = df["ID"]
    for src_col, dst_col in _MENTAL_HEALTH_COLUMNS.items():
        if src_col in df.columns:
            r[dst_col] = df[src_col]
        else:
            logger.warning("Mental health column not found: %s", src_col)
    for col in r.columns:
        if col != "ID":
            r[col] = r[col].replace(NAKO_SENTINEL_CODES, pd.NA)
    return r


def process_mental_health(paths: NAKOPaths, output_dir: Path) -> Path:
    """Process baseline mental health -> ``mental_health.parquet``."""
    logger.info("Processing baseline mental health ...")
    csv_path = paths.mental_health_csv
    df = _load_nako_csv(csv_path)
    result = _extract_mental_health(df)
    return _save_parquet(result, "mental_health", output_dir)


def process_psychometric(paths: NAKOPaths, output_dir: Path) -> Path:
    """Process psychometric questionnaires → ``psychometric_scores.parquet``.

    Extracts PHQ-9 (depression, range 0-27) and GAD-7 (anxiety, range 0-21)
    from the Corona-2 follow-up data.  These are used for descriptive
    characterisation only (not as predictors).
    """
    logger.info("Processing psychometric questionnaires ...")
    csv_path = paths.corona2_csv
    df = _load_nako_csv(csv_path)

    result = pd.DataFrame()
    result["ID"] = df["ID"]
    # PHQ-9: complete-case scoring (no missing items), matching how NAKO
    # derives ``a_emo_phq9_sum`` itself. The NAKO mental-health module computes
    # the PHQ-9 sum "for all participants without missing values on the
    # respective items following the manual" (Streit et al. 2023, World J Biol
    # Psychiatry, doi:10.1080/15622975.2021.2014152). No prorating is applied.
    result["phq9_total"] = _score_likert_questionnaire(
        df,
        _PHQ9_CO2_ITEMS,
        "PHQ-9",
        max_missing=0,
    )
    # GAD-7: same complete-case rule as PHQ-9. NAKO computes the GAD-7 sum
    # "for all participants without missing values on the respective items"
    # (Streit et al. 2023, doi:10.1080/15622975.2021.2014152). No prorating.
    result["gad7_total"] = _score_likert_questionnaire(
        df,
        _GAD7_CO2_ITEMS,
        "GAD-7",
        max_missing=0,
    )

    valid_phq9 = int(result["phq9_total"].notna().sum())
    valid_gad7 = int(result["gad7_total"].notna().sum())
    logger.info(
        "PHQ-9: %s valid scores (%.1f%%)",
        f"{valid_phq9:,}",
        valid_phq9 / len(result) * 100,
    )
    logger.info(
        "GAD-7: %s valid scores (%.1f%%)",
        f"{valid_gad7:,}",
        valid_gad7 / len(result) * 100,
    )

    return _save_parquet(result, "psychometric_scores", output_dir)
