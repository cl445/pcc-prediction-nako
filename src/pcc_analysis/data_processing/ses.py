"""Socioeconomic status: ISCED education, employment, income, ISCO/KldB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis.data_processing._common import (
    _load_nako_csv,
    _replace_nako_missing,
    _save_parquet,
)

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


def _extract_ses(df: DataFrame) -> DataFrame:
    """Extract and rename SES variables from the baseline export."""
    r = pd.DataFrame()
    r["ID"] = df["ID"]

    # Family structure
    r["marital_status"] = df["a_ses_famst"]
    r["has_partner"] = (df["a_ses_partner"] == 1).astype("boolean")
    r["household_size"] = df["a_ses_househ"]
    r["number_children"] = df["a_ses_child"]

    # Employment
    r["employment_status"] = df["a_ses_ewstat"]
    r["work_hours_category"] = df["a_ses_workh"]
    r["is_self_employed"] = (df["a_selfemp"] == 1).astype("boolean")
    r["number_employees"] = df["a_nempl"]

    # Education (ISCED 1997)
    r["education_isced_level"] = df["a_ses_isced97_level"]
    r["education_years"] = df["a_ses_isced97_years"]
    r["german_education_level"] = df["a_ses_deutsch"]

    # Income
    r["income_category"] = df["a_ses_inc"]
    r["income_position"] = df["a_ses_incpos"]
    r["income_weighted"] = df["a_ses_incgw"]
    r["needs_weighted"] = df["a_ses_bedarfgw"]

    # Occupational classification and prestige
    _isco_major_labels: dict[int, str] = {
        0: "armed_forces",
        1: "managers",
        2: "professionals",
        3: "technicians",
        4: "clerical_support",
        5: "service_sales",
        6: "agriculture_forestry",
        7: "craft_trades",
        8: "plant_machine_operators",
        9: "elementary_occupations",
    }
    r["isco_major"] = df["a_isco_major"].map(_isco_major_labels).astype("string")
    r["isei_score"] = df["a_isei"]

    # Occupation status
    r["occupation_status"] = df["a_ses_beruf"]

    # SIOPS prestige score
    r["siops_score"] = df["a_siops"]

    # ISCO hierarchy
    r["isco_code"] = df["a_isco_code"]
    r["isco_submajor"] = df["a_isco_submajor"]
    r["isco_minor"] = df["a_isco_minor"]
    r["isco_skill_level"] = df["a_isco_skill"]

    # KldB 2010 hierarchy
    r["kldb_code"] = df["a_kldb_code"]
    r["kldb_major"] = df["a_kldb_major"]
    r["kldb_skill_level"] = df["a_kldb_anf"]
    r["kldb_leadership"] = df["a_kldb_fuehr"]
    r["kldb_segment"] = df["a_kldb_seg"]
    r["kldb_sector"] = df["a_kldb_sek"]

    # Additional
    r["retirement_age"] = df["a_ses_rentenalter"]
    r["employment_duration_years"] = df["a_ses_el_seit_j"]
    r["employment_duration_total"] = df["a_ses_el_gesamt"]
    # NAKO encodes "no employment / not applicable" as the Excel-style
    # placeholder 1900-01-05 (carrying ~98 % of all rows in the analytic
    # sample as of 2026-04). Map it to NaT so the downstream date-feature
    # extractor sees an honest missing value rather than a 1900 timestamp
    # that gets turned into year/month features ≈ constants.
    employment_date = df["a_ses_el_datum"]
    sentinel_mask = (
        employment_date.astype("string").str.startswith("1900-01-05").fillna(False)
    )
    employment_date = employment_date.mask(sentinel_mask)
    r["employment_date"] = employment_date

    return _replace_nako_missing(r)


def _derive_ses_metrics(df: DataFrame) -> DataFrame:
    """Compute derived SES indicators."""
    df["employed"] = (df["employment_status"] == 1).astype("boolean")
    df["unemployed"] = (df["employment_status"] == 2).astype("boolean")
    df["retired"] = (df["employment_status"] == 3).astype("boolean")

    df["married"] = (df["marital_status"] == 2).astype("boolean")
    df["single"] = (df["marital_status"] == 1).astype("boolean")
    df["divorced_separated"] = df["marital_status"].isin([3, 4]).astype("boolean")
    df["widowed"] = (df["marital_status"] == 5).astype("boolean")

    df["living_alone"] = (df["household_size"] == 1).astype("boolean")
    df["has_children"] = (df["number_children"] > 0).astype("boolean")

    df["income_adequacy"] = df["income_weighted"] / df["needs_weighted"]

    return df


def process_ses(
    paths: NAKOPaths, output_dir: Path, *, baseline_df: DataFrame | None = None
) -> Path:
    """Process socioeconomic status -> ``socioeconomic_status.parquet``."""
    logger.info("Processing socioeconomic status ...")
    if baseline_df is not None:
        df = baseline_df
    else:
        csv_path = paths.baseline_dir / "export_baseline.csv"
        df = _load_nako_csv(csv_path)
    result = _extract_ses(df)
    result = _derive_ses_metrics(result)
    return _save_parquet(result, "socioeconomic_status", output_dir)
