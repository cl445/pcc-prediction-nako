"""Cardiovascular: blood pressure, heart rate, vascular stiffness."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis._plausibility import CARDIOVASCULAR_RANGES
from pcc_analysis.data_processing._common import (
    NAKO_MISSING_CODES,
    _enforce_plausibility,
    _load_nako_csv,
    _save_parquet,
)

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


def _extract_cardiovascular(df: DataFrame) -> DataFrame:
    """Extract cardiovascular variables from baseline export."""
    r = pd.DataFrame()
    r["ID"] = df["ID"]

    # Blood pressure
    if "a_rr_sys" in df.columns:
        r["systolic_bp_mean"] = df["a_rr_sys"]
    if "a_rr_dia" in df.columns:
        r["diastolic_bp_mean"] = df["a_rr_dia"]

    # Heart rate
    if "a_rr_puls_aut" in df.columns:
        r["heart_rate_rest"] = df["a_rr_puls_aut"]

    # Vascular stiffness (vascular assessment exam)
    if "vae_ex_ao_pwv_cf" in df.columns:
        r["pulse_wave_velocity"] = df["vae_ex_ao_pwv_cf"]
    if "vae_ex_aix_ao" in df.columns:
        r["augmentation_index"] = df["vae_ex_aix_ao"]
    if "vae_ex_l_abi" in df.columns:
        r["ankle_brachial_index_left"] = df["vae_ex_l_abi"]
    if "vae_ex_r_abi" in df.columns:
        r["ankle_brachial_index_right"] = df["vae_ex_r_abi"]

    # Replace NAKO sentinel codes. Exclude -5 for augmentation_index because
    # it falls within the valid physiological range (approx. -21 to +48).
    for col in r.columns:
        if col == "ID":
            continue
        codes = (
            [c for c in NAKO_MISSING_CODES if c != -5]
            if col == "augmentation_index"
            else NAKO_MISSING_CODES
        )
        r[col] = r[col].replace(codes, pd.NA)
    return r


def process_cardiovascular(
    paths: NAKOPaths, output_dir: Path, *, baseline_df: DataFrame | None = None
) -> Path:
    """Process cardiovascular measurements -> ``cardiovascular.parquet``."""
    logger.info("Processing cardiovascular measurements ...")
    if baseline_df is not None:
        df = baseline_df
    else:
        csv_path = paths.baseline_dir / "export_baseline.csv"
        df = _load_nako_csv(csv_path)
    result = _extract_cardiovascular(df)
    result = _enforce_plausibility(
        result, CARDIOVASCULAR_RANGES, module="cardiovascular"
    )
    return _save_parquet(result, "cardiovascular", output_dir)
