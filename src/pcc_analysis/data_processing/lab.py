"""Laboratory blood biomarkers (baseline ``sa_*`` columns)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis._plausibility import LAB_RANGES
from pcc_analysis.data_processing._common import (
    _enforce_plausibility,
    _load_nako_csv,
    _replace_nako_missing,
    _save_parquet,
)

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


def _extract_lab_values(df: DataFrame) -> DataFrame:
    """Extract laboratory values from baseline data.

    NAKO column prefix: ``sa_`` (Serum-Analyse).
    """
    r = pd.DataFrame()
    r["ID"] = df["ID"]

    # Inflammatory markers
    if "sa_hscrp" in df.columns:
        r["crp"] = df["sa_hscrp"]

    # Metabolic markers
    if "sa_chol" in df.columns:
        r["cholesterol"] = df["sa_chol"]
    if "sa_gluk" in df.columns:
        r["glucose"] = df["sa_gluk"]
    if "sa_hba1c" in df.columns:
        r["hba1c"] = df["sa_hba1c"]
    if "sa_ldlc" in df.columns:
        r["ldl"] = df["sa_ldlc"]
    if "sa_hdlc" in df.columns:
        r["hdl"] = df["sa_hdlc"]
    if "sa_trig" in df.columns:
        r["triglycerides"] = df["sa_trig"]

    # Kidney function
    if "sa_crea" in df.columns:
        r["creatinine"] = df["sa_crea"]
    if "sa_egfr" in df.columns:
        r["egfr"] = df["sa_egfr"]
    if "sa_urea" in df.columns:
        r["urea"] = df["sa_urea"]

    # Liver function
    if "sa_alat" in df.columns:
        r["alt"] = df["sa_alat"]
    if "sa_asat" in df.columns:
        r["ast"] = df["sa_asat"]
    if "sa_ggt" in df.columns:
        r["ggt"] = df["sa_ggt"]
    if "sa_bilit" in df.columns:
        r["bilirubin"] = df["sa_bilit"]

    # Thyroid function
    if "sa_tsh" in df.columns:
        r["tsh"] = df["sa_tsh"]
    if "sa_ft4" in df.columns:
        r["ft4"] = df["sa_ft4"]

    # Complete blood count
    if "sa_hb" in df.columns:
        r["hemoglobin"] = df["sa_hb"]
    if "sa_wbc" in df.columns:
        r["leukocytes"] = df["sa_wbc"]
    if "sa_plt" in df.columns:
        r["platelets"] = df["sa_plt"]

    # Electrolytes
    if "sa_na" in df.columns:
        r["sodium"] = df["sa_na"]
    if "sa_pot" in df.columns:
        r["potassium"] = df["sa_pot"]
    if "sa_ca" in df.columns:
        r["calcium"] = df["sa_ca"]

    return _replace_nako_missing(r)


def process_lab_values(
    paths: NAKOPaths, output_dir: Path, *, baseline_df: DataFrame | None = None
) -> Path:
    """Process laboratory values -> ``lab_values.parquet``."""
    logger.info("Processing lab values ...")
    if baseline_df is not None:
        df = baseline_df
    else:
        csv_path = paths.baseline_dir / "export_baseline.csv"
        df = _load_nako_csv(csv_path)
    result = _extract_lab_values(df)
    result = _enforce_plausibility(result, LAB_RANGES, module="lab_values")
    return _save_parquet(result, "lab_values", output_dir)
