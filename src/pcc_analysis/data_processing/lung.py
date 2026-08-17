"""Spirometry / lung-function measurements (baseline ``spiro_mw_*``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis._plausibility import LUNG_FUNCTION_RANGES
from pcc_analysis.data_processing._common import (
    NAKO_MISSING_CODES,
    _enforce_plausibility,
    _load_nako_csv,
    _replace_nako_missing,
    _save_parquet,
)

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


def _extract_lung_function(df: DataFrame) -> DataFrame:
    """Extract spirometry measurements from baseline data.

    NAKO column prefix: ``spiro_mw_`` (Spirometrie Messwert). Units and
    semantics per ``dd_nako882.xlsx`` (Spirometrie - Auswertung):

    * ``spiro_mw_fev1`` / ``spiro_mw_fvc`` — measured absolute volumes
      in litres.
    * ``spiro_mw_fev1_p`` / ``spiro_mw_fvc_p`` — **Sollwert (predicted
      value) in litres**, not percent-predicted. We store these verbatim
      as ``*_predicted_l`` and derive the true percent-predicted as
      ``100 * measured / predicted``.
    * ``spiro_mw_pef`` / ``spiro_mw_fef25_75`` — expiratory flows in
      **L/s** (NAKO native).
    """
    r = pd.DataFrame()
    r["ID"] = df["ID"]

    fev1 = df.get("spiro_mw_fev1")
    fvc = df.get("spiro_mw_fvc")
    fev1_ref = df.get("spiro_mw_fev1_p")
    fvc_ref = df.get("spiro_mw_fvc_p")

    if fev1 is not None:
        r["fev1"] = fev1
    if fvc is not None:
        r["fvc"] = fvc
    if fev1_ref is not None:
        r["fev1_predicted_l"] = fev1_ref
    if fvc_ref is not None:
        r["fvc_predicted_l"] = fvc_ref

    if fev1 is not None and fev1_ref is not None:
        # Replace NAKO missing sentinels before the ratio to avoid sentinel
        # arithmetic. Ratio stays NA where either side is NA.
        lhs = fev1.replace(NAKO_MISSING_CODES, pd.NA).astype("Float64")
        rhs = fev1_ref.replace(NAKO_MISSING_CODES, pd.NA).astype("Float64")
        r["fev1_percent_predicted"] = (lhs / rhs) * 100
    if fvc is not None and fvc_ref is not None:
        lhs = fvc.replace(NAKO_MISSING_CODES, pd.NA).astype("Float64")
        rhs = fvc_ref.replace(NAKO_MISSING_CODES, pd.NA).astype("Float64")
        r["fvc_percent_predicted"] = (lhs / rhs) * 100

    if "spiro_mw_fev1_fvc" in df.columns:
        r["fev1_fvc_ratio"] = df["spiro_mw_fev1_fvc"]
    if "spiro_mw_pef" in df.columns:
        r["peak_flow"] = df["spiro_mw_pef"]
    if "spiro_mw_fef25_75" in df.columns:
        r["fef25_75"] = df["spiro_mw_fef25_75"]

    return _replace_nako_missing(r)


def process_lung_function(
    paths: NAKOPaths, output_dir: Path, *, baseline_df: DataFrame | None = None
) -> Path:
    """Process lung function -> ``lung_function.parquet``."""
    logger.info("Processing lung function ...")
    if baseline_df is not None:
        df = baseline_df
    else:
        csv_path = paths.baseline_dir / "export_baseline.csv"
        df = _load_nako_csv(csv_path)
    result = _extract_lung_function(df)
    result = _enforce_plausibility(result, LUNG_FUNCTION_RANGES, module="lung_function")
    return _save_parquet(result, "lung_function", output_dir)
