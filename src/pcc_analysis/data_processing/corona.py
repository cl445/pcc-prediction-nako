"""Corona-2 PCC questionnaire and KVAD ambulatory diagnoses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pcc_analysis.data_processing._common import _load_nako_csv, _save_parquet
from pcc_analysis.data_processing.pcc_outcome import _determine_pcc_status

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


def nako_corona1_csv(paths: NAKOPaths) -> Path:
    """Resolve the Corona-1 questionnaire path.

    Thin wrapper around :pyattr:`NAKOPaths.corona1_csv`; kept as a
    free function so existing call sites (mh_longitudinal,
    pcs_corona1_acute) don't have to be touched.
    """
    return paths.corona1_csv


def process_corona2(paths: NAKOPaths, output_dir: Path) -> Path:
    """Process Corona-2 survey / PCC operationalisation -> ``corona2_pcc.parquet``."""
    logger.info("Processing Corona-2 survey (PCC) ...")
    csv_path = paths.corona2_csv
    df = _load_nako_csv(csv_path)
    result = _determine_pcc_status(df)
    return _save_parquet(result, "corona2_pcc", output_dir)


def process_kvad(paths: NAKOPaths, output_dir: Path) -> Path:
    """Process KVAD ambulatory diagnosis data -> ``kvad.parquet``."""
    logger.info("Processing KVAD (health insurance diagnoses) ...")
    csv_path = paths.kvad_csv
    df = _load_nako_csv(csv_path)

    # Remove completely empty rows
    feat_cols = [c for c in df.columns if c != "ID"]
    initial = len(df)
    df = df[~df[feat_cols].isna().all(axis=1)]
    dropped = initial - len(df)
    if dropped:
        logger.info("Dropped %d empty rows", dropped)

    return _save_parquet(df, "kvad", output_dir)
