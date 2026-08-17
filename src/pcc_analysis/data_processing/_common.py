"""Shared low-level helpers used across all NAKO data-processing modules.

Holds the canonical NAKO missing-value sentinels, generic CSV/Parquet I/O,
the dtype optimiser, and the plausibility-range enforcer. Domain-specific
helpers (e.g. MRI cleaning, sex mapping) live in their respective modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.arrays.integer import IntegerDtype

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common NAKO NA values recognised during CSV loading (string-based)
# ---------------------------------------------------------------------------
NAKO_NA_VALUES = [
    "",
    " ",
    "NA",
    "N/A",
    "na",
    "n/a",
    "missing",
    "Missing",
    "MISSING",
    "unknown",
    "Unknown",
    "UNKNOWN",
    "null",
    "NULL",
    "None",
    "-999",
    "-99",
    "-9",
    "999",
    "9999",
]

# ---------------------------------------------------------------------------
# Numeric sentinel codes indicating missing / invalid values
# ---------------------------------------------------------------------------
# General sentinels that can appear in ANY numeric column (survey or MRI).
# Includes the NAKO laboratory module's 999999 / 666666 "not measured" markers
# and the 444444 / 555555 "not assessed / invalid aliquot" markers, which
# occur in the baseline lab columns (ggt, tsh, ft4, hemoglobin, leukocytes,
# platelets, sodium, potassium, calcium, bilirubin).
_GENERAL_SENTINEL_CODES: list[int] = [
    -999,
    -99,
    -9,
    999,
    9999,
    99999,
    111111,
    444444,
    555555,
    666666,
    999999,
]

# Survey-instrument sentinels (questionnaire / assessment data only).
# These never appear in MRI volumetric columns.
_SURVEY_SENTINEL_CODES: list[int] = [
    -88,  # refused to answer
    -5,  # not applicable (NOTE: valid for cardiovascular augmentation_index)
    7775,  # question not shown
    7776,  # question skipped
    7777,  # not assessed
    8886,  # not evaluable
    8888,  # refused
    8889,  # don't know
]

# Combined canonical list: all codes recognised as missing across NAKO exports.
# Used by _replace_nako_missing() for mixed survey/clinical data.
NAKO_SENTINEL_CODES: list[int] = sorted(
    set(_GENERAL_SENTINEL_CODES + _SURVEY_SENTINEL_CODES)
)

# Backward-compatible alias
NAKO_MISSING_CODES = NAKO_SENTINEL_CODES


def _load_nako_csv(csv_path: Path, sep: str = ";") -> DataFrame:
    """Load a NAKO CSV with nullable dtypes and standard NA handling."""
    logger.info("Loading CSV from %s", csv_path)
    df = pd.read_csv(
        csv_path,
        sep=sep,
        na_values=NAKO_NA_VALUES,
        keep_default_na=True,
        dtype_backend="numpy_nullable",
        low_memory=False,
    )
    logger.info("Loaded %s rows x %s columns", f"{df.shape[0]:,}", df.shape[1])
    return df


def _apply_missingness_and_drop_m_columns(df: DataFrame) -> DataFrame:
    """Set feature values to NA where the companion ``_m`` column is not NA,
    then drop all ``_m`` columns."""
    m_cols = [c for c in df.columns if c.endswith("_m")]
    logger.info("Processing %d missingness-marker columns", len(m_cols))
    for m_col in m_cols:
        base_col = m_col[:-2]
        if base_col in df.columns:
            mask = df[m_col].notna()
            if mask.any():
                df.loc[mask, base_col] = pd.NA
    df = df.drop(columns=m_cols)
    return df


def _fix_hidden_na_values(df: DataFrame) -> DataFrame:
    """Replace general sentinel NA codes with ``pd.NA`` in numeric columns."""
    na_codes = _GENERAL_SENTINEL_CODES
    total_fixed = 0
    for col in df.columns:
        if col == "ID":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            mask = df[col].isin(na_codes)
            count = mask.sum()
            if count > 0:
                df.loc[mask, col] = pd.NA
                total_fixed += count
            if pd.api.types.is_float_dtype(df[col]):
                inf_mask = np.isinf(df[col])
                if inf_mask.any():
                    df.loc[inf_mask, col] = pd.NA
                    total_fixed += inf_mask.sum()
    if total_fixed:
        logger.info("Fixed %d hidden NA values", total_fixed)
    return df


def _optimize_dtypes(df: DataFrame) -> DataFrame:
    """Down-cast nullable integers and doubles to save memory.

    Integers: narrowest nullable integer dtype that fits the observed range.
    Floats: Float64 values whose magnitude fits the Float32 envelope
    (|x| <= 3.4e38) are down-cast to Float32. NAKO measurements do not
    require double precision, so the cast is lossless in practice.
    """
    for col in df.columns:
        if col == "ID":
            continue
        series = df[col]
        if series.isna().all():
            continue
        if isinstance(series.dtype, IntegerDtype) and series.notna().any():
            mi, ma = series.min(), series.max()
            if mi >= 0:
                if ma <= 255:
                    df[col] = series.astype("UInt8")
                elif ma <= 65535:
                    df[col] = series.astype("UInt16")
                elif ma <= 4294967295:
                    df[col] = series.astype("UInt32")
            elif mi >= -128 and ma <= 127:
                df[col] = series.astype("Int8")
            elif mi >= -32768 and ma <= 32767:
                df[col] = series.astype("Int16")
            elif mi >= -2147483648 and ma <= 2147483647:
                df[col] = series.astype("Int32")
        elif pd.api.types.is_float_dtype(series) and series.notna().any():
            finite = series[series.notna()]
            if finite.abs().max() <= np.finfo(np.float32).max:
                df[col] = series.astype("Float32")
    return df


def _enforce_plausibility(
    df: DataFrame,
    ranges: dict[str, tuple[float, float]],
    *,
    module: str,
) -> DataFrame:
    """Coerce out-of-range values to ``pd.NA`` and log the count per column.

    Parameters
    ----------
    df : DataFrame
        Frame to filter in-place (returned for chaining).
    ranges : mapping column → (min, max)
        Closed plausibility intervals. Columns absent from *ranges* are
        skipped.
    module : str
        Short identifier used in log messages so multi-module runs remain
        readable.
    """
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        mask = series.notna() & ((series < lo) | (series > hi))
        count = int(mask.sum())
        if count:
            df.loc[mask, col] = pd.NA
            logger.warning(
                "%s: %s had %d value(s) outside plausibility range [%s, %s]; set to NA",
                module,
                col,
                count,
                lo,
                hi,
            )
    return df


def _save_parquet(
    df: DataFrame,
    name: str,
    output_dir: Path,
    *,
    missing_threshold: float = 0.99,
    optimize_dtypes: bool = True,
) -> Path:
    """Save *df* as ``<output_dir>/<name>.parquet``, dropping near-empty columns.

    Dtype optimisation (narrowest nullable integer, Float32 for floats that
    fit) is applied by default — set ``optimize_dtypes=False`` to opt out for
    parquets where double precision is required.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{name}.parquet"

    pct = df.isna().mean()
    empty = pct[pct > missing_threshold].index.tolist()
    if empty:
        logger.info(
            "Dropping %d columns with >%.0f%% missing before saving",
            len(empty),
            missing_threshold * 100,
        )
        df = df.drop(columns=empty)

    if optimize_dtypes:
        df = _optimize_dtypes(df)

    df.to_parquet(out, index=False)
    logger.info(
        "Saved %s (%s rows x %s cols)", out.name, f"{len(df):,}", len(df.columns)
    )
    return out


def _replace_nako_missing(df: DataFrame, id_col: str = "ID") -> DataFrame:
    """Replace common NAKO missing-value sentinel codes with ``pd.NA``."""
    for col in df.columns:
        if col != id_col:
            df[col] = df[col].replace(NAKO_MISSING_CODES, pd.NA)
    return df
