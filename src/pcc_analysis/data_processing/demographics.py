"""Demographics: baseline sex (categorical) and age (Int8)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame
from pandas.core.arrays.integer import IntegerDtype

from pcc_analysis.data_processing._common import _save_parquet

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


_SEX_MAP: dict[str, str] = {"1": "male", "1.0": "male", "2": "female", "2.0": "female"}


def _map_sex(series: pd.Series) -> pd.Series:
    """Map NAKO ``basis_sex`` (1=male, 2=female) to string labels.

    Unknown or NA values are set to ``pd.NA`` and logged.
    """
    mapped = series.astype(str).map(_SEX_MAP, na_action="ignore")
    n_unknown = mapped.isna().sum() - series.isna().sum()
    if n_unknown > 0:
        logger.warning(
            "basis_sex: %d values could not be mapped (set to NA)", n_unknown
        )
    return mapped


def process_demographics(
    paths: NAKOPaths, output_dir: Path, *, baseline_df: DataFrame | None = None
) -> Path:
    """Process baseline sex and age into ``baseline_sex_age.parquet``."""
    logger.info("Processing demographics (sex / age) ...")

    if baseline_df is not None:
        df = baseline_df[["ID", "basis_age", "basis_sex"]].copy()
        df["basis_sex"] = _map_sex(df["basis_sex"])
    else:
        csv_path = paths.baseline_dir / "export_baseline.csv"
        df = pd.read_csv(
            csv_path,
            sep=";",
            usecols=["ID", "basis_age", "basis_sex"],
            dtype={"ID": "uint32"},
        )
        df["basis_sex"] = _map_sex(df["basis_sex"])
    df = df.rename(
        columns={"basis_age": "age", "basis_sex": "sex"},
    )
    df["sex"] = pd.Categorical(df["sex"], categories=["male", "female"], ordered=False)
    # Keep ID as an explicit column so the parquet conforms to the
    # "every processed parquet carries an ID" invariant checked by the
    # quality harness. Age fits Int8 for the NAKO enrolment envelope.
    if "age" in df.columns and isinstance(df["age"].dtype, IntegerDtype):
        df["age"] = df["age"].astype("Int8")
    return _save_parquet(df, "baseline_sex_age", output_dir)
