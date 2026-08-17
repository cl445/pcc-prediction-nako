"""MRI processors: Desikan-Killiany, Yeo, Subcortical, Cerebellar, Julich, Destrieux."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis.data_processing._common import (
    _apply_missingness_and_drop_m_columns,
    _fix_hidden_na_values,
    _load_nako_csv,
    _optimize_dtypes,
    _replace_nako_missing,
    _save_parquet,
)

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


def _clean_mri_dataframe(df: DataFrame) -> DataFrame:
    """Drop rows/columns that are entirely NA (except the ID column)."""
    initial = df.shape
    feat_cols = [c for c in df.columns if c != "ID"]
    if feat_cols:
        df = df[~df[feat_cols].isna().all(axis=1)]
    cols_to_check = [c for c in df.columns if c != "ID"]
    if cols_to_check:
        all_na = df[cols_to_check].isna().all()
        drop = all_na[all_na].index.tolist()
        if drop:
            df = df.drop(columns=drop)
    logger.info(
        "Cleaned MRI DataFrame: %s -> %s (-%d rows, -%d cols)",
        initial,
        df.shape,
        initial[0] - df.shape[0],
        initial[1] - df.shape[1],
    )
    return df


def _standard_mri_pipeline(df: DataFrame) -> DataFrame:
    """Apply the standard MRI cleaning pipeline (missingness, NA, clean, optimise).

    Survey-instrument sentinel codes (7775/7776/8888/...) occasionally appear
    in MRI volumetric columns where the segmentation inspector flagged the
    measurement as unusable; :func:`_replace_nako_missing` catches those in
    addition to the numeric sentinels handled by
    :func:`_fix_hidden_na_values`.
    """
    df = _apply_missingness_and_drop_m_columns(df)
    df = _fix_hidden_na_values(df)
    df = _replace_nako_missing(df)
    df = _clean_mri_dataframe(df)
    df = _optimize_dtypes(df)
    return df


def process_mri1(paths: NAKOPaths, output_dir: Path) -> list[Path]:
    """Process MRI-1 data (Desikan-Killiany + Yeo 7-Network).

    Produces two parquet files:
    * ``mri_cortical_desikan_killiany.parquet``
    * ``mri_cortical_yeo_networks.parquet``
    """
    logger.info("Processing MRI-1 (Desikan-Killiany + Yeo 7-Network) ...")
    csv_path = paths.mri1_csv
    df = _load_nako_csv(csv_path)
    df = _standard_mri_pipeline(df)

    # Desikan-Killiany: cortical volumes (_vo suffix, NOT in network columns)
    network_prefixes = ("con_", "dan_", "dmn_", "limn_", "smn_", "svan_", "vn_")
    desikan_cols = ["ID"] + [
        c
        for c in df.columns
        if c.startswith("ma_n_")
        and "_vo" in c
        and not any(x in c for x in network_prefixes)
    ]
    desikan_df = df[[c for c in desikan_cols if c in df.columns]]

    # Yeo 7-Network: functional networks
    yeo_tags = (
        "_lcon_",
        "_rcon_",
        "_ldan_",
        "_rdan_",
        "_ldmn_",
        "_rdmn_",
        "_llimn_",
        "_rlimn_",
        "_lsmn_",
        "_rsmn_",
        "_lsvan_",
        "_rsvan_",
        "_lvn_",
        "_rvn_",
    )
    yeo_cols = ["ID"] + [c for c in df.columns if any(t in c for t in yeo_tags)]
    yeo_df = df[[c for c in yeo_cols if c in df.columns]]

    logger.info("Desikan-Killiany: %d features", len(desikan_df.columns) - 1)
    logger.info("Yeo 7-Network: %d features", len(yeo_df.columns) - 1)

    dk_path = _save_parquet(desikan_df, "mri_cortical_desikan_killiany", output_dir)
    yeo_path = _save_parquet(yeo_df, "mri_cortical_yeo_networks", output_dir)
    return [dk_path, yeo_path]


def process_mri2(paths: NAKOPaths, output_dir: Path) -> list[Path]:
    """Process MRI-2 data (Subcortical + Cerebellum + Julich).

    Produces three parquet files:
    * ``mri_subcortical.parquet``
    * ``mri_cerebellar.parquet``
    * ``mri_cortical_julich.parquet``
    """
    logger.info("Processing MRI-2 (Subcortical + Cerebellum + Julich) ...")
    csv_path = paths.mri2_csv
    df = _load_nako_csv(csv_path)
    df = _standard_mri_pipeline(df)

    # Subcortical: white matter tracts and other subcortical structures
    subcortical_cols = ["ID"] + [
        c
        for c in df.columns
        if any(x in c for x in ["_cm", "_if", "_lb", "_mf", "_sf", "_vtm"])
        or (
            c.startswith("ma_n_")
            and not any(x in c for x in ["_ch", "ncl", "_hoc", "_op", "_fp", "_fg"])
        )
    ]
    subcortical_df = df[[c for c in subcortical_cols if c in df.columns]]

    # Cerebellar
    cerebellar_cols = ["ID"] + [
        c for c in df.columns if any(x in c for x in ["_ch", "ncl"])
    ]
    cerebellar_df = df[[c for c in cerebellar_cols if c in df.columns]]

    # Julich cytoarchitectonic
    julich_cols = ["ID"] + [
        c for c in df.columns if any(x in c for x in ["_hoc", "_op", "_fp", "_fg"])
    ]
    julich_df = df[[c for c in julich_cols if c in df.columns]]

    logger.info("Subcortical: %d features", len(subcortical_df.columns) - 1)
    logger.info("Cerebellar: %d features", len(cerebellar_df.columns) - 1)
    logger.info("Julich: %d features", len(julich_df.columns) - 1)

    paths_out = [
        _save_parquet(subcortical_df, "mri_subcortical", output_dir),
        _save_parquet(cerebellar_df, "mri_cerebellar", output_dir),
        _save_parquet(julich_df, "mri_cortical_julich", output_dir),
    ]
    return paths_out


def process_mri3(paths: NAKOPaths, output_dir: Path) -> list[Path]:
    """Process MRI-3 data (Destrieux Atlas + demographics covariates).

    Produces two parquet files:
    * ``mri_cortical_destrieux.parquet``
    * ``demographics.parquet``
    """
    logger.info("Processing MRI-3 (Destrieux + covariates) ...")
    csv_path = paths.mri3_csv
    df = _load_nako_csv(csv_path)
    df = _standard_mri_pipeline(df)

    # Destrieux: gyri and sulci
    destrieux_cols = ["ID"] + [
        c
        for c in df.columns
        if c.startswith(("ma_n_l", "ma_n_r"))
        and not c.startswith("basis_")
        and not c.startswith("d_co")
    ]
    destrieux_df = df[[c for c in destrieux_cols if c in df.columns]]

    # Drop non-MRI participants (all Destrieux features NA).
    # _clean_mri_dataframe cannot do this because MRI3 also contains
    # demographics columns that are present for all 117k participants.
    destrieux_feat = [c for c in destrieux_df.columns if c != "ID"]
    if destrieux_feat:
        destrieux_df = destrieux_df[~destrieux_df[destrieux_feat].isna().all(axis=1)]

    # Demographics: basis_ variables
    demo_cols = ["ID"] + [c for c in df.columns if c.startswith("basis_")]
    demo_df = df[[c for c in demo_cols if c in df.columns]].copy()

    # Validate basis_sex: NAKO codes 1=male, 2=female; flag unknown values
    if "basis_sex" in demo_df.columns:
        valid_sex = demo_df["basis_sex"].isin([1, 2, 1.0, 2.0])
        n_invalid = (~valid_sex & demo_df["basis_sex"].notna()).sum()
        if n_invalid > 0:
            logger.warning("basis_sex: %d values not in {1, 2} (set to NA)", n_invalid)
            demo_df.loc[~valid_sex, "basis_sex"] = pd.NA

    # eTIV: estimated total intracranial volume (for ICV adjustment)
    etiv_col = "ma_n_etiv"
    if etiv_col in df.columns:
        etiv_df = df[["ID", etiv_col]].copy()
        etiv_df = etiv_df.rename(columns={etiv_col: "etiv"})
        # Keep only participants with eTIV data
        etiv_df = etiv_df.dropna(subset=["etiv"])
        logger.info("eTIV: %d participants with data", len(etiv_df))
    else:
        etiv_df = None
        logger.warning("eTIV column '%s' not found in MRI-3 data", etiv_col)

    logger.info("Destrieux: %d features", len(destrieux_df.columns) - 1)
    logger.info("Demographics: %d features", len(demo_df.columns) - 1)

    paths_out = [
        _save_parquet(destrieux_df, "mri_cortical_destrieux", output_dir),
        _save_parquet(demo_df, "demographics", output_dir),
    ]
    if etiv_df is not None:
        paths_out.append(_save_parquet(etiv_df, "mri_etiv", output_dir))
    return paths_out
