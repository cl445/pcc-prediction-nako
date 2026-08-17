"""Longitudinal mental-health panel (PHQ-9 / GAD-7 across T0/T1/T2).

Long-format parquet with one row per participant x wave:

  wave = "T0"  -> Baseline (2014..2019); only pre-computed sums are
                  available (NAKO ``a_emo_phq9_sum`` / ``a_emo_gad7_sum``),
                  per-item PHQ-9/GAD-7 columns are NA because the raw
                  items are not in our data-access profile.
  wave = "T1"  -> Corona-1 mail-in questionnaire (2020); raw items are
                  ``d_co1_phq91_1..9`` (PHQ-9) and ``d_co1_gad7_1..7``
                  (GAD-7), encoded 1..4; rescaled to canonical 0..3.
  wave = "T2"  -> Corona-2 follow-up (2022); raw items are
                  ``d_co2_phq9_d1..d9`` and ``d_co2_gad7_d10..d16``,
                  same 1..4 encoding rescaled to 0..3.

Sum scores at T1 and T2 are computed from the items using complete-case
scoring (no missing items allowed) for both PHQ-9 and GAD-7, matching how
NAKO derives ``a_emo_phq9_sum`` / ``a_emo_gad7_sum`` itself: the sums are
computed "for all participants without missing values on the respective
items following the manual" (Streit et al. 2023, World J Biol Psychiatry,
doi:10.1080/15622975.2021.2014152). This keeps the reconstructed T1/T2
scores consistent with the pass-through NAKO sums at T0. Sum scores at T0
are passed through from the baseline mental-health module unchanged.

Columns produced:
  ID                       uint32
  wave                     Categorical (T0 < T1 < T2)
  phq9_sum                 Int8  (0..27, NA unless all 9 items answered)
  gad7_sum                 Int8  (0..21, NA unless all 7 items answered)
  phq9_positive            boolean  (phq9_sum >= 10)
  gad7_positive            boolean  (gad7_sum >= 10)
  mh_positive              boolean  (phq9_positive | gad7_positive)
  phq9_i1..phq9_i9         Int8  (0..3 per item; NA at T0 and on
                                  missing/sentinel responses)
  gad7_i1..gad7_i7         Int8  (0..3 per item; same)

The RI-CLPM model consumes ``phq9_sum`` / ``gad7_sum`` across waves;
the item-level bifactor-SEM consumes the per-item columns and
jointly fits on the T1/T2 subset where raw items are available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis._plausibility import MH_LONGITUDINAL_RANGES
from pcc_analysis.data_processing._common import (
    _enforce_plausibility,
    _load_nako_csv,
    _save_parquet,
)
from pcc_analysis.data_processing._likert import _PSYCHOMETRIC_MISSING
from pcc_analysis.data_processing.corona import nako_corona1_csv

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


MH_LONG_PHQ9_T1_ITEMS: list[str] = [f"d_co1_phq91_{i}" for i in range(1, 10)]
MH_LONG_GAD7_T1_ITEMS: list[str] = [f"d_co1_gad7_{i}" for i in range(1, 8)]
MH_LONG_PHQ9_T2_ITEMS: list[str] = [f"d_co2_phq9_d{i}" for i in range(1, 10)]
MH_LONG_GAD7_T2_ITEMS: list[str] = [f"d_co2_gad7_d{i}" for i in range(10, 17)]

# NAKO encodes Likert responses 1..4 on the raw export. Canonical PHQ/GAD
# scoring uses 0..3; we subtract 1 after sentinel replacement.
MH_LONG_LIKERT_OFFSET: int = 1

# Max missing items tolerated when computing per-wave sum scores.
# Both = 0 (complete-case): NAKO derives ``a_emo_phq9_sum`` /
# ``a_emo_gad7_sum`` "for all participants without missing values on the
# respective items following the manual" (Streit et al. 2023, World J Biol
# Psychiatry, doi:10.1080/15622975.2021.2014152). No prorating. Complete-case
# keeps the reconstructed T1/T2 scores consistent with the pass-through NAKO
# sums at T0.
MH_LONG_MAX_MISSING_PHQ9: int = 0
MH_LONG_MAX_MISSING_GAD7: int = 0


def _mh_score_items(
    df: DataFrame,
    item_cols: list[str],
    *,
    max_missing: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return (sum, items_0_to_3) from raw NAKO 1..4-encoded items.

    Sentinels are coerced to NA; remaining values are rescaled to 0..3.
    Rows with more than ``max_missing`` missing items get a NA sum.
    """
    items = df[item_cols].copy()
    items = items.replace(_PSYCHOMETRIC_MISSING, pd.NA)
    items = items.astype("Int16") - MH_LONG_LIKERT_OFFSET
    too_many_missing = items.isna().sum(axis=1) > max_missing
    sums = items.sum(axis=1, min_count=1).mask(too_many_missing).astype("Int16")
    # Cast items to Int8 for storage efficiency (values 0..3).
    items_int8 = items.astype("Int8")
    return sums, items_int8


def _mh_frame_t0(baseline_mh_df: DataFrame) -> DataFrame:
    """Long-format row set for T0 from the processed baseline MH parquet.

    Baseline items are not in our data-access profile; item columns are
    filled with NA to keep the long-format schema uniform across waves.
    """
    cols_needed = ["ID", "phq9_sum", "gad7_sum"]
    available = [c for c in cols_needed if c in baseline_mh_df.columns]
    t0 = baseline_mh_df[available].copy()
    for i in range(1, 10):
        t0[f"phq9_i{i}"] = pd.Series(pd.NA, index=t0.index, dtype="Int8")
    for i in range(1, 8):
        t0[f"gad7_i{i}"] = pd.Series(pd.NA, index=t0.index, dtype="Int8")
    t0["wave"] = "T0"
    return t0


def _mh_frame_from_items(
    raw_df: DataFrame,
    *,
    wave: str,
    phq9_items: list[str],
    gad7_items: list[str],
) -> DataFrame:
    """Long-format row set for T1 or T2 computed from raw items."""
    if "ID" not in raw_df.columns:
        raise ValueError("raw_df is missing required 'ID' column")
    n = len(raw_df)
    frames: list[DataFrame] = [
        pd.DataFrame({"ID": raw_df["ID"].reset_index(drop=True)})
    ]

    phq_present = [c for c in phq9_items if c in raw_df.columns]
    gad_present = [c for c in gad7_items if c in raw_df.columns]

    if phq_present:
        phq_sum, phq_items_0_3 = _mh_score_items(
            raw_df, phq_present, max_missing=MH_LONG_MAX_MISSING_PHQ9
        )
        phq_items_0_3 = phq_items_0_3.reset_index(drop=True)
        phq_items_0_3.columns = [f"phq9_i{i}" for i in range(1, len(phq_present) + 1)]
        frames.append(pd.DataFrame({"phq9_sum": phq_sum.reset_index(drop=True)}))
        frames.append(phq_items_0_3)
    else:
        frames.append(
            pd.DataFrame(
                {
                    "phq9_sum": pd.array([pd.NA] * n, dtype="Int16"),
                    **{
                        f"phq9_i{i}": pd.array([pd.NA] * n, dtype="Int8")
                        for i in range(1, 10)
                    },
                }
            )
        )

    if gad_present:
        gad_sum, gad_items_0_3 = _mh_score_items(
            raw_df, gad_present, max_missing=MH_LONG_MAX_MISSING_GAD7
        )
        gad_items_0_3 = gad_items_0_3.reset_index(drop=True)
        gad_items_0_3.columns = [f"gad7_i{i}" for i in range(1, len(gad_present) + 1)]
        frames.append(pd.DataFrame({"gad7_sum": gad_sum.reset_index(drop=True)}))
        frames.append(gad_items_0_3)
    else:
        frames.append(
            pd.DataFrame(
                {
                    "gad7_sum": pd.array([pd.NA] * n, dtype="Int16"),
                    **{
                        f"gad7_i{i}": pd.array([pd.NA] * n, dtype="Int8")
                        for i in range(1, 8)
                    },
                }
            )
        )

    out = pd.concat(frames, axis=1)
    out["wave"] = wave
    return out


def _mh_long_finalise(long_df: DataFrame) -> DataFrame:
    """Derive positive flags and coerce dtypes into the final schema."""
    phq_sum = long_df["phq9_sum"]
    gad_sum = long_df["gad7_sum"]
    phq_pos = (phq_sum >= 10).astype("boolean").mask(phq_sum.isna())
    gad_pos = (gad_sum >= 10).astype("boolean").mask(gad_sum.isna())
    long_df["phq9_positive"] = phq_pos
    long_df["gad7_positive"] = gad_pos
    mh_pos = (phq_pos.fillna(False) | gad_pos.fillna(False)).mask(
        phq_sum.isna() & gad_sum.isna()
    )
    long_df["mh_positive"] = mh_pos.astype("boolean")

    # Coerce sums to Int8; Int16 had been used during computation to avoid
    # overflow concerns on partial sums.
    long_df["phq9_sum"] = long_df["phq9_sum"].astype("Int8")
    long_df["gad7_sum"] = long_df["gad7_sum"].astype("Int8")

    long_df["wave"] = pd.Categorical(
        long_df["wave"], categories=["T0", "T1", "T2"], ordered=True
    )

    column_order = [
        "ID",
        "wave",
        "phq9_sum",
        "gad7_sum",
        "phq9_positive",
        "gad7_positive",
        "mh_positive",
        *[f"phq9_i{i}" for i in range(1, 10)],
        *[f"gad7_i{i}" for i in range(1, 8)],
    ]
    return long_df[[c for c in column_order if c in long_df.columns]]


def process_mh_longitudinal(
    paths: NAKOPaths,
    output_dir: Path,
    *,
    baseline_mh_df: DataFrame | None = None,
    corona1_df: DataFrame | None = None,
    corona2_df: DataFrame | None = None,
) -> Path:
    """Assemble the long-format longitudinal MH panel (T0/T1/T2).

    See the module docstring for the wave conventions and the items used
    per wave.
    """
    logger.info("Processing longitudinal MH panel (T0/T1/T2) ...")

    # T0: reuse the already-processed baseline mental-health parquet when
    # possible so the sum-score semantics stay in lockstep with the
    # cross-sectional mental_health module.
    if baseline_mh_df is None:
        mh_parquet = output_dir / "mental_health.parquet"
        if mh_parquet.exists():
            baseline_mh_df = pd.read_parquet(mh_parquet)
        else:
            raise FileNotFoundError(
                "process_mh_longitudinal requires mental_health.parquet; run "
                "process_mental_health first or pass baseline_mh_df explicitly."
            )

    # T1: Corona-1 mail-in questionnaire.
    if corona1_df is None:
        corona1_csv = nako_corona1_csv(paths)
        corona1_df = _load_nako_csv(corona1_csv)

    # T2: Corona-2 follow-up.
    if corona2_df is None:
        corona2_df = _load_nako_csv(paths.corona2_csv)

    t0 = _mh_frame_t0(baseline_mh_df)
    t1 = _mh_frame_from_items(
        corona1_df,
        wave="T1",
        phq9_items=MH_LONG_PHQ9_T1_ITEMS,
        gad7_items=MH_LONG_GAD7_T1_ITEMS,
    )
    t2 = _mh_frame_from_items(
        corona2_df,
        wave="T2",
        phq9_items=MH_LONG_PHQ9_T2_ITEMS,
        gad7_items=MH_LONG_GAD7_T2_ITEMS,
    )
    long_df = pd.concat([t0, t1, t2], ignore_index=True)
    long_df = _mh_long_finalise(long_df)
    long_df = _enforce_plausibility(
        long_df, MH_LONGITUDINAL_RANGES, module="mh_longitudinal"
    )
    return _save_parquet(long_df, "mh_longitudinal", output_dir)
