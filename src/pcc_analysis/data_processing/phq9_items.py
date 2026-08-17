"""Baseline PHQ-9 item-level responses (NAKO amendment NAM-45).

The baseline export in the primary delivery carries only NAKO's derived
``a_emo_phq9_sum``; the nine underlying items are requested separately and
arrive in the amendment delivery (``nako_amendment_dir`` in the config).

Item coding differs from every other PHQ-9 source in this pipeline. NAKO
stores the baseline items under the ``BeeintrDepr`` value list, which is
neither the canonical ``0..3`` nor the ``1..4`` encoding used for the
Corona-1/Corona-2 waves (see :mod:`mh_longitudinal`):

    2  -> 0   not at all
    11 -> 1   several days
    12 -> 2   more than half the days
    13 -> 3   nearly every day
    88        no answer      (missing)
    99        don't know     (missing)

``88`` and ``99`` are deliberately *not* added to
:data:`~pcc_analysis.data_processing._common.NAKO_SENTINEL_CODES`: both are
plausible measurements in other modules (ages, percentages, MET minutes), so
the mapping is applied locally to these nine columns only.

The reconstructed sum uses complete-case scoring, matching how NAKO derives
``a_emo_phq9_sum`` itself (Streit et al. 2023, World J Biol Psychiatry,
doi:10.1080/15622975.2021.2014152). Verified against the delivered sum on the
2026-08-06 export: 113,751 jointly scorable participants, zero discrepancies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from pandas import DataFrame

from pcc_analysis.data_processing._common import (
    _enforce_plausibility,
    _load_nako_csv,
    _save_parquet,
)

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


PHQ9_BASELINE_ITEMS: list[str] = [f"d_phq91_{i}" for i in range(1, 10)]

BEEINTR_DEPR_SCORE_MAP: dict[int, int] = {2: 0, 11: 1, 12: 2, 13: 3}
"""NAKO ``BeeintrDepr`` response codes mapped to canonical PHQ-9 scores."""

MIN_SCORABLE_ITEM_SHARE: float = 0.5
"""Floor on the share of responses **per item** that must match the value list.

The nine columns can be present and still decode to nothing. A quoted export,
or a single non-numeric token anywhere in a column, turns that column into
``object`` dtype; ``isin(valid_codes)`` is then False throughout it, the item
becomes NA for everyone, ``phq9_items_complete`` is False everywhere, the
99-%-missing rule in ``_save_parquet`` drops the whole block, and ``_attach``
later joins nothing into ``mental_health`` — leaving a normal-looking
full-stack result with no item data in it. Without this floor the only trace
is the INFO line below reporting 0.0 %.

Applied per item rather than to the nine pooled, because one broken column is
enough to produce exactly that outcome: it voids every complete-case sum while
the pooled share stays at 8/9. Item-level missingness on the 2026-08-06 export
runs about 3 % per item, so a floor at half the responses separates a decode
failure from any plausible non-response pattern by a wide margin.

This module already fails loudly when the columns are absent; the same
treatment belongs on columns that are present and decode to nothing.
"""

# Items keep their canonical Kroenke-2001 position, so ``phq9_item_1`` is
# anhedonia, _2 depressed mood, _3 sleep, _4 fatigue, _5 appetite,
# _6 worthlessness, _7 concentration, _8 psychomotor change and
# _9 suicidal ideation. Somatic/affective subscale splits are defined by item
# number in the literature, so numeric names stay closer to any published
# grouping than invented labels would.


def _extract_phq9_items(df: DataFrame) -> DataFrame:
    """Recode the nine baseline PHQ-9 items and rebuild the sum score."""
    r = pd.DataFrame()
    r["ID"] = df["ID"]

    valid_codes = list(BEEINTR_DEPR_SCORE_MAP)
    missing_cols = [c for c in PHQ9_BASELINE_ITEMS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"PHQ-9 item columns absent from the export: {missing_cols}. "
            "Check that the amendment delivery is configured."
        )

    raw_items = df[PHQ9_BASELINE_ITEMS]
    if len(df):
        scorable_share = raw_items.isin(valid_codes).mean()
        undecoded = scorable_share[scorable_share < MIN_SCORABLE_ITEM_SHARE]
        if not undecoded.empty:
            raise ValueError(
                f"{len(undecoded)} of {len(PHQ9_BASELINE_ITEMS)} PHQ-9 item "
                f"columns decode below {MIN_SCORABLE_ITEM_SHARE:.0%} of "
                f"responses against the BeeintrDepr value list "
                f"{sorted(valid_codes)}: "
                f"{ {c: f'{s:.1%}' for c, s in undecoded.items()} }. The "
                "columns are present but do not decode — check the delimiter, "
                "the quoting and the dtypes of the export (observed: "
                f"{sorted({str(d) for d in raw_items.dtypes})})."
            )

    scored: list[str] = []
    for position, src_col in enumerate(PHQ9_BASELINE_ITEMS, start=1):
        dst_col = f"phq9_item_{position}"
        # Codes outside the value list (88 no answer, 99 don't know) are
        # dropped to NA before the recode, so they can never enter a sum.
        raw = df[src_col]
        r[dst_col] = (
            raw.where(raw.isin(valid_codes))
            .replace(BEEINTR_DEPR_SCORE_MAP)
            .astype("Int8")
        )
        scored.append(dst_col)

    complete = cast("pd.Series", r[scored].notna().all(axis=1))
    r["phq9_sum_items"] = r[scored].sum(axis=1).where(complete).astype("Int8")
    r["phq9_items_complete"] = complete.astype("boolean")

    n_complete = int(complete.sum())
    logger.info(
        "PHQ-9 items: %s of %s participants with all nine items (%.1f%%)",
        f"{n_complete:,}",
        f"{len(r):,}",
        n_complete / len(r) * 100 if len(r) else 0.0,
    )
    return r


def process_phq9_items(
    paths: NAKOPaths, output_dir: Path, *, amendment_df: DataFrame | None = None
) -> Path:
    """Process baseline PHQ-9 items -> ``phq9_items.parquet``.

    ``amendment_df`` is the already-loaded amendment export, passed by the
    orchestrator so a full run parses that CSV once for all three
    amendment-derived processors.
    """
    logger.info("Processing baseline PHQ-9 items ...")
    from pcc_analysis._plausibility import PHQ9_ITEMS_RANGES

    df = (
        amendment_df
        if amendment_df is not None
        else _load_nako_csv(paths.amendment_baseline_csv)
    )
    result = _extract_phq9_items(df)
    result = _enforce_plausibility(result, PHQ9_ITEMS_RANGES, module="phq9_items")
    return _save_parquet(result, "phq9_items", output_dir)
