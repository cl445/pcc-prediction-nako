"""Unit tests for the baseline PHQ-9 item recode and the GPAQ extraction.

Both processors read the NAM-45 amendment export. The recode is the risky
part: NAKO stores the baseline items under the ``BeeintrDepr`` value list
(2/11/12/13) with 88 and 99 as missing markers, and neither 88 nor 99 is in
``NAKO_SENTINEL_CODES`` — a plain sentinel sweep would leave them in place
and let an 88 enter a sum score. The tests below pin that behaviour on
synthetic frames so the file-system processors stay out of the loop.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pytest

from pcc_analysis.data_processing.phq9_items import (
    BEEINTR_DEPR_SCORE_MAP,
    MIN_SCORABLE_ITEM_SHARE,
    PHQ9_BASELINE_ITEMS,
    _extract_phq9_items,
)
from pcc_analysis.data_processing.physical_activity import (
    GPAQ_MISSING_CODE,
    _extract_gpaq_activity,
)


def _item_frame(rows: Sequence[Sequence[object]]) -> pd.DataFrame:
    """Build a frame with ``ID`` plus the nine raw PHQ-9 item columns."""
    frame = pd.DataFrame(list(rows), columns=pd.Index(PHQ9_BASELINE_ITEMS))
    frame.insert(0, "ID", range(len(frame)))
    return frame


def test_beeintr_depr_codes_map_to_canonical_scores() -> None:
    """2/11/12/13 become 0/1/2/3 and the sum follows the canonical scale."""
    result = _extract_phq9_items(
        _item_frame(
            [
                [2] * 9,  # all "not at all"
                [13] * 9,  # all "nearly every day"
                [2, 11, 12, 13, 2, 11, 12, 13, 2],
            ]
        )
    )

    assert result["phq9_item_1"].tolist() == [0, 3, 0]
    assert result["phq9_item_4"].tolist() == [0, 3, 3]
    assert result["phq9_sum_items"].tolist() == [0, 27, 12]
    assert bool(result["phq9_items_complete"].all())


def test_score_map_covers_exactly_the_substantive_codes() -> None:
    """The map must not silently acquire a missing code such as 88 or 99."""
    assert set(BEEINTR_DEPR_SCORE_MAP) == {2, 11, 12, 13}
    assert set(BEEINTR_DEPR_SCORE_MAP.values()) == {0, 1, 2, 3}


@pytest.mark.parametrize("missing_code", [88, 99])
def test_missing_codes_never_enter_the_sum(missing_code: int) -> None:
    """88 (no answer) and 99 (don't know) drop to NA and void the sum.

    Without the explicit recode these codes would be summed as if they were
    responses, inflating a 0..27 score into the hundreds.
    """
    # Three rows so the affected item still decodes for two thirds of them and
    # the per-item decode floor reads this as non-response, not as a column
    # that failed to parse.
    rows: list[list[object]] = [[2] * 9 for _ in range(3)]
    rows[0][3] = missing_code
    result = _extract_phq9_items(_item_frame(rows))

    assert pd.isna(result["phq9_item_4"].iloc[0])
    assert pd.isna(result["phq9_sum_items"].iloc[0])
    assert not bool(result["phq9_items_complete"].iloc[0])
    assert result["phq9_item_4"].iloc[1] == 0


def test_sum_is_complete_case_not_prorated() -> None:
    """A single missing item voids the sum; no prorating is applied.

    Matches how NAKO derives ``a_emo_phq9_sum`` (Streit et al. 2023).
    """
    rows: list[list[object]] = [[11] * 9 for _ in range(3)]
    rows[0][8] = pd.NA
    result = _extract_phq9_items(_item_frame(rows))

    assert pd.isna(result["phq9_sum_items"].iloc[0])
    assert result["phq9_item_1"].iloc[0] == 1
    assert result["phq9_sum_items"].iloc[1] == 9


def test_absent_item_columns_raise() -> None:
    """A truncated export must fail loudly rather than emit a partial sum."""
    frame = _item_frame([[2] * 9]).drop(columns=[PHQ9_BASELINE_ITEMS[-1]])
    with pytest.raises(ValueError, match="PHQ-9 item columns absent"):
        _extract_phq9_items(frame)


@pytest.mark.parametrize("n_broken", [9, 1])
def test_items_present_but_undecodable_raise(n_broken: int) -> None:
    """Columns that are there and decode to nothing must fail as loudly.

    A quoted export, or a single non-numeric token, turns a column into
    ``object`` dtype and every ``isin`` test against the value list into
    False. Every item in it then goes NA, which voids the complete-case sum
    for everyone, ``_save_parquet`` drops the block at its 99-%-missing rule,
    and ``_attach`` joins nothing into ``mental_health`` -- producing a
    normal-looking full-stack run with no item data in it.

    One broken column of nine does that just as thoroughly as all nine, which
    is why the floor is applied per item: pooled over the nine, a single
    stringified column still decodes at 8/9.
    """
    frame = _item_frame([[2] * 9, [11] * 9])
    for col in PHQ9_BASELINE_ITEMS[:n_broken]:
        frame[col] = frame[col].astype(str)

    with pytest.raises(ValueError, match="do not decode"):
        _extract_phq9_items(frame)


def test_ordinary_item_missingness_stays_below_the_decode_floor() -> None:
    """The floor separates a decode failure from real non-response.

    Item-level missingness on the production export is about 3 %; the guard
    must not fire on that, or on anything like it.
    """
    rows: list[list[object]] = [[2] * 9 for _ in range(10)]
    rows[0][0] = 88
    rows[1][3] = 99
    rows[2][8] = pd.NA
    result = _extract_phq9_items(_item_frame(rows))

    assert int(result["phq9_items_complete"].sum()) == 7
    # The worst item here decodes for 9 of 10 participants. Pinning the floor
    # below that and above the 3 % seen in production is the whole margin the
    # constant has to hold.
    assert 0.1 < MIN_SCORABLE_ITEM_SHARE < 0.9


def test_gpaq_sentinel_becomes_missing() -> None:
    """``777777`` marks an uncomputable GPAQ total, not 777777 MET-minutes."""
    frame = pd.DataFrame(
        {"ID": [1, 2, 3], "a_gpaq_ptotalmet": [0, GPAQ_MISSING_CODE, 3360]}
    )
    result = _extract_gpaq_activity(frame)

    assert result["gpaq_met_total"].tolist()[0] == 0
    assert pd.isna(result["gpaq_met_total"].iloc[1])
    assert result["gpaq_met_total"].iloc[2] == 3360


def test_gpaq_survey_sentinels_become_missing() -> None:
    """The survey sentinels sit *inside* the plausible MET range.

    ``GPAQ_MET_WEEK_RANGE`` is (0, 80640), so a refusal code such as 8888
    passes the plausibility guard untouched; ``_join_gpaq`` would then set
    ``gpaq_reported = 1`` and the log1p transform would hand the base learner
    an above-average activity level. Only the sentinel sweep removes them.
    Nothing legitimate is at risk: GPAQ totals are multiples of 40 and no
    sentinel code is.
    """
    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6],
            "a_gpaq_ptotalmet": [7775, 7777, 8888, 8889, 9999, 3360],
        }
    )
    result = _extract_gpaq_activity(frame)

    assert result["gpaq_met_total"].isna().tolist() == [True] * 5 + [False]
    assert result["gpaq_met_total"].iloc[5] == 3360


def test_gpaq_absent_column_raises() -> None:
    with pytest.raises(ValueError, match="a_gpaq_ptotalmet absent"):
        _extract_gpaq_activity(pd.DataFrame({"ID": [1]}))
