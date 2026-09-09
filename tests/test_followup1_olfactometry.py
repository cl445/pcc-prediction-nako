"""Unit tests for the Sniffin' Sticks recode.

One rule carries almost all the risk. NAKO writes a literal 0 into
``olf_sum`` and ``olf_rslt`` for participants whose screening could not be
evaluated, and marks the row as unevaluable somewhere else entirely, in
``olf_kat`` and ``olf_norm``, with the 7777 sentinel. Nothing in the
shared sweep looks at a zero, so the zero survives into a column that
otherwise holds test results and reads as the worst possible score. In the
delivered data that is 17 568 rows against 9 genuine zeros, which is
enough to move any prevalence computed from the column.

The flags are therefore built from NAKO's own classification rather than
from a threshold of our own, and the scores are blanked wherever that
classification is absent. The tests below pin both halves, plus the
boundary the classification actually uses: the conventional "<= 6" cutoff
is NAKO's anosmia line, not its hyposmia line, so a flag derived from it
would count anosmia and call it hyposmia.
"""

from __future__ import annotations

import pandas as pd

from pcc_analysis._plausibility import OLFACTOMETRY_RANGES
from pcc_analysis.data_processing.followup1 import (
    _derive_olfactometry_indicators,
    _extract_followup1_olfactometry,
    _invalidate_unclassifiable_scores,
)

# Five participants as NAKO delivers them: normosmia, hyposmia, anosmia,
# one unevaluable row carrying the zero this module exists to catch, and
# one genuine zero that NAKO did classify as anosmia.
_NAKO_ROWS = pd.DataFrame(
    {
        "ID": [1, 2, 3, 4, 5],
        "olf_sum": [11, 8, 4, 0, 0],
        "olf_rslt": [12, 8, 5, 0, 0],
        "olf_kat": [1, 2, 3, 7777, 3],
        "olf_norm": [1, 0, 0, 7777, 0],
        "olf_frei": [1, 2, 3, 4, 5],
        "olf_method": [1, 2, 1, 2, 1],
        "olf_schnupfen": [0, 1, 9, 0, 0],
    }
)


def _process(frame: pd.DataFrame = _NAKO_ROWS) -> pd.DataFrame:
    out = _extract_followup1_olfactometry(frame)
    out = _invalidate_unclassifiable_scores(out)
    return _derive_olfactometry_indicators(out).set_index("ID")


def test_the_zero_of_an_unevaluable_screening_does_not_become_a_score() -> None:
    """Participant 4 was never scored; the 0 must not survive as one."""
    r = _process()
    assert pd.isna(r.loc[4, "olf_identification_sum"])
    assert pd.isna(r.loc[4, "olf_identification_sum_lenient"])
    assert pd.isna(r.loc[4, "olf_hyposmia_flag"])
    assert pd.isna(r.loc[4, "olf_anosmia_flag"])


def test_a_genuine_zero_is_kept() -> None:
    """Participant 5 scored zero and NAKO classified it; that is data."""
    r = _process()
    assert r.loc[5, "olf_identification_sum"] == 0
    assert bool(r.loc[5, "olf_anosmia_flag"]) is True
    assert bool(r.loc[5, "olf_hyposmia_flag"]) is True


def test_flags_follow_nakos_classification_not_a_threshold_of_our_own() -> None:
    """Hyposmia is category 2 or 3; anosmia is category 3 alone."""
    r = _process()
    assert [bool(v) for v in r.loc[[1, 2, 3], "olf_hyposmia_flag"]] == [
        False,
        True,
        True,
    ]
    assert [bool(v) for v in r.loc[[1, 2, 3], "olf_anosmia_flag"]] == [
        False,
        False,
        True,
    ]


def test_the_conventional_cutoff_would_have_mislabelled_hyposmia() -> None:
    """Guards the boundary: sum <= 6 is NAKO's anosmia line.

    Participant 2 scores 8 and is hyposmic by NAKO's reading, so a flag
    built from ``sum <= 6`` would have called that participant normal.
    """
    r = _process()
    assert r.loc[2, "olf_identification_sum"] == 8
    assert bool(r.loc[2, "olf_hyposmia_flag"]) is True


def test_the_no_answer_code_on_the_cold_question_becomes_missing() -> None:
    """9 is "keine Angabe" here, though a legal value elsewhere."""
    r = _process()
    assert r.loc[1, "olf_cold_recent"] == 0
    assert r.loc[2, "olf_cold_recent"] == 1
    assert pd.isna(r.loc[3, "olf_cold_recent"])


def test_renamed_columns_carry_plausibility_ranges() -> None:
    """A renamed column silently drops out of the quality harness."""
    r = _process()
    for column in (
        "olf_identification_sum",
        "olf_identification_sum_lenient",
        "olf_nasal_patency",
        "olf_function_category",
        "olf_cold_recent",
        "olf_normosmia",
    ):
        assert column in r.columns
        assert column in OLFACTOMETRY_RANGES
