"""Unit tests for the acute-infection recode.

Four rules carry the risk, and none of them falls out of the shared
sentinel sweep. The care-level boxes are a multi-select whose levels are
not nested, so the ordinal has to take the highest one ticked rather than
assume the ones below it — and, since the block arrives with nullable
dtypes, only a box somebody actually answered may move that ordinal.
``6666`` marks a day count as ruled out by the care answer and is not a
sentinel code, so it needs an explicit reading: zero for the counts whose
care answer implies one, and no column at all for the days until
admission, which has no such zero. And a "weiß nicht" or an empty block
has to leave the level unknown instead of dropping the participant into
the no-care reference group.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from pcc_analysis._plausibility import ACUTE_INFECTION_RANGES
from pcc_analysis.data_processing.acute_infection import (
    NOT_APPLICABLE_CODE,
    _extract_acute_infection,
)

# Five participants, coded the way NAKO delivers them: no care, outpatient
# only, a hospital ward, intensive care, and one who answered "weiß nicht".
_NAKO_ROWS: dict[str, list[object]] = {
    "ID": [1, 2, 3, 4, 5],
    "d_co2_i21_1": [1, 2, 2, 2, 2],  # Nein
    "d_co2_i21_2": [2, 1, 1, 1, 2],  # niedergelassener Arzt
    "d_co2_i21_3": [2, 2, 1, 2, 2],  # stationär
    "d_co2_i21_4": [2, 2, 2, 1, 2],  # Intensivstation
    "d_co2_i21_5": [2, 2, 1, 2, 2],  # Reha
    "d_co2_i21_6": [2, 2, 2, 2, 1],  # weiß nicht
    "d_co2_i22": [6666, 6666, 3, 1, 6666],
    "d_co2_i23": [6666, 6666, 12, 20, 6666],
    "d_co2_i24": [6666, 6666, 6666, 8, 6666],
    "d_co2_h3": [1, 2, 3, 4, 7775],
    "d_co2_h4": [
        "2022-03-01 00:00:00",
        "2020-12-01 00:00:00",
        "2019-12-01 00:00:00",
        "2021-06-01 00:00:00",
        "1800-01-01 00:00:00",
    ],
}


def _frame(**overrides: Sequence[object]) -> pd.DataFrame:
    """The rows as ``_load_nako_csv`` hands them over.

    The loader reads with ``dtype_backend="numpy_nullable"``, so an
    unanswered item arrives as ``pd.NA`` in an ``Int64`` column, not as a
    float ``NaN``. The distinction decides how a comparison against that
    item behaves — ``pd.NA`` rather than ``False`` — so the fixture matches
    the loader instead of leaving pandas to infer plain numpy dtypes.
    """
    data = {key: list(values) for key, values in _NAKO_ROWS.items()}
    data.update({key: list(values) for key, values in overrides.items()})
    return pd.DataFrame(data).convert_dtypes()


def test_care_level_takes_the_highest_box_ticked() -> None:
    result = _extract_acute_infection(_frame())

    assert result["acute_care_level"].tolist()[:4] == [0, 1, 2, 3]


def test_intensive_care_without_a_ward_still_counts_as_hospitalised() -> None:
    """55 participants ticked intensive care and not the ward below it.

    The boxes are a multi-select, not an ordinal with implied lower levels,
    so reading the hospital flag off ``i21_3`` alone would lose them.
    """
    result = _extract_acute_infection(_frame())

    icu_only = result.iloc[3]
    assert icu_only["acute_care_level"] == 3
    assert bool(icu_only["acute_hospitalised"]) is True
    assert bool(icu_only["acute_icu"]) is True


def test_unknown_care_stays_unknown_rather_than_no_care() -> None:
    """ "Weiß nicht" in the reference group would understate every level."""
    result = _extract_acute_infection(_frame())

    unknown = result.iloc[4]
    assert pd.isna(unknown["acute_care_level"])
    assert pd.isna(unknown["acute_hospitalised"])
    assert pd.isna(unknown["acute_icu"])


def test_nothing_ticked_is_unknown_too() -> None:
    """An all-Nein row contradicts itself; it is not an answer of "no care"."""
    result = _extract_acute_infection(
        _frame(
            d_co2_i21_1=[2, 2, 2, 2, 2],
            d_co2_i21_2=[2, 2, 2, 2, 2],
            d_co2_i21_3=[2, 2, 2, 2, 2],
            d_co2_i21_4=[2, 2, 2, 2, 2],
            d_co2_i21_6=[2, 2, 2, 2, 2],
        )
    )

    assert result["acute_care_level"].isna().all()


def test_an_unanswered_box_does_not_count_as_a_ticked_one() -> None:
    """Per-item missingness in the i21 block must not move the level.

    The block arrives with nullable dtypes, so comparing an unanswered box
    against Ja gives ``pd.NA``, and pandas reads an NA condition in
    ``mask`` as a hit. Taken at face value that treats every empty box as
    ticked: an unanswered intensive-care box would lift someone who
    answered "Nein" to the top level, and an unanswered "weiß nicht" would
    wipe out a real intensive-care tick. Both land on the severity axis
    that carries the interaction result.
    """
    result = _extract_acute_infection(
        _frame(
            # Row 0 ticks "Nein" and leaves the intensive-care box empty;
            # row 3 ticks intensive care and leaves "weiß nicht" empty.
            d_co2_i21_4=[None, 2, 2, 1, 2],
            d_co2_i21_6=[2, 2, 2, None, 1],
        )
    )

    no_care = result.iloc[0]
    assert no_care["acute_care_level"] == 0
    assert bool(no_care["acute_hospitalised"]) is False
    assert bool(no_care["acute_icu"]) is False

    icu = result.iloc[3]
    assert icu["acute_care_level"] == 3
    assert bool(icu["acute_icu"]) is True


def test_a_block_nobody_answered_stays_unknown() -> None:
    """43 % of the delivery carries NA in all six items at once.

    Those 50,794 participants were never shown the block, and every column
    keyed to the level has to follow them into NA. Reading an unseen block
    as "no medical care" would nearly double the reference group of the
    severity contrast with people who were never asked the question, and
    the day counts would take a structural zero on the strength of it.

    The all-answered and the all-missing row are the two shapes the current
    delivery actually has, so the level's reading of each is pinned
    separately from the per-item missingness a re-delivery could bring.
    """
    absent = [None] * 5
    result = _extract_acute_infection(
        _frame(
            d_co2_i21_1=absent,
            d_co2_i21_2=absent,
            d_co2_i21_3=absent,
            d_co2_i21_4=absent,
            d_co2_i21_5=absent,
            d_co2_i21_6=absent,
        )
    )

    assert result["acute_care_level"].isna().all()
    assert result["acute_hospitalised"].isna().all()
    assert result["acute_icu"].isna().all()
    assert result["acute_rehabilitation"].isna().all()
    # The structural zero is earned by a level below the threshold, so
    # without a level 6666 stays unresolved instead of becoming a zero.
    assert result["hospital_days"].take([0, 1, 4]).isna().all()
    assert result["icu_days"].take([0, 1, 4]).isna().all()
    # A day count somebody actually answered is a measurement in its own
    # right and survives the unreadable block.
    assert result["hospital_days"].iloc[2] == 12
    assert result["icu_days"].iloc[3] == 8


def test_not_applicable_becomes_zero_days_only_where_that_follows() -> None:
    """6666 is a structural skip for the day counts, driven by the level.

    Without the recode both counts would be missing for 99.4 % of the
    sample and get dropped inside every fold, rather than acting as the
    dose measure they are.
    """
    result = _extract_acute_infection(_frame())

    # Not hospitalised: zero hospital days and zero ICU days are the only
    # readings the care answer allows.
    assert result["hospital_days"].tolist()[:2] == [0, 0]
    assert result["icu_days"].tolist()[:2] == [0, 0]
    # Hospitalised but not in intensive care: zero ICU days, real ward days.
    assert result["hospital_days"].iloc[2] == 12
    assert result["icu_days"].iloc[2] == 0
    assert result["icu_days"].iloc[3] == 8


def test_days_to_admission_is_not_extracted() -> None:
    """``d_co2_i22`` is the day count with no structural zero.

    For someone never admitted the question has no numeric reading, so the
    column would sit at 99.4 % missing, fall below the retention threshold
    in ``_save_parquet`` and never reach the parquet. Extracting it would
    leave the plausibility catalogue and the data profile describing a
    column no consumer can load.
    """
    result = _extract_acute_infection(_frame())

    assert "days_to_hospital" not in result.columns
    assert "days_to_hospital" not in ACUTE_INFECTION_RANGES


def test_no_not_applicable_code_survives_into_the_parquet() -> None:
    result = _extract_acute_infection(_frame())

    for column in ("hospital_days", "icu_days"):
        assert NOT_APPLICABLE_CODE not in set(result[column].dropna())


def test_infection_count_keeps_the_open_ended_top_code() -> None:
    """Code 4 means "more than three", not exactly four."""
    result = _extract_acute_infection(_frame())

    assert result["n_infections"].tolist()[:4] == [1, 2, 3, 4]
    assert pd.isna(result["n_infections"].iloc[4])


def test_infection_month_is_counted_from_the_reference_month() -> None:
    """December 2019 is the questionnaire's own pre-pandemic boundary."""
    result = _extract_acute_infection(_frame())

    months = result["first_infection_months"]
    assert months.iloc[2] == 0  # 2019-12
    assert months.iloc[1] == 12  # 2020-12
    assert months.iloc[0] == 27  # 2022-03
    # 1800-01-01 is the missing marker, not a date 2,400 months early.
    assert pd.isna(months.iloc[4])
