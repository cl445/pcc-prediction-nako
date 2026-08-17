"""Unit tests for the baseline tobacco recode.

Two rules carry the risk here and neither falls out of the shared sentinel
sweep. ``a_smok_stat_qn == 4`` means "smoking status unknown" but is not a
sentinel code, so without an explicit mapping it would enter the model as a
fourth exposure level ordered above "current smoker". And ``7775``
("Erlaubt übersprungen") is a structural filter skip whose numeric reading
depends on the smoking status: zero pack-years for a never-smoker, genuinely
unknown for anyone else. The tests pin both on synthetic frames.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pcc_analysis.data_processing.smoking import (
    PERMITTED_SKIP_CODE,
    SMOKING_SOURCE_COLUMNS,
    SMOKING_STATUS_UNKNOWN,
    _extract_smoking,
)

# A never-smoker, a former smoker, a current smoker and an unknown, coded
# the way NAKO delivers them: the filter skips 7775 sit where the
# questionnaire never asked the question.
_NAKO_ROWS: dict[str, list[float]] = {
    "ID": [1, 2, 3, 4],
    "a_smok_stat_qn": [1, 2, 3, 4],
    "a_packyears": [7775, 12.5, 30.0, 7777],
    "a_smok_dur": [0, 20, 25, 7777],
    "a_smok_past_cig_d": [7775, 7775, 20.0, 7777],
    "a_f_smok_quit_age": [7775, 45, 7775, 7775],
}


def _frame(**overrides: list[float]) -> pd.DataFrame:
    data = {key: list(values) for key, values in _NAKO_ROWS.items()}
    data.update({key: list(values) for key, values in overrides.items()})
    frame = pd.DataFrame(data)
    # ``_load_nako_csv`` reads with ``dtype_backend="numpy_nullable"``, so the
    # status column reaches the extractor as nullable Int64. The distinction
    # matters for the sentinel tests below: numpy int64 raises on a narrowing
    # ``astype("Int8")``, nullable Int64 wraps silently — and it is the
    # silent wrap the extractor has to defend against.
    frame["a_smok_stat_qn"] = frame["a_smok_stat_qn"].astype("Int64")
    return frame


def test_unknown_status_is_missing_not_a_fourth_level() -> None:
    """Code 4 must not survive as an exposure level above "current"."""
    result = _extract_smoking(_frame())

    assert result["smoking_status"].tolist()[:3] == [1, 2, 3]
    assert pd.isna(result["smoking_status"].iloc[3])
    assert SMOKING_STATUS_UNKNOWN not in set(result["smoking_status"].dropna())


def test_permitted_skip_is_zero_pack_years_for_never_smokers() -> None:
    """7775 is a filter skip, not a missing answer, when nobody ever smoked.

    Blanket-NA'ing it would drop the never-smoker stratum out of the column
    and push it past the missingness threshold that removes a feature inside
    every CV fold.
    """
    result = _extract_smoking(_frame())

    assert result["pack_years"].iloc[0] == 0.0
    assert result["cigarettes_per_day"].iloc[0] == 0.0


def test_permitted_skip_stays_missing_where_zero_is_meaningless() -> None:
    """Age at quitting has no zero reading for someone who never started."""
    result = _extract_smoking(_frame())

    assert pd.isna(result["smoking_quit_age"].iloc[0])  # never smoked
    assert result["smoking_quit_age"].iloc[1] == 45  # former smoker
    assert pd.isna(result["smoking_quit_age"].iloc[2])  # still smoking


def test_permitted_skip_stays_missing_for_former_smokers_cigarette_count() -> None:
    """NAKO derives the cigarette count for current smokers only.

    A former smoker's skip is not a zero: the companion variable for their
    past consumption is not part of this delivery, so the value is unknown.
    """
    result = _extract_smoking(_frame())

    assert pd.isna(result["cigarettes_per_day"].iloc[1])
    assert result["cigarettes_per_day"].iloc[2] == 20.0


def test_no_sentinel_code_survives() -> None:
    """Every remaining global_missing code must be NA, in every column."""
    result = _extract_smoking(_frame())

    numeric = result.drop(columns=["ID", "current_smoker"])
    for col in numeric.columns:
        observed = set(numeric[col].dropna().tolist())
        assert not observed & {7775, 7776, 7777, 8886, 8888, 8889, 9999}, col


def test_permitted_skip_without_a_known_status_stays_missing() -> None:
    """The zero recode is conditional on the status, never unconditional."""
    result = _extract_smoking(
        _frame(a_smok_stat_qn=[4, 4, 4, 4], a_packyears=[PERMITTED_SKIP_CODE] * 4)
    )

    assert result["pack_years"].isna().all()


def test_current_smoker_tracks_the_status_and_keeps_unknowns_missing() -> None:
    result = _extract_smoking(_frame())

    assert result["current_smoker"].tolist()[:3] == [False, False, True]
    assert pd.isna(result["current_smoker"].iloc[3])


@pytest.mark.parametrize("sentinel", [7777, 8888, 9999, 111111])
def test_status_sentinels_do_not_wrap_into_a_smoking_level(sentinel: int) -> None:
    """A sentinel in the status column must leave both derived columns missing.

    The narrowing cast to ``Int8`` wraps rather than raises: 8888 lands on
    -72, 7777 on 97, 111111 on 7 — one code away from passing as a real
    status. ``_enforce_plausibility`` catches the out-of-range wraps in
    ``smoking_status`` afterwards but never touches ``current_smoker``, which
    belongs to no range catalogue, so a wrapped row would enter the model as
    "status unknown, but certainly not a current smoker". Only sweeping
    before the cast keeps the pair consistent.
    """
    result = _extract_smoking(_frame(a_smok_stat_qn=[sentinel] * 4))

    assert result["smoking_status"].isna().all()
    assert result["current_smoker"].isna().all()


def test_only_the_three_substantive_status_codes_survive() -> None:
    """Anything outside never/former/current is missing, not a fourth level."""
    result = _extract_smoking(_frame(a_smok_stat_qn=[1, 2, 3, 5]))

    assert result["smoking_status"].tolist()[:3] == [1, 2, 3]
    assert pd.isna(result["smoking_status"].iloc[3])
    assert pd.isna(result["current_smoker"].iloc[3])


def test_absent_source_columns_raise() -> None:
    """A delivery without the tobacco block must fail loudly."""
    frame = _frame().drop(columns=[SMOKING_SOURCE_COLUMNS[0]])
    with pytest.raises(ValueError, match="Smoking columns absent"):
        _extract_smoking(frame)


def test_status_without_intensity_is_not_carried() -> None:
    """``a_smok_stat_noint`` agrees 98.9 % with the status and is dropped.

    Pinned so a future edit does not reintroduce a near-duplicate feature.
    """
    assert "a_smok_stat_noint" not in SMOKING_SOURCE_COLUMNS
