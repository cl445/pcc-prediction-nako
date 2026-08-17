"""Unit tests for the smoke-test profiler's derived-column bookkeeping.

The profiler decides, per column, whether the generator samples that
column on its own or recomputes it from its sources. Get that wrong for a
column the pipeline recodes and the synthetic parquet holds rows the real
extractor could never produce — a care level of "no medical care" next to
a hospital stay — and the smoke run of the script that reads those columns
checks nothing, while still passing.

``acute_infection`` is the case where the two sides can drift apart
unnoticed: the recode lives in ``data_processing.acute_infection`` and the
list of columns it produces lives here, with nothing between them. These
tests pin both halves — that the profiler marks the recoded columns, and
that the extractor still derives exactly those columns from the care
level.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from smoke_test.profiler import DERIVED_COLUMNS, extract_profile

from pcc_analysis.data_processing.acute_infection import (
    HOSPITALISED_LEVEL,
    ICU_LEVEL,
    _extract_acute_infection,
)

# Five participants as NAKO delivers them: no care, outpatient only, a
# hospital ward, intensive care, and one who answered "weiß nicht". The
# five cover every branch of the recode — both sides of each threshold,
# plus the unknown level that has to stay NA.
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


def _corona2_frame() -> pd.DataFrame:
    """The rows as ``_load_nako_csv`` hands them over.

    The loader reads with ``dtype_backend="numpy_nullable"``, so an
    unanswered item arrives as ``pd.NA`` in an ``Int64`` column rather than
    a float ``NaN``, and a comparison against it yields ``pd.NA`` instead
    of ``False``. The recode depends on that distinction, so the fixture
    reproduces the loader's dtypes instead of letting pandas infer numpy
    ones.
    """
    return pd.DataFrame(_NAKO_ROWS).convert_dtypes()


def test_profiler_marks_the_care_level_recodes_as_derived(tmp_path: Path) -> None:
    """The four recoded columns carry ``derived``; their source does not.

    The flag is the whole interface to the generator: a column without it
    is drawn from its own marginal, independently of every other column in
    the modality. For a threshold on the care level that means a synthetic
    participant can be hospitalised without having been in hospital.
    """
    _extract_acute_infection(_corona2_frame()).to_parquet(
        tmp_path / "acute_infection.parquet"
    )

    columns = extract_profile(tmp_path)["modalities"]["acute_infection"]["columns"]

    assert columns["acute_care_level"].get("derived") is None
    for column in ("acute_hospitalised", "acute_icu", "hospital_days", "icu_days"):
        assert columns[column].get("derived") is True, column


def test_extractor_derives_exactly_the_columns_the_profiler_marks() -> None:
    """Every marked column is a function of ``acute_care_level``.

    The profiler's list is a copy of a rule that lives in the extractor, so
    it is only correct as long as the extractor keeps deriving those
    columns that way. Checking the rule against real extractor output
    means a change on either side fails here rather than quietly producing
    synthetic data whose severity axes disagree.
    """
    result = _extract_acute_infection(_corona2_frame())
    level = result["acute_care_level"]
    known = level.notna()

    assert set(DERIVED_COLUMNS["acute_infection"]) <= set(result.columns)

    assert (
        result.loc[known, "acute_hospitalised"].tolist()
        == (level[known] >= HOSPITALISED_LEVEL).tolist()
    )
    assert (
        result.loc[known, "acute_icu"].tolist() == (level[known] == ICU_LEVEL).tolist()
    )
    assert (
        result.loc[level.isna(), ["acute_hospitalised", "acute_icu"]]
        .isna()
        .all(axis=None)
    )

    below_ward = known & (level < HOSPITALISED_LEVEL)
    assert (result.loc[below_ward, "hospital_days"] == 0).all()
    below_icu = known & (level < ICU_LEVEL)
    assert (result.loc[below_icu, "icu_days"] == 0).all()
