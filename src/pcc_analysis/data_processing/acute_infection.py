"""Acute course of the SARS-CoV-2 infection, from the Corona-2 survey.

The second-hit framing says a pre-infection vulnerability needs an
infection to become post-COVID condition. Whether the *severity* of that
infection modifies the effect is the obvious next question, and it is only
identified if a severity measure exists.

``export_corona2.csv`` carries the acute-care block (``d_co2_i21_1`` ...
``i21_6``), the hospital and ICU day counts, the number of infections and
the month of the first one. This module lifts them into
``acute_infection.parquet``.

**Not a predictor.** Everything here is measured after the infection, so
it belongs to the post-exposure side of the design and cannot enter a
model that predicts a post-infection outcome from pre-infection
information — the same circularity rule that keeps post-infection symptom
items out of the modality stack. The parquet is deliberately not
registered as a modality in :mod:`pcc_analysis.data_manager`; its consumer
is the interaction sensitivity in
``scripts/supplementary/severity_interaction.py``.

Three coding details, all verified against the delivery:

``d_co2_i21_*`` is a multi-select of Ja/Nein columns, not an ordinal, and
the levels are not nested: 55 participants ticked intensive care without
ticking a hospital ward. ``acute_care_level`` therefore takes the highest
level ticked rather than assuming the boxes below it were ticked too.

``6666`` marks the day counts as not applicable and is not in
``NAKO_SENTINEL_CODES``. It appears for exactly the participants who were
not hospitalised — 66,207 of them, and for none of the 403 who were — so
the reading is unambiguous, and the hospital and ICU day counts take the
structural zero that follows from it (no hospital stay means zero hospital
days), on the same principle as the permitted skips in :mod:`smoking`.
Without the recode both would be 99.4 % missing and get dropped inside
every fold rather than acting as the dose measure they are. ``d_co2_i22``,
the days until admission, is the one count with no such zero — for someone
never admitted the question has no numeric reading at all, only an empty
one — so it would stay 99.4 % missing, fall below the column-retention
threshold of :func:`~pcc_analysis.data_processing._common._save_parquet`
and never reach the parquet. It stays out of the extract, so that neither
the plausibility catalogue nor the data profile describes a column no
consumer can load.

``d_co2_h4`` is a month-precision date with ``1800-01-01`` as its missing
marker. It is carried as months since December 2019 — the reference month
the questionnaire itself uses to separate pre-pandemic complaints — so the
column is a plain number the plausibility harness can bound.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis.data_processing._common import (
    _enforce_plausibility,
    _load_nako_csv,
    _replace_nako_missing,
    _save_parquet,
)

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


CARE_LEVEL_COLUMNS: dict[str, int] = {
    "d_co2_i21_1": 0,  # Nein (no medical care)
    "d_co2_i21_2": 1,  # Ja, von einem niedergelassenen Arzt
    "d_co2_i21_3": 2,  # Ja, stationär im Krankenhaus
    "d_co2_i21_4": 3,  # Ja, im Krankenhaus auf einer Intensivstation
}
"""Acute-care columns and the severity level each one marks.

Rehabilitation (``d_co2_i21_5``) is deliberately absent: it is post-acute
care and sits on a different axis from the acuity of the illness itself,
so it is carried as its own flag rather than folded into the ordinal.
``d_co2_i21_6`` is "weiß nicht" and marks the level as unknown.
"""

CARE_LEVEL_REHAB_COLUMN: str = "d_co2_i21_5"
CARE_LEVEL_UNKNOWN_COLUMN: str = "d_co2_i21_6"

YES: int = 1
NO: int = 2
"""``JaNein`` value list: 1 = Ja, 2 = Nein."""

HOSPITALISED_LEVEL: int = 2
ICU_LEVEL: int = 3

NOT_APPLICABLE_CODE: int = 6666
"""Day-count marker for a question the acute-care answer ruled out."""

INFECTION_COUNT_COLUMN: str = "d_co2_h3"
INFECTION_COUNT_CODES: tuple[int, ...] = (1, 2, 3, 4)
"""``cl_co2_23``: 1 = once, 2 = twice, 3 = three times, 4 = more than three.

Kept on NAKO's numbering, which is also the natural order; ``4`` is
open-ended and must not be read as exactly four infections.
"""

FIRST_INFECTION_DATE_COLUMN: str = "d_co2_h4"
INFECTION_DATE_EPOCH: pd.Timestamp = pd.Timestamp("2019-12-01")
"""December 2019, the questionnaire's own pre-pandemic reference month."""

MIN_PLAUSIBLE_INFECTION_YEAR: int = 2019
"""Anything earlier is the ``1800-01-01`` missing marker, not a date."""


def _care_level(df: DataFrame) -> pd.Series:
    """Highest acute-care level ticked, or NA where none can be read.

    NA covers three states that all mean the same thing for an ordinal:
    "weiß nicht" was ticked, nothing was ticked, or the block was never
    shown. Guessing at any of them would put someone in the no-care
    reference group on the strength of a non-answer.

    The delivery is read with nullable dtypes, so comparing an unanswered
    box against Ja yields ``pd.NA``, and :meth:`~pandas.Series.mask` counts
    an NA condition as a hit. Each condition is therefore resolved to plain
    ``False`` first: a box nobody answered is not a box that was ticked.
    Read the other way round, per-item missingness in the block would both
    invent intensive care for someone who ticked "Nein" and erase a real
    intensive-care tick behind an unanswered "weiß nicht" — on the very
    axis the severity interaction is estimated over.
    """
    level = pd.Series(pd.NA, index=df.index, dtype="Int8")
    for column, value in CARE_LEVEL_COLUMNS.items():
        ticked = pd.to_numeric(df[column], errors="coerce") == YES
        level = level.mask(ticked.fillna(False), value)

    unknown = pd.to_numeric(df[CARE_LEVEL_UNKNOWN_COLUMN], errors="coerce") == YES
    return level.mask(unknown.fillna(False), pd.NA)


def _first_infection_months(df: DataFrame) -> pd.Series:
    """Months from December 2019 to the first reported infection."""
    dates = pd.to_datetime(df[FIRST_INFECTION_DATE_COLUMN], errors="coerce")
    dated = dates.where(dates.dt.year >= MIN_PLAUSIBLE_INFECTION_YEAR)
    months = (dated.dt.year - INFECTION_DATE_EPOCH.year) * 12 + (
        dated.dt.month - INFECTION_DATE_EPOCH.month
    )
    return months.astype("Float64")


def _extract_acute_infection(df: DataFrame) -> DataFrame:
    """Recode the Corona-2 acute-course block into modelling columns."""
    required = [
        *CARE_LEVEL_COLUMNS,
        CARE_LEVEL_REHAB_COLUMN,
        CARE_LEVEL_UNKNOWN_COLUMN,
        INFECTION_COUNT_COLUMN,
        FIRST_INFECTION_DATE_COLUMN,
        "d_co2_i23",
        "d_co2_i24",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Acute-infection columns absent from the Corona-2 export: "
            f"{missing}.  Check that the primary delivery is configured."
        )

    r = pd.DataFrame()
    r["ID"] = df["ID"]

    level = _care_level(df)
    r["acute_care_level"] = level
    r["acute_hospitalised"] = (level >= HOSPITALISED_LEVEL).astype("boolean")
    r["acute_icu"] = (level == ICU_LEVEL).astype("boolean")
    r.loc[level.isna(), ["acute_hospitalised", "acute_icu"]] = pd.NA

    rehab = pd.to_numeric(df[CARE_LEVEL_REHAB_COLUMN], errors="coerce")
    r["acute_rehabilitation"] = (
        (rehab == YES).astype("boolean").where(rehab.isin([YES, NO]))
    )

    # Resolve the not-applicable code against the care level before the
    # shared sweep, so a structural zero is distinguishable from a refusal.
    known_level = level.notna()
    for destination, source, zero_below in (
        ("hospital_days", "d_co2_i23", HOSPITALISED_LEVEL),
        ("icu_days", "d_co2_i24", ICU_LEVEL),
    ):
        days = pd.to_numeric(df[source], errors="coerce")
        structural = (
            (days == NOT_APPLICABLE_CODE) & known_level & (level < zero_below)
        ).fillna(False)
        days = days.mask(structural, 0)
        r[destination] = days.mask(days == NOT_APPLICABLE_CODE)

    count = pd.to_numeric(df[INFECTION_COUNT_COLUMN], errors="coerce")
    r["n_infections"] = count.where(count.isin(INFECTION_COUNT_CODES)).astype("Int8")

    r["first_infection_months"] = _first_infection_months(df)

    r = _replace_nako_missing(r)

    n_level = int(r["acute_care_level"].notna().sum())
    logger.info(
        "Acute infection: care level known for %s participants; "
        "hospitalised %s, of those in intensive care %s; "
        "reinfections reported by %s",
        f"{n_level:,}",
        f"{int((r['acute_hospitalised'] == True).sum()):,}",  # noqa: E712
        f"{int((r['acute_icu'] == True).sum()):,}",  # noqa: E712
        f"{int((r['n_infections'] > 1).sum()):,}",
    )
    return r


def process_acute_infection(paths: NAKOPaths, output_dir: Path) -> Path:
    """Process the Corona-2 acute course -> ``acute_infection.parquet``."""
    logger.info("Processing acute infection course ...")
    from pcc_analysis._plausibility import ACUTE_INFECTION_RANGES

    df = _load_nako_csv(paths.corona2_csv)
    result = _extract_acute_infection(df)
    result = _enforce_plausibility(
        result, ACUTE_INFECTION_RANGES, module="acute_infection"
    )
    return _save_parquet(result, "acute_infection", output_dir)
