"""Baseline tobacco exposure from the supplementary delivery.

The primary delivery carries no tobacco variable at all, which is why
smoking reaches the model only through this module. The supplementary export
(``nako_supplementary_dir`` in the config) adds NAKO's six derived tobacco
variables for the full baseline sample.

Two coding conventions in this block need care, and neither is handled by
the shared sentinel sweep:

``a_smok_stat_qn`` (Rauchstatus) uses the ``rauch_status`` value list, where
``4`` means "smoking status unknown". Four is not a sentinel code, so it
survives ``_replace_nako_missing`` and would otherwise enter the model as a
fourth exposure level ordered above "current smoker". It is mapped to NA
here, the same way :mod:`phq9_items` handles the ``BeeintrDepr`` codes.

``7775`` ("Erlaubt übersprungen") marks a *structural* skip, not a missing
answer: the questionnaire never asked never-smokers for their pack-years,
and never asked never- or current-smokers when they quit. Blanket-NA'ing it
would discard the "never smoked" information for 56,076 participants and
push ``pack_years`` past the 50 % missingness threshold that drops a column
inside every CV fold. Where the skip has an unambiguous numeric reading —
zero pack-years, zero cigarettes for someone who never smoked — it is
recoded to 0; where it does not (age at quitting for someone who never
started), it stays NA. The mapping is pinned per column below and covered by
unit tests.

``a_smok_stat_noint`` is deliberately not carried. It is the same construct
derived without the intensity questions and agrees with ``a_smok_stat_qn``
on 98.9 % of participants, so it would add a near-duplicate feature rather
than information.
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


SMOKING_SOURCE_COLUMNS: list[str] = [
    "a_smok_stat_qn",
    "a_packyears",
    "a_smok_dur",
    "a_smok_past_cig_d",
    "a_f_smok_quit_age",
]

SMOKING_STATUS_UNKNOWN: int = 4
"""``rauch_status`` code 4 = "Rauchstatus unbekannt" -> missing, not a level.

The other codes are 1 = never (also not formerly), 2 = former, 3 = current.
They are kept on NAKO's own numbering so the parquet stays traceable to the
value list; 1 < 2 < 3 also orders the three groups by current exposure,
which is how the linear base learner reads an unencoded numeric column.
"""

SMOKING_STATUS_CODES: tuple[int, ...] = (1, 2, 3)
"""The three substantive ``rauch_status`` levels, as a membership test.

Mirrors ``SMOKING_STATUS_RANGE`` in :mod:`pcc_analysis._plausibility`, but is
applied inside the extractor rather than after it, because ``current_smoker``
is derived here and belongs to no range catalogue: deriving it from a status
that has not yet been restricted would leave the pair inconsistent.
"""

PERMITTED_SKIP_CODE: int = 7775
"""``global_missing`` code 7775 = "Erlaubt übersprungen" (filter skip)."""

# Columns where a permitted skip has an unambiguous numeric reading, and
# the smoking-status group for which that reading holds. Everything else
# stays NA. `a_smok_dur` is absent because NAKO already delivers 0 for
# never-smokers rather than skipping the question.
ZERO_ON_PERMITTED_SKIP: dict[str, tuple[int, ...]] = {
    # Someone who never smoked has accumulated zero pack-years.
    "a_packyears": (1,),
    # Someone who never smoked smokes zero cigarettes per day. Former
    # smokers are skipped too, but their reading is genuinely unknown:
    # NAKO derives this column only for current smokers, and the
    # companion variable for former smokers (a_f_smok_cig_d) is not part
    # of this delivery.
    "a_smok_past_cig_d": (1,),
}


def _extract_smoking(df: DataFrame) -> DataFrame:
    """Recode NAKO's six tobacco variables into modelling columns."""
    missing_cols = [c for c in SMOKING_SOURCE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Smoking columns absent from the export: {missing_cols}. "
            "Check that the supplementary delivery is configured."
        )

    r = pd.DataFrame()
    r["ID"] = df["ID"]

    status_raw = pd.to_numeric(df["a_smok_stat_qn"], errors="coerce")
    status = status_raw.where(status_raw != SMOKING_STATUS_UNKNOWN)

    # Resolve the permitted skips against the status column while the raw
    # sentinel values are still present, then let the shared sweep turn
    # every remaining sentinel into NA.
    df = df.copy()
    for src_col, zero_groups in ZERO_ON_PERMITTED_SKIP.items():
        skipped = df[src_col] == PERMITTED_SKIP_CODE
        structural = (skipped & status_raw.isin(zero_groups)).fillna(False)
        df.loc[structural, src_col] = 0

    r["smoking_status"] = status
    for dst_col, src_col in (
        ("pack_years", "a_packyears"),
        ("smoking_duration_years", "a_smok_dur"),
        ("cigarettes_per_day", "a_smok_past_cig_d"),
        ("smoking_quit_age", "a_f_smok_quit_age"),
    ):
        r[dst_col] = pd.to_numeric(df[src_col], errors="coerce")

    r = _replace_nako_missing(r)

    # Narrow to Int8 only after the sweep, and only on the three substantive
    # codes. ``_load_nako_csv`` hands over nullable Int64, and ``astype`` wraps
    # a four- or six-digit sentinel into the Int8 range silently instead of
    # raising: 8888 lands on -72, 7777 on 97, 111111 on 7. The first two are
    # outside [1, 3] and would still be caught by ``_enforce_plausibility``,
    # but 7 is only one code away from surviving, and ``current_smoker`` below
    # belongs to no range catalogue at all — a wrapped row would read as
    # "status unknown, but certainly not a current smoker" and land in the
    # reference category. Sweeping first keeps the pair consistent by
    # construction rather than by luck about which sentinels happen to wrap
    # out of range.
    status_clean = r["smoking_status"]
    unexpected = status_clean.notna() & ~status_clean.isin(SMOKING_STATUS_CODES)
    if unexpected.any():
        # Reported here because narrowing inside the extractor takes the
        # column out of reach of the ``SMOKING_STATUS_RANGE`` guard, which
        # would otherwise be what announces an unexpected code.
        logger.warning(
            "smoking: %d value(s) in a_smok_stat_qn outside the rauch_status "
            "codes %s; set to NA. Observed: %s",
            int(unexpected.sum()),
            list(SMOKING_STATUS_CODES),
            sorted(status_clean[unexpected].unique().tolist()),
        )
    r["smoking_status"] = status_clean.where(~unexpected).astype("Int8")

    # One derived binary. The 1/2/3 status column enters the linear base
    # learner as a numeric with a monotone never < former < current
    # ordering; this lets the model separate current smokers from the rest
    # without needing that ordering to hold. It is a deterministic function
    # of the status and is deliberately exempt from the collinearity drops in
    # ``create_medical_history_preprocessor``; see that docstring.
    r["current_smoker"] = (r["smoking_status"] == 3).astype("boolean")
    r.loc[r["smoking_status"].isna(), "current_smoker"] = pd.NA

    n_status = int(r["smoking_status"].notna().sum())
    logger.info(
        "Smoking: status known for %s of %s participants (%.1f%%); "
        "never %s, former %s, current %s",
        f"{n_status:,}",
        f"{len(r):,}",
        n_status / len(r) * 100 if len(r) else 0.0,
        f"{int((r['smoking_status'] == 1).sum()):,}",
        f"{int((r['smoking_status'] == 2).sum()):,}",
        f"{int((r['smoking_status'] == 3).sum()):,}",
    )
    return r


def process_smoking(paths: NAKOPaths, output_dir: Path) -> Path:
    """Process baseline tobacco exposure -> ``smoking.parquet``."""
    logger.info("Processing smoking ...")
    from pcc_analysis._plausibility import SMOKING_RANGES

    df = _load_nako_csv(paths.supplementary_baseline_csv)
    result = _extract_smoking(df)
    result = _enforce_plausibility(result, SMOKING_RANGES, module="smoking")
    return _save_parquet(result, "smoking", output_dir)
