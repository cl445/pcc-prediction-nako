"""Physical activity: QUAP MET-minutes, sitting, stairs, plus the GPAQ total.

NAKO administered two parallel activity instruments, and they are not
interchangeable: on the participants who answered both they correlate at
only r = 0.25 (Spearman 0.38), with GPAQ medians running about 1.7 times the
QUAP ones. GPAQ is therefore carried as a second measurement rather than as
a stand-in for the missing QUAP responses.

Both feed the ``physical_activity`` modality, but under different
missing-data conventions: the QUAP block is zero-filled for the ~65 % who
never returned the questionnaire (see :func:`_extract_physical_activity`),
while ``gpaq_met_total`` stays genuinely NA where it is absent. GPAQ arrives
with the NAM-45 amendment and also keeps its own parquet
(:func:`process_gpaq_activity`) so the two instruments stay separable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis.data_processing._common import (
    _load_nako_csv,
    _replace_nako_missing,
    _save_parquet,
)

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)

QUAP_MEASUREMENT_COUNT: int = 35
"""Number of QUAP measurement columns subject to the non-respondent fill.

Pinned so that adding a column to :func:`_extract_physical_activity` has to
be a deliberate act rather than an accidental extension of the zero-fill.
"""


def _extract_physical_activity(df: DataFrame) -> DataFrame:
    """Extract physical activity questionnaire variables.

    Uses MET-minutes where available (intensity-weighted), raw minutes
    only for activities without MET equivalents.
    """
    r = pd.DataFrame()
    r["ID"] = df["ID"]

    # Occupational activity
    r["occupational_activity_level"] = df["a_quap_beruf"]

    # Household (minutes per week, no MET equivalent)
    r["household_minutes_week"] = df["a_quap_hausarbeit"]

    # Active transportation (minutes per week, no MET equivalent)
    r["active_transport_summer"] = df["a_quap_fortbew_so"]
    r["active_transport_winter"] = df["a_quap_fortbew_wi"]

    # Walking (minutes per week, no MET equivalent)
    r["walking_summer"] = df["a_quap_spazier_so"]
    r["walking_winter"] = df["a_quap_spazier_wi"]

    # Cycling leisure (minutes per week, no MET equivalent)
    r["cycling_summer"] = df["a_quap_radtour_so"]
    r["cycling_winter"] = df["a_quap_radtour_wi"]

    # Seasonal sports (raw minutes, NAKO-provided)
    r["sports_spring"] = df["a_quap_sport_frue"]
    r["sports_summer"] = df["a_quap_sport_somm"]
    r["sports_autumn"] = df["a_quap_sport_herb"]
    r["sports_winter"] = df["a_quap_sport_wint"]
    r["sports_combined"] = df["a_quap_sport_comb"]

    # Participation flags
    r["household_participation"] = df["a_quap_hausarbeit_pr"]
    r["active_transport_participation"] = df["a_quap_fortbeweg_pr"]
    r["sports_participation"] = df["a_quap_sport_pr"]
    r["sitting_participation"] = df["a_quap_sitzen_pr"]
    r["stairs_participation"] = df["a_quap_treppenst_pr"]
    r["leisure_participation"] = df["a_quap_freizeit_pr"]

    # Sports (MET-minutes per week)
    r["met_sports_combined"] = df["a_quap_sp_met_comb"]

    # Seasonal sports MET
    r["met_sports_spring"] = df["a_quap_sp_met_frue"]
    r["met_sports_summer"] = df["a_quap_sp_met_somm"]
    r["met_sports_autumn"] = df["a_quap_sp_met_herb"]
    r["met_sports_winter"] = df["a_quap_sp_met_wint"]

    # Sedentary behaviour (minutes per day)
    r["sitting_weekday"] = df["a_quap_seed_week"]
    r["sitting_saturday"] = df["a_quap_seed_sa"]
    r["sitting_sunday"] = df["a_quap_seed_so"]

    # Stair climbing (floors per day)
    r["stairs_weekday"] = df["a_quap_tr_metr_week"]
    r["stairs_weekend"] = df["a_quap_tr_metr_saso"]

    # Total physical activity (raw minutes, NAKO-provided)
    r["pa_total"] = df["a_quap_patotal"]
    r["pa_summer"] = df["a_quap_patotal_so"]
    r["pa_winter"] = df["a_quap_patotal_wi"]

    # Total physical activity (MET-minutes per week)
    r["met_total"] = df["a_quap_mettotal"]
    r["met_summer"] = df["a_quap_mettotal_so"]
    r["met_winter"] = df["a_quap_mettotal_wi"]

    for col in r.columns:
        if col != "ID":
            r[col] = r[col].replace([7777, 777777, -9], pd.NA)

    # Questionnaire-return indicator + zero-fill for QUAP non-respondents.
    # Only about 35 % of the analytic sample returned the QUAP
    # questionnaire at all (38.8 % of the MRI cohort; 32.5 % with a usable
    # ``met_total``), so every raw QUAP column sits between 61 % and 68 %
    # missing. At that level ``MissingnessThreshold(0.5)`` drops the entire
    # QUAP block inside each CV fold and the base learner collapses to its
    # indicator. The zero-fill prevents that by encoding the
    # deployment-time reading "no QUAP minutes or MET on record", and
    # ``pa_self_reported`` keeps the missingness recoverable to the linear
    # classifier. Note what the indicator does and does not mean: it marks
    # that the questionnaire came back, not that the participant was
    # inactive.
    #
    # The convention is QUAP-specific by construction, so the fill is
    # confined to the columns assigned above. An instrument with different
    # coverage — GPAQ at 94 % is the case in point — must not inherit it:
    # letting such a column into the non-response test would make almost
    # every row count as a respondent, leave the QUAP block above the
    # missingness threshold, and silently reduce the modality to two
    # features. GPAQ is therefore joined downstream, in
    # ``process_physical_activity``, after this block has run. The count
    # check makes an accidental addition here fail loudly instead.
    quap_cols = [c for c in r.columns if c != "ID"]
    if len(quap_cols) != QUAP_MEASUREMENT_COUNT:
        raise ValueError(
            f"Expected {QUAP_MEASUREMENT_COUNT} QUAP measurement columns, "
            f"got {len(quap_cols)}. A column added to this extractor is "
            "about to be zero-filled for QUAP non-respondents; confirm that "
            "is intended and update QUAP_MEASUREMENT_COUNT."
        )
    any_reported = r[quap_cols].notna().any(axis=1)
    r["pa_self_reported"] = any_reported.astype("UInt8")
    nonrespondent = ~any_reported
    if nonrespondent.any():
        r.loc[nonrespondent, quap_cols] = r.loc[nonrespondent, quap_cols].fillna(0)
    return r


def _join_gpaq(
    result: DataFrame, paths: NAKOPaths, *, amendment_df: DataFrame | None = None
) -> DataFrame:
    """Attach ``gpaq_met_total`` from the amendment delivery, if present.

    Left join, so the QUAP row set is authoritative and the modality costs
    no additional attrition. GPAQ keeps true NA where it is missing: unlike
    the QUAP block it is available for 94 % of participants, so the rows
    without it are a small genuine-missing stratum that the imputer should
    handle, not a non-response stratum to be encoded as zero activity.

    ``amendment_df`` is the already-loaded amendment export. The orchestrator
    passes it so a full run parses that ~117,000-row CSV once rather than once
    per amendment-derived processor.
    """
    from pcc_analysis._plausibility import GPAQ_RANGES
    from pcc_analysis.data_processing._common import _enforce_plausibility

    if amendment_df is None:
        amendment_csv = paths.amendment_baseline_csv
        if not amendment_csv.exists():
            logger.warning(
                "Amendment baseline CSV not found at %s; physical_activity "
                "written without gpaq_met_total.",
                amendment_csv,
            )
            return result
        amendment_df = _load_nako_csv(amendment_csv)

    gpaq = _extract_gpaq_activity(amendment_df)
    # Range check before the merge, so the indicator below counts what
    # actually survives rather than what was delivered: the guard nulls a
    # handful of values above the instrument's arithmetic ceiling, and a
    # gpaq_reported of 1 next to a missing MET total would be a lie.
    gpaq = _enforce_plausibility(gpaq, GPAQ_RANGES, module="physical_activity")
    merged = result.merge(gpaq[["ID", "gpaq_met_total"]], on="ID", how="left")
    merged["gpaq_reported"] = merged["gpaq_met_total"].notna().astype("UInt8")

    n_gpaq = int(merged["gpaq_met_total"].notna().sum())
    logger.info(
        "Physical activity: QUAP returned by %s participants (%.1f%%), "
        "GPAQ total present for %s (%.1f%%)",
        f"{int(result['pa_self_reported'].sum()):,}",
        float(result["pa_self_reported"].mean()) * 100,
        f"{n_gpaq:,}",
        n_gpaq / len(merged) * 100 if len(merged) else 0.0,
    )
    return merged


def process_physical_activity(
    paths: NAKOPaths,
    output_dir: Path,
    *,
    baseline_df: DataFrame | None = None,
    amendment_df: DataFrame | None = None,
) -> Path:
    """Process physical activity -> ``physical_activity.parquet``."""
    logger.info("Processing physical activity ...")
    if baseline_df is not None:
        df = baseline_df
    else:
        csv_path = paths.baseline_dir / "export_baseline.csv"
        df = _load_nako_csv(csv_path)
    result = _extract_physical_activity(df)
    result = _join_gpaq(result, paths, amendment_df=amendment_df)
    return _save_parquet(result, "physical_activity", output_dir)


# GPAQ is a second, parallel activity instrument in NAKO: QUAP and GPAQ were
# both administered, and ``a_gpaq_ptotalmet`` covers far more participants
# than the QUAP MET total (94 % versus ~35 % on the 2026-08-06 export). It
# arrives in the NAM-45 amendment delivery. The column feeds the
# ``physical_activity`` modality via ``_join_gpaq``; the standalone parquet
# below is kept so the two instruments can still be compared against each
# other without unpicking the merged frame.
GPAQ_MISSING_CODE: int = 777777
"""``a_gpaq_ptotalmet`` sentinel: computation impossible (missing subdomains).

Distinct from the ``7777`` used by the QUAP columns and absent from
``NAKO_SENTINEL_CODES``, so it has to be handled here.
"""


def _extract_gpaq_activity(df: DataFrame) -> DataFrame:
    """Extract the GPAQ total MET-minutes/week column."""
    if "a_gpaq_ptotalmet" not in df.columns:
        raise ValueError(
            "a_gpaq_ptotalmet absent from the export. Check that the "
            "amendment delivery is configured."
        )
    r = pd.DataFrame()
    r["ID"] = df["ID"]
    r["gpaq_met_total"] = df["a_gpaq_ptotalmet"].replace(GPAQ_MISSING_CODE, pd.NA)

    # The shared sweep on top of the 777777 special case. Every survey
    # sentinel (7775/7776/7777/8886/8888/8889) sits inside
    # ``GPAQ_MET_WEEK_RANGE``, so ``_enforce_plausibility`` would pass it
    # through and ``_join_gpaq`` would then record gpaq_reported = 1 for a
    # refusal code — entering the base learner as an above-average activity
    # level after log1p.
    #
    # Nothing legitimate is lost: GPAQ weights moderate activity at 4 METs and
    # vigorous at 8, over 10-minute blocks, so every honest total is a multiple
    # of 40. All 110,583 values the sweep sees on the 2026-08-06 export are
    # (110,564 survive the plausibility ceiling), and no NAKO sentinel code is.
    r = _replace_nako_missing(r)

    n_valid = int(r["gpaq_met_total"].notna().sum())
    logger.info(
        "GPAQ total MET: %s of %s participants (%.1f%%)",
        f"{n_valid:,}",
        f"{len(r):,}",
        n_valid / len(r) * 100 if len(r) else 0.0,
    )
    return r


def process_gpaq_activity(
    paths: NAKOPaths, output_dir: Path, *, amendment_df: DataFrame | None = None
) -> Path:
    """Process GPAQ total activity -> ``gpaq_activity.parquet``."""
    logger.info("Processing GPAQ physical activity ...")
    from pcc_analysis._plausibility import GPAQ_RANGES
    from pcc_analysis.data_processing._common import _enforce_plausibility

    df = (
        amendment_df
        if amendment_df is not None
        else _load_nako_csv(paths.amendment_baseline_csv)
    )
    result = _extract_gpaq_activity(df)
    result = _enforce_plausibility(result, GPAQ_RANGES, module="gpaq_activity")
    return _save_parquet(result, "gpaq_activity", output_dir)
