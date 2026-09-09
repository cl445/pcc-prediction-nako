"""Follow-up 1 (FU1) processors: olfactometry, visit-meta, lab, sleep, FeNO, misc.

All FU1 processors share the same ``export_followup1.csv`` source file and
are grouped here for that reason.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis._plausibility import (
    FENO_RANGES,
    FOLLOWUP1_VISIT_META_RANGES,
    FU1_LAB_RANGES,
    FU1_MARKERS_MISC_RANGES,
    OLFACTOMETRY_RANGES,
    SLEEP_OBJECTIVE_RANGES,
)
from pcc_analysis.data_processing._common import (
    NAKO_NA_VALUES,
    _enforce_plausibility,
    _load_nako_csv,
    _replace_nako_missing,
    _save_parquet,
)

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Olfactometry (12-item Sniffin'-Sticks)
# ═══════════════════════════════════════════════════════════════════════════
#
# NAKO-882 administers a 12-item identification-only Sniffin'-Sticks
# screening, not the full 48-point TDI battery, and it ships its own
# clinical reading of that screening. Taking the raw sum at face value
# gets the domain wrong three times over, so the coding is spelled out
# here against the data dictionary (dd_Nako_882_20260410.xlsx ->
# Variablen / Codierung, "Geruchstestung").
#
# THE SUM IS NOT ALWAYS A TEST RESULT
# -----------------------------------
# ``olf_sum`` and ``olf_rslt`` carry a literal 0 for participants whose
# screening could not be evaluated, and 0 is not in the sentinel list, so
# nothing sweeps it. NAKO marks those rows in ``olf_kat`` and
# ``olf_norm`` with 7777, "Berechnung nicht möglich wegen fehlender oder
# unplausibler Daten". Observed 2026-08-24: 17 568 rows carry a zero sum
# and 17 559 of them are 7777 rows, leaving 9 genuine zeros. A zero is
# implausible as a result in any case, the screening being forced choice
# from four options, where guessing alone returns about three. The scores
# are therefore set to NA wherever NAKO declines to classify the row.
#
# NAKO ALREADY CLASSIFIES OLFACTORY FUNCTION
# ------------------------------------------
# ``olf_kat`` is 1 Normosmie / 2 Hyposmie / 3 Anosmie, derived by NAKO
# from ``olf_rslt``: normosmia at 10..12, hyposmia at 7..9, anosmia at
# 0..6 (verified against the delivered data, no row crosses a boundary).
# The flags below come from that classification rather than from a
# threshold of our own, so the parquet agrees with anything else computed
# from NAKO's reading. Note where the conventional "<= 6" cutoff actually
# falls: it is NAKO's anosmia boundary, not its hyposmia boundary.
#
# TWO COLUMNS DO NOT MEAN WHAT THEIR NAMES SUGGEST
# ------------------------------------------------
# ``olf_frei`` is not free-recall identification; it is "Wie frei ist Ihre
# Atmung durch die Nase im Moment?", 1 völlig frei .. 10 völlig verstopft,
# and it is the better congestion covariate of the two. ``olf_schnupfen``
# asks about a cold within the past six weeks rather than on the test day,
# coded 0 nein / 1 ja / 9 keine Angabe.

FU1_OLFACTOMETRY_COLUMNS: list[str] = [
    "ID",
    "olf_frei",
    "olf_kat",
    "olf_method",
    "olf_norm",
    "olf_rslt",
    "olf_schnupfen",
    "olf_sum",
]

FU1_OLFACTOMETRY_COLUMN_RENAME: dict[str, str] = {
    # Pen 6 is scored as one correct answer in olf_sum and as two in
    # olf_rslt; NAKO's own classification runs off the lenient variant.
    "olf_sum": "olf_identification_sum",
    "olf_rslt": "olf_identification_sum_lenient",
    "olf_frei": "olf_nasal_patency",
    "olf_kat": "olf_function_category",
    "olf_method": "olf_method_code",
    "olf_norm": "olf_normosmia",
    "olf_schnupfen": "olf_cold_recent",
}

# olf_kat levels, in NAKO's coding.
OLF_CATEGORY_NORMOSMIA: int = 1
OLF_CATEGORY_HYPOSMIA: int = 2
OLF_CATEGORY_ANOSMIA: int = 3

# The score columns invalidated alongside an unclassifiable row.
OLF_SCORE_COLUMNS: tuple[str, ...] = (
    "olf_identification_sum",
    "olf_identification_sum_lenient",
)


def _extract_followup1_olfactometry(df: DataFrame) -> DataFrame:
    """Extract the FU1 olfactometry block and standardise column names."""
    available = [c for c in FU1_OLFACTOMETRY_COLUMNS if c in df.columns]
    r = df[available].copy()
    r = r.rename(columns=FU1_OLFACTOMETRY_COLUMN_RENAME)
    # Sentinel codes (7777 "not computable", -9 "not documented") become NA
    # via the shared NAKO-missing handler.
    r = _replace_nako_missing(r)
    # The "keine Angabe" 9 on olf_schnupfen is not in the shared sentinel
    # list because 9 is a legal code in other modules; handle it locally.
    if "olf_cold_recent" in r.columns:
        r["olf_cold_recent"] = r["olf_cold_recent"].replace(9, pd.NA)
    return r


def _invalidate_unclassifiable_scores(df: DataFrame) -> DataFrame:
    """Blank the sums NAKO declined to classify.

    The 7777 that says so lives in ``olf_kat``, not in the score columns,
    which is why the sweep leaves a literal zero standing in a field that
    otherwise holds a test result.
    """
    if "olf_function_category" not in df.columns:
        return df
    unclassified = df["olf_function_category"].isna()
    for column in OLF_SCORE_COLUMNS:
        if column in df.columns:
            blanked = int((unclassified & df[column].notna()).sum())
            if blanked:
                logger.info(
                    "Olfactometry: %s zeroed on %s unclassifiable row(s); set to NA",
                    column,
                    f"{blanked:,}",
                )
            df[column] = df[column].mask(unclassified)
    return df


def _derive_olfactometry_indicators(df: DataFrame) -> DataFrame:
    """Add hyposmia/anosmia flags from NAKO's own olfactory classification."""
    if "olf_function_category" not in df.columns:
        return df
    category = df["olf_function_category"]
    df["olf_hyposmia_flag"] = (
        (category >= OLF_CATEGORY_HYPOSMIA).astype("boolean").mask(category.isna())
    )
    df["olf_anosmia_flag"] = (
        (category == OLF_CATEGORY_ANOSMIA).astype("boolean").mask(category.isna())
    )
    return df


def process_followup1_olfactometry(
    paths: NAKOPaths, output_dir: Path, *, followup1_df: DataFrame | None = None
) -> Path:
    """Process FU1 olfactometry -> ``followup1_olfactometry.parquet``."""
    logger.info("Processing FU1 olfactometry ...")
    if followup1_df is not None:
        df = followup1_df
    else:
        csv_path = paths.followup1_csv
        df = _load_nako_csv(csv_path)
    result = _extract_followup1_olfactometry(df)
    result = _enforce_plausibility(
        result, OLFACTOMETRY_RANGES, module="followup1_olfactometry"
    )
    # Order matters: the scores are blanked before anything is derived, so
    # no flag can be built from a zero that stands for an absent test.
    result = _invalidate_unclassifiable_scores(result)
    result = _derive_olfactometry_indicators(result)
    return _save_parquet(result, "followup1_olfactometry", output_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  Visit timing: reconstructed FU1 year, plus a GEFU1 age proxy
# ═══════════════════════════════════════════════════════════════════════════
#
# WHY THIS EXISTS
# ---------------
# The NAKO data model declares, for each survey wave (baseline, GEFU1,
# Corona-2, FU1, ...), a family of per-visit ``basis_*`` variables
# including ``basis_udat`` (examination date), ``basis_year``,
# ``basis_monthsdiff`` (months since baseline), and ``basis_age`` (age
# at that wave's examination date). These are documented in
# ``dd_Nako_882_20260410.xlsx`` -> Basisdaten.
#
# Which of them survive redaction differs BETWEEN THE TWO DELIVERIES,
# and that difference is the whole reason this module has two timing
# columns rather than one:
#
#   supplementary (2026-04-10) export_followup1.csv
#       117 434 rows, carries every olf_/sa_/sow_ measurement block,
#       but only basis_uort / basis_lvl / basis_status_mrt. No age,
#       no date. This is the file all measurement processors read.
#   primary (2025-02-17) export_followup1.csv
#       51 587 rows, carries basis_age at the FU1 visit.
#
# RECONSTRUCTED FU1 YEAR (preferred)
# ----------------------------------
# For participants present in the primary delivery the FU1 examination
# year follows from three fully populated fields:
#
#   fu1_year_reconstructed = basis_year + (fu1_age - baseline_age)
#
# with basis_year from the supplementary baseline export (2014..2019)
# and both ages from their respective visits. Observed 2026-08-24 the
# baseline-to-FU1 delta is 3..8 years (mean 4.46), and the resulting
# years fall in 2017..2023 with the mass on 2019..2021.
#
# Resolution is one calendar year: the inputs are integer ages, so a
# visit late in one year and early in the next are indistinguishable.
# That is enough to place the FU1 examination before or after a
# reported infection month, which is what the olfactory objective-anchor
# analysis needs, and not enough for "within N months of infection".
#
# COVERAGE LIMIT: the reconstruction is available for the 51 587
# participants of the primary delivery, not for all 117 434 FU1
# participants. Analyses using it must report that restriction.
#
# GEFU1 AGE PROXY (fallback, retained)
# ------------------------------------
# ``basis_age`` in ``export_gefu1.csv`` (the Corona-1 mail-in
# questionnaire) anchors a post-baseline age for a wider set of
# participants. Across the 117 434 FU1 participants the offset vs.
# baseline ``basis_age`` was (observed 2026-04-24):
#
#   delta = gefu1.basis_age - baseline.basis_age
#   delta = 0 years:     21 participants   (<0.1 %)
#   delta = 1 year:     159                (0.1 %)
#   delta = 2 years: 78 941                (67.9 %)
#   delta = 3 years: 27 520                (23.7 %)
#   delta = 4..9 years: ~2 900             (2.5 %)
#   delta <= -1 year:    6 participants    (data-entry errors; set NA)
#
# GEFU1 is therefore administered ~2-3 years after baseline. It dates
# the QUESTIONNAIRE, not the clinical FU1 examination, which is a
# separate event; treat it as a coarse pre/post-baseline indicator only
# and prefer ``fu1_year_reconstructed`` wherever it is populated.
#
# WHAT THIS PARQUET CONTAINS
# --------------------------
#   baseline_age                 UInt8     age at baseline visit
#   gefu1_age_proxy              UInt8     age at GEFU1 questionnaire
#   years_since_baseline_proxy   Int8      gefu1_age_proxy - baseline_age
#   fu1_age                      UInt8     age at the FU1 visit
#   years_baseline_to_fu1        Int8      fu1_age - baseline_age
#   fu1_year_reconstructed       Int16     basis_year + years_baseline_to_fu1
#
# Rows are restricted to the intersection of FU1 IDs, baseline IDs,
# and GEFU1 IDs; the last three columns are additionally NA outside the
# primary delivery. Implausible deltas become NA on the derived columns
# (coerced by the shared plausibility enforcer); raw ages are retained.


def _load_visit_anchor(csv_path: Path, column: str) -> DataFrame | None:
    """Read ``ID`` plus one timing column, or ``None`` when unavailable.

    Both anchors are optional: the primary FU1 delivery and the
    supplementary baseline export are absent on smoke configurations, and
    a delivery may be re-cut without the column. Returning ``None`` lets
    the visit-meta parquet fall back to the GEFU1 proxy alone instead of
    failing the whole run.
    """
    if not csv_path.exists():
        logger.warning("Visit anchor %s not found at %s", column, csv_path)
        return None
    wanted = {"ID", column}

    def _keep(name: str) -> bool:
        return name in wanted

    df = pd.read_csv(
        csv_path,
        sep=";",
        usecols=_keep,
        na_values=NAKO_NA_VALUES,
        keep_default_na=True,
        dtype_backend="numpy_nullable",
        low_memory=False,
    )
    if column not in df.columns:
        logger.warning("Column %r absent from %s", column, csv_path)
        return None
    logger.info("Visit anchor %s: %s rows from %s", column, f"{len(df):,}", csv_path)
    return df


def _extract_followup1_visit_meta(
    baseline_df: DataFrame,
    gefu1_df: DataFrame,
    fu1_ids: pd.Index,
    *,
    fu1_age_df: DataFrame | None = None,
    baseline_year_df: DataFrame | None = None,
) -> DataFrame:
    """Join baseline, GEFU1 and FU1 ages; restrict to FU1 IDs; derive deltas."""
    bl = baseline_df[["ID", "basis_age"]].copy()
    bl = bl.rename(columns={"basis_age": "baseline_age"})

    gefu = gefu1_df[["ID", "basis_age"]].copy()
    gefu = gefu.rename(columns={"basis_age": "gefu1_age_proxy"})

    merged = bl.merge(gefu, on="ID", how="inner")
    merged = merged[merged["ID"].isin(fu1_ids)].reset_index(drop=True)

    if fu1_age_df is not None:
        fu1 = fu1_age_df[["ID", "basis_age"]].rename(columns={"basis_age": "fu1_age"})
        merged = merged.merge(fu1, on="ID", how="left")
    if baseline_year_df is not None:
        merged = merged.merge(
            baseline_year_df[["ID", "basis_year"]], on="ID", how="left"
        )

    merged = _replace_nako_missing(merged)
    merged["years_since_baseline_proxy"] = merged["gefu1_age_proxy"].astype(
        "Int64"
    ) - merged["baseline_age"].astype("Int64")

    if "fu1_age" in merged.columns:
        merged["years_baseline_to_fu1"] = merged["fu1_age"].astype("Int64") - merged[
            "baseline_age"
        ].astype("Int64")
        if "basis_year" in merged.columns:
            merged["fu1_year_reconstructed"] = (
                merged["basis_year"].astype("Int64") + merged["years_baseline_to_fu1"]
            )
    return merged.drop(columns=["basis_year"], errors="ignore")


def process_followup1_visit_meta(
    paths: NAKOPaths,
    output_dir: Path,
    *,
    baseline_df: DataFrame | None = None,
    gefu1_df: DataFrame | None = None,
    followup1_df: DataFrame | None = None,
) -> Path:
    """Process FU1 visit timing -> ``followup1_visit_meta.parquet``.

    Emits two anchors of different quality, both documented in the module
    block comment above. ``fu1_year_reconstructed`` dates the FU1
    examination to the calendar year and is the one to use, but covers
    only the participants of the primary delivery. ``gefu1_age_proxy``
    dates the Corona-1 questionnaire instead of the examination and is a
    **coarse pre/post-baseline indicator only**; it is retained because it
    reaches participants the reconstruction cannot.
    """
    logger.info("Processing FU1 visit timing ...")

    if baseline_df is None:
        baseline_df = _load_nako_csv(paths.baseline_dir / "export_baseline.csv")
    if gefu1_df is None:
        gefu1_df = _load_nako_csv(paths.gefu1_csv)
    if followup1_df is None:
        followup1_df = _load_nako_csv(paths.followup1_csv)

    fu1_ids = pd.Index(followup1_df["ID"].dropna().unique())
    result = _extract_followup1_visit_meta(
        baseline_df,
        gefu1_df,
        fu1_ids,
        fu1_age_df=_load_visit_anchor(paths.primary_followup1_csv, "basis_age"),
        baseline_year_df=_load_visit_anchor(
            paths.supplementary_baseline_csv, "basis_year"
        ),
    )
    result = _enforce_plausibility(
        result, FOLLOWUP1_VISIT_META_RANGES, module="followup1_visit_meta"
    )
    return _save_parquet(result, "followup1_visit_meta", output_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  Laboratory panel (sa_* block)
# ═══════════════════════════════════════════════════════════════════════════
#
# NAKO FU1 provides 56 ``sa_*`` (Sofortanalyse) laboratory variables per
# participant. We extract the clinically-interpretable subset with
# documented units and plausibility ranges; the dense haematology panel
# (differential WBC, red-cell indices, platelet indices) is included for
# the biological-mediator analyses.
#
# Units (NAKO German clinical convention):
#   sa_hscrp      -> mg/L
#   sa_hba1c      -> mmol/mol (IFCC; not %; range ~20..150)
#   sa_gluk       -> mmol/L
#   sa_chol/hdl/ldl/trig -> mmol/L
#   sa_crea       -> µmol/L (serum)
#   sa_cysc       -> mg/L
#   sa_alat/asat  -> µkat/L (1 U/L ≈ 0.01667 µkat/L; confirms the
#                   Phase 0.5 "ALT looks wrong" finding for baseline)
#   sa_tsh        -> mU/L
#   sa_urate      -> µmol/L
#   sa_urea       -> mmol/L
#   sa_hb         -> mmol/L (1 mmol/L ≈ 1.61 g/dL)
#   sa_plt, sa_wbc -> Gpt/L (= 10⁹/L)
#   sa_rbc        -> Tpt/L (= 10¹²/L)

FU1_LAB_COLUMN_RENAME: dict[str, str] = {
    "sa_hscrp": "hscrp",
    "sa_hba1c": "hba1c_ifcc",
    "sa_hba1crel": "hba1c_relative",
    "sa_gluk": "glucose",
    "sa_chol": "cholesterol_total",
    "sa_hdlc": "hdl_cholesterol",
    "sa_ldlc": "ldl_cholesterol",
    "sa_ldlc_calc": "ldl_cholesterol_friedewald",
    "sa_nonhdlc_calc": "non_hdl_cholesterol",
    "sa_trig": "triglycerides",
    "sa_crea": "creatinine_serum",
    "sa_crea_u": "creatinine_urine",
    "sa_alb": "albumin_serum",
    "sa_alb_u": "albumin_urine",
    "sa_acr_u": "albumin_creatinine_ratio_urine",
    "sa_cysc": "cystatin_c",
    "sa_alat": "alt",
    "sa_asat": "ast",
    "sa_ggt": "ggt",
    "sa_ap": "alkaline_phosphatase",
    "sa_bilid": "bilirubin_direct",
    "sa_bilit": "bilirubin_total",
    "sa_ldh": "ldh",
    "sa_tsh": "tsh",
    "sa_ft3": "free_t3",
    "sa_ft4": "free_t4",
    "sa_urate": "urate",
    "sa_urate_u": "urate_urine",
    "sa_urea": "urea",
    "sa_prot": "total_protein",
    "sa_che": "cholinesterase",
    "sa_na": "sodium",
    "sa_cl": "chloride",
    "sa_pot": "potassium",
    "sa_ca": "calcium",
    "sa_mg": "magnesium",
    "sa_hb": "haemoglobin",
    "sa_hk": "haematocrit",
    "sa_plt": "platelets",
    "sa_wbc": "wbc",
    "sa_rbc": "rbc",
    "sa_ne": "neutrophils",
    "sa_ly": "lymphocytes",
    "sa_mo": "monocytes",
    "sa_eo": "eosinophils",
    "sa_ba": "basophils",
    "sa_mch": "mch",
    "sa_mchc": "mchc",
    "sa_mcv": "mcv",
    "sa_mpv": "mpv",
    "sa_pdw": "pdw",
    "sa_rdw": "rdw",
    "sa_lipa": "lipase",
    "sa_tk": "tk",
    "sa_egfr": "egfr",
}


def _extract_followup1_lab_values(df: DataFrame) -> DataFrame:
    """Extract FU1 sa_* lab markers and rename to canonical names."""
    available = [c for c in FU1_LAB_COLUMN_RENAME if c in df.columns]
    out = df[["ID", *available]].copy()
    out = out.rename(columns={c: FU1_LAB_COLUMN_RENAME[c] for c in available})
    out = _replace_nako_missing(out)
    return out


def process_followup1_lab_values(
    paths: NAKOPaths, output_dir: Path, *, followup1_df: DataFrame | None = None
) -> Path:
    """Process FU1 laboratory panel -> ``followup1_lab_values.parquet``."""
    logger.info("Processing FU1 laboratory panel ...")
    if followup1_df is not None:
        df = followup1_df
    else:
        df = _load_nako_csv(paths.followup1_csv)
    result = _extract_followup1_lab_values(df)
    result = _enforce_plausibility(
        result, FU1_LAB_RANGES, module="followup1_lab_values"
    )
    return _save_parquet(result, "followup1_lab_values", output_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  SOMNOwatch objective sleep
# ═══════════════════════════════════════════════════════════════════════════
#
# ``sow_brief_*`` are SOMNOwatch result-letter values aggregated across
# the monitored nights; -1 is a sleep-module-specific missing marker
# that is not in the shared NAKO_SENTINEL_CODES list (it is a legal
# value in other modules such as NTC age-of-onset distributions),
# so we coerce it locally here.

FU1_SLEEP_COLUMN_RENAME: dict[str, str] = {
    "sow_brief_tst": "sleep_tst_hours",
    "sow_brief_waso": "sleep_waso_hours",
    "sow_brief_effiz": "sleep_efficiency_pct",
    "sow_brief_naso": "sleep_n_awakenings",
}
FU1_SLEEP_LOCAL_MISSING: int = -1


def _extract_followup1_sleep_objective(df: DataFrame) -> DataFrame:
    available = [c for c in FU1_SLEEP_COLUMN_RENAME if c in df.columns]
    out = df[["ID", *available]].copy()
    out = out.rename(columns={c: FU1_SLEEP_COLUMN_RENAME[c] for c in available})
    out = _replace_nako_missing(out)
    for col in out.columns:
        if col == "ID":
            continue
        out[col] = out[col].replace(FU1_SLEEP_LOCAL_MISSING, pd.NA)
    return out


def process_followup1_sleep_objective(
    paths: NAKOPaths, output_dir: Path, *, followup1_df: DataFrame | None = None
) -> Path:
    """Process FU1 objective sleep -> ``followup1_sleep_objective.parquet``."""
    logger.info("Processing FU1 objective sleep (SOMNOwatch) ...")
    if followup1_df is not None:
        df = followup1_df
    else:
        df = _load_nako_csv(paths.followup1_csv)
    result = _extract_followup1_sleep_objective(df)
    result = _enforce_plausibility(
        result, SLEEP_OBJECTIVE_RANGES, module="followup1_sleep_objective"
    )
    return _save_parquet(result, "followup1_sleep_objective", output_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  Fractional exhaled nitric oxide (FeNO)
# ═══════════════════════════════════════════════════════════════════════════
#
# Single-value airway-inflammation marker; > 50 ppb is the clinical
# cut-off for elevated/eosinophilic inflammation (ATS 2011).

FENO_ELEVATED_CUTOFF_PPB: float = 50.0


def _extract_followup1_feno(df: DataFrame) -> DataFrame:
    if "hp_feno_wert" not in df.columns:
        return pd.DataFrame({"ID": df["ID"]})
    out = df[["ID", "hp_feno_wert"]].rename(columns={"hp_feno_wert": "feno_ppb"})
    out = _replace_nako_missing(out)
    return out


def _derive_feno_flags(df: DataFrame) -> DataFrame:
    if "feno_ppb" in df.columns:
        values = df["feno_ppb"]
        df["feno_elevated_flag"] = (
            (values > FENO_ELEVATED_CUTOFF_PPB).astype("boolean").mask(values.isna())
        )
    return df


def process_followup1_feno(
    paths: NAKOPaths, output_dir: Path, *, followup1_df: DataFrame | None = None
) -> Path:
    """Process FU1 FeNO -> ``followup1_feno.parquet``."""
    logger.info("Processing FU1 FeNO ...")
    if followup1_df is not None:
        df = followup1_df
    else:
        df = _load_nako_csv(paths.followup1_csv)
    result = _extract_followup1_feno(df)
    result = _enforce_plausibility(result, FENO_RANGES, module="followup1_feno")
    result = _derive_feno_flags(result)
    return _save_parquet(result, "followup1_feno", output_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  Miscellaneous markers (AGE-Reader + teeth)
# ═══════════════════════════════════════════════════════════════════════════
#
# Skin autofluorescence (advanced glycation end-products, oxidative-
# stress proxy) and dental-status counts. Grouped into one parquet
# because each is a single-value marker too small to justify its own
# module.

FU1_MARKERS_MISC_COLUMN_RENAME: dict[str, str] = {
    "hp_age_afr": "skin_age_autofluorescence",
    "hp_age_refl": "skin_age_reflection",
    "d_zz_zz": "n_natural_teeth",
    "d_zz_pz": "n_prostheses",
}


def _extract_followup1_markers_misc(df: DataFrame) -> DataFrame:
    available = [c for c in FU1_MARKERS_MISC_COLUMN_RENAME if c in df.columns]
    out = df[["ID", *available]].copy()
    out = out.rename(columns={c: FU1_MARKERS_MISC_COLUMN_RENAME[c] for c in available})
    out = _replace_nako_missing(out)
    return out


def process_followup1_markers_misc(
    paths: NAKOPaths, output_dir: Path, *, followup1_df: DataFrame | None = None
) -> Path:
    """Process FU1 miscellaneous markers -> ``followup1_markers_misc.parquet``."""
    logger.info("Processing FU1 miscellaneous markers (AGE-Reader + teeth) ...")
    if followup1_df is not None:
        df = followup1_df
    else:
        df = _load_nako_csv(paths.followup1_csv)
    result = _extract_followup1_markers_misc(df)
    result = _enforce_plausibility(
        result, FU1_MARKERS_MISC_RANGES, module="followup1_markers_misc"
    )
    return _save_parquet(result, "followup1_markers_misc", output_dir)
