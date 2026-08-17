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
# screening, not the full 48-point TDI battery. The key outcome is
# ``olf_sum`` (sum of correctly identified odours, 0..12); a score of
# <= 6 is conventionally used to flag hyposmia/anosmia. The 10-item
# ``olf_frei`` captures free-recall identification without cueing and is
# administered only for the method code 2 subset. ``olf_schnupfen``
# records a cold/runny nose on the test day and is used as an
# adjustment covariate in the olfactory reporting-bias analysis.

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
    "olf_sum": "olf_identification_sum",
    "olf_rslt": "olf_identification_result",
    "olf_frei": "olf_free_identification",
    "olf_kat": "olf_category_identification",
    "olf_method": "olf_method_code",
    "olf_norm": "olf_normosmia_pass",
    "olf_schnupfen": "olf_cold_on_test_day",
}

# Hyposmia threshold for the 12-item NAKO Sniffin'-Sticks screening
# (Hummel 2001): sum <= 6 flags anosmia/hyposmia.
OLF_HYPOSMIA_CUTOFF: int = 6


def _extract_followup1_olfactometry(df: DataFrame) -> DataFrame:
    """Extract the FU1 olfactometry block and standardise column names."""
    available = [c for c in FU1_OLFACTOMETRY_COLUMNS if c in df.columns]
    r = df[available].copy()
    r = r.rename(columns=FU1_OLFACTOMETRY_COLUMN_RENAME)
    # Sentinel codes (7777 "not shown", -9 "n/a", 9 "don't know" on
    # olf_schnupfen) become NA via the shared NAKO-missing handler.
    r = _replace_nako_missing(r)
    # The 'don't know' 9 on olf_schnupfen is not in the shared sentinel
    # list because 9 is a legal code in other modules; handle it locally.
    if "olf_cold_on_test_day" in r.columns:
        r["olf_cold_on_test_day"] = r["olf_cold_on_test_day"].replace(9, pd.NA)
    return r


def _derive_olfactometry_indicators(df: DataFrame) -> DataFrame:
    """Add hyposmia/anosmia flags derived from the identification sum."""
    if "olf_identification_sum" in df.columns:
        sums = df["olf_identification_sum"]
        df["olf_hyposmia_flag"] = (
            (sums <= OLF_HYPOSMIA_CUTOFF).astype("boolean").mask(sums.isna())
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
    result = _derive_olfactometry_indicators(result)
    return _save_parquet(result, "followup1_olfactometry", output_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  Visit-timing proxy via GEFU1 basis_age
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
# OUR CONCRETE DATA ACCESS (NAKO-882 export) STRIPS THE DATE/YEAR
# FIELDS FROM FU1 FOR PRIVACY. ``export_followup1.csv`` only carries
# ``basis_uort``, ``basis_lvl``, ``basis_status_mrt`` — no date, no
# ``basis_age``, no ``basis_monthsdiff``. We therefore cannot compute
# FU1 visit timing directly from FU1 alone.
#
# PROXY CHOICE
# ------------
# The closest approximation available to us is the ``basis_age`` column
# in ``export_gefu1.csv`` (the Corona-1 mail-in questionnaire). Across
# the 117 434 FU1 participants it shows the following offset vs.
# baseline ``basis_age`` (observed 2026-04-24):
#
#   delta = gefu1.basis_age - baseline.basis_age
#   delta = 0 years:     21 participants   (<0.1 %)
#   delta = 1 year:     159                (0.1 %)
#   delta = 2 years: 78 941                (67.9 %)
#   delta = 3 years: 27 520                (23.7 %)
#   delta = 4..9 years: ~2 900             (2.5 %)
#   delta <= -1 year:    6 participants    (data-entry errors; set NA)
#
# So GEFU1 is administered ~2-3 years after baseline for the vast
# majority of participants. The CLINICAL FU1 examination (Sniffin'-
# Sticks, FeNO, vascular etc.) may be a separate event at a different
# calendar time; GEFU1 is the closest post-baseline age anchor we have.
# For pre-/post-infection classification at the COARSE level
# (baseline is deeply pre-pandemic 2014..2019; any follow-up is
# post-2020) the proxy is adequate. For FINE timing (e.g. "did the
# olfactory test fall within 12 months of the reported infection?")
# it is not adequate; such analyses must be deferred until NAKO
# releases the FU1 date fields.
#
# WHAT THIS PARQUET CONTAINS
# --------------------------
#   baseline_age                 UInt8     age at baseline visit (known)
#   gefu1_age_proxy              UInt8     age at GEFU1 questionnaire
#   years_since_baseline_proxy   Int8      gefu1_age_proxy - baseline_age
#
# Rows are restricted to the intersection of FU1 IDs, baseline IDs,
# and GEFU1 IDs. Rows with implausible negative deltas become NA on
# the derived ``years_since_baseline_proxy`` column (coerced by the
# shared plausibility enforcer); the raw ages are retained.


def _extract_followup1_visit_meta(
    baseline_df: DataFrame,
    gefu1_df: DataFrame,
    fu1_ids: pd.Index,
) -> DataFrame:
    """Join baseline and GEFU1 ages; restrict to FU1 IDs; derive delta."""
    bl = baseline_df[["ID", "basis_age"]].copy()
    bl = bl.rename(columns={"basis_age": "baseline_age"})

    gefu = gefu1_df[["ID", "basis_age"]].copy()
    gefu = gefu.rename(columns={"basis_age": "gefu1_age_proxy"})

    merged = bl.merge(gefu, on="ID", how="inner")
    merged = merged[merged["ID"].isin(fu1_ids)].reset_index(drop=True)

    merged = _replace_nako_missing(merged)
    merged["years_since_baseline_proxy"] = merged["gefu1_age_proxy"].astype(
        "Int64"
    ) - merged["baseline_age"].astype("Int64")
    return merged


def process_followup1_visit_meta(
    paths: NAKOPaths,
    output_dir: Path,
    *,
    baseline_df: DataFrame | None = None,
    gefu1_df: DataFrame | None = None,
    followup1_df: DataFrame | None = None,
) -> Path:
    """Process FU1 visit-timing proxy -> ``followup1_visit_meta.parquet``.

    Uses GEFU1 ``basis_age`` as a proxy anchor because the FU1 export
    lacks its own age field (stripped for privacy). See the module
    block comment above for caveats. Callers needing precise FU1
    visit timing should treat this parquet as a **coarse pre/post-
    baseline indicator only**.
    """
    logger.info("Processing FU1 visit-meta proxy (via GEFU1 basis_age) ...")

    if baseline_df is None:
        baseline_df = _load_nako_csv(paths.baseline_dir / "export_baseline.csv")
    if gefu1_df is None:
        gefu1_df = _load_nako_csv(paths.gefu1_csv)
    if followup1_df is None:
        followup1_df = _load_nako_csv(paths.followup1_csv)

    fu1_ids = pd.Index(followup1_df["ID"].dropna().unique())
    result = _extract_followup1_visit_meta(baseline_df, gefu1_df, fu1_ids)
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
