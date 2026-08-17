"""Plausibility ranges for NAKO-derived measurements.

Each constant is a ``(min, max)`` closed interval on the physically/clinically
plausible domain of a measurement. Values outside the range are coerced to
``pd.NA`` by :func:`pcc_analysis.data_processing._enforce_plausibility` and
counted in the processing log.

Ranges are deliberately wide — they are a guard against data-entry and
decoding errors, not against genuine but unusual values. Where a clinical
threshold exists (e.g.\\ anosmia at TDI < 30.5) it is applied downstream as
a derived feature, not as a plausibility filter.

References are inline per constant where a canonical threshold exists.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Mental-health screeners
# ---------------------------------------------------------------------------
# PHQ-9 sum: 9 items × 0..3 (Kroenke 2001). Individual items are 0..3.
PHQ9_SUM_RANGE: tuple[float, float] = (0.0, 27.0)
PHQ9_ITEM_RANGE: tuple[float, float] = (0.0, 3.0)

# GAD-7 sum: 7 items × 0..3 (Spitzer 2006). Individual items are 0..3.
GAD7_SUM_RANGE: tuple[float, float] = (0.0, 21.0)
GAD7_ITEM_RANGE: tuple[float, float] = (0.0, 3.0)

# MINI major-depression screen: binary 0/1.
MINI_BINARY_RANGE: tuple[float, float] = (0.0, 1.0)

# ---------------------------------------------------------------------------
# Demographics / anthropometrics
# ---------------------------------------------------------------------------
# NAKO baseline enrolment: 19..75 years (design); leave a safety margin.
AGE_YEARS_RANGE: tuple[float, float] = (18.0, 90.0)

# BMI: WHO extremes span ~10..70; ICU-survivor cohorts rarely exceed 70.
BMI_RANGE: tuple[float, float] = (10.0, 70.0)
HEIGHT_CM_RANGE: tuple[float, float] = (120.0, 220.0)
WEIGHT_KG_RANGE: tuple[float, float] = (30.0, 250.0)

# ---------------------------------------------------------------------------
# Cardiovascular measurements
# ---------------------------------------------------------------------------
SBP_MMHG_RANGE: tuple[float, float] = (70.0, 250.0)
DBP_MMHG_RANGE: tuple[float, float] = (40.0, 150.0)
HEART_RATE_RANGE: tuple[float, float] = (30.0, 220.0)
# Pulse-wave velocity (Vicorder): physiological 3..20 m/s, extreme stiffness up to 30.
PWV_MS_RANGE: tuple[float, float] = (3.0, 30.0)
# Augmentation index: typical -50..80 %.
AUG_INDEX_PCT_RANGE: tuple[float, float] = (-50.0, 80.0)
# Ankle-brachial index: 0.3 (critical ischaemia) .. 1.6 (incompressible arteries).
ABI_RANGE: tuple[float, float] = (0.3, 1.6)

# ---------------------------------------------------------------------------
# Lung function (spirometry)
# ---------------------------------------------------------------------------
FEV1_L_RANGE: tuple[float, float] = (0.3, 8.0)
FVC_L_RANGE: tuple[float, float] = (0.3, 10.0)
# Percent-predicted for adults: rarely outside 20..200 %.
PCT_PREDICTED_RANGE: tuple[float, float] = (20.0, 200.0)
FEV1_FVC_RATIO_RANGE: tuple[float, float] = (0.1, 1.0)
# NAKO stores peak-flow and mid-expiratory flow values in L/s, not L/min.
PEAK_FLOW_L_PER_SECOND_RANGE: tuple[float, float] = (0.5, 20.0)
FEF_L_PER_SECOND_RANGE: tuple[float, float] = (0.1, 16.0)
PEAK_FLOW_L_PER_MIN_RANGE: tuple[float, float] = (
    50.0,
    900.0,
)

# ---------------------------------------------------------------------------
# Laboratory values (clinical reference envelope, not ref-interval)
# ---------------------------------------------------------------------------
CRP_MG_PER_L_RANGE: tuple[float, float] = (0.01, 300.0)
HSCRP_MG_PER_L_RANGE: tuple[float, float] = (0.01, 300.0)
CHOLESTEROL_MMOL_PER_L_RANGE: tuple[float, float] = (1.0, 15.0)
HDL_MMOL_PER_L_RANGE: tuple[float, float] = (0.2, 5.0)
LDL_MMOL_PER_L_RANGE: tuple[float, float] = (0.2, 12.0)
TRIGLYCERIDES_MMOL_PER_L_RANGE: tuple[float, float] = (0.2, 30.0)
GLUCOSE_MMOL_PER_L_RANGE: tuple[float, float] = (1.0, 40.0)
HBA1C_PCT_RANGE: tuple[float, float] = (3.0, 18.0)
CREATININE_UMOL_PER_L_RANGE: tuple[float, float] = (20.0, 2000.0)
EGFR_ML_PER_MIN_RANGE: tuple[float, float] = (0.0, 200.0)
UREA_MMOL_PER_L_RANGE: tuple[float, float] = (1.0, 100.0)
ALT_U_PER_L_RANGE: tuple[float, float] = (1.0, 5000.0)
# NAKO reports transaminases in µkat/L (German clinical convention).
# 1 U/L ≈ 0.01667 µkat/L; typical adult ranges are 0.05–0.9 µkat/L.
ALT_UKAT_PER_L_RANGE: tuple[float, float] = (0.01, 80.0)
AST_UKAT_PER_L_RANGE: tuple[float, float] = (0.01, 80.0)
# HbA1c in mmol/mol (IFCC) rather than %. Adult range 20..150 covers
# healthy subjects through severe poorly-controlled diabetes.
HBA1C_MMOL_PER_MOL_RANGE: tuple[float, float] = (20.0, 150.0)
# Thyroid-stimulating hormone in mU/L.
TSH_MU_PER_L_RANGE: tuple[float, float] = (0.01, 100.0)
# Cystatin C in mg/L.
CYSC_MG_PER_L_RANGE: tuple[float, float] = (0.3, 10.0)
# Uric acid in µmol/L (German convention).
URATE_UMOL_PER_L_RANGE: tuple[float, float] = (50.0, 1200.0)
# Haemoglobin in mmol/L (German convention; 1 mmol/L ≈ 1.611 g/dL).
HB_MMOL_PER_L_RANGE: tuple[float, float] = (3.0, 15.0)
# Platelets / white-blood cells in Gpt/L (= 10⁹/L).
PLATELETS_GPT_PER_L_RANGE: tuple[float, float] = (20.0, 1500.0)
WBC_GPT_PER_L_RANGE: tuple[float, float] = (1.0, 100.0)
RBC_TPT_PER_L_RANGE: tuple[float, float] = (2.0, 8.0)
# Free thyroxine in pmol/L (German convention).
FT4_PMOL_PER_L_RANGE: tuple[float, float] = (1.0, 100.0)
# γ-Glutamyl transferase in µkat/L (German convention; 1 U/L ≈ 0.01667 µkat/L).
GGT_UKAT_PER_L_RANGE: tuple[float, float] = (0.01, 200.0)
# Serum sodium in mmol/L (clinically tolerated 110..170).
SODIUM_MMOL_PER_L_RANGE: tuple[float, float] = (110.0, 170.0)
# Serum potassium in mmol/L (clinically tolerated 1.5..9).
POTASSIUM_MMOL_PER_L_RANGE: tuple[float, float] = (1.5, 9.0)
# Serum calcium in mmol/L (clinically tolerated 1.5..4).
CALCIUM_MMOL_PER_L_RANGE: tuple[float, float] = (1.5, 4.0)
# Total bilirubin in µmol/L (clinically tolerated 1..600).
BILIRUBIN_UMOL_PER_L_RANGE: tuple[float, float] = (1.0, 600.0)
# Serum albumin in g/L (German convention; 1 g/dL = 10 g/L).
ALBUMIN_G_PER_L_RANGE: tuple[float, float] = (10.0, 80.0)
# Alkaline phosphatase in µkat/L (German convention).
ALP_UKAT_PER_L_RANGE: tuple[float, float] = (0.05, 50.0)
# Lactate dehydrogenase in µkat/L.
LDH_UKAT_PER_L_RANGE: tuple[float, float] = (0.5, 100.0)
# Free triiodothyronine in pmol/L.
FT3_PMOL_PER_L_RANGE: tuple[float, float] = (1.0, 50.0)
# Total protein in g/L.
TOTAL_PROTEIN_G_PER_L_RANGE: tuple[float, float] = (30.0, 120.0)
# Cholinesterase in µkat/L.
CHE_UKAT_PER_L_RANGE: tuple[float, float] = (5.0, 500.0)
# Serum chloride in mmol/L.
CHLORIDE_MMOL_PER_L_RANGE: tuple[float, float] = (70.0, 130.0)
# Serum magnesium in mmol/L.
MAGNESIUM_MMOL_PER_L_RANGE: tuple[float, float] = (0.3, 2.0)
# Lipase in µkat/L (clinically tolerated up to ~25, severe pancreatitis to ~100).
LIPASE_UKAT_PER_L_RANGE: tuple[float, float] = (0.05, 200.0)

# ---------------------------------------------------------------------------
# Sniffin'-Sticks olfactometry (NAKO 12-item screening form)
# ---------------------------------------------------------------------------
# NAKO-882 administers the 12-item Sniffin'-Sticks identification screening
# (Hummel 2001), not the full 48-point TDI battery. The sum score is 0..12;
# anosmia is indicated at sum <= 6 (downstream clinical derivation).
OLF_IDENTIFICATION_SUM_RANGE: tuple[float, float] = (0.0, 12.0)
# Free-identification count (which odours the participant named without cueing)
# is bounded by the test design.
OLF_FREE_IDENTIFICATION_RANGE: tuple[float, float] = (0.0, 10.0)
# Category-identification items (multi-choice answer).
OLF_CATEGORY_IDENTIFICATION_RANGE: tuple[float, float] = (0.0, 3.0)
# Administration method code (test-type identifier).
OLF_METHOD_CODE_RANGE: tuple[float, float] = (1.0, 2.0)
# Cold on the test day: binary 0/1.
OLF_COLD_RANGE: tuple[float, float] = (0.0, 1.0)
# Normosmia pass/fail derived by the exam software: binary 0/1.
OLF_NORMOSMIA_PASS_RANGE: tuple[float, float] = (0.0, 1.0)

# ---------------------------------------------------------------------------
# Objective sleep (SOMNOwatch — "brief" export values are in hours)
# ---------------------------------------------------------------------------
# NAKO ``sow_brief_*`` are the result-letter-format values derived per
# monitored night: TST and WASO in hours, efficiency in percent, NASO
# is the count of awakening events during the sleep period.
SLEEP_TST_HOURS_RANGE: tuple[float, float] = (1.0, 15.0)
SLEEP_WASO_HOURS_RANGE: tuple[float, float] = (0.0, 8.0)
SLEEP_EFFICIENCY_PCT_RANGE: tuple[float, float] = (0.0, 100.0)
SLEEP_N_AWAKENINGS_RANGE: tuple[float, float] = (0.0, 200.0)

# ---------------------------------------------------------------------------
# Breath test
# ---------------------------------------------------------------------------
FENO_PPB_RANGE: tuple[float, float] = (5.0, 300.0)

# ---------------------------------------------------------------------------
# Module-level catalogues
# ---------------------------------------------------------------------------
# Mapping column-name → plausibility-range for parquets currently in
# production. Extraction functions call ``_enforce_plausibility`` with the
# relevant catalogue before the parquet is written. The catalogue is the
# single source of truth for what counts as an outlier.
#
# When a new measurement is added, extend the corresponding catalogue below
# (and, if needed, add a new module-level constant above).

CARDIOVASCULAR_RANGES: dict[str, tuple[float, float]] = {
    "systolic_bp_mean": SBP_MMHG_RANGE,
    "diastolic_bp_mean": DBP_MMHG_RANGE,
    "heart_rate_rest": HEART_RATE_RANGE,
    "pulse_wave_velocity": PWV_MS_RANGE,
    "augmentation_index": AUG_INDEX_PCT_RANGE,
    "ankle_brachial_index_left": ABI_RANGE,
    "ankle_brachial_index_right": ABI_RANGE,
}

LUNG_FUNCTION_RANGES: dict[str, tuple[float, float]] = {
    "fev1": FEV1_L_RANGE,
    "fvc": FVC_L_RANGE,
    # NAKO ``spiro_mw_*_p`` columns carry reference/predicted values in
    # litres (not % predicted). We keep them verbatim as *_predicted_l
    # and compute true percent-predicted in the extractor.
    "fev1_predicted_l": FEV1_L_RANGE,
    "fvc_predicted_l": FVC_L_RANGE,
    "fev1_percent_predicted": PCT_PREDICTED_RANGE,
    "fvc_percent_predicted": PCT_PREDICTED_RANGE,
    "fev1_fvc_ratio": FEV1_FVC_RATIO_RANGE,
    # NAKO stores peak expiratory flow in L/s (cf. dd_nako882.xlsx).
    "peak_flow": PEAK_FLOW_L_PER_SECOND_RANGE,
    "fef25_75": FEF_L_PER_SECOND_RANGE,
}

LAB_RANGES: dict[str, tuple[float, float]] = {
    # NAKO baseline labs use the same µkat/L (transaminases) and IFCC
    # (HbA1c in mmol/mol) conventions as FU1; cf. FU1_LAB_RANGES.
    "crp": CRP_MG_PER_L_RANGE,
    "cholesterol": CHOLESTEROL_MMOL_PER_L_RANGE,
    "ldl": LDL_MMOL_PER_L_RANGE,
    "hdl": HDL_MMOL_PER_L_RANGE,
    "triglycerides": TRIGLYCERIDES_MMOL_PER_L_RANGE,
    "glucose": GLUCOSE_MMOL_PER_L_RANGE,
    "hba1c": HBA1C_MMOL_PER_MOL_RANGE,
    "creatinine": CREATININE_UMOL_PER_L_RANGE,
    "egfr": EGFR_ML_PER_MIN_RANGE,
    "urea": UREA_MMOL_PER_L_RANGE,
    "alt": ALT_UKAT_PER_L_RANGE,
    "ast": AST_UKAT_PER_L_RANGE,
    # Defence in depth against the 444444/555555 sentinel leak observed
    # across the lab panel. _GENERAL_SENTINEL_CODES already replaces those
    # values with NA at extraction time; these ranges catch any future
    # sentinel that slips through via _enforce_plausibility.
    "ggt": GGT_UKAT_PER_L_RANGE,
    "bilirubin": BILIRUBIN_UMOL_PER_L_RANGE,
    "tsh": TSH_MU_PER_L_RANGE,
    "ft4": FT4_PMOL_PER_L_RANGE,
    "hemoglobin": HB_MMOL_PER_L_RANGE,
    "leukocytes": WBC_GPT_PER_L_RANGE,
    "platelets": PLATELETS_GPT_PER_L_RANGE,
    "sodium": SODIUM_MMOL_PER_L_RANGE,
    "potassium": POTASSIUM_MMOL_PER_L_RANGE,
    "calcium": CALCIUM_MMOL_PER_L_RANGE,
}

MEDICAL_HISTORY_ANTHROPOMETRIC_RANGES: dict[str, tuple[float, float]] = {
    "bmi_self_reported": BMI_RANGE,
    "height_cm_self_reported": HEIGHT_CM_RANGE,
    "weight_kg_self_reported": WEIGHT_KG_RANGE,
}

MENTAL_HEALTH_RANGES: dict[str, tuple[float, float]] = {
    "phq9_sum": PHQ9_SUM_RANGE,
    "gad7_sum": GAD7_SUM_RANGE,
    "mini_major_depression": MINI_BINARY_RANGE,
    "mini_major_depression_imputed": MINI_BINARY_RANGE,
    "mini_screen_depression": MINI_BINARY_RANGE,
}

PSYCHOMETRIC_RANGES: dict[str, tuple[float, float]] = {
    "phq9_total": PHQ9_SUM_RANGE,
    "gad7_total": GAD7_SUM_RANGE,
}

# Baseline PHQ-9 items after recoding from the NAKO ``BeeintrDepr`` value
# list to canonical 0..3 scores (see ``data_processing/phq9_items.py``).
PHQ9_ITEMS_RANGES: dict[str, tuple[float, float]] = {
    "phq9_sum_items": PHQ9_SUM_RANGE,
    **{f"phq9_item_{position}": PHQ9_ITEM_RANGE for position in range(1, 10)},
}

DEMOGRAPHICS_RANGES: dict[str, tuple[float, float]] = {
    "age": AGE_YEARS_RANGE,
    "basis_age": AGE_YEARS_RANGE,
}

# GPAQ total MET-minutes/week. The upper bound is the arithmetic ceiling of
# the instrument rather than a clinical cut-off: a week has 10,080 minutes
# and GPAQ weights vigorous activity at 8 METs, so no honest response can
# exceed 10,080 x 8. Values above it are decoding or reporting errors, which
# is what this guard is for. Down-weighting implausible-but-in-range
# over-reporting is an analysis decision, not a plausibility filter.
GPAQ_MET_WEEK_RANGE: tuple[float, float] = (0.0, 80_640.0)

GPAQ_RANGES: dict[str, tuple[float, float]] = {
    "gpaq_met_total": GPAQ_MET_WEEK_RANGE,
}

# ``gpaq_met_total`` is joined into ``physical_activity.parquet`` as well as
# kept in its own parquet, so the same bound has to be registered under both
# stems or the harness stops checking it on the frame the model reads. The
# QUAP columns carry no ranges: NAKO derives them and they are consumed as
# delivered.
PHYSICAL_ACTIVITY_RANGES: dict[str, tuple[float, float]] = {
    "gpaq_met_total": GPAQ_MET_WEEK_RANGE,
}

# Baseline tobacco exposure (see ``data_processing/smoking.py``). Bounds
# follow the same principle as the GPAQ range: they catch decoding errors
# and stray sentinel codes, not implausible-but-honest over-reporting,
# which is an analysis decision rather than a plausibility filter. NAKO
# already deletes values its own quality control rejects (code 8886).
SMOKING_STATUS_RANGE: tuple[float, float] = (1.0, 3.0)
SMOKING_YEARS_RANGE: tuple[float, float] = (0.0, AGE_YEARS_RANGE[1])
# Six packs a day sustained for sixty years. The delivered components cap
# out well below this (120 cigarettes/day, 59 smoking-years), so the bound
# deletes nothing observed while still catching every sentinel code.
PACK_YEARS_RANGE: tuple[float, float] = (0.0, 360.0)
CIGARETTES_PER_DAY_RANGE: tuple[float, float] = (0.0, 200.0)

SMOKING_RANGES: dict[str, tuple[float, float]] = {
    "smoking_status": SMOKING_STATUS_RANGE,
    "pack_years": PACK_YEARS_RANGE,
    "smoking_duration_years": SMOKING_YEARS_RANGE,
    "cigarettes_per_day": CIGARETTES_PER_DAY_RANGE,
    "smoking_quit_age": SMOKING_YEARS_RANGE,
}

# Acute infection course (see ``data_processing/acute_infection.py``). The
# day counts are bounded by their own question, which asks about the first
# month after the positive test: across the two counts the range bounds,
# all 547 answered values sit in [0, 31] — 452 ward days and 95 intensive-
# care days — so the bound is the instrument's rather than a judgement
# about plausibility. The month index runs from the questionnaire's
# December 2019 reference to a survey that closed in 2023; five years of
# headroom keeps a later delivery from tripping it while still catching the
# 1800-01-01 missing marker, which lands 240 months below zero.
ACUTE_CARE_LEVEL_RANGE: tuple[float, float] = (0.0, 3.0)
ACUTE_DAYS_FIRST_MONTH_RANGE: tuple[float, float] = (0.0, 31.0)
INFECTION_COUNT_RANGE: tuple[float, float] = (1.0, 4.0)
INFECTION_MONTHS_RANGE: tuple[float, float] = (0.0, 60.0)

ACUTE_INFECTION_RANGES: dict[str, tuple[float, float]] = {
    "acute_care_level": ACUTE_CARE_LEVEL_RANGE,
    "hospital_days": ACUTE_DAYS_FIRST_MONTH_RANGE,
    "icu_days": ACUTE_DAYS_FIRST_MONTH_RANGE,
    "n_infections": INFECTION_COUNT_RANGE,
    "first_infection_months": INFECTION_MONTHS_RANGE,
}

OLFACTOMETRY_RANGES: dict[str, tuple[float, float]] = {
    "olf_identification_sum": OLF_IDENTIFICATION_SUM_RANGE,
    "olf_identification_result": OLF_IDENTIFICATION_SUM_RANGE,
    "olf_free_identification": OLF_FREE_IDENTIFICATION_RANGE,
    "olf_category_identification": OLF_CATEGORY_IDENTIFICATION_RANGE,
    "olf_method_code": OLF_METHOD_CODE_RANGE,
    "olf_cold_on_test_day": OLF_COLD_RANGE,
    "olf_normosmia_pass": OLF_NORMOSMIA_PASS_RANGE,
}

SLEEP_OBJECTIVE_RANGES: dict[str, tuple[float, float]] = {
    "sleep_tst_hours": SLEEP_TST_HOURS_RANGE,
    "sleep_waso_hours": SLEEP_WASO_HOURS_RANGE,
    "sleep_efficiency_pct": SLEEP_EFFICIENCY_PCT_RANGE,
    "sleep_n_awakenings": SLEEP_N_AWAKENINGS_RANGE,
}

FENO_RANGES: dict[str, tuple[float, float]] = {
    "feno_ppb": FENO_PPB_RANGE,
}

# ---------------------------------------------------------------------------
# AGE-Reader (skin autofluorescence proxy of advanced glycation end-products)
# ---------------------------------------------------------------------------
# Autofluorescence: AU (arbitrary units), typical adult range 1..4.
AGE_AFR_RANGE: tuple[float, float] = (0.1, 10.0)
AGE_REFL_RANGE: tuple[float, float] = (0.0, 5.0)

# ---------------------------------------------------------------------------
# Dental status
# ---------------------------------------------------------------------------
# Natural tooth count (0..32 for adult dentition).
TEETH_COUNT_RANGE: tuple[float, float] = (0.0, 32.0)
# Prosthesis count (upper + lower jaw).
PROSTHESIS_COUNT_RANGE: tuple[float, float] = (0.0, 2.0)

FU1_MARKERS_MISC_RANGES: dict[str, tuple[float, float]] = {
    "skin_age_autofluorescence": AGE_AFR_RANGE,
    "skin_age_reflection": AGE_REFL_RANGE,
    "n_natural_teeth": TEETH_COUNT_RANGE,
    "n_prostheses": PROSTHESIS_COUNT_RANGE,
}

# ---------------------------------------------------------------------------
# FU1 visit-timing proxy (followup1_visit_meta)
# ---------------------------------------------------------------------------
# Time between NAKO baseline visit (2014..2019) and the Corona-1
# questionnaire (gefu1, administered 2020). Empirically 68 % of
# participants have a 2-year delta and 24 % have 3 years, with a tail up
# to ~9 years for late Level-2/3 re-examinations. Negative deltas
# (gefu1 age < baseline age) have been observed in ~0.01 % of rows and
# are treated as implausible (likely data-entry errors).
VISIT_META_YEARS_SINCE_BASELINE_RANGE: tuple[float, float] = (0.0, 12.0)

FOLLOWUP1_VISIT_META_RANGES: dict[str, tuple[float, float]] = {
    "baseline_age": AGE_YEARS_RANGE,
    "gefu1_age_proxy": AGE_YEARS_RANGE,
    "years_since_baseline_proxy": VISIT_META_YEARS_SINCE_BASELINE_RANGE,
}

# ---------------------------------------------------------------------------
# Longitudinal mental-health panel (T0 baseline / T1 Corona-1 / T2 Corona-2)
# ---------------------------------------------------------------------------
# Per-wave sum scores and per-item values for PHQ-9 and GAD-7. Items are
# stored on the canonical 0..3 scale (after subtracting the NAKO 1..4
# encoding) so all three waves share one plausibility range.
FU1_LAB_RANGES: dict[str, tuple[float, float]] = {
    "hscrp": HSCRP_MG_PER_L_RANGE,
    "hba1c_ifcc": HBA1C_MMOL_PER_MOL_RANGE,
    "glucose": GLUCOSE_MMOL_PER_L_RANGE,
    "cholesterol_total": CHOLESTEROL_MMOL_PER_L_RANGE,
    "non_hdl_cholesterol": CHOLESTEROL_MMOL_PER_L_RANGE,
    "hdl_cholesterol": HDL_MMOL_PER_L_RANGE,
    "ldl_cholesterol": LDL_MMOL_PER_L_RANGE,
    "ldl_cholesterol_friedewald": LDL_MMOL_PER_L_RANGE,
    "triglycerides": TRIGLYCERIDES_MMOL_PER_L_RANGE,
    "creatinine_serum": CREATININE_UMOL_PER_L_RANGE,
    "cystatin_c": CYSC_MG_PER_L_RANGE,
    "alt": ALT_UKAT_PER_L_RANGE,
    "ast": AST_UKAT_PER_L_RANGE,
    "ggt": GGT_UKAT_PER_L_RANGE,
    "alkaline_phosphatase": ALP_UKAT_PER_L_RANGE,
    "ldh": LDH_UKAT_PER_L_RANGE,
    "cholinesterase": CHE_UKAT_PER_L_RANGE,
    "lipase": LIPASE_UKAT_PER_L_RANGE,
    "bilirubin_total": BILIRUBIN_UMOL_PER_L_RANGE,
    "albumin_serum": ALBUMIN_G_PER_L_RANGE,
    "total_protein": TOTAL_PROTEIN_G_PER_L_RANGE,
    "tsh": TSH_MU_PER_L_RANGE,
    "free_t3": FT3_PMOL_PER_L_RANGE,
    "free_t4": FT4_PMOL_PER_L_RANGE,
    "urea": UREA_MMOL_PER_L_RANGE,
    "urate": URATE_UMOL_PER_L_RANGE,
    "haemoglobin": HB_MMOL_PER_L_RANGE,
    "platelets": PLATELETS_GPT_PER_L_RANGE,
    "wbc": WBC_GPT_PER_L_RANGE,
    "rbc": RBC_TPT_PER_L_RANGE,
    "sodium": SODIUM_MMOL_PER_L_RANGE,
    "chloride": CHLORIDE_MMOL_PER_L_RANGE,
    "potassium": POTASSIUM_MMOL_PER_L_RANGE,
    "calcium": CALCIUM_MMOL_PER_L_RANGE,
    "magnesium": MAGNESIUM_MMOL_PER_L_RANGE,
}

MH_LONGITUDINAL_RANGES: dict[str, tuple[float, float]] = {
    "phq9_sum": PHQ9_SUM_RANGE,
    "gad7_sum": GAD7_SUM_RANGE,
    **{f"phq9_i{i}": PHQ9_ITEM_RANGE for i in range(1, 10)},
    **{f"gad7_i{i}": GAD7_ITEM_RANGE for i in range(1, 8)},
}

# Single lookup dict: parquet stem → plausibility catalogue. Used by the
# quality-validation harness. Catalogues may be partial — columns not
# listed are not plausibility-checked.
#
PARQUET_RANGE_CATALOGUES: dict[str, dict[str, tuple[float, float]]] = {
    "baseline_sex_age": DEMOGRAPHICS_RANGES,
    "cardiovascular": CARDIOVASCULAR_RANGES,
    "lab_values": LAB_RANGES,
    "lung_function": LUNG_FUNCTION_RANGES,
    "medical_history": MEDICAL_HISTORY_ANTHROPOMETRIC_RANGES,
    "mental_health": MENTAL_HEALTH_RANGES,
    "phq9_items": PHQ9_ITEMS_RANGES,
    "gpaq_activity": GPAQ_RANGES,
    "physical_activity": PHYSICAL_ACTIVITY_RANGES,
    "smoking": SMOKING_RANGES,
    "psychometric_scores": PSYCHOMETRIC_RANGES,
    "demographics": DEMOGRAPHICS_RANGES,
    "followup1_olfactometry": OLFACTOMETRY_RANGES,
    "followup1_sleep_objective": SLEEP_OBJECTIVE_RANGES,
    "followup1_feno": FENO_RANGES,
    "followup1_visit_meta": FOLLOWUP1_VISIT_META_RANGES,
    "mh_longitudinal": MH_LONGITUDINAL_RANGES,
    "followup1_lab_values": FU1_LAB_RANGES,
    "followup1_markers_misc": FU1_MARKERS_MISC_RANGES,
    "acute_infection": ACUTE_INFECTION_RANGES,
}
