# Feature Preprocessing Decisions

This document describes how raw NAKO data is preprocessed for the PCC
prediction pipeline.  Each modality section explains which features are
**excluded**, which are **retained**, and **why** — focusing on redundancy,
encoding correctness, and information content.

---

## Missing Data Strategy

Missing data is handled in three stages, each at the appropriate level:

### Stage 1 — Preprocessing (`_save_parquet`): Drop near-empty columns

During parquet export, columns with **>99% missing** are dropped.  These are
structurally absent variables (e.g. age-at-onset columns where <1% of
participants had the condition).  This is a conservative filter that only
removes columns with virtually no data.

### Stage 2 — Pipeline (`MissingnessThreshold`): Drop high-missingness columns

At runtime, **before imputation**, the `MissingnessThreshold` transformer
drops columns with >50% missing **in the training fold**.  This is critical
because:

- Missingness rates differ between the full cohort (117k) and the MRI
  subsample (~19k) that forms the actual analysis sample.
- The decision must be based on the relevant population, not the storage
  population.
- The threshold is learned on training data only (no leakage).

### Stage 3 — Pipeline (`KNNImputer`): Impute remaining values

Columns with ≤50% missing are imputed via KNN (k=5) within each CV fold.
The imputer is fit on training data and applied to both train and test,
preventing data leakage.

### Row-level handling

Participants with **all features missing** for a modality (e.g. Physical
Activity questionnaire not completed, or no MRI scan) receive a NaN
prediction for that modality.  The XGBoost meta-learner handles NaN
predictions natively via `tree_method="hist"`.

---

## Socioeconomic Status (SES)

**43 in parquet → 27 after pipeline drops** (after one-hot and date expansion: ~75)

### Parquet contents (43 features + ID)

The parquet contains the complete NAKO SES variable set: 30 extracted columns
(family, employment, education, income, ISCO/KLDB hierarchies, prestige
scores, occupation status) plus 13 derived columns (dummies for marital/
employment status, living arrangement, income adequacy).

### Encoding

NAKO stores several nominal variables as integers, which a model would
incorrectly treat as ordinal:

| Variable | NAKO encoding | Preprocessing | Rationale |
|----------|--------------|---------------|-----------|
| `a_ses_partner` | Int (1=yes, 2=no) | → `has_partner` (boolean) | Binary variable |
| `a_selfemp` | Int (0/1) | → `is_self_employed` (boolean) | Binary variable |
| `a_ses_fstand` (marital status) | Int (1-5) | Dummy-encoded → `married`, `single`, `divorced_separated`, `widowed`; original retained in parquet | 5 nominal categories stored as numbers |
| `a_ses_erwstat` (employment status) | Int (1-4) | Dummy-encoded → `employed`, `unemployed`, `retired`; original retained in parquet | 4 nominal categories stored as numbers |
| `a_isco_major` | Int (0-9) | → `isco_major` (string labels, one-hot encoded) | 10 nominal occupation groups |
| `a_isco_submajor` | Int (01-96) | → `isco_submajor` (one-hot encoded) | 43 nominal occupation sub-groups |

Nominal columns with missing values are cast to string and NA is replaced
with `"_missing"` before one-hot encoding, to avoid mixed-type errors in
scikit-learn's `OneHotEncoder`.

All binary flag variables use `boolean` dtype.

### Dropped in pipeline preprocessor (16)

| Column | Reason |
|--------|--------|
| `retired` | Reference category for employment status (employed, unemployed retained) |
| `widowed` | Reference category for marital status (married, single, divorced_separated retained) |
| `has_partner` | Redundant with marital status dummies |
| `income_weighted` | Component of `income_adequacy` (= income_weighted / needs_weighted) |
| `needs_weighted` | Component of `income_adequacy` (= income_weighted / needs_weighted) |
| `marital_status` | Nominal original; dummies (`married`, `single`, `divorced_separated`) retained |
| `employment_status` | Nominal original; dummies (`employed`, `unemployed`) retained |
| `occupation_status` | Worker/employee/civil servant/self-employed; `isco_submajor` provides finer granularity |
| `isco_code` | Detailed ISCO code; `isco_major` and `isco_submajor` retain the relevant groupings |
| `isco_minor` | Determined by `isco_code`; multicollinear |
| `isco_skill_level` | Determined by ISCO major group; multicollinear |
| `kldb_code` | KldB 2010 detailed code; redundant with ISCO |
| `kldb_major` | KldB major group; redundant with `isco_major` |
| `kldb_leadership` | KldB leadership flag |
| `kldb_segment` | KldB segment |
| `kldb_sector` | KldB sector |

### Retained occupational features (kept from ISCO/KLDB hierarchy)

| Column | Reason |
|--------|--------|
| `isco_major` | 10 broad occupation groups (nominal, one-hot encoded) |
| `isco_submajor` | 43 occupation sub-groups (nominal, one-hot encoded); provides occupational granularity that survives orthogonalisation against demographics |
| `isei_score` | International Socio-Economic Index of occupational status (continuous) |
| `siops_score` | Standard International Occupational Prestige Score; complements ISEI with a prestige dimension |
| `kldb_skill_level` | KldB skill requirement level; captures formal qualification requirements orthogonal to occupation group |

### Final schema (27 features after drops + ID)

**Extraction (19):** `household_size`, `number_children`,
`work_hours_category`, `is_self_employed`, `number_employees`,
`education_isced_level`, `education_years`, `german_education_level`,
`income_category`, `income_position`,
`isco_major`, `isco_submajor`, `isei_score`, `siops_score`,
`kldb_skill_level`, `retirement_age`, `employment_duration_years`,
`employment_duration_total`, `employment_date`

**Derivation (8):** `employed`, `unemployed`, `married`, `single`,
`divorced_separated`, `living_alone`, `has_children`,
`income_adequacy`

---

## Cognitive Tests

**12 in parquet → 8 features after pipeline drops**

### Parquet contents (12 features + ID)

All 12 NAKO neuropsychological test variables, including the composite score
(`cognitive_composite_score` from `a_npsy_sumts6`), difference scores, and
individual test scores.

### Dropped in pipeline preprocessor (4)

| Column | Reason |
|--------|--------|
| `word_list_learning` | Difference score (recall2 − recall1), linearly dependent on retained raw scores |
| `word_list_forgetting` | Difference score (recall2 − delayed), linearly dependent on retained raw scores |
| `stroop_interference_effect` | Difference score (interference − colors), linearly dependent on retained raw scores |
| `cognitive_composite_score` | NAKO-provided weighted linear combination of individual test scores; a regularised model learns its own optimal weighting |

These difference scores are exact linear combinations of their component
scores, which are all retained.  A regularised linear model can recover
these contrasts from the raw inputs.

### Final schema (8 features after drops + ID)

**Word list:** `word_list_recall1`, `word_list_recall2`, `word_list_delayed`
**Verbal fluency:** `verbal_fluency_animals`
**Stroop:** `stroop_colors_time`, `stroop_interference_time`
**Working memory:** `digit_span_backwards`
**Reasoning:** `number_series`

---

## Physical Activity

**35 in parquet → 17 features after pipeline drops**

### Parquet contents (35 features + ID)

All NAKO physical activity questionnaire variables: occupational activity,
household/transport/walking/cycling minutes, seasonal sports (raw minutes
and MET-minutes), participation flags, sedentary behaviour, stair climbing,
and total physical activity (raw minutes and MET-minutes).

### Dropped in pipeline preprocessor (18)

| Column(s) | Count | Reason |
|-----------|-------|--------|
| `sports_spring`, `sports_summer`, `sports_autumn`, `sports_winter`, `sports_combined` | 5 | Raw minutes where MET-minutes exist (near-perfect collinearity) |
| `pa_total`, `pa_summer`, `pa_winter` | 3 | Raw total PA minutes; MET equivalents (`met_total`, `met_summer`, `met_winter`) preferred |
| `household_participation`, `active_transport_participation`, `sports_participation`, `sitting_participation`, `stairs_participation`, `leisure_participation` | 6 | Participation flags encode missingness patterns, not physical activity |
| `met_sports_spring`, `met_sports_summer`, `met_sports_autumn`, `met_sports_winter` | 4 | Seasonal sports MET; aggregate `met_sports_combined` suffices |

**MET-minutes are preferred** over raw minutes because they weight by
physiological intensity — 30 min jogging contributes more than 30 min walking.
WHO physical activity guidelines are also MET-based.

### Final schema (17 features after drops + ID)

`occupational_activity_level`, `household_minutes_week`,
`active_transport_summer`, `active_transport_winter`,
`walking_summer`, `walking_winter`, `cycling_summer`, `cycling_winter`,
`met_sports_combined`, `sitting_weekday`, `sitting_saturday`,
`sitting_sunday`, `stairs_weekday`, `stairs_weekend`,
`met_total`, `met_summer`, `met_winter`

---

## Laboratory Values

**22 in parquet → 20 features after pipeline drops**

### Parquet contents (22 features + ID)

All 22 NAKO serum analysis biomarkers including `cholesterol` (`sa_chol`)
and `creatinine` (`sa_crea`).

### Dropped in pipeline preprocessor (2)

| Column | Reason |
|--------|--------|
| `cholesterol` | ≈ LDL + HDL + TG/5 per Friedewald equation; subfraction markers (`ldl`, `hdl`, `triglycerides`) are clinically more informative |
| `creatinine` | eGFR is derived from creatinine + age/sex (CKD-EPI); eGFR is the clinical standard (mL/min/1.73 m²) |

### Final schema (20 features after drops + ID)

The 20 biomarkers span 7 clinical domains with no further redundancies.
All are continuous and require no special encoding.

**Inflammation:** `crp`
**Metabolic:** `glucose`, `hba1c`, `ldl`, `hdl`, `triglycerides`
**Kidney:** `egfr`, `urea`
**Liver:** `alt`, `ast`, `ggt`, `bilirubin`
**Thyroid:** `tsh`, `ft4`
**Blood count:** `hemoglobin`, `leukocytes`, `platelets`
**Electrolytes:** `sodium`, `potassium`, `calcium`

---

## Cardiovascular

**7 features**

### Excluded: `pulse_pressure` and `mean_arterial_pressure`

Both are exact linear combinations of `systolic_bp_mean` and
`diastolic_bp_mean`:

- PP = SBP − DBP
- MAP = DBP + (SBP − DBP) / 3

A regularised linear model can learn these transformations directly from the
two blood pressure inputs.  Keeping derived linear combinations adds
multicollinearity without new information.

### Final schema (7 features + ID)

**Blood pressure:** `systolic_bp_mean`, `diastolic_bp_mean`
**Heart rate:** `heart_rate_rest`
**Vascular stiffness:** `pulse_wave_velocity`, `augmentation_index`,
`ankle_brachial_index_left`, `ankle_brachial_index_right`

---

## Lung Function

**7 in parquet → 4 features after pipeline drops**

### Parquet contents (7 features + ID)

All NAKO spirometry variables: absolute volumes (`fev1`, `fvc`), their
percent-predicted equivalents, the FEV1/FVC ratio (`fev1_fvc_ratio` from
`spiro_mw_fev1_fvc`), plus flow parameters (`peak_flow`, `fef25_75`).

### Dropped in pipeline preprocessor (3)

| Column | Reason |
|--------|--------|
| `fev1` | Subsumed by `fev1_percent_predicted` (reference-normalised, includes height correction) |
| `fvc` | Subsumed by `fvc_percent_predicted` (reference-normalised, includes height correction) |
| `fev1_fvc_ratio` | Exact quotient of `fev1` / `fvc`; the model can learn this ratio from the inputs |

The percent-predicted values correct for age, sex, **and height** using
GLI-2012 reference equations.  While age and sex are handled by the
orthogonalisation step (confounders), height is not included elsewhere
in the model.  Keeping only percent-predicted retains the height
information that absolute volumes would lose after orthogonalisation.

### Final schema (4 features after drops + ID)

**Reference-normalised:** `fev1_percent_predicted`, `fvc_percent_predicted`
**Flow parameters:** `peak_flow`, `fef25_75`

---

## Medical History

**~35 in parquet → ~29 after pipeline drops**

The code defines 48 features (39 extracted + 9 derived), but many age-at-event
columns have >99% missing and are dropped by `_save_parquet`, leaving ~35 in
the parquet file (exact count depends on the cohort).

### Encoding

NAKO stores binary disease and medication flags using integer coding
(1=yes, 2=no or 0/1).  All are converted to `boolean` dtype:

- **Disease flags:** `has_hypertension`, `has_psoriasis`,
  `has_psoriasis_arthritis`
- **Medications:** `medication_antihypertensive`, `medication_antiparkinson`,
  `medication_antiepileptic`, `medication_antidiabetic`,
  `medication_betablocker`, `medication_lipid_lowering`,
  `medication_anticoagulant`
- **Derived flags:** `has_infection_history`, `has_neurological_disease`

### Dropped in pipeline preprocessor (6)

| Column | Reason |
|--------|--------|
| `bmi_self_reported` | Height + weight retained; BMI = weight/height² is derivable |
| `bmi_category` | Ordinal binning of BMI (now dropped with BMI itself) |
| `number_medications` | Count variable; individual medication flags retained |
| `polypharmacy` | Binary threshold (≥5) of `number_medications` |
| `has_surgery_history` | Redundant with `number_surgeries` (>0 equivalent) |
| `has_cancer_history` | Redundant with `number_cancers` (>0 equivalent) |

### Retained: Age-at-event variables

Individual `cancer_N_age`, `infection_*_age`, `surgery_*_age`, and
neurological `*_age` columns encode both whether an event occurred (non-null)
and when.  The count variables `number_cancers` and `number_surgeries`
aggregate these (the corresponding boolean flags `has_cancer_history` and
`has_surgery_history` are dropped as redundant — see above).
High-missingness columns (>50% missing in the analysis sample) are dropped
at pipeline runtime by `MissingnessThreshold`.

### Retained: `cancer_1_type` (nominal, numeric)

ICD-based cancer type code.  High-cardinality nominal variable stored as
integer.  The model treats it numerically which is imperfect, but the column
is mostly NA (only ~5 % of participants have cancer history) so impact is
minimal.

---

## Demographics

**3 raw features → ~20 after one-hot encoding**

`basis_age` (continuous), `basis_sex` (categorical: male/female),
`basis_uort` (study center, 1–18).  Used as confounders for
orthogonalisation in all other modality pipelines.  No redundancies.

### Encoding

Both `basis_sex` and `basis_uort` are nominal variables that would be
incorrectly treated as ordinal if passed as integers.  They are one-hot
encoded (drop-first) in the pipeline preprocessor:

- `basis_sex`: 1 dummy (2 categories, drop first)
- `basis_uort`: 17 dummies (18 study centres, drop first)
- `basis_age`: passed through as continuous

This yields ~20 features after encoding (1 age + 1 sex + 17 centre + 1
intercept absorbed by drop-first).

---

## Corona-2 / PCC

**Target variable — not a predictor modality**

Contains the PCC operationalisation (symptom matrix, Diexer symptom
count, Bahmer weighted PCS).  The primary outcome is
`bahmer_any_pcs = (bahmer_pcs_score > 10.75)`.  Processed separately,
no feature selection applied.

---

## Psychometric (PHQ-9, GAD-7)

**2 features — descriptive only, not used as predictors**

`phq9_total` (depression, 0–27) and `gad7_total` (anxiety, 0–21) are sum
scores from the Corona-2 follow-up.  Used for Table 1 characterisation only.
Item-level recoding (NAKO 1-based → standard 0-based Likert) and missing-item
tolerance (max 2 missing) are applied during scoring.

---

## MRI (6 atlases)

**599 features**

Raw regional brain volumes from FreeSurfer parcellations:

| Atlas | Features | Content |
|-------|----------|---------|
| Desikan-Killiany | 68 | Cortical volumes |
| Destrieux | 187 | Cortical gyri/sulci |
| Julich | 50 | Cytoarchitectonic regions |
| Yeo 7-Network | 96 | Functional networks |
| Subcortical | 186 | White matter + subcortical |
| Cerebellar | 12 | Cerebellum |

All features are continuous (mm³).  Cleaning handles missingness markers,
sentinel NA codes, empty row/column removal, and dtype optimisation.

### ICV correction (residual method)

Head-size differences between participants are corrected by including
estimated total intracranial volume (eTIV) as an additional covariate in
the orthogonalisation step, alongside age, sex, and study centre.  eTIV
is loaded from `mri_etiv.parquet` and merged by ID.

The `FeatureToConfounder` transformer extracts the eTIV column from the
feature matrix and routes it to the `Orthogonalizer` via
`ConfoundedPipeline`.  This implements the **residual method** — the
Orthogonalizer regresses out the eTIV-predictable component from each
brain volume, rather than dividing all volumes by eTIV (proportion
method).

**Why the residual method:**

- The proportion method (volume ÷ eTIV) assumes a strictly proportional
  relationship between regional volume and ICV, which does not hold for
  all structures (e.g. caudate, putamen; Voevodskaya et al. 2014).
- Since eTIV correlates with sex (~10–12 % larger in males), dividing by
  eTIV partially removes sex effects.  The orthogonaliser then removes
  age/sex/centre again, causing **double correction** of the sex-related
  component.
- The residual method handles all confounders (age, sex, centre, ICV) in
  a single step, partitioning variance correctly.

### Dimensionality reduction

PCA (retaining 95 % of variance) is applied to atlases with >20 features
after orthogonalisation.  The cerebellar atlas (12 features) skips PCA
and relies solely on elastic net regularisation in the classifier.

| Atlas | Features | PCA |
|-------|----------|-----|
| Desikan-Killiany | 68 | Yes |
| Destrieux | 187 | Yes |
| Julich | 50 | Yes |
| Subcortical | 186 | Yes |
| Yeo 7-Network | 96 | Yes |
| Cerebellar | 12 | No |

Stability selection is **not** used for MRI modalities.  Brain regional
volumes are highly correlated (bilateral homologues r > 0.8, neighbouring
regions r > 0.6).  L1-based stability selection arbitrarily picks one
variable from a correlated group and zeros the rest; across bootstrap
samples different regions get selected, diluting every region's selection
probability below the threshold (Zou & Hastie 2005; confirmed in GWAS
literature for correlated features).  Instead, elastic net regularisation
in the classifier handles both feature selection (L1 component) and
correlated-feature grouping (L2 component) in a single step, which is
the standard approach in the neuroimaging prediction literature.
