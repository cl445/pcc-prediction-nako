# Design Decisions

Methodological and technical decisions in this pipeline, with rationale.
Ordered by pipeline stage, not chronologically.

---

## 1. Data Quality & Sample Definition

### 1.1 Two groups of sentinel codes, not one

**Decision:** NAKO sentinel codes are split into *general* (`-999, -99, -9, 999,
9999`) and *survey-specific* (`-88, -5, 7775, 7776, 7777, 8886, 8888, 8889`).

**Rationale:** The NAKO exports use different codes depending on the data type.
MRI volumes carry only the general sentinels, while questionnaires add their own
(`-88` = refused to answer, `7777` = not assessed). A single flat list produced
false-positive NA replacements in numeric columns — `-5` is a valid augmentation
index in the cardiovascular data, and a flat list silently deleted it. The split
makes sentinel handling depend on where the column came from.

**Where:** `data_processing/_common.py` (`_GENERAL_SENTINEL_CODES`,
`_SURVEY_SENTINEL_CODES`, `NAKO_SENTINEL_CODES`)

### 1.2 Symptom validity filter (routing plus an 80 % threshold)

**Decision:** Infected participants are excluded through the `valid_symptoms`
flag when either (a) the kmatrix routing question `d_co2_k0` was not asked or was
refused, or (b) the kmatrix was started but fewer than 80 % of its items were
answered validly. Downstream consumers filter on that flag.

**Rationale:** PCC symptom status comes from the kmatrix (21 items, 4–12 months
post-infection), which is gated by a routing question:

| `d_co2_k0` | Meaning | N (infected) | Symptom status |
|---|---|---|---|
| 1 | symptoms reported, kmatrix completed | 19,429 | observable → valid |
| 2 | no symptoms reported, kmatrix skipped | 35,826 | observable → symptom-free (`fillna(False)` is correct) |
| 7775 | routing question never shown (questionnaire break-off) | 10,803 | **not observable** → excluded |
| 8888 | routing question refused | 608 | **not observable** → excluded |

Without the routing filter, 11,411 infected participants of unknown symptom
status would be coded symptom-free by `fillna(False)` — a systematic bias toward
PCC-negative. The break-off (miss_sec 5–6 in 98 % of the k0=7775 cases) is likely
to correlate with health status, so it is missing not at random and any
imputation introduces a bias nobody controls.

Infected participants who *started* the kmatrix (k0=1, at least one item
answered) but answered fewer than 80 % of items validly (N=2,614) are excluded as
well. The 80 % threshold follows ordinary psychometric practice.

**Where:** `data_processing/pcc_outcome.py` (routing filter, 80 % filter,
`valid_symptoms`), `data_manager.py` (downstream filtering)

### 1.3 Restriction to the infected, not adjustment for infection

**Decision:** The analytic sample is unconditionally restricted to participants
reporting a SARS-CoV-2 infection (`had_covid == 1`). The restriction is central
and hard-coded in `load_pipeline_data()`, so every downstream step — cohort
intersection, nested CV, SHAP, feature importance — operates natively on the
restricted cohort. There is no config flag, and `had_covid` is not passed through
the pipeline.

**Rationale:** The PCC outcome is defined only for infected participants: the
kmatrix symptom items are structurally never put to the uninfected. Keeping the
uninfected in the sample and forcing `y=0` gives an estimator that mixes two
questions — who was infected, and who develops PCC *given* infection. DML
orthogonalization cannot repair that, because infection is not a classical
confounder but a gatekeeper between predictors and Y: for an uninfected
participant Y is structurally zero, not zero by chance. The estimator the
second-hit hypothesis asks for is the conditional prediction given infection,
which means restriction rather than adjustment.

**Where:** `data_manager.py:load_pipeline_data()`

### 1.4 Sex encoding accepts only the two explicit values

**Decision:** `basis_sex` accepts `1` (male) and `2` (female). Everything else
becomes NA and is logged, rather than being coded implicitly as female.

**Rationale:** A test phrased as `basis_sex != 1 -> female` treats unknown values,
NA and unexpected codes alike as female, silently. Sex is a confounder in the DML
orthogonalization, so a wrong value there propagates into every residualised
feature of every modality.

**Where:** `data_processing/demographics.py` (`_SEX_MAP`, `_map_sex()`)

### 1.5 Demographic completeness

**Decision:** Participants missing age, sex or study centre are excluded.

**Rationale:** Demographics are the confounders in the DML orthogonalization, and
a missing confounder corrupts the X-residualisation for every modality. They come
from the baseline assessment, where completeness is expected, so an exclusion
here points at a data problem rather than at a participant.

**Where:** `data_manager.py`

### 1.6 Sample attrition is logged at every intersection

**Decision:** Every index intersection logs cumulatively how many participants
are lost.

**Rationale:** The pipeline intersects the indices of all modalities. Without the
log there is no way to say which modality costs how many participants — which
matters most for modalities with high missingness (physical activity: 62.4 %) and
for the MRI subsample.

**Where:** `data_manager.py`

### 1.7 `medical_history` is already a self-report stack

**Observation (no code change):** With exactly one exception, the
`medical_history` modality consists of self-reported questionnaire items from
the NAKO baseline.

**Contents:**
- Self-reported: anthropometry (BMI, height, weight), chronic diagnoses
  (hypertension, psoriasis, psoriatic arthritis), medications, infection
  history, cancer history, surgical procedures, neurological diagnoses
  (stroke, epilepsy, Parkinson's).
- The one measured column: `body_fat_percentage`, from a bioimpedance
  measurement.

**Why this note exists:** The Lean Stack modality is labelled "Medical History
(Self-Report)". That is a qualifier on the mode of assessment — it distinguishes
the modality from laboratory, MRI and cognitive testing — not an instruction to
apply a sub-filter. The loader needs no self-report filter, because roughly 95 %
of the modality already is one.

**What that means for transportability:** nothing has to change in the loader
when this modality is used in the Lean Stack. A cohort that lacks the
bioimpedance measurement would have to decide between imputing
`body_fat_percentage` as missing-not-at-random and dropping it from the Lean
variant beforehand — a decision that belongs to whoever builds that loader.

**Where:** `data_processing/medical_history.py` (`_extract_medical_history`)

### 1.8 Cohort and stack variant

**Decision:** `load_pipeline_data()` takes `cohort: Literal["mri", "non_mri",
"mri_plus_non_mri", "all"]` and `stack_variant: Literal["full", "lean"]`.
Together they select one of the analysis configurations below.

**Supported combinations:**

| cohort | stack_variant | Meaning |
|---|---|---|
| `all` | `full` | default; no explicit cohort filter, N ≈ the MRI sample |
| `mri` | `full` | primary analysis, explicit MRI cohort |
| `mri` | `lean` | Lean Stack on the MRI cohort — the source run a transfer is compared against |
| `non_mri` | `lean` | within-study replication on the non-MRI sample |
| `non_mri` | `full` | **`ValueError`** — the MRI modalities are structurally empty |
| `mri_plus_non_mri` | `lean` | pooled Lean training (MRI ∪ non-MRI); a candidate source for an external transfer |
| `mri_plus_non_mri` | `full` | **`ValueError`** — the MRI intersection would silently drop every non-MRI participant and collapse the pool to `cohort='mri'` |

**Rationale:**

- **An explicit filter, not an implicit one.** The MRI sample can arise
  implicitly, as the intersection of all modality indices in the Full Stack.
  That is not enough for a replication across cohorts: the implicit
  intersection is empty for a non-MRI run, and the attrition log cannot say
  which filter cost which participants.
- **The Lean Stack is pinned in code.** Its four modalities (`demographics`,
  `ses`, `medical_history`, `mental_health`) are a `frozenset`, not a config
  value. The selection rule was fixed before the first run and must not change
  afterwards, or the transportability argument fails: a modality set adjusted
  after seeing the results turns a replication into a fit.
- **`non_mri + full` raises rather than degrades.** Empty MRI modalities
  without a warning are a silent loss of sample size, and the pipeline would
  keep running for hours on a configuration error that was visible at the
  start. Failing early is cheaper.
- **`mri_plus_non_mri`** exists because the union of the two sub-cohorts is the
  natural training sample for the deployment-oriented Lean variant. Pooling
  with the Full Stack is rejected for the same reason as `non_mri + full`: the
  intersection over the six MRI atlases drops every non-MRI participant, so the
  pool would quietly fall back to `cohort='mri'`. The log counts the split per
  sub-cohort, so an unexpectedly empty one shows up immediately rather than in
  the cross-validation aggregate.
- **The cohort filter runs after `clean_controls`,** so both arms use the same
  outcome definition. Applied the other way round, the MRI arm would reach a
  subtly different prevalence through the intersection pruning.
- **`cohort` and `stack_variant` are recorded in `config.csv`** and feed
  `_config_hash()`. Without them a distributed fold run cannot be attributed to
  an analysis variant, and folds from different cohorts could be merged.

**Where:** `data_manager.py` (`CohortName`, `StackVariant`,
`LEAN_STACK_MODALITIES`, `_mri_participant_ids`, `load_pipeline_data` and its
guards), `orchestration.py` (`_get_config` → `config.csv`),
`scripts/pipeline/02_run_pipeline.py` (`--cohort`, `--stack`),
`scripts/pipeline/05_compute_statistics.py` (`--cohort`),
`tests/test_data_manager.py` (one test per combination)

See also §1.7: `medical_history` is already self-report, so the Lean Stack
needs no additional sub-filter.

### 1.9 Transfer validation, not a second fit

**Decision:** The within-study replication is a **transfer**: fit the Lean Stack
once on the MRI cohort, then score the same model on the non-MRI sample without
retraining. The refit path (`--cohort non_mri --stack lean`) remains available as
a sensitivity variant.

**How it works:**

1. **`final_model.pkl` carries what inference needs** — `FinalModelResult` adds:
   - `modality_pipelines` — one pipeline per modality, fitted on the *whole*
     source sample rather than per CV fold.
   - `modality_column_layout` — the exact column order each pipeline saw when
     it was fitted. Target data is brought to that layout by
     `reindex(columns=expected)`, missing columns arriving as NaN, which the
     KNN imputer handles exactly as it did in the source cohort.
   - `confounders_train` — a snapshot of the training cohort's confounders, for
     the KS-distance comparison in the transfer report (see §1.10).
2. **`scripts/pipeline/03_apply_transfer.py`** loads the pickle, calls
   `ConfoundedPipeline.predict_proba(X_target, confounders=target_confounders)`
   per modality, and applies the meta-learner to the resulting matrix.
3. **The residualiser transfers too.** The `Orthogonalizer` is an internal step
   of the `ConfoundedPipeline`, so the `f(C)` fitted on the source cohort is
   applied to the target's confounders — source-conditioned residualisation on
   a new population. It is not refitted on the target.
4. **What the transfer output reports:**
   - Standard metrics: ROC-AUC, PR-AUC, Brier, calibration slope, calibration
     intercept, ECE, decision curve, bootstrap CIs.
   - `shap_modality_importance_transfer.csv` — SHAP values on the target
     predictions.
   - `transfer_summary.json`, whose verdict is a **TOST equivalence test** on
     three transportability metrics (Steyerberg 2019; Van Calster 2019;
     TRIPOD-AI). The target's 95 % CI must lie entirely inside a pre-specified
     margin in every case:
     - ΔROC-AUC: ±0.03 around the source point, the usual
       non-inferiority margin for a clinical prediction model
     - calibration slope: [0.85, 1.15], Van Calster's "acceptable"
     - calibration-in-the-large: [−0.05, +0.05]

     All three must pass — a conservative conjunction.
   - Descriptive, and deliberately not a gate: a per-cohort **SHAP-share CI**
     (1000 bootstrap resamples on source and target separately, same fitted
     meta-learner, 95 % percentile interval), the Spearman rank correlation of
     modality SHAP values between cohorts, and per-column KS distances on the
     confounders.

**Why the SHAP shares are descriptive.** Two earlier formulations were tried
and both fail as gates. A ±30 % rule on the share is an arbitrary threshold,
calibrated against nothing, least of all bootstrap noise. Requiring the two
share CIs to overlap is structurally too strict at this sample size: at tens of
thousands of participants every statistically detectable drift fails, including
drift far too small to matter. TOST on discrimination and calibration is the
established transportability language, its margins are pre-specified and come
from the literature, and it tests the thing a reader cares about.

**Rationale:**

- **Transfer rather than refit** makes the stronger claim — same model, different
  population — and costs one training run plus minutes of inference instead of
  two training runs.
- **The column layout is persisted explicitly** rather than reconstructed from
  `feature_names_in_` on the pipeline steps: it survives sklearn internals
  changing, and there is one place that answers "which columns does this model
  expect".
- **Calibration is measured, not refitted.** Refitting Platt scaling on the
  target would stop the exercise being a transfer; the miscalibration *is* the
  signal.
- **An old pickle fails loudly.** The transfer script raises when
  `final_model.pkl` lacks the fields above, rather than running through with
  incomplete predictions.

**Where:** `_types.py` (`FinalModelResult`), `orchestration.py`
(`_train_final_model`), `scripts/pipeline/03_apply_transfer.py`,
`tests/test_transfer.py`, `Makefile`
(`smoke-test-transfer-nonmri-lean`, `smoke-test-all`)


### 1.10 The confounder snapshot in `final_model.pkl` is shuffled per column

**Decision:** The `confounders_train` snapshot from §1.9 is not pickled as it
stands. Each column is permuted independently with the pipeline seed and the
frame gets a fresh `RangeIndex`. The per-column marginal distributions — all the
transfer report's `_ks_confounder_drift` needs — are unchanged, but the
per-participant tuple `(age, sex, basis_uort)` can no longer be reconstructed
from the pickle.

**Rationale:** Written straight, the snapshot carries one complete demographic
tuple per participant of the source cohort. At roughly 8,500 participants, age
to the year combined with sex and one of eighteen study centres is enough to
make re-identification against an external source practical. And the pickle is
precisely the object that travels: it is how a source run reaches a target run,
including onto a machine outside the perimeter where the data access lives.

KS distance is a *column-wise* statistic — `Series.dropna().to_numpy()` per
column, then `ks_2samp` — so row order is irrelevant to it. A per-column shuffle
is therefore the least invasive way to remove the re-identification risk without
changing a single number in the transfer report.

Two alternatives were rejected:

- *Dropping the snapshot entirely* would force the transfer script to re-read
  the source parquets, which breaks any transfer onto a machine that does not
  have them.
- *Storing only per-column quantiles* would change the statistic: KS on 100
  quantiles is not KS on the full empirical distribution, which makes it
  unusable as the drift measure a reader is asked to trust.

**Where:** `_types.py` (`FinalModelResult` — the field name is unchanged, so
existing readers keep working), `orchestration.py` (`_train_final_model`,
seeded per-column shuffle before `pickle.dump`)

**Verification:** the round-trip and KS-helper tests in `tests/test_transfer.py`
and the `confounders_train` assertion in `tests/test_orchestration.py` pass
unchanged, and the transfer smoke run produces numerically identical KS
distances with and without the shuffle.

### 1.11 The claims-based proxy indexes every participant, not every case

**Decision:** `kvad_pcs_proxy.parquet` contains exactly those participants who
have at least one claims diagnosis row surviving the pre-pandemic and certainty
filters. A participant whose codes map to none of the Bahmer symptom groups gets
an explicit `pcs_t0_proxy_count = 0` with all indicators at zero, rather than
being dropped from the output.

**Rationale:** Building the output index from the union of per-symptom hits
loses every participant whose pre-pandemic claims carry only codes outside the
mapping — 2,788 participants on production data, taking the output from 35,372
to 38,160 rows. The score is documented as a *floor* on somatic symptom load,
and a floor with no zero-valued observations is not a floor: it is a restriction
to the symptomatic sub-cohort. Downstream, that would pool the healthy controls
with the participants who have no claims linkage at all, which is exactly the
distinction the proxy exists to make.

One structural property follows: `pcs_t0_proxy_n_diagnoses` is ≥ 1 for every
output row by construction (NA ids are filtered out of the index defensively),
and reaches about 5,200 in practice. Values like 999 or 9999 are therefore
legitimate counts here rather than NAKO sentinels, and the quality test's
allowlist marks the column accordingly.

**Where:** `data_processing/pcs_baseline_proxy.py`
(`_aggregate_to_bahmer_indicators`), `tests/test_pcs_baseline_proxy.py`
(`test_aggregate_keeps_zero_score_participants` as the regression guard),
`tests/test_data_processing_quality.py` (`SENTINEL_EXEMPTIONS`)

### 1.12 `current_smoker` stays alongside `smoking_status`

**Decision:** The `medical_history` block carries both tobacco encodings — the
numeric `smoking_status` (1/2/3) and the derived binary `current_smoker`
(`smoking_status == 3`). `current_smoker` is *not* in the drop list of
`create_medical_history_preprocessor`, even though it is a deterministic
function of another column.

**Rationale:** The drop list removes `has_surgery_history` and
`has_cancer_history` as collinear, and `phq9_sum_items` as a literal duplicate.
`current_smoker` is neither: it holds different values, and it is linearly
independent of `smoking_status`. The redundancy is the point. The
`medical_history` base learner is an elastic-net `LogisticRegressionCV`, and
`smoking_status` enters it unencoded as a number — which asserts a
never < former < current spacing that nothing justifies. The binary supplies
that one contrast without the assumption.

**The cost, stated plainly:** stability selection scores the two columns
separately and splits the tobacco signal between them. The *effect* is therefore
not read off the selection frequencies; it is reported by
`scripts/supplementary/smoking_adjustment.py`, which expands the status into
never/former/current indicators for exactly this reason. Per-feature SHAP
attribution does not exist here either way — `shap_analysis.py` runs on the
meta-learner over modality-level out-of-fold scores.

**The alternative, for a future rerun:** replace `smoking_status` with
`former_smoker` plus `current_smoker` — the same information without the double
encoding. It is a change to the model with visible consequences, so it is not
available without a rerun.

**Where:** `preprocessing.py` (`create_medical_history_preprocessor`),
`data_processing/smoking.py` (`_extract_smoking`)

---

## 2. Pipeline-Methodik

### 2.1 The primary outcome is the Bahmer weighted score, not an IRT score

**Decision:** The primary PCC outcome is
`bahmer_any_pcs = (bahmer_pcs_score > 10.75)`. IRT-based scoring is not part of
this codebase.

**Rationale:** The kmatrix routing question `d_co2_k0` splits the infected in
two: those who reported complaints (k0=1, kmatrix completed) and those who did
not (k0=2, kmatrix skipped). The roughly 36,000 participants in the second group
contribute structural zeros to an IRT model — not item responses, but artefacts
of the questionnaire's routing. That violates the IRT assumption that every
response is a function of the latent trait. Since IRT and the Bahmer weighted
score already agree at κ = 0.913, the latent measurement model buys nothing over
the established score, while costing an external R/mirt dependency and a
separate argument to defend.

**Where:** `data_manager.py` (`load_pipeline_data`), `orchestration.py`,
`scripts/pipeline/05_compute_statistics.py`,
`scripts/supplementary/outcome_agreement.py`, `pyproject.toml`

---

### 2.2 Clean controls: PCC-negative means k0=2 only

**Decision:** The PCC-negative control arm is restricted to participants with
`d_co2_k0 == 2` — an explicit self-declaration of no post-COVID complaints.
Infected participants who declared complaints (k0=1) but fall below the outcome
threshold are excluded from the analytic sample rather than carried as controls.

**Rationale:** The analysis tests a hypothesis rather than estimating a
population quantity. When a continuous outcome is dichotomised, a design that
treats everything below the threshold as a control mixes the sub-threshold
symptomatic with the genuinely symptom-free. That makes the control
distribution bimodal and attenuates effect sizes systematically (Steyerberg
2019; Riley 2021). At roughly 8,500 participants there is power for the clean
contrast, and losing the sub-threshold group — a few hundred participants under
the Bahmer target, about a thousand under the neurocognitive one — is
statistically affordable.

**Where:** `data_processing/corona.py` (`process_corona2`, which carries
`d_co2_k0` into the parquet), `data_manager.py` (`clean_controls`, default
True), `scripts/pipeline/02_run_pipeline.py` (`--no-clean-controls` for the
mixed-controls sensitivity run)

---

### 2.3 Ten outer folds, not five

**Decision:** `n_outer_folds = 10` for production runs. The inner CV stays at 5.

**Rationale:** The Nadeau-Bengio test on 5 outer folds runs at df = 4, which is
too little power for the pairwise modality comparisons it is used for. Ten folds
give df = 9. The inner CV stays at 5: raising it costs time in every outer fold
and buys no stability in hyperparameter selection. The outer change costs about
1.5× runtime per target.

**Where:** `config.toml` (`n_outer_folds`)

---

### 2.4 Fourteen base learners, not nine

**Decision:** The architecture has 14 base learners: 9 modality groups, with
brain MRI contributing six of them.

**Rationale:** MRI is one modality group, but each of the six atlases
(Desikan-Killiany, Destrieux, Julich, subcortical, Yeo networks, cerebellar)
has its own base learner with its own pipeline. Counting modality groups gives
nine and describes something other than what is fitted.

**Where:** `modality_pipelines.py`

---

### 2.5 Solver choice per modality

**Decision:** Base learners use SAGA for L1 and elastic-net penalties, and
LBFGS for the pure-L2 modalities (demographics, cardiovascular, lung function).

**Rationale:** SAGA is the only sklearn solver that supports L1, so it is
required wherever the penalty selects features. For pure L2, LBFGS converges
faster and uses less memory. The choice is not automatic: `_lr_cv()` defaults to
SAGA and the three L2 modalities pass `solver="lbfgs"` explicitly. Removing one
of those arguments as redundant refits that modality under SAGA — no error, and
different coefficients.

**Where:** `modality_pipelines.py` (`_lr_cv` and the three pipeline
constructors that override the solver)

---

### 2.6 A failed residualisation is a warning, not a debug line

**Decision:** A feature that cannot be residualised is logged at WARNING, not
DEBUG.

**Rationale:** When residualisation fails the original feature is passed through
unchanged — with no confounder adjustment at all. That is a silent methodological
deviation, and it has to be visible. At DEBUG level it would not appear in a
production log.

**Where:** `orthogonalization.py`

---

### 2.7 Class balancing in the meta-learner (`scale_pos_weight`)

**Decision:** The XGBoost meta-learner gets
`scale_pos_weight = n_neg / n_pos`, computed inside `fit()`.

**Rationale:** The base learners balance classes through sklearn's
`class_weight="balanced"`; `scale_pos_weight` is the XGBoost equivalent, and
without it the meta-learner is the one stage of the stack that does not balance
at all. It is computed per fit rather than fixed because the class ratio differs
slightly between folds — prevalence is stable enough that a constant would be
defensible, but there is no reason to prefer one.

**Where:** `meta_learner.py`

---

### 2.8 Calibration is skipped on small samples

**Decision:** `MetaLearner.fit()` skips Platt calibration when `n < 100` or
`n_positive < 30`, and falls back to the uncalibrated model.

**Rationale:** Platt scaling over a 3-fold CV needs enough positives per fold for
a stable sigmoid. Below that the calibration is unstable and can make the model
worse than leaving it alone. The condition never fires on the production sample,
where prevalence is around 12 % of some sixteen thousand participants; it exists
for anyone reusing this code on a smaller dataset.

**Where:** `meta_learner.py`

---

### 2.9 Split-MH sub-modalities match what the preprocessor leaves

**Decision:** Under `split_mh_submodalities=True`, the five sub-modality column
lists in `data_manager.MH_SUBMODALITY_COLUMNS` are aligned exactly with the
layout `create_mental_health_preprocessor` produces. Columns the global mental
health preprocessor drops anyway do not appear in the sub-modality frames
either.

The consequence is that `mh_gad7`, `mh_panic` and `mh_stress` are
single-feature modalities, `mh_phq9` has two features before the amendment
items and `mh_mini` has seven.

**Rationale:** Listing a column that the preprocessor removes does not add it to
the model — it only makes the documentation disagree with the fit. `mh_gad7`,
for instance, named three columns of which the preprocessor kept one: the
binary cut-off is redundant with the sum score, and the diagnosis flag
correlates with it at r = 0.91. The promised three-feature GAD-7 sub-model was a
one-feature model, and the SHAP-share attribution was reported against columns
the classifier never saw.

Two alternatives were considered and rejected:

- *A sub-modality-specific preprocessor* that bypasses the global one would
  override drops made for statistical reasons — zero variance,
  multicollinearity above 0.9, missing-not-at-random imputation — and give the
  sub-modalities unstable features. That is worse than the single-feature
  result, not better.
- *Re-specifying the drops in the mental health preprocessor* would change the
  monolithic `mental_health` modality in the default stack, and with it the
  primary analysis. Not a reasonable price for a question about a secondary
  mode.

Single-feature sub-modalities are not a problem in a stacking setup: stability
selection reduces to keeping the feature, the logistic base learner becomes a
univariate indicator, and the meta-learner attributes SHAP to exactly that
score. That is the quantity the split mode exists to report — which mental
health screener contributes most.

**Where:** `data_manager.py` (`MH_SUBMODALITY_COLUMNS`), `preprocessing.py`
(`create_mental_health_preprocessor`, which supplies the canonical drop list
both modes align to), `tests/test_data_manager.py`

---

### 2.10 One definition of "has MRI"

**Decision:** `05_compute_statistics.py` determines `has_mri` from presence in
any of the six atlas parquets — the same definition
`data_manager._mri_participant_ids` uses to select the cohort. The extra
condition "and at least one non-missing value" is gone.

**Rationale:** There were two definitions, two apart. The pipeline trained on
19,242 participants while the descriptive tables described 19,240, and a
participant-flow figure that subtracts one from the other does not add up. The
two participants in the difference carry a completely empty row in the
Desikan and Yeo parquets: they are in the analytic sample and drop out later, at
the modality-readiness cut, not at cohort selection.

**What this changes:** descriptive tables now run on N = 8,464 rather than
8,462 — the same sample the models are fitted on. Table 1 gains two
participants, and the MRI participation table reads 19,242/98,219 instead of
19,240/98,221. No reported percentage moves by more than one decimal place.

**Where:** `scripts/pipeline/05_compute_statistics.py`

---

### 2.11 Modality ablation, not "permutation dropout"

**Decision:** The method is called modality ablation. It replaces a modality's
out-of-fold predictions with `NaN` rather than permuting or removing the column.

**Rationale:** "Permutation dropout" would describe shuffling the values, which
is not what happens. Setting the column to NaN uses XGBoost's native missing
handling: the samples take the default split path, which is what actually
happens when a modality is unavailable, and it matches how systematic
missingness is treated during training. Zeros would be a value the model can
learn from, and shuffling would keep the marginal distribution while destroying
the association — a different question.

**Where:** `orchestration.py` (`_modality_ablation`), `_types.py`
(`PipelineResult`)

---

### 2.12 Out-of-sample ablation and incremental performance

**Decision:** `_modality_ablation_oos` and `_incremental_performance_oos`
compute both analyses out-of-sample at the meta-learner level, alongside the
in-sample versions.

**Background:** The in-sample variants train a final meta-learner on the
out-of-fold prediction matrix and evaluate on that same matrix. That gives every
modality a positive contribution before any signal is involved, and the ranking
it produces is not the out-of-sample ranking. Reporting those numbers means
saying that they are in-sample.

**How it works:**

- `_nested_cv` keeps, per outer fold and in memory only: the train and test
  indices, both label vectors, the inner out-of-fold matrix on the training
  fold, the base learners' predictions on the test fold, the modality names,
  and the fitted meta-learner.
- **Ablation** masks the test-fold base-learner predictions column by column
  with NaN, rescores them with that fold's meta-learner, concatenates across
  folds and computes ROC and PR against the concatenated labels. Writes
  `modality_ablation_oos.csv`.
- **Incremental** trains a baseline-only meta-learner (demographics) on the
  training fold's out-of-fold matrix, plus one [baseline + modality] learner per
  non-baseline modality, evaluates both on the test fold's predictions, and
  concatenates across folds. Hyperparameter search is off so the modalities stay
  comparable to each other, as in the in-sample version. Writes
  `incremental_performance_oos.csv`.

**When it is skipped:** a distributed fold run and `merge_fold_results` never
populate the per-fold structures, so both analyses are skipped silently in those
paths. A full run gets them automatically.

**Verification:** four tests in `tests/test_orchestration.py` cover shape, CSV
output and the empty-fold case. On synthetic data the out-of-sample metrics come
out well below the in-sample ones, which is the expected direction — the
meta-learner has not seen the test samples during training.

**Where:** `_types.py` (`PerFoldOoS`, `PipelineResult`), `orchestration.py`
(collection in `_nested_cv`, the two methods, their calls in
`_run_post_cv_steps`), `tests/test_orchestration.py`

---

### 2.13 The Nadeau-Bengio comparison ran against the wrong reference

**Decision:** `_compare_modalities` passes
`reference_modality="mental_health"`, and the parameter has no default — every
call site has to name it. When the reference modality is absent from the fold
scores, `compare_modalities_pairwise` raises instead of returning an empty
table.

**Rationale:** The call was set to `"demographics"` from the start, while the
table it feeds describes mental health as the reference throughout and marks it
as the reference row. The reported p-values were therefore the comparisons
against demographics, entered one row off: the value printed beside
"demographics" belonged to mental health.

**What this changes:** under the reference that was described, mental health
separates from **all** fourteen other modalities — the largest Holm-adjusted p
is .002, with SES at .001 and medical history at .002. The qualification
"except socioeconomic status and medical history" falls away entirely. It was
never a statement about mental health; it said that SES and medical history do
not separate from *demographics*.

**Why there is no default any more:** the returned table does not contain the
reference, so without knowing it no column means anything. A default invites
exactly the error that survived here unnoticed.

**Regenerated:** `results/runs/*/modality_comparisons_nb.csv` — a pure function
of `modality_scores_cv.csv`, so no CV run was needed. The split-MH run gets no
file at all: mental health exists there only as five sub-scales, and that run
supplies no pairwise p-values downstream.

**Where:** `orchestration.py`, `statistical_tests.py`,
`tests/test_statistical_tests.py`

---

### 2.14 Stratified bootstrap rather than plain

**Decision:** Bootstrap confidence intervals resample the positive and negative
classes separately.

**Rationale:** At roughly 12 % prevalence a uniform bootstrap sample can contain
no positive cases at all. ROC-AUC and PR-AUC are undefined on such a sample,
which produces NaNs and a broken interval. Stratified resampling keeps every
bootstrap sample at the original class ratio.

The cost is worth naming: fixing the prevalence removes one real source of
variation, so the resulting interval is narrower than an unstratified one — for
PR-AUC noticeably so, because that metric depends on prevalence directly. At
this sample size the degenerate case the stratification protects against cannot
actually occur, which makes the narrowing the only effect it has.

**Where:** `evaluation.py` (`compute_bootstrap_ci`, `stratify`)

---

### 2.15 ECE over equal-count bins, not equal-width

**Decision:** `expected_calibration_error` and `quantile_bin_edges` live in
`pcc_analysis.evaluation` and bin by quantile. `EvaluationMetrics.compute_ece`
delegates there, as do the figure scripts that each held their own copy.

**Rationale:** There were two definitions in the repository — the pipeline
binned equal-width over [0,1], the calibration figure by quantile — and they
can agree closely enough on one dataset to go unnoticed while disagreeing by a
factor of three on another. The predictions here fall between 0.07 and 0.62, so
equal-width binning puts 8,461 cases into six bins, one of them holding three
people, and deviations in opposite directions inside a wide bin cancel. The
quantile variant is the stricter one, the usual choice when the predicted
probabilities are concentrated, and it is the partition the plotted curve
actually uses.

**What this changes:** the primary ECE moves from 0.005 to 0.015, split-MH from
0.004 to 0.011, no-DML from 0.007 to 0.011, Lean-MRI from 0.006 to 0.007,
Lean-non-MRI from 0.007 to 0.004, the transfer from 0.003 to 0.004, and the
pooled Lean run not at all. The direction is not uniform: this is a different
partition, not a systematically stricter yardstick. Calibration slope and
intercept are untouched, and they carry the calibration claim.

**Regenerated:** `ece` in `results/runs/*/metrics.csv` and in
`transfer_summary.json` — a deterministic function of `predictions.csv`, so no
CV run was needed. Before overwriting, every other column was recomputed and
checked against its stored value; all eleven runs reproduced to 1e-9, which is
what establishes that only the ECE moved.

**Where:** `evaluation.py`,
`scripts/figures/regenerate_evaluation_curves.py`,
`scripts/pipeline/04_generate_figures.py`, `tests/test_evaluation.py`

---

### 2.16 The interaction's confidence interval came from a stratum

**Decision:** `analysis_difference_in_association` binds the interaction's
confidence bounds to their own names (`inter_lo`, `inter_hi`) before the
stratum loop rebinds `lo` and `hi`.

**Rationale:** The loop over the two arms ran after the interaction estimate and
rebound `lo`/`hi`, so `interaction_ci` carried the interval of the uninfected
stratum: an odds ratio of 0.89 reported with a confidence interval of
[6.73, 7.72]. Each number is plausible on its own, the JSON schema is satisfied,
and only the arithmetic between them gives it away. The correct interval is
[0.81, 0.97].

**Where:** `scripts/supplementary/reporting_style_robustness.py`,
`tests/test_reporting_style_robustness.py`

---

### 2.17 A sensitivity run without DML orthogonalization

**Decision:** An additional run with `--no-orthogonalize`, to check whether
residualising the demographics out masks an age- or sex-dependent second-hit
effect.

**Rationale:** DML removes demographic association from the features, linear and
non-linear alike. If the second-hit effect is itself modified by age or sex —
plausible, since both are candidate modifiers of PCC risk — the orthogonalised
analysis would attenuate it. The non-DML run is the robustness check: the
modality ranking and the MRI null should be invariant to it.

**Where:** `scripts/pipeline/02_run_pipeline.py` (`--no-orthogonalize`); the
pipeline API has always carried `orthogonalize` as a parameter.

---

### 2.18 Random-seed propagation

**Decision:** Every stochastic component takes its seed from
`config.toml`'s `random_state`. None of them draws from global state.

**The chain:** Pipeline → ModalityPipelineFactory → {StabilitySelection,
LogisticRegressionCV, Orthogonalizer, PCA}; Pipeline → MetaLearner →
{XGBClassifier, RandomizedSearchCV, StratifiedKFold for the inner folds};
Pipeline → `_nested_cv` → outer StratifiedKFold; Pipeline →
`_permutation_test_cv` → `np.random.default_rng`; Pipeline → `_evaluate` →
`evaluate_model_comprehensive` → `compute_bootstrap_ci` →
`np.random.default_rng`. `KNNImputer` is deterministic and needs no seed.

A single break in that chain would not be visible: the run completes, and only
a second run on the same data shows the difference. That is why the chain is
written out here rather than asserted.

---

### 2.19 Figure reproducibility

**Decision:** The jitter in the modality-contribution strip plot draws from a
global `np.random` state seeded to a fixed value, and the previous state is
restored afterwards.

**Rationale:** seaborn takes its jitter from the legacy global stream rather
than from a generator it is handed, so two runs over identical data placed the
points differently. `modality_contributions.{pdf,pgf,png}` was then the one part
of a run directory that could not be diffed against another run — a gap in a
repository that uses byte-identity as a check (§2.20). Everything else draws
from its own `default_rng`, which is why saving and restoring the global state
here has no other consequence.

**What remains:** `evaluation_curves.pdf` and `decision_curve_analysis.pdf`
still differ between two identical runs, but only in the font-subset prefix
matplotlib assigns when rendering through TeX (`ZAKFNR+CMMI8` against
`ZSQGSQ+CMMI8`); the content stream is identical. `PYTHONHASHSEED=0` does not
fix it — that was measured, not assumed. Comparing two runs therefore means
comparing the 21 non-PDF files.

**Where:** `evaluation.py` (`_seeded_global_rng`,
`plot_modality_contribution_boxplot`), `tests/test_evaluation.py`

---

### 2.20 One dependency moves the numbers: XGBoost

**Decision:** Every runtime and development dependency is pinned to the latest
release at the time of pinning, **including** XGBoost 3.4.0. The bump precedes a
full rerun deliberately, so that the results shipped and the environment
documented are the same one.

**Rationale, and what it costs:** Of all the packages bumped, exactly one
changes the results. On the smoke graph — identical synthetic fixtures, same
seed — against the previous state:

- numpy 2.4.3 → 2.5.2 **and** scipy 1.17.1 → 1.18.0: 16 of 16 output files
  byte-identical.
- sklearn 1.8.0 → 1.9.0: checked separately, the same 16 files byte-identical.
  The one file that is not is `mri_positive_control.json`, which is outside the
  smoke set: its `RidgeCV` drifts from the 13th significant digit
  (`r2` 0.3480007246595168 → …5267) as a consequence of the 1.9 switch from
  `auto` to `eigen`. The selected `alpha_` is unchanged and the rounded output
  is byte-identical, so nothing reported moves.
- xgboost 3.2.0 → 3.4.0: **13 of 16 files differ**, including
  `predictions.csv`, `metrics.csv` and every SHAP output. The full bump
  reproduces the XGBoost-only bump exactly, which is what identifies XGBoost as
  the sole source of the drift.

The most plausible trigger is the `min_child_weight` bugfix in 3.4.0 — "could
produce an empty root node" — with a search space of `[1, 3, 5]`, together with
the histogram rework in 3.3.0. The categorical handling enabled by default in
3.3.0 does not reach this pipeline: the meta-learner receives a float
`ndarray`, not categorical columns.

The sampling-reproducibility fix mentioned in the 3.3.0 notes is **not** an
argument here. Measured, the same model is already order-independent under
3.2.0 — bit-identical predictions whether fitted cold or after five unrelated
sampling fits — because `random_state` is set explicitly throughout. Both
versions are deterministic. 3.4.0 is not more reproducible; it is different.

**What follows for any rerun:** the cheapest sanity check of the whole chain
would be that a rerun without feature changes reproduces the previous run's
ROC-AUC exactly. That check does **not** survive an XGBoost change: a deviation
can no longer be attributed to the feature change rather than to the library.
Checking the chain requires a run pinned to the version the earlier result came
from. This was a deliberate trade in favour of a current, uniformly pinned
stack.

**Side decision — declaring supported platforms instead of overriding numba:**
shap caps numba below 0.63, but only on Intel macOS. Resolving that platform
too forks the lock and writes numba 0.53.1 / llvmlite 0.36.0 alongside the
current pair; llvmlite 0.36.0 does not build on Python 3.14, so the fork is not
an installable resolution there — only noise in the lock. Intel macOS is out of
scope (development on arm64, CI on ubuntu), so `[tool.uv] environments` names
the supported ones. That collapses the fork to 56 packages at exactly one
version each, which is the point of pinning, without discarding anyone else's
constraints: a future numba lower bound from shap still resolves normally. An
`override-dependencies` entry would not — it would silently outvote every later
numba requirement.

**Known ceiling — matplotlib:** seaborn 0.13.2, the current release, calls
`ax.bxp(vert=…)`, deprecated in matplotlib 3.11 and scheduled for removal in
3.13; the test run raises 38 deprecation warnings from
`evaluation.py:plot_modality_comparison`. Until seaborn follows, matplotlib
3.12 is the upper bound — a bump past it makes figure generation fail at the
end of a run that takes hours.

**Where:** `pyproject.toml` (`dependencies`, `dependency-groups`, the
`[tool.uv]` block, `PLR0917` in the ruff ignore list), `uv.lock`

**Verification:** lint, typecheck and the test suite pass, and the smoke
pipeline runs through. In addition, the supplementary analyses were run against
both stacks, the pre-bump one from a worktree via `uv sync --frozen`: every
script runs under both, eleven of the twelve generated files are byte-identical,
and the twelfth is the `mri_positive_control.json` described above.

---

### 2.21 The library stack is stamped into `config.csv`

**Decision:** Every run writes the versions of the result-relevant libraries as
a `package_versions` field in `config.csv` — Python, numpy, pandas,
scikit-learn, scipy, shap, statsmodels, xgboost. The stamp feeds
`_config_hash()`.

**Rationale:** §2.20 establishes that an XGBoost change moves the numbers, and
that only a run pinned to the earlier version can check the chain. Without a
stamp a `results/…` directory cannot be attributed to a version at all: a run
that predates the stamp carries no evidence of the stack it ran under. The
lockfile does not help — it describes the checkout, not the run.

Feeding the stamp into the config hash is deliberate. Distributed fold runs on
machines with unequal stacks now fail at the merge with a config mismatch
instead of quietly mixing folds from two environments. With dependencies pinned
exactly, that can only fire when something has genuinely diverged.

**Where:** `_provenance.py` (`package_versions`), `orchestration.py`
(`_get_config`), `_types.py` (`PipelineConfig`),
`scripts/pipeline/03_apply_transfer.py`, `tests/test_orchestration.py`

---

### 2.22 UTC in the provenance record, local wall clock in directory names

**Decision:** A naive `datetime.now()` no longer appears anywhere (enforced by
ruff's `DTZ` rules). The two stamps that go into a provenance record —
`config.csv` and `transfer_summary.json` — use `datetime.now(UTC)`. The three
that form a directory or log-file name use `datetime.now().astimezone()` and
keep the local wall clock.

**Rationale:** A bare `2026-01-15T23:57:41` in a record read by someone in
another time zone cannot be resolved; that is where the offset belongs. A run
directory, by contrast, is browsed by hand and should sit next to the clock of
the machine that started the run. Switching it to UTC would also have shifted
new directory names against existing ones by the daylight-saving offset.
`astimezone()` makes local time explicitly zone-aware without changing the
digits, and satisfies the same lint rule the UTC variant does.

**Where:** `orchestration.py` (`_setup_logging`, `_get_config`),
`scripts/pipeline/02_run_pipeline.py`, `scripts/pipeline/03_apply_transfer.py`

---

### 2.23 Splitting a rerun across two machines

**Decision:** A full rerun can be split across two machines, and five additions
support it: step selection in `run_analysis.sh`, a comparability check across
run directories, a data fingerprint and device stamp in `config.csv`, a stack
check in the transfer script, and an exact Python pin.

**The runtime distribution decides the split.** The two Lean fits on the large
cohorts take roughly 47 h and 25 h, while each MRI-cohort run takes 1.5–2 h —
about 84 h in total, 71 of them in two runs. The split that balances is
therefore by configuration (`--only lean_pooled` against
`--except lean_pooled`), not by fold: the post-CV part is serial, so splitting
the longest configuration by fold buys around 10 % of makespan over the
configuration split and mixes folds from two machines into one model to get it.

**Why the existing guards were not enough.** `_config_hash()` (§2.21) protects
exactly one comparison: merging the folds of a *single* run. That is the one
operation a rerun split by configuration never performs. Each run directory is
internally consistent; what can diverge lies between them — and that is where
the control delta, the transport panel, the clinical-utility panel and the
sub-modality panel compute differences, none of which compared library stacks.

**The five additions:**

1. *Step selection.* `run_analysis.sh --only/--except/--list`. Without it the
   `02_run_pipeline.py` commands are run by hand, which loses precisely the
   guards built for this split — the split-MH check, the control deltas, the
   constants snapshot and its diff. A selected step whose input lives on the
   other machine stops and names the directory rather than running on nothing,
   and a misspelt step name exits 2 rather than passing as a successful no-op.
2. *`run_comparability.py`.* Reads `config.csv` and `provenance.json` from
   several run directories and separates disagreements into *decisive*
   (`package_versions`, `device` — either one makes any delta unreadable) and
   *worth reporting* (`machine`, `platform`, `commit`). The three panels that
   read two run directories against each other refuse to compute on a decisive
   difference, and `scripts/check_run_stacks.py` checks all present directories
   against each other at the end of every run. It exits non-zero even when
   everything was computed: the artifacts are usable, they are just not
   readable against each other, and an exit code is the only signal that
   survives an overnight `nohup` run.
3. *`device` in the config hash.* `cuda_available()` decides the XGBoost
   backend, but it is *detected* rather than configured and appeared nowhere in
   the run directory. GPU and CPU folds would have mixed silently.
4. *`data_fingerprint` in the config hash.* The merge checks `y_te` per fold —
   the labels, not the feature matrix. Two machines each running
   `01_process_data.py` themselves can hold parquets at different states and
   still pass every label check. The fingerprint is content-based
   (`hash_pandas_object` over `y` and every modality frame, including column
   names and order) rather than over file bytes, so regenerated parquets with
   identical values raise no false alarm. It is deliberately **not** compared
   across run directories: the MRI and non-MRI runs are supposed to see
   different data.
5. *An exact Python pin.* `.python-version` named the series while
   `package_versions()` stamps the patch and feeds it into the hash. Two
   machines that ran `uv sync` weeks apart would have computed folds under
   different patch releases and discovered it hours later, at a refused merge.
   `requires-python` and pyrefly stay on the series.

**Also in the transfer script:** `03_apply_transfer.py` unpickles the source's
`final_model.pkl` — the one artifact that carries a *fitted model* between two
runs, and therefore between two machines. The source's stack is now compared
against the running one and a difference aborts. A source run from before the
stamp existed is reported, not rejected.

**What the guards key on.** Every step guard tests for `config.csv`, not for
the existence of the directory. A run creates its directory and
`provenance.json` immediately and writes `config.csv` and `metrics.csv` only at
the end — so an aborted run, an rsync in flight and a fold-subset directory all
look like a finished run. Testing the last artifact written is the only check
that tells the four cases apart.

**A pre-flight for the data.** `02_run_pipeline.py --fingerprint-only` loads a
configuration's data, prints the overall digest plus one per modality, and
exits. The fingerprint is otherwise compared only at the fold merge — which a
configuration split never reaches — so it would have no effect on exactly this
division of work. This makes it comparable on both machines *before* the run,
and a mismatch names the modality rather than merely its existence.

**What was deliberately left alone:** modality parallelism in
`create_oof_predictions`. Both callers leave `n_jobs` at 1, and that step is 41
of `lean_pooled`'s 47 hours — about 4 h per fold, against 10 s for the
meta-learner. Enabling it is plausible, since the modalities are independent
and the backend is threads, but the stability selection underneath already runs
at `n_jobs=-1`, so oversubscription is as likely as a speed-up. That is a
measurement on one fold, not an assumption, and it belongs before a rerun
rather than inside one. Evidence that a fold still produces a bit-identical
`predictions.csv` would be the precondition.

**Where:** `scripts/run_analysis.sh`, `scripts/check_run_stacks.py`,
`run_comparability.py`, `orchestration.py` (`_get_config`, `run`,
`merge_fold_results`), `_provenance.py`, `_types.py`,
`scripts/pipeline/03_apply_transfer.py`, the three panel scripts,
`.python-version`, `README.md`, and the corresponding tests

---

### 2.24 Ask XGBoost about the GPU instead of hoping for an exception

**Decision:** `cuda_available()` checks two things — whether the installed
XGBoost build supports CUDA at all (`build_info()["USE_CUDA"]`) and, only then,
whether a minimal fit completes without the fallback warning. The alternative is
to fit a tiny model with `device="cuda"` and read *any* exception as "no GPU".

**Rationale:** XGBoost 3.x does not raise. It moves the fit to the CPU and warns
(`Device is changed from GPU to CPU as we couldn't find any available GPU on the
system`). A probe built on the exception therefore answers `True` on every
machine, including a macOS wheel built with `USE_CUDA=false`. It became visible
only when §2.23 wrote the device stamp into `config.csv` and it read `cuda` on a
machine with no CUDA support of any kind.

Three things depended on the answer. `train_meta_learner` passes `n_jobs=1` to
the `RandomizedSearchCV` when it believes it is on a GPU, and `_nested_cv` does
the same for the permutation importance — so both ran single-core, at 50
hyperparameter draws rather than 10. The third is the worse one: a stamp that
says `cuda` everywhere compares one untruth against another and lets a GPU
machine and a CPU machine pass as the same environment, which is exactly what
§2.23 exists to prevent.

**Verification:** an A/B on the smoke graph. The `primary` configuration
produces **byte-identical** `predictions.csv`, `metrics.csv` and twelve further
result files before and after, while the runtime falls from 81 s to 75 s on a
fixture where both affected steps are tiny. The correction moves no number; it
returns the parallelism.

**Limit of the construction:** the second condition is a substring match on an
upstream warning. If XGBoost rewords it, `cuda_available()` will again answer
`True` on a CUDA build with no GPU — the dangerous direction. The first
condition is unaffected and covers the case that actually occurs here.
`tests/test_gpu_utils.py` pins both branches.

**Where:** `_gpu_utils.py`, `tests/test_gpu_utils.py`

---

### 2.25 Type-check structurally, do not replace modules with `Any`

**Decision:** `replace-imports-with-any` is empty. It previously held
`sklearn.base.*` and `sklearn.pipeline.*` — two modules whose every use in the
project went unchecked as a result. What that exposed is typed rather than
suppressed:

- `protocols.py` gains `Regressor`; `Classifier` gains `get_params`/`set_params`,
  and both require `y` in `fit` — a model that can fit without a target is
  neither.
- `_sklearn_typing.clone()` passes the estimator type through, where sklearn's
  own annotation collapses it to `BaseEstimator` and loses `predict` and
  `predict_proba` at all seven call sites.
- `preprocessing.FrameTransformer` is the common base of the four DataFrame
  transformers. It holds `set_output` (previously copied four times) and
  declares `fit_transform`, because `TransformerMixin.fit_transform` is typed
  `-> ndarray`: correct for sklearn's own transformers, wrong for these, and the
  reason every caller here reasoned about an array that never existed. `fit` and
  `transform` are abstract, so the DataFrame contract is stated once rather than
  implied four times.
- The factory methods in `modality_pipelines.py` declare `ConfoundedPipeline`
  instead of `Pipeline` — which is what they return anyway.

**Rationale:** sklearn describes estimators nominally, this repository
structurally. An `Any` shim absorbs that conflict, but it absorbs everything
else too: `Pipeline` is the class subclassed here, and `BaseEstimator` and
`TransformerMixin` are the base of every transformer. Two configuration lines
looked cheaper than suppressions and cost the checking of 93 files.

Eleven suppressions remain, each at the point where the decision is made: four
signature narrowings in `ConfoundedPipeline` (the class accepts only what its
transformers handle, where `Pipeline` accepts any iterable), three
`fit_transform` declarations that depart from the mixin signature (two of them
taking `confounders`), one sklearn signature pyrefly cannot see through the
decorated `__init__`, and three tests that pass wrong types on purpose.

Two casts mark the nominal/structural boundary explicitly: in
`_sklearn_typing.clone()` and at the `CalibratedClassifierCV` constructor.

**Where:** `pyproject.toml` (`[tool.pyrefly]`), `protocols.py`,
`_sklearn_typing.py`, `preprocessing.py` (`FrameTransformer`),
`custom_pipeline.py`, `modality_pipelines.py`, `orthogonalization.py`,
`stability_selection.py`, `meta_learner.py`

**Verification:** `pyrefly check` reports 0 errors with 11 suppressions, where
it previously reported 0 with 4 but two sklearn modules unchecked; the test
suite passes; and the smoke pipeline is byte-identical in all 21 non-figure
files against the state before.

---

### 2.26 The Lean panel reads named run directories, not a timestamp glob

**Decision:** `lean_clinical_utility.py` resolves its runs as
`results/runs/lean_mri` and `results/runs/lean_non_mri`, rather than taking the
most recently modified match of a `pcc_pipeline_*_lean_bahmer_*` glob.

**Rationale:** Superseded timestamped directories stay on disk. The glob picks
them up, and the script then writes a complete, plausible panel from the wrong
run with nothing to indicate it. A panel that is silently about a different
analysis is worse than one that fails to build.

**Where:** `scripts/supplementary/lean_clinical_utility.py`

---

### 2.27 No feature count beats a plausible wrong one

**Decision:** `MODALITIES` in `05_compute_statistics.py` carries no
`n_features` values. When a parquet is missing the count is `None` and the row
prints "n/a"; if even one of the six atlas parquets is missing, the same applies
to the MRI row.

**Rationale:** The entries were declared a fallback and documented as
unmaintained, which is exactly why they were stale — MRI read 314 against an
actual 599, lung function 7 against 9. They became visible only when a parquet
was missing: the one moment the row has nothing to describe, and it put a
plausible number next to the words "not available". The MRI total had the same
property more quietly, summing the atlases that were present and looking like a
complete count.

**What this changes:** nothing in the reported numbers. With complete parquets
the script produces byte-identical tables, verified by diff.

**Where:** `scripts/pipeline/05_compute_statistics.py`,
`tests/test_compute_statistics.py`

---

### 2.28 The severity interaction emits its own constants block

**Decision:** `severity_interaction.py` writes `severity_interaction_tex.tex`
alongside its JSON — 61 `\newcommand` definitions from the primary pooled Lean
panel plus the MRI-cohort hospitalisation estimate.

**Rationale:** Every other supplementary analysis here feeds downstream text
through a generated macro block. A hand-typed section carrying forty numbers is
precisely the surface that goes stale on the next run, without anything
failing.

**Where:** `scripts/supplementary/severity_interaction.py`

---

### 2.29 "Published" claims something this repository cannot

**Decision:** The word *published* does not describe this project's own results
anywhere in the repository. The control check compares against
`CONTROL_REFERENCE_ROC` and `CONTROL_REFERENCE_XGBOOST`, both read from the
environment and both optional.

**Rationale:** Calling an earlier run "published" asserts a reference that has
to be reproduced. A constant that presents a previous run as a published state
reads to every later reader — including the author in six months — as an
obligation that does not exist. The number those constants hold is one earlier
run's, and nothing more.

Hard-coding it is the second half of the problem. A number baked into the script
is inherited by everyone who runs it, including people who have no access to the
run that produced it and no way to interpret it. Reading it from the environment
makes the comparison available to whoever has a reference and silent for
everyone else.

**What this changes:** the control runs keep their purpose, but it is a
different one. They do not defend a published result; they explain why numbers
move between two specifications. And the reference run against the older XGBoost
mentioned in §2.20, the one that would check the chain end to end, is an option
rather than a duty — it costs a full run.

Left alone: `publish` as a verb meaning "emit" (`fit and publish a model`,
`nothing to publish`) and references to external literature.

**Where:** `scripts/run_analysis.sh`, `scripts/pipeline/02_run_pipeline.py`,
`scripts/supplementary/evalue.py`, `data_manager.py`, `orchestration.py`,
`_provenance.py`, `Makefile`, `README.md`, and the corresponding tests

### 2.30 The transfer intercept CI is stratified, and that is the wrong scheme

**Decision:** The reported calibration intercept CI stays as the pipeline
computes it for now, and no downstream text quotes the size of the gap to the
equivalence band. `_bootstrap_target_calibration` should resample unstratified
for the intercept and raise `CALIB_BOOTSTRAP_ITERATIONS` before those numbers
are reported anywhere final.

**Rationale:** two separate problems surfaced when the reported bound was
checked against its own precision.

The bound is the 97.5th percentile of `CALIB_BOOTSTRAP_ITERATIONS` = 1000
draws. On the transfer run it lands at 0.05021 against a band top of 0.05.
Repeating the same bootstrap under 100 seeds puts that bound between 0.0469
and 0.0529, a Monte-Carlo SD of 0.0014 and roughly seven times the 0.0002 gap
the manuscript used to report; 46 of the 100 seeds pass. A boundary decision
at B = 1000 is therefore not reproducible, and a gap of that size is not a
quantity, it is simulation noise.

The scheme is the deeper issue. Resampling positives among positives and
negatives among negatives holds the observed outcome prevalence fixed. But
with a slope near 1 the calibration intercept is approximately
logit(observed prevalence) minus the mean predicted logit, so stratification
removes part of the very variability the interval expresses. Measured on the
same predictions: stratified [-0.019, +0.0504], unstratified
[-0.026, +0.0563], model-based Wald [-0.026, +0.0563]. The last two agree to
four decimals, as they should, and both are about 18 % wider.

**Why the current output is still usable:** the narrower interval makes a PASS
*easier*, and the sub-test fails anyway. The reported result therefore does not
flatter the transfer, and the two-of-three verdict holds under all three
methods. What is not defensible is quoting the 0.0002 overshoot as a quantity:
it invites the opposite reading and it rests on a seed.

The slope CI is unaffected in practice: the slope is not prevalence-facing.

**The same scheme, elsewhere:** §2.14 records the identical trade for the
bootstrap intervals in `evaluation.py`. There, average precision is
prevalence-facing and is resampled unstratified for exactly this reason, while
ROC-AUC is invariant and is not. The calibration intercept belongs on the
unstratified side too.

**Where:** `scripts/pipeline/03_apply_transfer.py`
(`_bootstrap_target_calibration`, `CALIB_BOOTSTRAP_ITERATIONS`)

### 2.31 ROC-AUC is resampled stratified, average precision is not

**Decision:** `compute_bootstrap_ci` takes a `stratify` flag. The ROC-AUC
interval is drawn stratified by outcome, the PR-AUC interval unstratified.
`scripts/recompute_confidence_intervals.py` rewrites
`confidence_intervals.json` for stored runs without retraining anything.

**Rationale:** stratified resampling draws positives among positives and
negatives among negatives, so every resample reproduces the observed
prevalence exactly. That buys one thing: it cannot produce an all-one-class
sample, on which both metrics are undefined. At a few dozen positives that is
a real risk; at several thousand it cannot happen, so the protection is worth
nothing here.

What the same property costs is prevalence as a source of variation.
Average precision is a function of prevalence, so fixing it suppresses
variation the estimate genuinely has. ROC-AUC is invariant to prevalence and
does not notice either way. Across the eleven stored runs the PR-AUC interval
widens by roughly a quarter once stratification is dropped, the primary run
from [0.3974, 0.4304] to [0.3921, 0.4338], while every ROC-AUC interval is
unchanged to the last digit. That last part is the check that the two branches
do what they claim.

**Why a separate script rewrites the artifacts:** the bootstrap consumes only
`y_true` and `y_pred_proba`, both in each run's `predictions.csv`, so no model
has to be refitted. Nothing in the pipeline writes that file outside a full
run, though. The script therefore recomputes every run under the *stored*
settings first and refuses to write unless all of them reproduce the file on
disk bit for bit. Only then does it write the new interval, which is what
makes the resulting diff attributable to the one thing that changed.

**Relation to §2.30:** same defect, different metric. §2.30 found it on the
transfer calibration intercept, where stratification suppresses the variation
of the prevalence-facing quantity itself. This entry generalises it to average
precision. The transfer calibration bootstrap in
`scripts/pipeline/03_apply_transfer.py` is deliberately *not* changed yet;
§2.30 records why and what it would cost.

**Downstream:** anything quoting these intervals carries the widened ones, and
any Methods sentence or table caption saying "stratified bootstrap" has to say
for which metric — it is now true of one and not the other.

**Where:** `src/pcc_analysis/evaluation.py`,
`scripts/recompute_confidence_intervals.py`, `scripts/compare_constants.py`
