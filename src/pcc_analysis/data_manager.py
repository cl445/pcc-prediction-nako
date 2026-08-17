"""Load processed Parquet files and align for analysis."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import get_processed_dir

logger = logging.getLogger(__name__)


TargetName = Literal["bahmer", "neurocog"]
CohortName = Literal["mri", "non_mri", "mri_plus_non_mri", "all"]
"""Analytic cohort filter relative to MRI availability.

- ``mri`` — only infected participants with at least one MRI modality
  (the primary analysis sample).  N≈8.461 on production data after
  clean-controls + symptom-validity filters.
- ``non_mri`` — infected participants WITHOUT any MRI modality
  (within-study replication).  Used together with
  ``stack_variant='lean'`` for the non-MRI replication run.
- ``mri_plus_non_mri`` — explicit union of the MRI and non-MRI
  cohorts (the full infected, clean-controls analytic sample).
  Intended for Lean-Stack pooled training where the transportability
  argument benefits from both sub-cohorts contributing simultaneously.
  Restricted to ``stack_variant='lean'`` because the ``full`` stack
  would silently drop the non-MRI subjects via the MRI-modality
  intersection (effectively reducing pool+full to ``mri``).
- ``all`` — no cohort filter; MRI participants are implicitly
  restricted by the MRI-modality intersection in the ``full`` stack.
  It is the default, so a caller that names no cohort runs under it —
  which is why it stays alongside ``mri``, even though on production
  data the two select the same participants.
"""

StackVariant = Literal["full", "lean"]
"""Model specification (multi-modality vs. Lean Universal Classifier).

- ``full`` — all ten modalities (demographics, ses, cognitive,
  physical_activity, medical_history, lab_values, cardiovascular,
  lung_function, mental_health, plus six MRI atlases).  Primary
  specification.
- ``lean`` — four self-report/questionnaire modalities: demographics,
  ses, medical_history, mental_health.  Designed to transport
  across the three cohorts (MRI, non-MRI, DigiHero) without
  modality drift.  Must be combined with any ``cohort`` value.
"""

LEAN_STACK_MODALITIES: frozenset[str] = frozenset(
    {"demographics", "ses", "medical_history", "mental_health"}
)
"""Modality keys that make up the Lean Universal Classifier.

Used by ``load_pipeline_data`` when ``stack_variant='lean'`` to filter
the modality-loader list.  The set is pinned in code (rather than a
config value) because it is pre-specified and must
remain fixed across all three cohorts for the transportability
argument to hold.

Pinning the *names* is not by itself enough. Three of the four are
composite frames, so a column joined onto one of them enters the Lean
stack just as surely as a new entry in this set would. Two live cases:
tobacco exposure onto ``medical_history`` and the baseline PHQ-9 items
onto ``mental_health``. Both are joined only when
``stack_variant == 'full'``, because whether DigiHero carries a
comparable variable is unresolved in each case and the Lean stack must
not come to depend on the answer. Anything else attached to a modality
named here has to make the same decision explicitly.
"""

CONFOUNDER_COLUMNS: list[str] = ["basis_age", "basis_sex", "basis_uort"]
"""The columns ``demographics.parquet`` contributes to the model.

``demographics.parquet`` also carries ``basis_lvl`` and ``basis_status_mrt``,
which are sample-description fields rather than predictors:
``load_pipeline_data`` selects this whitelist and nothing else. Exported so
that a downstream description of the model — Table 2 in
``scripts/pipeline/05_compute_statistics.py`` — can report the selection
instead of re-deriving it from the parquet's column list.
"""

FULL_STACK_JOINS: dict[str, str] = {
    "mental_health": "phq9_items",
    "medical_history": "smoking",
}
"""Supplementary parquets left-joined onto a host modality in the Full Stack.

Host modality key -> parquet stem. These frames never become their own
``X_dict`` key (see ``_attach``), so the modality the model reads is wider
than the parquet it is named after: ``mental_health`` gains eleven columns
(the nine PHQ-9 items plus their reconstructed sum and completeness flag),
``medical_history`` the six tobacco ones.

Declared here rather than inline at the join site because two consumers need
the same answer, and a feature count derived from the parquets alone
describes an artifact the model never sees. Full Stack only — see
``LEAN_STACK_MODALITIES`` for why.
"""

AMENDMENT_PARQUET_COLUMNS: dict[str, tuple[str, ...]] = {
    "physical_activity": ("gpaq_met_total", "gpaq_reported"),
}
"""Amendment features that a host parquet already contains on disk.

``FULL_STACK_JOINS`` covers the amendment features attached at load time,
where withholding them is a matter of not joining. GPAQ is different: it is
merged into ``physical_activity.parquet`` during data processing
(``_join_gpaq``), so by the time the loader sees the frame the columns are
indistinguishable from the QUAP ones NAKO delivered in the first place.
Reproducing the pre-amendment feature set therefore takes two mechanisms,
and a control run that used only the first would quietly keep GPAQ and
report a difference it had not actually removed.
"""

MH_SUBMODALITY_COLUMNS: dict[str, list[str]] = {
    "mh_phq9": [
        "phq9_sum",
        "phq9_dsm_criteria",
        *[f"phq9_item_{position}" for position in range(1, 10)],
    ],
    "mh_gad7": [
        "gad7_sum",
    ],
    "mh_mini": [
        "mini_major_depression",
        "mini_screen_depression",
        "age_depression_onset",
        "age_anxiety_onset",
        "depression_duration",
        "depression_duration_past20y",
        "depression_episode_duration",
    ],
    "mh_panic": [
        "phq_panic",
    ],
    "mh_stress": [
        "phq_stress",
    ],
}
"""Column assignment for the five Mental-Health sub-modalities.

Used by ``load_pipeline_data`` when ``split_mh_submodalities=True`` to
replace the monolithic ``mental_health`` modality with five independent
base-learners. The mapping is pinned by instrument family (PHQ-9, GAD-7,
MINI, PHQ-Panic, PHQ-Stress), and the partition into those five families
is what has to stay stable across reruns for the SHAP-share comparison to
mean anything. Any extra column not listed here is dropped when split mode
is active — the list is a whitelist, so a column added to
``mental_health`` without a matching entry here is silently invisible in
split mode while remaining visible in the monolithic run.

The composition of a family is not frozen: ``mh_phq9`` gained the nine
baseline PHQ-9 items when the NAM-45 amendment delivered them, which is
why sub-scale metrics are comparable across reruns only in their ranking,
not in their absolute level. The items belong to this family and to no
other, so the partition is unchanged. What the addition does change is
that ``phq9_sum`` is the exact arithmetic sum of the nine items: the
family is rank-deficient by construction, and while that leaves the
sub-modality's predictive performance intact, it means credit within the
family cannot be attributed to an individual item. Item-level
decomposition (somatic versus affective subscales) needs a citable item
assignment first and is deliberately not implemented here.

The items reach ``mental_health`` only in the Full Stack (see
``LEAN_STACK_MODALITIES``), so a split run combined with
``stack_variant='lean'`` yields the pre-amendment two-column ``mh_phq9``.
That is intended, and the whitelist filter below handles it without a
warning because the family still has columns present.

The lists are aligned with the columns that survive
``create_mental_health_preprocessor`` (in ``preprocessing.py``); columns
the global MH preprocessor drops (e.g. ``gad7_diagnosis``,
``phq_stress_moderate``, the redundant binary cutoffs and
``mini_major_depression_imputed``) are excluded here too, otherwise
the sub-modality DataFrame would contain features that are silently
removed before the classifier ever sees them. The drop rules in
``preprocessing.py`` are statistically justified (zero-variance,
collinearity ≥0.9 with the sum score, MNAR imputation) and apply
equally in split mode — this is alignment, not feature loss.

Sub-modalities ``mh_gad7``, ``mh_panic`` and ``mh_stress`` end up with
one feature each. That is by design: the GAD-7 sum, the PHQ panic
flag and the PHQ stress sum are themselves the instrument-level
indicators, and stability selection on a single feature reduces to
"keep it" — which is the correct meta-level signal for SHAP-share
attribution.
"""
"""Supported PCC operationalisations.

- ``bahmer`` — primary outcome: Bahmer weighted PCS > 10.75
  (column ``bahmer_any_pcs``).
- ``neurocog`` — secondary outcome (neurocognitive subtype): at least two
  of four neurocognitive items (fatigue, reduced physical capacity,
  memory problems, concentration problems).  Intended as a
  hypothesis-aligned sensitivity analysis for the second-hit framing;
  the ``>=2``-of-four threshold is syndrome-like and empirically
  distinct from the primary Bahmer outcome (κ≈0.78 on the analytic
  sample, versus κ≈0.86 for a ``>=1`` definition).
"""

NEUROCOG_SYMPTOM_ITEMS: tuple[str, ...] = (
    "symptom_fatigue",
    "symptom_reduced_physical_capacity",
    "symptom_memory_problems",
    "symptom_concentration_problems",
)
NEUROCOG_MIN_ITEMS: int = 2
"""Minimum number of neurocognitive items endorsed for a positive label.

We use ``>=2`` rather than ``>=1`` so the subtype is syndrome-like and
empirically distinct from the primary Bahmer outcome (at ``>=1`` the two
targets agree at κ≈0.86 on the analytic sample, leaving little room for
the sensitivity analysis to add information).
"""


class NakoDataManager:
    """Load and manage processed NAKO data modalities."""

    def __init__(self, processed_dir: Path | None = None) -> None:
        self.processed_dir = (
            Path(processed_dir) if processed_dir else get_processed_dir()
        )
        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed data directory not found: {self.processed_dir}"
            )
        logger.info("NakoDataManager: %s", self.processed_dir)

    def _load(self, name: str) -> pd.DataFrame:
        fp = self.processed_dir / f"{name}.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"Data file not found: {fp}")
        df = pd.read_parquet(fp)
        logger.info("Loaded %s: %d rows x %d cols", name, len(df), len(df.columns))
        return df

    # Individual loaders
    def load_demographics(self) -> pd.DataFrame:
        return self._load("demographics")

    def load_socioeconomic_status(self) -> pd.DataFrame:
        return self._load("socioeconomic_status")

    def load_cognitive_tests(self) -> pd.DataFrame:
        return self._load("cognitive_tests")

    def load_physical_activity(self) -> pd.DataFrame:
        return self._load("physical_activity")

    def load_medical_history(self) -> pd.DataFrame:
        return self._load("medical_history")

    def load_lab_values(self) -> pd.DataFrame:
        return self._load("lab_values")

    def load_cardiovascular(self) -> pd.DataFrame:
        return self._load("cardiovascular")

    def load_lung_function(self) -> pd.DataFrame:
        return self._load("lung_function")

    def load_corona2_pcc(self) -> pd.DataFrame:
        return self._load("corona2_pcc")

    def load_acute_infection(self) -> pd.DataFrame:
        """Acute course of the infection — a sensitivity input, not a modality.

        Measured after the exposure, so it cannot enter a model that
        predicts a post-infection outcome from pre-infection information.
        Deliberately absent from the modality loader list in
        ``load_pipeline_data``; see
        :mod:`pcc_analysis.data_processing.acute_infection`.
        """
        return self._load("acute_infection")

    def load_mri_cortical_desikan_killiany(self) -> pd.DataFrame:
        return self._load("mri_cortical_desikan_killiany")

    def load_mri_cortical_destrieux(self) -> pd.DataFrame:
        return self._load("mri_cortical_destrieux")

    def load_mri_cortical_julich(self) -> pd.DataFrame:
        return self._load("mri_cortical_julich")

    def load_mri_cortical_yeo_networks(self) -> pd.DataFrame:
        return self._load("mri_cortical_yeo_networks")

    def load_mri_subcortical(self) -> pd.DataFrame:
        return self._load("mri_subcortical")

    def load_mri_cerebellar(self) -> pd.DataFrame:
        return self._load("mri_cerebellar")

    def load_mental_health(self) -> pd.DataFrame:
        return self._load("mental_health")

    def load_phq9_items(self) -> pd.DataFrame:
        return self._load("phq9_items")

    def load_smoking(self) -> pd.DataFrame:
        return self._load("smoking")

    def load_mri_etiv(self) -> pd.DataFrame:
        return self._load("mri_etiv")

    def list_available_datasets(self) -> list[str]:
        return sorted(f.stem for f in self.processed_dir.glob("*.parquet"))


def prepare_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Set participant ID as index, drop duplicates."""
    for col in ["nako_id", "ID", "id", "participant_id"]:
        if col in df.columns:
            df = df.set_index(col)
            logger.info("%s: index set to '%s'", label, col)
            break
    if df.index.name is None:
        logger.warning("%s: no explicit ID column; keeping existing index", label)
    else:
        dups = df.index.duplicated()
        if dups.any():
            logger.warning("%s: dropping %d duplicate IDs", label, int(dups.sum()))
            df = df[~dups]
    return df.sort_index()


BAHMER_TARGET_COLUMN = "bahmer_any_pcs"


def _mri_participant_ids(mgr: NakoDataManager) -> pd.Index:
    """Union of participant IDs across all six MRI-atlas parquets.

    Used to split the analytic sample into MRI vs. non-MRI cohorts
    without requiring MRI data to be in ``X_dict``.  Returns an empty
    index if no MRI parquets are available (e.g. smoke-test data
    without MRI generators).
    """
    ids_union: pd.Index | None = None
    mri_loaders = (
        mgr.load_mri_cortical_desikan_killiany,
        mgr.load_mri_cortical_destrieux,
        mgr.load_mri_cortical_julich,
        mgr.load_mri_cortical_yeo_networks,
        mgr.load_mri_subcortical,
        mgr.load_mri_cerebellar,
    )
    for loader in mri_loaders:
        try:
            df = prepare_index(loader(), "MRI cohort scan")
        except FileNotFoundError:
            continue
        ids_union = df.index if ids_union is None else ids_union.union(df.index)
    return pd.Index([]) if ids_union is None else ids_union


def _compute_neurocog_label(pcc: pd.DataFrame) -> pd.Series:
    """Build the neurocognitive subtype label (>=NEUROCOG_MIN_ITEMS of four).

    Raises when any of the four symptom columns is missing.
    """
    missing = [c for c in NEUROCOG_SYMPTOM_ITEMS if c not in pcc.columns]
    if missing:
        raise ValueError(
            "corona2_pcc data is missing neurocognitive symptom columns: "
            f"{missing}.  Re-run scripts/pipeline/01_process_data.py."
        )
    items = pcc[list(NEUROCOG_SYMPTOM_ITEMS)].fillna(0)
    return (items.sum(axis=1) >= NEUROCOG_MIN_ITEMS).astype("int8")


def _withhold_amendment_columns(X_dict: dict[str, pd.DataFrame]) -> None:
    """Drop amendment features their host parquet carries on disk.

    In place, because the caller is assembling ``X_dict`` and a copy here
    would silently apply to nothing. A column that is already absent is not
    an error: a checkout configured against the primary delivery alone
    never had it, and that checkout's feature set is the one being
    reproduced.
    """
    for modality, columns in AMENDMENT_PARQUET_COLUMNS.items():
        frame = X_dict.get(modality)
        if frame is None:
            continue
        present = [column for column in columns if column in frame.columns]
        if not present:
            continue
        X_dict[modality] = frame.drop(columns=present)
        logger.info(
            "Amendment columns withheld from %s: %s (%d -> %d columns)",
            modality,
            ", ".join(present),
            frame.shape[1],
            X_dict[modality].shape[1],
        )


def load_pipeline_data(
    processed_dir: Path | None = None,
    target: TargetName = "bahmer",
    clean_controls: bool = True,
    cohort: CohortName = "all",
    stack_variant: StackVariant = "full",
    split_mh_submodalities: bool = False,
    amendment_features: bool = True,
) -> tuple[pd.Series, dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all modality data aligned to the PCC target index.

    The analytic sample is restricted to participants with a reported
    prior SARS-CoV-2 infection.  This restriction is unconditional
    because the PCC outcome is only defined for infected participants
    (the 4-12 month post-infection symptom items are structurally not
    administered to the non-infected).  Including non-infected as
    PCC-negative would conflate infection prediction with PCC
    prediction and cannot be adjusted away by DML, since infection is
    a mediator/gatekeeper between predictors and the outcome, not a
    classical confounder.

    Cohort and stack variant
    ------------------------
    The ``cohort`` and ``stack_variant`` parameters together select one
    of the analysis configurations (sample architecture and model
    specifications):

    - ``cohort='all'`` + ``stack_variant='full'`` (default) — the
      analytic sample is the intersection across all ten modalities,
      which on production data coincides with the MRI cohort (since
      non-MRI participants have no MRI data).
    - ``cohort='mri'`` + ``stack_variant='full'`` — A1 primary.
      Explicit MRI-cohort restriction; equivalent to default on
      production data but safer semantics.
    - ``cohort='mri'`` + ``stack_variant='lean'`` — Lean Stack on the MRI
      cohort (specification sensitivity; the source run the transfer is
      compared against).
    - ``cohort='non_mri'`` + ``stack_variant='lean'`` — the within-study
      replication on the non-MRI sample.
    - ``cohort='non_mri'`` + ``stack_variant='full'`` — rejected
      with ``ValueError``.  The MRI modalities are structurally empty
      in the non-MRI cohort, so a full-stack run is a configuration
      error.
    - ``cohort='mri_plus_non_mri'`` + ``stack_variant='lean'`` —
      pooled-cohort Lean-Stack training.  The analytic sample is the
      union of the MRI and non-MRI cohorts; the four self-report
      modalities transport across both, so the resulting model can
      then be inferred on either sub-cohort separately.
    - ``cohort='mri_plus_non_mri'`` + ``stack_variant='full'`` —
      rejected with ``ValueError``.  The full-stack intersection
      across the six MRI atlases would drop every non-MRI subject,
      collapsing the pool back to ``cohort='mri'``; we reject the
      configuration to surface this silently-degenerate behaviour.

    Two PCC operationalisations are supported:

    - ``target="bahmer"`` (default) — Bahmer weighted post-COVID
      syndrome score binarised at ``> 10.75`` (column
      ``bahmer_any_pcs``).  Primary outcome.
    - ``target="neurocog"`` — secondary outcome motivated by the
      second-hit hypothesis: at least two of four
      neurocognitive items (fatigue, reduced physical capacity, memory
      problems, concentration problems).  Syndrome-like cut-off;
      distinct from Bahmer at κ≈0.78 on the analytic sample.

    Both thresholds are only meaningful on infected participants; the
    infected restriction is applied regardless of which target is
    selected, so this loader remains the single supported entry point.

    Clean-controls design
    ---------------------
    When ``clean_controls=True`` (default), sub-threshold-symptomatic
    infected participants are dropped from the analytic sample:

    - Bahmer target: drop rows with ``d_co2_k0 == 1`` (routed to the
      21-item inventory, i.e.\\ participant endorsed ``had post-COVID
      symptoms``) *and* ``bahmer_any_pcs == 0`` (their weighted PCS
      stayed at or below the moderate-severity threshold of 10.75).
      PCC-negative controls are then restricted to ``d_co2_k0 == 2``
      (participants who explicitly declared no persistent post-infection
      symptoms via the routing gate).
    - Neurocog target: analogously, drop rows with ``d_co2_k0 == 1`` and
      fewer than ``NEUROCOG_MIN_ITEMS`` neurocognitive items endorsed.

    Rationale: the paper frames a hypothesis test about pre-infection
    predictors of a clinically meaningful post-COVID outcome.  Mixing
    "truly asymptomatic" participants with "mildly symptomatic but
    below-threshold" participants in the control arm attenuates
    effect-size estimates because the control distribution is bimodal
    and partially overlaps the case distribution on the very predictors
    we want to measure.  This matters less for a deployment validator
    than for a mechanism-testing design.  Set ``clean_controls=False``
    to recover the older mixed-controls definition for a sensitivity
    analysis.

    Control runs
    ------------
    ``amendment_features=False`` reproduces the feature set as it stood
    before the NAKO amendment deliveries: no baseline PHQ-9 items, no
    tobacco block, no GPAQ. It exists for one job. A rerun that changes the
    feature set moves the headline ROC-AUC, and a moved number is only
    interpretable next to a run in which nothing changed — otherwise a shift
    caused by the new features cannot be told apart from one caused by a
    library upgrade, a scoring fix, or a widened hyperparameter search. Pair
    it with ``hyperparam_iterations`` at the value the earlier run used:
    that holds both parts of the specification the primary run moves — the
    feature set and the search width — at the earlier setting, so the delta
    between the two runs is attributable to them.

    Reproducing an earlier figure end to end is a different check, and this
    run does not deliver it across a library change: DECISIONS §2.9
    measures xgboost 3.2.0 → 3.4.0 as changing 13 of 16 smoke outputs while
    every other bumped package leaves them byte-identical, so an earlier
    ROC-AUC comes back only from a run pinned to the version that produced
    it. The library effect is already isolated by that measurement, and this
    run isolates the other two by construction.

    Returns
    -------
    y : Series
        Binary PCC labels for the selected target and analytic sample.
    X_dict : dict
        Mapping modality name → feature DataFrame (indexed by participant).
    confounders : DataFrame
        Age, sex, center (indexed by participant).
    """
    if cohort == "non_mri" and stack_variant == "full":
        raise ValueError(
            "cohort='non_mri' is incompatible with stack_variant='full': "
            "MRI modalities are structurally empty in the non-MRI "
            "sample, so a full-stack run in that cohort is a configuration "
            "error.  Use stack_variant='lean' for the non-MRI "
            "replication (within-study)."
        )
    if cohort == "mri_plus_non_mri" and stack_variant == "full":
        raise ValueError(
            "cohort='mri_plus_non_mri' is incompatible with "
            "stack_variant='full': the MRI-modality intersection would "
            "drop every non-MRI subject and silently collapse the pool "
            "back to cohort='mri'.  Use stack_variant='lean' for the "
            "pooled-cohort Lean-Stack training."
        )

    mgr = NakoDataManager(processed_dir)

    pcc_raw = mgr.load_corona2_pcc()
    if target == "bahmer" and BAHMER_TARGET_COLUMN not in pcc_raw.columns:
        raise ValueError(
            f"corona2_pcc data is missing the '{BAHMER_TARGET_COLUMN}' "
            "column.  Re-run scripts/pipeline/01_process_data.py to compute the "
            "Bahmer weighted PCS."
        )
    pcc = prepare_index(pcc_raw, "PCC")

    if "had_covid" not in pcc.columns:
        raise ValueError(
            "corona2_pcc data is missing the 'had_covid' column, which is "
            "required to restrict the analytic sample to SARS-CoV-2 infected "
            "participants."
        )

    # Build a single combined keep-mask so all row filters apply in one
    # step, and every per-filter drop count refers to the same starting N.
    n_start = len(pcc)
    keep_mask = pd.Series(True, index=pcc.index)

    # Exclude infected participants with <80% valid symptom responses
    if "valid_symptoms" in pcc.columns:
        valid_mask = pcc["valid_symptoms"] == 1
        keep_mask &= valid_mask
        n_dropped = int((~valid_mask).sum())
        if n_dropped > 0:
            logger.info(
                "Symptom validity filter: dropped %d/%d participants with "
                "<80%% valid responses",
                n_dropped,
                n_start,
            )

    # Restrict to SARS-CoV-2 infected participants (unconditional).
    infected_mask = pcc["had_covid"] == 1
    n_before = int(keep_mask.sum())
    keep_mask &= infected_mask
    n_after = int(keep_mask.sum())
    logger.info(
        "Restricted analytic sample to SARS-CoV-2 infected: "
        "N=%d -> N=%d (dropped %d non-infected)",
        n_before,
        n_after,
        n_before - n_after,
    )

    # Apply combined mask once
    pcc = pcc.loc[keep_mask]

    if target == "bahmer":
        y = pcc[BAHMER_TARGET_COLUMN].astype("int8")
        y.name = BAHMER_TARGET_COLUMN
    elif target == "neurocog":
        y = _compute_neurocog_label(pcc)
        y.name = "neurocog_ge2"
    else:  # pragma: no cover — Literal exhausts the options
        raise ValueError(f"Unsupported target: {target!r}")

    # Clean-controls design: remove subthreshold k0=1 participants from
    # the PCC-negative pool so the control arm is restricted to
    # participants who explicitly declared symptom-free via the routing
    # gate (k0=2).  See the module docstring for rationale.
    if clean_controls:
        if "d_co2_k0" not in pcc.columns:
            raise ValueError(
                "clean_controls=True requires the 'd_co2_k0' column in "
                "corona2_pcc.parquet.  Re-run scripts/pipeline/01_process_data.py "
                "with the updated data_processing that stores the "
                "routing-gate code."
            )
        k0 = pcc["d_co2_k0"]
        subthreshold = (k0 == 1) & (y == 0)
        n_sub = int(subthreshold.sum())
        if n_sub > 0:
            keep = ~subthreshold
            pcc = pcc.loc[keep]
            y = y.loc[keep]
            logger.info(
                "Clean-controls filter (target=%s): dropped %d subthreshold "
                "k0=1 participants (symptomatic but below threshold); "
                "controls are now restricted to k0=2 declared symptom-free",
                target,
                n_sub,
            )
    # Target columns can still carry residual missingness (e.g. a
    # ``None`` Bahmer score on a respondent whose item matrix could
    # not be resolved).  Drop those now so downstream CV never sees
    # NaN labels.
    missing = y.isna()
    if bool(missing.any()):
        logger.info(
            "Dropping %d participants with missing %s label", int(missing.sum()), target
        )
        y = y.loc[~missing]

    # Cohort filter by MRI availability.  Split
    # the analytic sample by MRI-availability so the Lean-Stack
    # replication can be run separately in MRI and non-MRI, or pool
    # them for combined Lean-Stack training.  Applied AFTER
    # clean-controls so both cohorts use identical outcome definitions.
    if cohort in ("mri", "non_mri"):
        mri_ids = _mri_participant_ids(mgr)
        if cohort == "mri":
            cohort_mask = y.index.isin(mri_ids)
        else:  # non_mri
            cohort_mask = ~y.index.isin(mri_ids)
        n_before_cohort = len(y)
        y = y.loc[cohort_mask]
        logger.info(
            "Cohort filter (cohort=%s): N=%d -> N=%d (dropped %d)",
            cohort,
            n_before_cohort,
            len(y),
            n_before_cohort - len(y),
        )
    elif cohort == "mri_plus_non_mri":
        # Pooled-cohort training: keep the union (no row drop), but
        # log the partition so the run-config is auditable and so an
        # unexpectedly empty sub-cohort surfaces immediately.
        mri_ids = _mri_participant_ids(mgr)
        n_mri = int(y.index.isin(mri_ids).sum())
        n_non = len(y) - n_mri
        logger.info(
            "Cohort filter (cohort=mri_plus_non_mri): N=%d (MRI=%d + non-MRI=%d)",
            len(y),
            n_mri,
            n_non,
        )

    initial_n = len(y)
    logger.info(
        "Initial analytic sample (target=%s, cohort=%s, stack=%s): N=%d",
        target,
        cohort,
        stack_variant,
        initial_n,
    )

    def _subset(df: pd.DataFrame, label: str) -> pd.DataFrame | None:
        overlap = y.index.intersection(df.index)
        if len(overlap) == 0:
            logger.warning("%s: no overlap with target sample", label)
            return None
        return df.loc[overlap]

    # Confounders
    demo = prepare_index(mgr.load_demographics(), "Demographics")
    confounders_src = demo[CONFOUNDER_COLUMNS].copy()

    # A3: Require complete demographics (age, sex, center)
    demo_missing = confounders_src.isna().any(axis=1)
    n_demo_missing = int(demo_missing.sum())
    if n_demo_missing > 0:
        logger.warning(
            "Dropping %d participants with incomplete demographics (age/sex/center)",
            n_demo_missing,
        )
        confounders_src = confounders_src.loc[~demo_missing]

    confounders = _subset(confounders_src, "Confounders")
    if confounders is None:
        raise ValueError("Confounders have no overlap with symptom data")

    # Build modality dict.  ``demographics`` is always present (it is
    # the confounders frame) and is part of both stack variants.
    X_dict: dict[str, pd.DataFrame] = {"demographics": confounders.copy()}

    # Non-MRI modality loaders.  For the Lean Universal Classifier
    # (stack_variant='lean'), keep only modalities in
    # ``LEAN_STACK_MODALITIES``.
    _all_non_mri_loaders: list[tuple[str, str]] = [
        ("ses", "Socioeconomic Status"),
        ("cognitive", "Cognitive"),
        ("physical_activity", "Physical Activity"),
        ("medical_history", "Medical History"),
        ("lab_values", "Lab Values"),
        ("cardiovascular", "Cardiovascular"),
        ("lung_function", "Lung Function"),
        ("mental_health", "Mental Health"),
    ]
    if stack_variant == "lean":
        loaders = [
            (key, label)
            for key, label in _all_non_mri_loaders
            if key in LEAN_STACK_MODALITIES
        ]
    else:
        loaders = _all_non_mri_loaders
    loader_methods = {
        "ses": mgr.load_socioeconomic_status,
        "cognitive": mgr.load_cognitive_tests,
        "physical_activity": mgr.load_physical_activity,
        "medical_history": mgr.load_medical_history,
        "lab_values": mgr.load_lab_values,
        "cardiovascular": mgr.load_cardiovascular,
        "lung_function": mgr.load_lung_function,
        "mental_health": mgr.load_mental_health,
    }
    for key, label in loaders:
        try:
            raw = prepare_index(loader_methods[key](), f"{label} raw")
            sub = _subset(raw, label)
            if sub is not None:
                X_dict[key] = sub
        except Exception as e:
            logger.warning("Could not load %s: %s", label, e)

    def _attach(modality: str, loader: Callable[[], pd.DataFrame], label: str) -> None:
        """Left-join a supplementary parquet onto an existing modality.

        Follows the eTIV pattern: the extra frame never becomes its own
        ``X_dict`` key, so it needs no base learner of its own and costs no
        attrition through the common-index intersection below. A left join
        also keeps the modality's own row set authoritative.
        """
        if modality not in X_dict:
            return
        try:
            extra = prepare_index(loader(), f"{label} raw")
        except Exception as e:
            logger.warning("Could not load %s: %s", label, e)
            return
        # A left join onto a disjoint index yields an all-NA block, and the
        # ``dropna(axis=1, how="all")`` sweep below removes it again without
        # comment — the run then reports the widened column count and produces
        # a normal-looking result with none of the joined data in it. The
        # trigger is an ID space that does not match: a differently
        # pseudonymised delivery, a truncated export, or a ``prepare_index``
        # that fell back to a RangeIndex for want of an ID column. Report it
        # the way ``_subset`` does for the modalities that go through it.
        overlap = X_dict[modality].index.intersection(extra.index)
        if len(overlap) == 0:
            logger.warning(
                "%s: no overlap with %s (%d vs %d participants); nothing "
                "joined. Check that both parquets share an ID space.",
                label,
                modality,
                len(extra),
                len(X_dict[modality]),
            )
            return
        before = X_dict[modality].shape[1]
        X_dict[modality] = X_dict[modality].join(extra, how="left")
        logger.info(
            "%s joined into %s: %d -> %d columns, present for %d of %d participants",
            label,
            modality,
            before,
            X_dict[modality].shape[1],
            len(overlap),
            len(X_dict[modality]),
        )

    # Features the NAKO amendment deliveries added after the Lean stack was
    # pre-specified. Both are Full-Stack only, for the same reason: their
    # host modalities are themselves in LEAN_STACK_MODALITIES, so an
    # unconditional join would quietly extend the Lean Universal Classifier
    # and make it depend on variables whose availability in DigiHero is
    # unresolved — item-level PHQ-9 responses in one case, any tobacco
    # variable at all in the other. That dependency is what the pinned Lean
    # set exists to prevent.
    #
    # The PHQ-9 items are joined before the sub-modality split below, so
    # both the monolithic mental_health modality and the mh_phq9
    # sub-modality see them. Note that phq9_sum is the exact arithmetic sum
    # of the nine items, so that block is rank-deficient by construction:
    # the sub-modality's predictive performance is unaffected, but credit
    # for it is not attributable to an individual item.
    #
    # ``amendment_features=False`` withholds every amendment-derived feature
    # at once; see ``AMENDMENT_PARQUET_COLUMNS`` for why that takes two
    # mechanisms rather than one.
    if stack_variant == "full" and amendment_features:
        join_loaders: dict[str, tuple[Callable[[], pd.DataFrame], str]] = {
            "phq9_items": (mgr.load_phq9_items, "PHQ-9 items"),
            "smoking": (mgr.load_smoking, "Smoking"),
        }
        for host, parquet_stem in FULL_STACK_JOINS.items():
            loader, join_label = join_loaders[parquet_stem]
            _attach(host, loader, join_label)
    if not amendment_features:
        _withhold_amendment_columns(X_dict)

    # Optional mental-health sub-modality split. Replaces the single
    # "mental_health" modality
    # with five instrument-specific DataFrames so the meta-learner can
    # attribute SHAP to each screener separately. Columns not assigned
    # in MH_SUBMODALITY_COLUMNS are dropped.
    if split_mh_submodalities and "mental_health" in X_dict:
        full_mh = X_dict.pop("mental_health")
        for sub_key, sub_cols in MH_SUBMODALITY_COLUMNS.items():
            present = [c for c in sub_cols if c in full_mh.columns]
            if present:
                X_dict[sub_key] = full_mh[present].copy()
            else:
                logger.warning(
                    "split_mh_submodalities: %s has no columns present in "
                    "mental_health.parquet; sub-modality skipped.",
                    sub_key,
                )

    # MRI modalities (Full-Stack only).  The Lean Universal Classifier
    # deliberately drops MRI to preserve transportability across the
    # three cohorts (MRI / non-MRI / DigiHero).  Also short-
    # circuit when cohort='non_mri' even in full stack, although that
    # combination is rejected earlier.
    if stack_variant == "full" and cohort != "non_mri":
        # Load eTIV for ICV adjustment (joined to each MRI modality)
        try:
            etiv_raw = prepare_index(mgr.load_mri_etiv(), "eTIV raw")
            etiv_sub = _subset(etiv_raw, "eTIV")
            logger.info(
                "eTIV loaded: %d participants",
                len(etiv_sub) if etiv_sub is not None else 0,
            )
        except Exception as e:
            etiv_sub = None
            logger.warning("Could not load eTIV: %s", e)

        mri_loaders = {
            "mri_desikan": ("MRI Desikan", mgr.load_mri_cortical_desikan_killiany),
            "mri_destrieux": ("MRI Destrieux", mgr.load_mri_cortical_destrieux),
            "mri_julich": ("MRI Julich", mgr.load_mri_cortical_julich),
            "mri_yeo": ("MRI Yeo", mgr.load_mri_cortical_yeo_networks),
            "mri_subcortical": ("MRI Subcortical", mgr.load_mri_subcortical),
            "mri_cerebellar": ("MRI Cerebellar", mgr.load_mri_cerebellar),
        }
        for mod_name, (label, loader) in mri_loaders.items():
            try:
                raw = prepare_index(loader(), f"{label} raw")
                sub = _subset(raw, label)
                if sub is not None:
                    # Join eTIV for ICV adjustment in pipeline
                    if etiv_sub is not None:
                        sub = sub.join(etiv_sub[["etiv"]], how="left")
                    X_dict[mod_name] = sub
            except Exception as e:
                logger.warning("Could not load %s: %s", label, e)

    # Drop empty modalities (preserve object/categorical columns
    # for ModalityPreprocessor encoding)
    for mod_name in list(X_dict):
        df = X_dict[mod_name].dropna(axis=1, how="all")
        if df.empty:
            del X_dict[mod_name]
            continue
        X_dict[mod_name] = df

    # Align to common index (A5: log attrition at each step)
    common_idx = y.index
    logger.info("Attrition — target sample: N=%d", len(common_idx))
    for mod_name, df in X_dict.items():
        prev_n = len(common_idx)
        common_idx = common_idx.intersection(df.index)
        lost = prev_n - len(common_idx)
        if lost > 0:
            logger.info(
                "Attrition — after %s intersection: N=%d (-%d)",
                mod_name,
                len(common_idx),
                lost,
            )

    y = y.loc[common_idx]
    confounders = confounders.loc[common_idx]
    for mod_name in X_dict:
        X_dict[mod_name] = X_dict[mod_name].loc[common_idx]

    logger.info(
        "Final sample (target=%s): N=%d (%.1f%% attrition), prevalence=%.1f%%, %d modalities",
        target,
        len(common_idx),
        (1.0 - len(common_idx) / initial_n) * 100 if initial_n > 0 else 0,
        float(y.mean()) * 100 if len(y) else 0.0,
        len(X_dict),
    )
    return y, X_dict, confounders
