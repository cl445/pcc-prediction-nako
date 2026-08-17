"""Tests for the ``load_pipeline_data`` cohort/stack_variant logic.

The fixture builds a tiny set of parquet files in ``tmp_path`` so we
can exercise every (cohort, stack_variant) combination without depending
on production data or the full smoke-test generator.  Only the columns
the loader reads are populated — we are testing filter behaviour, not
data realism.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcc_analysis.data_manager import (
    LEAN_STACK_MODALITIES,
    load_pipeline_data,
)


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    """Tiny parquet fixture with 100 participants.

    Half of them (IDs 0-49) have MRI modalities; the other half (50-99)
    do not.  All are flagged infected.  PCC prevalence is ~30 % so every
    stratum has both positive and negative labels.
    """
    rng = np.random.default_rng(42)
    n = 100
    ids = pd.Series(range(n), name="nako_id")

    # corona2_pcc — needs had_covid, d_co2_k0, bahmer_any_pcs,
    # valid_symptoms, and the four neurocognitive symptom columns.
    bahmer = rng.choice([0, 1], size=n, p=[0.7, 0.3]).astype("int8")
    # d_co2_k0 routing: positives tend to have k0=1, negatives k0=2,
    # with a few k0=1-but-negative (subthreshold) participants that
    # clean_controls should drop.
    k0 = np.where(bahmer == 1, 1, 2)
    k0[rng.choice(n, size=5, replace=False)] = 1  # a handful of subthreshold

    corona2 = pd.DataFrame(
        {
            "nako_id": ids,
            "had_covid": np.ones(n, dtype="int8"),
            "d_co2_k0": pd.array(k0, dtype="Int8"),
            "bahmer_any_pcs": bahmer,
            "valid_symptoms": np.ones(n, dtype="int8"),
            # Four neurocog items — not used for bahmer target but loader
            # validates their presence when target='neurocog' is selected.
            "symptom_fatigue": rng.integers(0, 2, n, dtype="int8"),
            "symptom_reduced_physical_capacity": rng.integers(0, 2, n, dtype="int8"),
            "symptom_memory_problems": rng.integers(0, 2, n, dtype="int8"),
            "symptom_concentration_problems": rng.integers(0, 2, n, dtype="int8"),
        }
    )
    corona2.to_parquet(tmp_path / "corona2_pcc.parquet", index=False)

    # Demographics — confounders read basis_age/sex/uort.
    demographics = pd.DataFrame(
        {
            "nako_id": ids,
            "basis_age": rng.integers(30, 70, n).astype("int16"),
            "basis_sex": rng.choice([1, 2], size=n).astype("int8"),
            "basis_uort": rng.choice([1, 2, 3], size=n).astype("int8"),
        }
    )
    demographics.to_parquet(tmp_path / "demographics.parquet", index=False)

    # Four "lean" non-MRI modalities (plus three that are full-only).
    for name in (
        "socioeconomic_status",
        "cognitive_tests",
        "physical_activity",
        "medical_history",
        "lab_values",
        "cardiovascular",
        "lung_function",
        "mental_health",
    ):
        df = pd.DataFrame(
            {
                "nako_id": ids,
                f"{name}_feat_a": rng.standard_normal(n),
                f"{name}_feat_b": rng.standard_normal(n),
            }
        )
        df.to_parquet(tmp_path / f"{name}.parquet", index=False)

    # Supplementary frames joined onto an existing modality rather than
    # loaded as modalities of their own (the eTIV pattern).
    phq9_items = pd.DataFrame(
        {
            "nako_id": ids,
            **{
                f"phq9_item_{position}": rng.integers(0, 4, n, dtype="int8")
                for position in range(1, 10)
            },
        }
    )
    phq9_items.to_parquet(tmp_path / "phq9_items.parquet", index=False)

    smoking = pd.DataFrame(
        {
            "nako_id": ids,
            "smoking_status": rng.integers(1, 4, n, dtype="int8"),
            "pack_years": rng.uniform(0, 40, n),
        }
    )
    smoking.to_parquet(tmp_path / "smoking.parquet", index=False)

    # GPAQ is the third amendment feature and the one that is not joined at
    # load time: data processing merges it into physical_activity.parquet,
    # so on disk it is indistinguishable from the QUAP columns.
    physical_activity = pd.read_parquet(tmp_path / "physical_activity.parquet")
    physical_activity["gpaq_met_total"] = rng.uniform(0, 5000, n)
    physical_activity["gpaq_reported"] = 1
    physical_activity.to_parquet(tmp_path / "physical_activity.parquet", index=False)

    # MRI modalities — only available for IDs 0-49 (half the sample).
    mri_ids = pd.Series(range(50), name="nako_id")
    mri_files = (
        "mri_cortical_desikan_killiany",
        "mri_cortical_destrieux",
        "mri_cortical_julich",
        "mri_cortical_yeo_networks",
        "mri_subcortical",
        "mri_cerebellar",
    )
    for name in mri_files:
        df = pd.DataFrame(
            {
                "nako_id": mri_ids,
                f"{name}_region_a": rng.standard_normal(50),
                f"{name}_region_b": rng.standard_normal(50),
            }
        )
        df.to_parquet(tmp_path / f"{name}.parquet", index=False)

    # eTIV — joined onto MRI modalities for ICV adjustment.
    etiv = pd.DataFrame(
        {
            "nako_id": mri_ids,
            "etiv": rng.normal(1.5e6, 1e5, 50),
        }
    )
    etiv.to_parquet(tmp_path / "mri_etiv.parquet", index=False)

    return tmp_path


# --- cohort x stack_variant matrix ---


def test_mri_full_is_default_shape(processed_dir: Path) -> None:
    """cohort='mri' + stack='full' → MRI sample, all modalities."""
    y, X_dict, conf = load_pipeline_data(
        processed_dir=processed_dir, cohort="mri", stack_variant="full"
    )
    assert len(y) > 0
    assert set(y.index).issubset(set(range(50)))  # MRI-only participants
    # Full stack should include MRI modalities.
    mri_keys = {k for k in X_dict if k.startswith("mri_")}
    assert len(mri_keys) == 6, f"expected 6 MRI modalities, got {mri_keys}"
    assert "cognitive" in X_dict
    assert "lab_values" in X_dict
    assert conf.shape[1] == 3


def test_mri_lean_keeps_mri_cohort_but_drops_mri_modalities(
    processed_dir: Path,
) -> None:
    """cohort='mri' + stack='lean' → MRI sample, only 4 modalities."""
    y, X_dict, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="mri", stack_variant="lean"
    )
    assert len(y) > 0
    assert set(y.index).issubset(set(range(50)))
    assert set(X_dict) == LEAN_STACK_MODALITIES
    assert not any(k.startswith("mri_") for k in X_dict)


def test_non_mri_lean_excludes_mri_cohort(processed_dir: Path) -> None:
    """cohort='non_mri' + stack='lean' → IDs 50-99, 4 modalities."""
    y, X_dict, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="non_mri", stack_variant="lean"
    )
    assert len(y) > 0
    assert set(y.index).issubset(set(range(50, 100)))
    assert set(y.index).isdisjoint(set(range(50)))
    assert set(X_dict) == LEAN_STACK_MODALITIES


def test_non_mri_full_raises(processed_dir: Path) -> None:
    """cohort='non_mri' + stack='full' is a configuration error."""
    with pytest.raises(ValueError, match=r"non_mri.*full"):
        load_pipeline_data(
            processed_dir=processed_dir, cohort="non_mri", stack_variant="full"
        )


def test_all_full_default_matches_mri_full_on_this_fixture(
    processed_dir: Path,
) -> None:
    """cohort='all' + stack='full' should preserve legacy behaviour.

    On this fixture non-MRI participants are dropped via the MRI-
    intersection step, so the final sample equals the MRI cohort.
    """
    y_all, X_all, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="all", stack_variant="full"
    )
    y_mri, X_mri, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="mri", stack_variant="full"
    )
    assert set(y_all.index) == set(y_mri.index)
    assert set(X_all) == set(X_mri)


def test_mri_lean_disjoint_from_non_mri_lean(processed_dir: Path) -> None:
    """The two Lean-Stack cohorts partition the analytic sample."""
    y_mri, _, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="mri", stack_variant="lean"
    )
    y_non, _, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="non_mri", stack_variant="lean"
    )
    assert set(y_mri.index).isdisjoint(set(y_non.index))


def test_non_mri_lean_has_no_etiv_join(processed_dir: Path) -> None:
    """Lean stack must not pull eTIV or MRI loaders even incidentally."""
    _, X_dict, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="non_mri", stack_variant="lean"
    )
    for mod_name, df in X_dict.items():
        assert "etiv" not in df.columns, (
            f"Lean modality {mod_name} unexpectedly contains eTIV column"
        )


# --- Pooled (mri_plus_non_mri) cohort ---


def test_pool_lean_sample_equals_mri_plus_non_mri(processed_dir: Path) -> None:
    """cohort='mri_plus_non_mri' + stack='lean' yields the union of both
    sub-cohort samples, with identical outcome handling.
    """
    y_pool, X_pool, _ = load_pipeline_data(
        processed_dir=processed_dir,
        cohort="mri_plus_non_mri",
        stack_variant="lean",
    )
    y_mri, _, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="mri", stack_variant="lean"
    )
    y_non, _, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="non_mri", stack_variant="lean"
    )
    assert set(y_pool.index) == set(y_mri.index) | set(y_non.index)
    assert len(y_pool) == len(y_mri) + len(y_non)
    # Lean modality set is preserved on the pool.
    assert set(X_pool) == LEAN_STACK_MODALITIES


def test_pool_lean_preserves_per_cohort_prevalence(processed_dir: Path) -> None:
    """Outcome prevalence in each sub-cohort matches its standalone load.

    Confirms clean-controls + valid-symptoms filters are applied
    uniformly when the pooled mode short-circuits the per-cohort
    drop step.
    """
    y_pool, _, _ = load_pipeline_data(
        processed_dir=processed_dir,
        cohort="mri_plus_non_mri",
        stack_variant="lean",
    )
    y_mri, _, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="mri", stack_variant="lean"
    )
    y_non, _, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="non_mri", stack_variant="lean"
    )
    pool_mri = y_pool.loc[y_pool.index.intersection(y_mri.index)]
    pool_non = y_pool.loc[y_pool.index.intersection(y_non.index)]
    assert (pool_mri.sort_index() == y_mri.sort_index()).all()
    assert (pool_non.sort_index() == y_non.sort_index()).all()


def test_pool_full_raises(processed_dir: Path) -> None:
    """cohort='mri_plus_non_mri' + stack='full' is rejected to avoid
    silently collapsing the pool back to the MRI cohort via the
    MRI-modality intersection.
    """
    with pytest.raises(ValueError, match=r"mri_plus_non_mri.*full"):
        load_pipeline_data(
            processed_dir=processed_dir,
            cohort="mri_plus_non_mri",
            stack_variant="full",
        )


# --- supplementary joins onto existing modalities ---
#
# Both the baseline PHQ-9 items and the tobacco block arrived after the
# Lean stack was pre-specified, and both are joined onto modalities that
# are themselves part of it. Nothing raises when that gating is removed,
# so the tests below are the only thing standing between a one-line edit
# and a Lean Universal Classifier that silently depends on variables
# DigiHero may not have.


def test_full_stack_joins_phq9_items_and_smoking(processed_dir: Path) -> None:
    y, X_dict, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="mri", stack_variant="full"
    )
    mh_cols = set(X_dict["mental_health"].columns)
    assert {f"phq9_item_{i}" for i in range(1, 10)} <= mh_cols

    med_cols = set(X_dict["medical_history"].columns)
    assert {"smoking_status", "pack_years"} <= med_cols

    # A left join must not cost the modality any participants.
    assert len(X_dict["mental_health"]) == len(y)
    assert len(X_dict["medical_history"]) == len(y)


def test_lean_stack_excludes_both_supplementary_joins(processed_dir: Path) -> None:
    """The pinned Lean feature set must not move with a new delivery."""
    _, X_dict, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="mri", stack_variant="lean"
    )
    mh_cols = set(X_dict["mental_health"].columns)
    assert not any(c.startswith("phq9_item_") for c in mh_cols)

    med_cols = set(X_dict["medical_history"].columns)
    assert not any(
        c in med_cols for c in ("smoking_status", "pack_years", "current_smoker")
    )


def test_split_mh_carries_the_items_into_the_phq9_submodality(
    processed_dir: Path,
) -> None:
    """MH_SUBMODALITY_COLUMNS is a whitelist — an unlisted column vanishes.

    Split mode drops an unlisted item without warning: the sub-modality
    still has phq9_sum present, so the "no columns" branch never fires.
    """
    _, X_dict, _ = load_pipeline_data(
        processed_dir=processed_dir,
        cohort="mri",
        stack_variant="full",
        split_mh_submodalities=True,
    )
    assert "mental_health" not in X_dict
    phq9_cols = set(X_dict["mh_phq9"].columns)
    assert {f"phq9_item_{i}" for i in range(1, 10)} <= phq9_cols


def test_a_disjoint_join_is_reported_rather_than_silently_dropped(
    processed_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A joined parquet from a foreign ID space must not pass for a join.

    The left join produces an all-NA block, the empty-column sweep removes it
    again, and the run reports the widened column count and then trains with
    none of the data. Every other loader goes through ``_subset``, which
    reports exactly this; ``_attach`` has to as well.
    """
    items = pd.read_parquet(processed_dir / "phq9_items.parquet")
    items["nako_id"] = items["nako_id"] + 10_000
    items.to_parquet(processed_dir / "phq9_items.parquet", index=False)

    with caplog.at_level(logging.WARNING, logger="pcc_analysis.data_manager"):
        _, X_dict, _ = load_pipeline_data(
            processed_dir=processed_dir, cohort="mri", stack_variant="full"
        )

    assert "no overlap with mental_health" in caplog.text
    assert not any(c.startswith("phq9_item_") for c in X_dict["mental_health"].columns)
    # The host modality itself must survive untouched.
    assert "mental_health_feat_a" in X_dict["mental_health"].columns


def test_control_run_withholds_every_amendment_feature(
    processed_dir: Path,
) -> None:
    """The control run has to reproduce the pre-amendment feature set.

    All three amendment features, not the two that happen to be joined at
    load time. GPAQ reaches the model inside ``physical_activity.parquet``,
    so a control run that only skipped the joins would keep it and then
    report a delta it had not actually isolated.
    """
    _, X_dict, _ = load_pipeline_data(
        processed_dir=processed_dir,
        cohort="mri",
        stack_variant="full",
        amendment_features=False,
    )

    mh_cols = set(X_dict["mental_health"].columns)
    assert not any(c.startswith("phq9_item_") for c in mh_cols)
    med_cols = set(X_dict["medical_history"].columns)
    assert not {"smoking_status", "pack_years"} & med_cols
    pa_cols = set(X_dict["physical_activity"].columns)
    assert not {"gpaq_met_total", "gpaq_reported"} & pa_cols

    # Only the amendment columns go: the host modalities are the reference
    # specification and have to arrive intact.
    assert "mental_health_feat_a" in mh_cols
    assert "medical_history_feat_a" in med_cols
    assert "physical_activity_feat_a" in pa_cols


def test_control_run_keeps_the_sample_it_compares_against(
    processed_dir: Path,
) -> None:
    """Same participants, fewer columns — otherwise the delta is confounded."""
    y_full, X_full, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="mri", stack_variant="full"
    )
    y_control, X_control, _ = load_pipeline_data(
        processed_dir=processed_dir,
        cohort="mri",
        stack_variant="full",
        amendment_features=False,
    )

    assert set(y_control.index) == set(y_full.index)
    assert set(X_control) == set(X_full)
    assert X_control["mental_health"].shape[1] < X_full["mental_health"].shape[1]


def test_amendment_features_are_on_by_default(processed_dir: Path) -> None:
    """The withholding switch must not become the accidental default."""
    _, X_dict, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="mri", stack_variant="full"
    )

    assert "gpaq_met_total" in X_dict["physical_activity"].columns
    assert "smoking_status" in X_dict["medical_history"].columns


def test_missing_supplementary_parquet_is_not_fatal(
    processed_dir: Path,
) -> None:
    """A checkout without the amendment deliveries must still load.

    The processors that write these two parquets are themselves guarded
    on the delivery existing, so their absence is a supported state.
    """
    (processed_dir / "phq9_items.parquet").unlink()
    (processed_dir / "smoking.parquet").unlink()
    y, X_dict, _ = load_pipeline_data(
        processed_dir=processed_dir, cohort="mri", stack_variant="full"
    )
    assert len(y) > 0
    assert "mental_health" in X_dict
    assert "medical_history" in X_dict
