"""Unit tests for ``pcs_baseline_proxy`` (KVAD → Bahmer-style T0 PCS proxy).

The processor reads ``exportfile_kv_amb_diag_mapped.csv`` and emits
``kvad_pcs_proxy.parquet``. We test the building blocks
(``_filter_pre_pandemic``, ``_filter_certainty``,
``_aggregate_to_bahmer_indicators``) on synthetic frames so the file-
system processor stays out of the loop.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pcc_analysis.data_processing.pcs_baseline_proxy import (
    _CERTAINTY_DROP_STRICT,
    _ICD3_CHRONIC,
    BAHMER_TO_ICD3,
    _aggregate_to_bahmer_indicators,
    _filter_certainty,
    _filter_pre_pandemic,
    _restrict_mapping_to_chronic,
    process_kvad_pcs_proxy,
    process_kvad_pcs_proxy_all_variants,
)


def _diag_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Helper: build a minimal KVAD-like DataFrame with the required columns."""
    base_cols = {
        "ID": pd.NA,
        "kvad_year": pd.NA,
        "kvad_amb_diag_sure": pd.NA,
        "kvad_a_amb_diag_3": pd.NA,
    }
    return pd.DataFrame([{**base_cols, **r} for r in rows])


# ---------------------------------------------------------------------------
# _filter_pre_pandemic
# ---------------------------------------------------------------------------


def test_pre_pandemic_filter_drops_post_pandemic_rows() -> None:
    diag = _diag_frame(
        [
            {
                "ID": 1,
                "kvad_year": 2015,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "R51",
            },
            {
                "ID": 2,
                "kvad_year": 2020,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "R51",
            },
            {
                "ID": 3,
                "kvad_year": 2021,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "R51",
            },
        ]
    )
    out = _filter_pre_pandemic(diag)
    assert list(out["ID"]) == [1]


def test_pre_pandemic_filter_drops_pre_2010_rows() -> None:
    """The lower bound is the documented coverage start (2010)."""
    diag = _diag_frame(
        [
            {
                "ID": 1,
                "kvad_year": 2009,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "R51",
            },
            {
                "ID": 2,
                "kvad_year": 2010,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "R51",
            },
        ]
    )
    out = _filter_pre_pandemic(diag)
    assert list(out["ID"]) == [2]


def test_pre_pandemic_filter_drops_missing_year() -> None:
    diag = _diag_frame(
        [
            {
                "ID": 1,
                "kvad_year": pd.NA,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "R51",
            },
            {
                "ID": 2,
                "kvad_year": 2015,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "R51",
            },
        ]
    )
    out = _filter_pre_pandemic(diag)
    # NaN year rows are silently False on Series.between → dropped.
    assert list(out["ID"]) == [2]


# ---------------------------------------------------------------------------
# _filter_certainty
# ---------------------------------------------------------------------------


def test_certainty_filter_drops_ausgeschlossen_and_neg9() -> None:
    diag = _diag_frame(
        [
            {"ID": 1, "kvad_amb_diag_sure": 1},  # gesichert
            {"ID": 2, "kvad_amb_diag_sure": 2},  # Verdacht
            {"ID": 3, "kvad_amb_diag_sure": 3},  # ausgeschlossen → drop
            {"ID": 4, "kvad_amb_diag_sure": 4},  # Zustand nach
            {"ID": 5, "kvad_amb_diag_sure": -9},  # explicit missing → drop
        ]
    )
    out = _filter_certainty(diag)
    assert sorted(out["ID"].tolist()) == [1, 2, 4]


# ---------------------------------------------------------------------------
# _aggregate_to_bahmer_indicators
# ---------------------------------------------------------------------------


def test_aggregate_keeps_zero_score_participants() -> None:
    """Participants with KVAD claims but no Bahmer-mapped ICD code must
    appear in the output with all indicators = 0 (regression guard against
    the silent-drop bug)."""
    diag = _diag_frame(
        [
            {"ID": 1, "kvad_a_amb_diag_3": "R51"},  # → headache hit
            {"ID": 2, "kvad_a_amb_diag_3": "Z00"},  # not in any Bahmer mapping
            {"ID": 3, "kvad_a_amb_diag_3": "R50"},  # → fever hit
        ]
    )
    out = _aggregate_to_bahmer_indicators(diag, mapping=BAHMER_TO_ICD3)
    assert sorted(out["ID"].tolist()) == [1, 2, 3]
    # ID 2 has count 0 (no Bahmer-mapped code) — must not be silently
    # dropped from the output.
    row_2 = out.loc[out["ID"] == 2].iloc[0]
    assert int(row_2["pcs_t0_proxy_count"]) == 0
    assert int(row_2["pcs_t0_proxy_n_diagnoses"]) == 1
    # ID 1 / ID 3 have count 1 each via R51 / R50.
    assert int(out.loc[out["ID"] == 1, "pcs_t0_proxy_count"].iloc[0]) == 1
    assert int(out.loc[out["ID"] == 3, "pcs_t0_proxy_count"].iloc[0]) == 1


def test_aggregate_r20_double_counts_circulation_and_nerve() -> None:
    """R20 maps to both ``circulation_problems`` and ``nerve_problems``;
    the Bahmer-style count therefore advances by 2 (documented design)."""
    diag = _diag_frame([{"ID": 7, "kvad_a_amb_diag_3": "R20"}])
    out = _aggregate_to_bahmer_indicators(diag, mapping=BAHMER_TO_ICD3)
    row = out.loc[out["ID"] == 7].iloc[0]
    assert int(row["pcs_t0_proxy_circulation_problems"]) == 1
    assert int(row["pcs_t0_proxy_nerve_problems"]) == 1
    assert int(row["pcs_t0_proxy_count"]) == 2


def test_aggregate_r43_double_counts_smell_and_taste() -> None:
    """R43 maps to both smell and taste — same shared-ICD pattern as R20."""
    diag = _diag_frame([{"ID": 8, "kvad_a_amb_diag_3": "R43"}])
    out = _aggregate_to_bahmer_indicators(diag, mapping=BAHMER_TO_ICD3)
    row = out.loc[out["ID"] == 8].iloc[0]
    assert int(row["pcs_t0_proxy_loss_of_smell"]) == 1
    assert int(row["pcs_t0_proxy_loss_of_taste"]) == 1


def test_aggregate_n_diagnoses_audit_count() -> None:
    """``pcs_t0_proxy_n_diagnoses`` is the raw row count in the filtered
    diag frame — not the count of Bahmer-mapped hits."""
    diag = _diag_frame(
        [
            {"ID": 9, "kvad_a_amb_diag_3": "R51"},
            {"ID": 9, "kvad_a_amb_diag_3": "Z00"},
            {"ID": 9, "kvad_a_amb_diag_3": "Z00"},
        ]
    )
    out = _aggregate_to_bahmer_indicators(diag, mapping=BAHMER_TO_ICD3)
    row = out.loc[out["ID"] == 9].iloc[0]
    assert int(row["pcs_t0_proxy_n_diagnoses"]) == 3
    assert int(row["pcs_t0_proxy_count"]) == 1  # only R51 contributes


def test_aggregate_drops_na_ids_defensively() -> None:
    """NA participant IDs in the diag frame must not surface as their own
    output row (defensive against malformed exports)."""
    diag = _diag_frame(
        [
            {"ID": 1, "kvad_a_amb_diag_3": "R51"},
            {"ID": pd.NA, "kvad_a_amb_diag_3": "R51"},
        ]
    )
    out = _aggregate_to_bahmer_indicators(diag, mapping=BAHMER_TO_ICD3)
    assert list(out["ID"]) == [1]


# ---------------------------------------------------------------------------
# process_kvad_pcs_proxy (end-to-end with synthetic input)
# ---------------------------------------------------------------------------


def test_process_kvad_pcs_proxy_writes_parquet(tmp_path: Path) -> None:
    """End-to-end: the processor writes a parquet that round-trips."""
    diag = _diag_frame(
        [
            {
                "ID": 1,
                "kvad_year": 2015,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "R51",
            },
            {
                "ID": 2,
                "kvad_year": 2018,
                "kvad_amb_diag_sure": 2,
                "kvad_a_amb_diag_3": "R50",
            },
            {
                "ID": 3,
                "kvad_year": 2020,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "R51",
            },
            {
                "ID": 4,
                "kvad_year": 2017,
                "kvad_amb_diag_sure": 3,
                "kvad_a_amb_diag_3": "R51",
            },
        ]
    )
    paths = pytest.importorskip("pcc_analysis.config").NAKOPaths.__new__(
        pytest.importorskip("pcc_analysis.config").NAKOPaths
    )
    out_path = process_kvad_pcs_proxy(paths, tmp_path, diag_df=diag)
    assert out_path.exists()
    parquet = pd.read_parquet(out_path)
    # ID 3 dropped (post-pandemic), ID 4 dropped (ausgeschlossen).
    assert sorted(parquet["ID"].tolist()) == [1, 2]


# ---------------------------------------------------------------------------
# Sensitivity variants: chronic / strict
# ---------------------------------------------------------------------------


def test_certainty_strict_drops_verdacht_and_zustand_nach() -> None:
    """Strict-certainty variant retains only ``gesichert`` (1)."""
    diag = _diag_frame(
        [
            {"ID": 1, "kvad_amb_diag_sure": 1},  # gesichert     → keep
            {"ID": 2, "kvad_amb_diag_sure": 2},  # Verdacht      → drop in strict
            {"ID": 3, "kvad_amb_diag_sure": 3},  # ausgeschlossen → drop
            {"ID": 4, "kvad_amb_diag_sure": 4},  # Zustand nach  → drop in strict
        ]
    )
    out = _filter_certainty(diag, drop_codes=_CERTAINTY_DROP_STRICT)
    assert sorted(out["ID"].tolist()) == [1]


def test_chronic_mapping_drops_r_chapter_codes() -> None:
    """Chronic-only mapping retains F/G/I/J/K/L/M codes; R-chapter removed."""
    chronic = _restrict_mapping_to_chronic(BAHMER_TO_ICD3)

    # Items whose mapping is entirely R-chapter become empty.
    assert chronic["symptom_fever"] == ()  # only R50
    assert chronic["symptom_concentration_problems"] == ()  # only R41
    assert chronic["symptom_cough"] == ()  # only R05
    assert chronic["symptom_loss_of_smell"] == ()  # only R43
    assert chronic["symptom_loss_of_taste"] == ()  # only R43
    assert chronic["symptom_loss_of_appetite"] == ()  # only R63
    assert chronic["symptom_chest_tightness"] == ()  # only R07
    assert chronic["symptom_sweating"] == ()  # only R61

    # Mixed-chapter items keep their chronic codes only.
    assert chronic["symptom_headache"] == ("G43", "G44")
    assert chronic["symptom_breathing_problems"] == ("J45", "J44")
    assert chronic["symptom_heart_problems"] == ("I49",)
    assert chronic["symptom_gastrointestinal_problems"] == ("K58", "K59")
    assert chronic["symptom_circulation_problems"] == ("I73",)
    assert chronic["symptom_nerve_problems"] == ("G50", "G56", "G57", "G58")
    assert chronic["symptom_fatigue"] == ("F48",)
    assert chronic["symptom_reduced_physical_capacity"] == ("F48",)
    assert chronic["symptom_memory_problems"] == ("F06",)

    # Pure-chronic items are unchanged.
    assert chronic["symptom_runny_nose"] == ("J30", "J31")
    assert chronic["symptom_joint_muscle_pain"] == ("M25", "M54", "M79", "M62")
    assert chronic["symptom_sleep_problems"] == ("G47", "F51")
    assert chronic["symptom_hair_loss"] == ("L65",)


def test_aggregate_with_chronic_mapping_zeros_out_acute_items() -> None:
    """A participant with only R-chapter codes scores 0 under chronic mapping."""
    diag = _diag_frame(
        [
            {"ID": 1, "kvad_a_amb_diag_3": "R51"},  # headache (acute)
            {"ID": 1, "kvad_a_amb_diag_3": "G43"},  # migraine (chronic) — same item
            {"ID": 2, "kvad_a_amb_diag_3": "R50"},  # fever — only acute mapping
        ]
    )
    chronic = _restrict_mapping_to_chronic(BAHMER_TO_ICD3)
    out = _aggregate_to_bahmer_indicators(diag, mapping=chronic)
    # ID 1 still scores headache via G43 (chronic survives).
    assert int(out.loc[out["ID"] == 1, "pcs_t0_proxy_headache"].iloc[0]) == 1
    assert int(out.loc[out["ID"] == 1, "pcs_t0_proxy_count"].iloc[0]) == 1
    # ID 2 has only R50 → fever has no chronic ICD → score 0.
    assert int(out.loc[out["ID"] == 2, "pcs_t0_proxy_fever"].iloc[0]) == 0
    assert int(out.loc[out["ID"] == 2, "pcs_t0_proxy_count"].iloc[0]) == 0


def test_chronic_filter_breaks_r20_r43_cross_mapping() -> None:
    """In chronic-only, R20/R43 cross-mappings (circulation/nerve;
    smell/taste) disappear because R-chapter is acute by CCI 2023."""
    diag = _diag_frame(
        [
            {"ID": 1, "kvad_a_amb_diag_3": "R20"},  # default: +2 (circulation+nerve)
            {"ID": 2, "kvad_a_amb_diag_3": "R43"},  # default: +2 (smell+taste)
        ]
    )
    chronic = _restrict_mapping_to_chronic(BAHMER_TO_ICD3)
    out = _aggregate_to_bahmer_indicators(diag, mapping=chronic)
    # Default would give count=2 for both; chronic gives 0 because R20/R43
    # have no chronic mapping.
    assert int(out.loc[out["ID"] == 1, "pcs_t0_proxy_count"].iloc[0]) == 0
    assert int(out.loc[out["ID"] == 2, "pcs_t0_proxy_count"].iloc[0]) == 0


def test_icd3_chronic_set_is_disjoint_from_r_chapter() -> None:
    """Sanity guard: no R-chapter code may appear in the chronic set
    (mismatch would silently re-include acute codes in the chronic
    variant)."""
    r_codes = [c for c in _ICD3_CHRONIC if c.startswith("R")]
    assert r_codes == []


def test_process_kvad_pcs_proxy_chronic_variant(tmp_path: Path) -> None:
    """End-to-end chronic variant: only F/G/I/J/K/L/M codes contribute."""
    diag = _diag_frame(
        [
            # ID 1: G43 (chronic migraine)         → headache hit
            {
                "ID": 1,
                "kvad_year": 2015,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "G43",
            },
            # ID 2: R51 (acute headache)           → no hit in chronic
            {
                "ID": 2,
                "kvad_year": 2015,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "R51",
            },
            # ID 3: F45 (somatoform) + M54 (back)  → fatigue + joint_muscle hits via F48 only? F45 isn't in any item; M54 → joint_muscle
            {
                "ID": 3,
                "kvad_year": 2017,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "M54",
            },
        ]
    )
    paths = pytest.importorskip("pcc_analysis.config").NAKOPaths.__new__(
        pytest.importorskip("pcc_analysis.config").NAKOPaths
    )
    out_path = process_kvad_pcs_proxy(paths, tmp_path, diag_df=diag, variant="chronic")
    assert out_path.name == "kvad_pcs_proxy_chronic.parquet"
    parquet = pd.read_parquet(out_path)
    # ID 1: G43 → headache=1, count=1.
    row1 = parquet.loc[parquet["ID"] == 1].iloc[0]
    assert int(row1["pcs_t0_proxy_headache"]) == 1
    assert int(row1["pcs_t0_proxy_count"]) == 1
    # ID 2: R51 has no chronic mapping → all zero.
    row2 = parquet.loc[parquet["ID"] == 2].iloc[0]
    assert int(row2["pcs_t0_proxy_count"]) == 0
    # ID 3: M54 → joint_muscle_pain=1, count=1.
    row3 = parquet.loc[parquet["ID"] == 3].iloc[0]
    assert int(row3["pcs_t0_proxy_joint_muscle_pain"]) == 1
    assert int(row3["pcs_t0_proxy_count"]) == 1


def test_process_kvad_pcs_proxy_strict_variant(tmp_path: Path) -> None:
    """End-to-end strict variant: only ``gesichert`` (1) contributes."""
    diag = _diag_frame(
        [
            {
                "ID": 1,
                "kvad_year": 2015,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "R51",
            },
            {
                "ID": 2,
                "kvad_year": 2015,
                "kvad_amb_diag_sure": 2,
                "kvad_a_amb_diag_3": "R51",
            },
            {
                "ID": 3,
                "kvad_year": 2015,
                "kvad_amb_diag_sure": 4,
                "kvad_a_amb_diag_3": "R51",
            },
        ]
    )
    paths = pytest.importorskip("pcc_analysis.config").NAKOPaths.__new__(
        pytest.importorskip("pcc_analysis.config").NAKOPaths
    )
    out_path = process_kvad_pcs_proxy(paths, tmp_path, diag_df=diag, variant="strict")
    assert out_path.name == "kvad_pcs_proxy_strict.parquet"
    parquet = pd.read_parquet(out_path)
    # Only ID 1 (gesichert) survives the strict certainty filter.
    assert sorted(parquet["ID"].tolist()) == [1]


def test_process_kvad_pcs_proxy_rejects_unknown_variant(tmp_path: Path) -> None:
    """Guard against typos in the variant argument."""
    diag = _diag_frame([])
    paths = pytest.importorskip("pcc_analysis.config").NAKOPaths.__new__(
        pytest.importorskip("pcc_analysis.config").NAKOPaths
    )
    with pytest.raises(ValueError, match="variant"):
        # Passing an unknown variant is the point of the test.
        process_kvad_pcs_proxy(
            paths,
            tmp_path,
            diag_df=diag,
            variant="bogus",  # pyrefly: ignore[bad-argument-type]
        )


def test_process_kvad_pcs_proxy_all_variants_emits_three_parquets(
    tmp_path: Path,
) -> None:
    """Orchestrator emits all three sensitivity variants from one input."""
    diag = _diag_frame(
        [
            {
                "ID": 1,
                "kvad_year": 2015,
                "kvad_amb_diag_sure": 1,
                "kvad_a_amb_diag_3": "G43",
            },
            {
                "ID": 2,
                "kvad_year": 2015,
                "kvad_amb_diag_sure": 2,
                "kvad_a_amb_diag_3": "R51",
            },
            {
                "ID": 3,
                "kvad_year": 2015,
                "kvad_amb_diag_sure": 4,
                "kvad_a_amb_diag_3": "M54",
            },
        ]
    )
    paths = pytest.importorskip("pcc_analysis.config").NAKOPaths.__new__(
        pytest.importorskip("pcc_analysis.config").NAKOPaths
    )
    out = process_kvad_pcs_proxy_all_variants(paths, tmp_path, diag_df=diag)
    assert set(out.keys()) == {"default", "chronic", "strict"}
    assert all(p.exists() for p in out.values())
    # Default keeps all three; strict keeps only ID 1; chronic keeps the
    # chronic-mapped contributions (ID 1 via G43, ID 3 via M54).
    default = pd.read_parquet(out["default"])
    strict = pd.read_parquet(out["strict"])
    chronic = pd.read_parquet(out["chronic"])
    assert sorted(default["ID"].tolist()) == [1, 2, 3]
    assert sorted(strict["ID"].tolist()) == [1]
    # All three IDs appear in chronic too, but ID 2's R51 contributes zero.
    assert sorted(chronic["ID"].tolist()) == [1, 2, 3]
    assert int(chronic.loc[chronic["ID"] == 2, "pcs_t0_proxy_count"].iloc[0]) == 0
    assert int(chronic.loc[chronic["ID"] == 1, "pcs_t0_proxy_count"].iloc[0]) == 1
    assert int(chronic.loc[chronic["ID"] == 3, "pcs_t0_proxy_count"].iloc[0]) == 1
