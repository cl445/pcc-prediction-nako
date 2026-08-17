"""Tests for the modality feature counts in ``05_compute_statistics.py``.

Both tables in that script report a feature count per modality, and the
count has to come from what the model reads rather than from the modality's
own parquet. ``load_pipeline_data`` selects a whitelist for demographics and
joins two separately delivered blocks onto their host modalities, so for
three of the ten rows the parquet's width is not the model's feature count.
It is also the kind of number that stays plausible when it is wrong —
nothing downstream contradicts it — so it is pinned here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pcc_analysis.data_manager import CONFOUNDER_COLUMNS, FULL_STACK_JOINS

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "pipeline"
    / "05_compute_statistics.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("compute_statistics_cli", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stats = _load_module()

N = 40
_IDS = list(range(N))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Processed parquets with the two columns Table 2 must not count."""
    pd.DataFrame(
        {
            "ID": _IDS,
            "basis_age": [50] * N,
            "basis_sex": [1, 2] * (N // 2),
            "basis_uort": [1] * N,
            # Present in the real parquet, never selected by the loader.
            "basis_lvl": [3] * N,
            "basis_status_mrt": [1] * N,
        }
    ).to_parquet(tmp_path / "demographics.parquet", index=False)

    pd.DataFrame({"ID": _IDS, "phq9_sum": [4] * N, "gad7_sum": [3] * N}).to_parquet(
        tmp_path / "mental_health.parquet", index=False
    )
    pd.DataFrame(
        {
            "ID": _IDS,
            **{f"phq9_item_{i}": [0] * N for i in range(1, 10)},
            "phq9_sum_items": [0] * N,
            "phq9_items_complete": [True] * N,
        }
    ).to_parquet(tmp_path / "phq9_items.parquet", index=False)

    pd.DataFrame({"ID": _IDS, "has_hypertension": [0] * N}).to_parquet(
        tmp_path / "medical_history.parquet", index=False
    )
    pd.DataFrame(
        {"ID": _IDS, "smoking_status": [1] * N, "pack_years": [0.0] * N}
    ).to_parquet(tmp_path / "smoking.parquet", index=False)
    return tmp_path


def _counts(data_dir: Path) -> dict[str, int]:
    frame = pd.DataFrame({"ID": _IDS, "has_mri": [True] * N})
    _, results = _stats.generate_table2(frame, data_dir)
    return {name: entry["n_features"] for name, entry in results.items()}


def test_demographics_counts_the_confounder_whitelist_only(data_dir: Path) -> None:
    """basis_lvl and basis_status_mrt describe the sample, they do not predict."""
    assert _counts(data_dir)["Demographics"] == len(CONFOUNDER_COLUMNS) == 3


def test_a_joined_block_counts_toward_its_host_modality(data_dir: Path) -> None:
    """The model reads mental_health eleven columns wider than its parquet."""
    counts = _counts(data_dir)
    assert counts["Mental Health"] == 2 + 11
    assert counts["Medical History"] == 1 + 2


def test_an_absent_joined_parquet_falls_back_to_the_host(data_dir: Path) -> None:
    """A checkout without the amendment delivery is a supported state."""
    (data_dir / "phq9_items.parquet").unlink()
    assert _counts(data_dir)["Mental Health"] == 2


def test_an_absent_modality_parquet_is_reported_as_unavailable(
    data_dir: Path,
) -> None:
    (data_dir / "mental_health.parquet").unlink()
    frame = pd.DataFrame({"ID": _IDS, "has_mri": [True] * N})
    latex, results = _stats.generate_table2(frame, data_dir)

    assert results["Mental Health"]["pattern"] == "Not available"
    assert results["Mental Health"]["n_complete"] == 0
    # No count at all rather than a plausible one. The dict used to carry a
    # per-modality fallback, and two of those had drifted years behind the
    # data, so a missing parquet printed a wrong number next to the words
    # "Not available" instead of admitting it had nothing to count.
    assert results["Mental Health"]["n_features"] is None
    assert "Mental Health & n/a &" in latex


def test_a_partial_mri_delivery_does_not_report_a_full_count(
    data_dir: Path,
) -> None:
    """Summing the atlases that happen to be present would look complete."""
    for name, cols in (("mri_subcortical", 3), ("mri_cerebellar", 2)):
        pd.DataFrame(
            {"ID": _IDS, **{f"{name}_{i}": [1.0] * N for i in range(cols)}}
        ).to_parquet(data_dir / f"{name}.parquet", index=False)
    frame = pd.DataFrame({"ID": _IDS, "has_mri": [True] * N})

    latex, results = _stats.generate_table2(frame, data_dir)

    assert results["MRI"]["n_features"] is None
    assert "MRI & n/a &" in latex


def test_the_supplementary_table_counts_the_same_features(data_dir: Path) -> None:
    """Table S1 is the one the manuscript transcribes; it must not disagree.

    The two tables were computed by separate code paths, and only one of them
    was ever corrected — leaving the paper quoting the uncorrected count.
    """
    frame = pd.DataFrame({"ID": _IDS, "has_mri": [True] * N})
    table2 = {
        name: e["n_features"]
        for name, e in _stats.generate_table2(frame, data_dir)[1].items()
    }
    table_s1 = _stats.generate_table_s1_missingness(frame, data_dir)

    for modality in ("Demographics", "Mental Health", "Medical History"):
        assert table_s1[modality]["n_features"] == table2[modality], modality


def test_the_join_declaration_is_the_loaders_own(data_dir: Path) -> None:
    """Pinned so the table cannot drift from what load_pipeline_data joins."""
    assert set(FULL_STACK_JOINS) <= set(_stats.MODALITY_PIPELINE_KEYS.values())
    for host, stem in FULL_STACK_JOINS.items():
        assert (data_dir / f"{stem}.parquet").exists(), host


def test_the_mri_row_counts_every_atlas_column(tmp_path: Path) -> None:
    """The MRI count reads the parquet footer; it must still be the real count."""
    files = _stats.MODALITIES["MRI"]["files"]
    assert isinstance(files, list)
    expected = 0
    for position, fname in enumerate(files):
        width = position + 2
        pd.DataFrame(
            {"ID": _IDS, **{f"region_{position}_{c}": [1.0] * N for c in range(width)}}
        ).to_parquet(tmp_path / fname, index=False)
        expected += width
    pd.DataFrame({"ID": _IDS, "x": [1.0] * N}).to_parquet(
        tmp_path / "demographics.parquet", index=False
    )

    frame = pd.DataFrame({"ID": _IDS, "has_mri": [True] * N})
    _, results = _stats.generate_table2(frame, tmp_path)

    assert results["MRI"]["n_features"] == expected


def test_results_serialise_for_the_constants_payload(data_dir: Path) -> None:
    """descriptive_stats.json is written with json.dump(default=float)."""
    frame = pd.DataFrame({"ID": _IDS, "has_mri": [True] * N})
    _, results = _stats.generate_table2(frame, data_dir)
    assert json.loads(json.dumps(results, default=float))
