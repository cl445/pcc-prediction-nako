"""Tests for the data fingerprint and the cross-run comparability check."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from pcc_analysis.run_comparability import (
    IncomparableRunsError,
    assert_comparable_runs,
    fingerprint_analysis_data,
    fingerprint_modality,
    read_run_stack,
)

STACK = "python==3.14.6 numpy==2.5.2 xgboost==3.4.0"
OTHER_STACK = "python==3.14.6 numpy==2.5.2 xgboost==3.2.0"


def _write_run(
    run_dir: Path,
    *,
    package_versions: str | None = STACK,
    device: str | None = "cpu",
    data_fingerprint: str = "0123456789abcdef",
    machine: str = "x86_64",
) -> Path:
    """Write the two records a run directory carries about its environment."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        "random_state": 42,
        "data_fingerprint": data_fingerprint,
    }
    if package_versions is not None:
        config["package_versions"] = package_versions
    if device is not None:
        config["device"] = device
    pd.DataFrame([config]).to_csv(run_dir / "config.csv", index=False)
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "git": {"commit": "abc123", "branch": "main", "dirty": False},
                "platform": "Linux-6.8.0-x86_64",
                "machine": machine,
            }
        )
    )
    return run_dir


def _analysis_data() -> tuple[pd.Series, dict[str, pd.DataFrame]]:
    y = pd.Series([0, 1, 0, 1], name="bahmer_any_pcs")
    X_dict = {
        "demographics": pd.DataFrame({"age": [40.0, 50.0, 60.0, 30.0]}),
        "mental_health": pd.DataFrame({"phq9_sum": [3.0, 12.0, 1.0, 8.0]}),
    }
    return y, X_dict


def test_fingerprint_is_stable_for_equal_data() -> None:
    """Two machines given the same parquets must agree, or the guard is noise."""
    first = fingerprint_analysis_data(*_analysis_data())
    second = fingerprint_analysis_data(*_analysis_data())
    assert first == second


def test_fingerprint_moves_with_a_feature_value() -> None:
    """The case the per-fold label check cannot see."""
    y, X_dict = _analysis_data()
    baseline = fingerprint_analysis_data(y, X_dict)

    moved = {name: frame.copy() for name, frame in X_dict.items()}
    moved["mental_health"].loc[0, "phq9_sum"] = 4.0

    assert fingerprint_analysis_data(y, moved) != baseline


def test_fingerprint_moves_with_the_feature_set() -> None:
    """A joined or dropped column changes the model, so it changes the digest."""
    y, X_dict = _analysis_data()
    baseline = fingerprint_analysis_data(y, X_dict)

    joined = {name: frame.copy() for name, frame in X_dict.items()}
    joined["mental_health"]["gad7_sum"] = [1.0, 9.0, 0.0, 6.0]

    assert fingerprint_analysis_data(y, joined) != baseline


def test_fingerprint_moves_with_the_labels() -> None:
    y, X_dict = _analysis_data()
    baseline = fingerprint_analysis_data(y, X_dict)
    relabelled = pd.Series([0, 1, 1, 1], name=y.name)
    assert fingerprint_analysis_data(relabelled, X_dict) != baseline


def test_read_run_stack_reads_both_records(tmp_path: Path) -> None:
    """config.csv decides comparability, provenance.json explains it."""
    stack = read_run_stack(_write_run(tmp_path / "primary"))

    assert stack.package_versions == STACK
    assert stack.device == "cpu"
    assert stack.data_fingerprint == "0123456789abcdef"
    assert stack.machine == "x86_64"
    assert stack.commit == "abc123"
    assert stack.has_config


def test_read_run_stack_on_a_directory_that_is_not_a_run(tmp_path: Path) -> None:
    """Never raises: this runs at the end of a multi-hour analysis."""
    empty = tmp_path / "not_a_run"
    empty.mkdir()
    stack = read_run_stack(empty)

    assert not stack.has_config
    assert stack.package_versions is None


def test_a_differing_stack_stops_the_comparison(tmp_path: Path) -> None:
    """Two halves of one rerun, produced against different library stacks."""
    here = _write_run(tmp_path / "primary")
    there = _write_run(tmp_path / "control", package_versions=OTHER_STACK)

    with pytest.raises(IncomparableRunsError) as excinfo:
        assert_comparable_runs([here, there], comparison="The control delta")

    message = str(excinfo.value)
    assert "The control delta" in message
    assert "package_versions" in message
    # Both values, so the reader can see which machine is the odd one out
    # instead of being told only that something differs.
    assert OTHER_STACK in message
    assert "control" in message


def test_a_differing_device_stops_the_comparison(tmp_path: Path) -> None:
    here = _write_run(tmp_path / "lean_mri", device="cpu")
    there = _write_run(tmp_path / "lean_non_mri", device="cuda")

    with pytest.raises(IncomparableRunsError, match="device"):
        assert_comparable_runs([here, there], comparison="The utility panel")


def test_matching_runs_pass(tmp_path: Path) -> None:
    """A check that also stops the workflow it protects is not a check."""
    assert_comparable_runs(
        [_write_run(tmp_path / "a"), _write_run(tmp_path / "b")],
        comparison="A comparison of two runs from one machine",
    )


def test_different_data_does_not_stop_a_comparison(tmp_path: Path) -> None:
    """The MRI and non-MRI runs are *meant* to see different data.

    The fingerprint guards the fold merge from inside the config hash.
    Comparing it across run directories would reject the cohort
    comparisons the paper is built on.
    """
    assert_comparable_runs(
        [
            _write_run(tmp_path / "lean_mri", data_fingerprint="1111111111111111"),
            _write_run(tmp_path / "lean_non_mri", data_fingerprint="2222222222222222"),
        ],
        comparison="The clinical-utility panel",
    )


def test_a_different_architecture_is_reported_but_allowed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Architecture explains a delta; it does not invalidate one.

    ``package_versions`` is identical across an arm64 and an x86_64 machine
    — the pins are the same — so this is the field that says why two runs
    under one stack still moved.
    """
    here = _write_run(tmp_path / "primary", machine="x86_64")
    there = _write_run(tmp_path / "split_mh", machine="arm64")

    with caplog.at_level(logging.WARNING):
        assert_comparable_runs([here, there], comparison="The sub-modality panel")

    assert "machine" in caplog.text
    assert "arm64" in caplog.text


def test_an_unstamped_run_is_not_read_as_agreement(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A run from before the stamp cannot be checked, and says so.

    Failing on it would invalidate every pre-2.17 run directory; passing
    silently would let one of them stand in for a verified match.
    """
    here = _write_run(tmp_path / "primary")
    there = _write_run(tmp_path / "old_run", package_versions=None)

    with caplog.at_level(logging.WARNING):
        assert_comparable_runs([here, there], comparison="A delta against an old run")

    assert "package_versions not recorded" in caplog.text


def test_an_unreadable_stamp_blocks_the_comparison(tmp_path: Path) -> None:
    """A stamp that cannot be read is not a stamp that agrees.

    A config.csv truncated by a full disk or a killed copy parses to
    pandas' EmptyDataError — a ValueError, not a ParserError. Reporting it
    like a missing file would let a corrupt run directory into a delta.
    """
    here = _write_run(tmp_path / "primary")
    truncated = _write_run(tmp_path / "control")
    (truncated / "config.csv").write_text("")

    stack = read_run_stack(truncated)
    assert stack.unreadable is not None
    assert stack.package_versions is None
    # provenance.json is written first, so it is still readable and still
    # says which machine this directory came from.
    assert stack.machine == "x86_64"

    with pytest.raises(IncomparableRunsError, match="unreadable"):
        assert_comparable_runs([here, truncated], comparison="The control delta")


def test_a_run_without_a_config_still_reports_its_machine(tmp_path: Path) -> None:
    """The interrupted-run case: provenance exists, config.csv does not."""
    run_dir = _write_run(tmp_path / "lean_pooled")
    (run_dir / "config.csv").unlink()

    stack = read_run_stack(run_dir)

    assert not stack.has_config
    assert stack.unreadable is None
    assert stack.machine == "x86_64"
    assert stack.commit == "abc123"


def test_a_dirty_tree_does_not_read_as_the_same_code(tmp_path: Path) -> None:
    """A commit hash alone claims the run used exactly the committed code."""
    clean = read_run_stack(_write_run(tmp_path / "primary"))
    dirty_dir = _write_run(tmp_path / "split_mh")
    (dirty_dir / "provenance.json").write_text(
        json.dumps(
            {
                "git": {"commit": "abc123", "branch": "main", "dirty": True},
                "platform": "Linux-6.8.0-x86_64",
                "machine": "x86_64",
            }
        )
    )
    dirty = read_run_stack(dirty_dir)

    assert clean.commit == dirty.commit
    assert clean.field("commit") != dirty.field("commit")
    assert (dirty.field("commit") or "").endswith("-dirty")


def test_per_modality_digests_locate_a_divergence() -> None:
    """The run-level digest says *that*; this says *where*."""
    _, X_dict = _analysis_data()
    moved = {name: frame.copy() for name, frame in X_dict.items()}
    moved["mental_health"].loc[0, "phq9_sum"] = 4.0

    assert fingerprint_modality("demographics", X_dict["demographics"]) == (
        fingerprint_modality("demographics", moved["demographics"])
    )
    assert fingerprint_modality("mental_health", X_dict["mental_health"]) != (
        fingerprint_modality("mental_health", moved["mental_health"])
    )


def test_a_modality_digest_covers_its_name() -> None:
    """Two modalities with identical contents are still two modalities."""
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    assert fingerprint_modality("ses", frame) != fingerprint_modality(
        "cognitive", frame
    )
