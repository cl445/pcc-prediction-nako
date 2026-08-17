"""Tests for the run-comparability CLI.

The tool runs at the end of a rerun and again before numbers move into the
manuscript, and its failure mode is silence: a directory it does not list,
or a summary line that counts a run which recorded nothing.

The script is imported rather than shelled out to, so a failure points at
the function and not at CLI plumbing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_run_stacks.py"

STACK = "python==3.14.6 numpy==2.5.2 xgboost==3.4.0"


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location("check_run_stacks", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finished_run(
    run_dir: Path,
    *,
    package_versions: str = STACK,
    device: str = "cpu",
    data_fingerprint: str = "0123456789abcdef",
    dirty: bool = False,
) -> Path:
    """A run directory as it looks once the run has written its last file."""
    _interrupted_run(run_dir, dirty=dirty)
    pd.DataFrame(
        [
            {
                "random_state": 42,
                "package_versions": package_versions,
                "device": device,
                "data_fingerprint": data_fingerprint,
            }
        ]
    ).to_csv(run_dir / "config.csv", index=False)
    return run_dir


def _interrupted_run(run_dir: Path, *, dirty: bool = False) -> Path:
    """What a run killed mid-flight leaves: provenance, but no config.csv."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "git": {"commit": "abc1234567", "branch": "main", "dirty": dirty},
                "platform": "Linux-6.8.0-x86_64",
                "machine": "x86_64",
            }
        )
    )
    return run_dir


def _run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    """Invoke the CLI's main(), returning its exit status.

    The console width is pinned because rich wraps its table to the
    terminal, and the width differs between a laptop and a CI runner.
    """
    monkeypatch.setenv("COLUMNS", "200")
    module = _load_cli()
    monkeypatch.setattr("sys.argv", ["check_run_stacks.py", *argv])
    try:
        module.main()
    except SystemExit as exit_status:
        return int(exit_status.code or 0)
    return 0


def _plain(capsys: pytest.CaptureFixture[str]) -> str:
    """Captured output with rich's line wrapping collapsed."""
    return " ".join(capsys.readouterr().out.split())


def test_matching_runs_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    _finished_run(runs / "primary")
    _finished_run(runs / "control")

    assert _run_cli(monkeypatch, "--runs-dir", str(runs)) == 0
    assert "2 run(s) agree" in _plain(capsys)


def test_a_differing_stack_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    _finished_run(runs / "primary")
    _finished_run(runs / "control", package_versions="python==3.14.6 xgboost==3.2.0")

    assert _run_cli(monkeypatch, "--runs-dir", str(runs)) == 1
    assert "not comparable" in _plain(capsys)


def test_an_interrupted_run_is_listed_not_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """config.csv is written last, so filtering on it looks away from these.

    An interrupted run and an rsync still in flight are the states a
    two-machine rerun produces, and omitting them reports agreement over a
    set that was never checked.
    """
    runs = tmp_path / "runs"
    _finished_run(runs / "primary")
    _interrupted_run(runs / "lean_pooled")

    status = _run_cli(monkeypatch, "--runs-dir", str(runs))
    out = _plain(capsys)

    assert status == 0  # incomplete is not incompatible
    assert "lean_pooled" in out
    assert "no config.csv" in out
    # And it must not be counted as agreeing with anything.
    assert "1 of 2 run(s) agree" in out


def test_an_unreadable_stamp_is_decisive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A truncated config.csv must not pass for "nothing to compare"."""
    runs = tmp_path / "runs"
    _finished_run(runs / "primary")
    truncated = _interrupted_run(runs / "control")
    (truncated / "config.csv").write_text("")

    assert _run_cli(monkeypatch, "--runs-dir", str(runs)) == 1
    out = _plain(capsys)
    assert "unreadable" in out
    assert "not comparable" in out


def test_a_dirty_working_tree_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same commit, one of them with edits, is not the same code."""
    runs = tmp_path / "runs"
    _finished_run(runs / "primary", dirty=False)
    _finished_run(runs / "split_mh", dirty=True)

    assert _run_cli(monkeypatch, "--runs-dir", str(runs)) == 0
    assert "dirty" in _plain(capsys)


def test_no_run_directories_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """run_analysis.sh calls this unconditionally, including on a fresh box."""
    assert _run_cli(monkeypatch, "--runs-dir", str(tmp_path / "absent")) == 0
    assert "Nothing to check" in _plain(capsys)
