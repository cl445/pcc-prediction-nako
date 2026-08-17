"""Tests for the step selection in ``scripts/run_analysis.sh``.

The selection exists so a rerun can be split across two machines without
anyone assembling the pipeline commands by hand, which is what loses the
guards the script carries. Its failure modes are therefore silent ones: a
misspelt step that runs nothing and exits 0, or a step that runs on an
input the other machine was supposed to produce.

The tests run the script against a copy in a scratch directory, because the
script resolves its run directories relative to its own location — pointing
it at the real repository would risk the check starting a multi-hour step.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_analysis.sh"


def _isolated_script(tmp_path: Path) -> Path:
    """Copy the script into an empty tree, so no run directory exists."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    copy = scripts_dir / SCRIPT.name
    shutil.copy(SCRIPT, copy)
    return copy


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["bash", str(script), *args],  # noqa: S607 - bash from PATH
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_list_names_every_step(tmp_path: Path) -> None:
    """The names are the interface: `--only` is useless without them."""
    result = _run(_isolated_script(tmp_path), "--list")

    assert result.returncode == 0
    steps = result.stdout.split()
    for step in ("data", "primary", "split_mh", "lean_pooled", "transfer", "panels"):
        assert step in steps


def test_a_misspelt_step_stops_before_any_work(tmp_path: Path) -> None:
    """`--only lean-pooled` must not look like a run that succeeded."""
    result = _run(_isolated_script(tmp_path), "--only", "lean-pooled")

    assert result.returncode == 2
    assert "Unknown step" in result.stderr
    # The valid names, so the fix does not need a second command.
    assert "lean_pooled" in result.stderr


def test_an_unknown_flag_stops(tmp_path: Path) -> None:
    result = _run(_isolated_script(tmp_path), "--onlyy", "primary")

    assert result.returncode == 2
    assert "Unknown argument" in result.stderr


def test_a_step_whose_input_is_missing_names_it(tmp_path: Path) -> None:
    """The other machine's half is absent, and the message has to say so.

    Running the figures without the primary run is the shape of every
    two-machine mistake: the step is selected, its input was produced
    elsewhere, and nothing local says which.
    """
    result = _run(_isolated_script(tmp_path), "--only", "figures")

    assert result.returncode == 1
    assert "results/runs/primary" in result.stderr
    assert "primary" in result.stderr
    assert "copy" in result.stderr


def test_selection_is_reported(tmp_path: Path) -> None:
    """A partial run has to be recognisable as one in the log."""
    result = _run(_isolated_script(tmp_path), "--only", "figures")

    assert "Partial run" in result.stdout


def _stub_uv(tmp_path: Path) -> Path:
    """A `uv` on PATH that records its arguments instead of computing.

    The selection is a property of the whole graph, and the graph is days of
    real work, so the script walks it against a stub and the recorded
    commands stand in for the runs.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "uv"
    stub.write_text('#!/bin/sh\necho "UV $*"\nexit 0\n')
    stub.chmod(0o755)
    return bin_dir


def _finished_runs(root: Path, runs: str = "results/runs") -> None:
    """Finished run directories for every configuration the graph reads.

    The dependency guards check for `config.csv` — a run's last file — so a
    walk of the full graph needs one per directory, plus the sub-modality
    artifact the split-MH guard counts rows in.
    """
    names = [
        "primary",
        "control",
        "neurocog",
        "mixed_controls",
        "no_dml",
        "split_mh",
        "lean_mri",
        "control_lean",
        "lean_non_mri",
        "lean_pooled",
        "transfer_lean_mri_to_non_mri",
    ]
    for name in names:
        run_dir = root / runs / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.csv").write_text("random_state\n42\n")
    shap = root / runs / "split_mh" / "shap_modality_importance.csv"
    shap.write_text(
        "modality,importance\n"
        + "".join(f"mh_{i},0.1\n" for i in ("phq9", "gad7", "mini", "panic", "stress"))
    )


def _pipeline_runs(output: str) -> set[str]:
    """Which pipeline configurations a recorded run would have computed."""
    return {
        line.rsplit("/", 1)[-1].strip()
        for line in output.splitlines()
        if "02_run_pipeline.py" in line and "--output-dir" in line
    }


def _walk(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    script = _isolated_script(tmp_path)
    _finished_runs(script.parent.parent)
    env = {
        **os.environ,
        "PATH": f"{_stub_uv(tmp_path)}:{os.environ['PATH']}",
    }
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["bash", str(script), *args],  # noqa: S607 - bash from PATH
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=env,
    )


def test_the_two_halves_of_the_split_are_complementary(tmp_path: Path) -> None:
    """`--only X` and `--except X` have to partition the configurations.

    Anything in both is computed twice; anything in neither is missing from
    the assembled set. Both mistakes surface a day and a half later, when
    the two halves are read against each other.
    """
    whole = _walk(tmp_path / "whole")
    long_half = _walk(tmp_path / "long", "--only", "lean_pooled")
    rest = _walk(tmp_path / "rest", "--except", "lean_pooled")

    assert whole.returncode == 0, whole.stderr
    computed_whole = _pipeline_runs(whole.stdout)
    computed_long = _pipeline_runs(long_half.stdout)
    computed_rest = _pipeline_runs(rest.stdout)

    assert len(computed_whole) == 10
    assert computed_long == {"lean_pooled"}
    assert computed_long & computed_rest == set()
    assert computed_long | computed_rest == computed_whole


def test_an_unfinished_run_does_not_stop_an_unrelated_step(tmp_path: Path) -> None:
    """A directory is not a finished run.

    `config.csv` is written last, so an interrupted run, a killed copy and
    an rsync in flight all leave a directory behind. Reading one of those as
    a finished run takes down the whole invocation under `set -e`, including
    one that selected neither of the runs being compared.
    """
    root = tmp_path / "interrupted"
    script = _isolated_script(root)
    _finished_runs(script.parent.parent)
    # What a run killed at hour 30 leaves: a directory, no config.csv.
    for stale in ("control", "split_mh"):
        (script.parent.parent / "results" / "runs" / stale / "config.csv").unlink()
    env = {**os.environ, "PATH": f"{_stub_uv(root)}:{os.environ['PATH']}"}
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["bash", str(script), "--only", "lean_pooled"],  # noqa: S607 - bash from PATH
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert _pipeline_runs(result.stdout) == {"lean_pooled"}
    assert "no delta" in result.stdout


def test_an_empty_selection_is_refused(tmp_path: Path) -> None:
    """`--only=` parsed, selected nothing, and exited 0 having done nothing."""
    script = _isolated_script(tmp_path)

    for argv in (["--only="], ["--only", ""], ["--only", ","]):
        result = _run(script, *argv)
        assert result.returncode == 2, argv
        assert "empty step list" in result.stderr


def test_a_step_in_both_selections_is_refused(tmp_path: Path) -> None:
    """`--only primary --except primary` runs nothing at all."""
    result = _run(
        _isolated_script(tmp_path), "--only", "primary", "--except", "primary"
    )

    assert result.returncode == 2
    assert "both --only and --except" in result.stderr
