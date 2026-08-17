"""Tests for run provenance.

Two things are pinned here. First, that a run directory records the code
and environment that produced it — a reported ROC-AUC that cannot be tied
to a commit is not reproducible on purpose. Second, that collecting that
record can never take a run down: the analysis is the point, the
description of it is not.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pytest

from pcc_analysis._provenance import (
    RESULT_AFFECTING_PACKAGES,
    collect_provenance,
    write_provenance,
)
from pcc_analysis.orchestration import PCCMultimodalPipeline


def _init_repo(root: Path) -> None:
    """Create a git repository with one commit."""

    def run(*argv: str) -> None:
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv, cwd=root, check=True, capture_output=True
        )

    run("git", "init", "--quiet")
    run("git", "config", "user.email", "test@example.invalid")
    run("git", "config", "user.name", "Test")
    (root / "tracked.txt").write_text("one\n")
    run("git", "add", "tracked.txt")
    run("git", "commit", "--quiet", "-m", "initial")


def test_records_commit_and_clean_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    git = collect_provenance(repo)["git"]

    assert git["commit"] is not None
    assert len(git["commit"]) == 40
    assert git["dirty"] is False


def test_reports_a_dirty_working_tree(tmp_path: Path) -> None:
    """A commit hash alone would claim the run used the committed code."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "tracked.txt").write_text("edited\n")

    assert collect_provenance(repo)["git"]["dirty"] is True


def test_missing_git_metadata_leaves_a_gap_rather_than_failing(
    tmp_path: Path,
) -> None:
    """An exported checkout has no history; the run still has to start."""
    git = collect_provenance(tmp_path)["git"]

    assert git == {"commit": None, "branch": None, "dirty": None}


def test_an_enclosing_repository_is_not_mistaken_for_the_code_that_ran(
    tmp_path: Path,
) -> None:
    """``git -C`` searches upward, and a virtualenv sits inside a checkout.

    A non-editable install resolves ``repo_root`` to a directory under the
    virtualenv rather than to a checkout. Because virtualenvs are habitually
    created inside the working tree, the enclosing repository answers for
    it, and the record would pin the run to whatever is checked out today —
    not to the older code that was installed and executed. Recording nothing
    is the honest answer.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    installed_into = repo / ".venv" / "lib" / "python3.14"
    installed_into.mkdir(parents=True)

    git = collect_provenance(installed_into)["git"]

    assert git == {"commit": None, "branch": None, "dirty": None}


def test_the_checkout_is_still_recognised_through_a_symlinked_path(
    tmp_path: Path,
) -> None:
    """Rejecting an enclosing repository must not reject the real one.

    git reports the top level as a real path, while the interpreter may well
    have imported the package through a symlink — ``/var`` against
    ``/private/var`` on macOS is the everyday case. Both sides are resolved
    before they are compared, so a checkout reached by another name is not
    read as a foreign repository and silently dropped from the record.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    reached_by_symlink = tmp_path / "link"
    reached_by_symlink.symlink_to(repo, target_is_directory=True)

    git = collect_provenance(reached_by_symlink)["git"]

    assert git["commit"] is not None
    assert git["dirty"] is False


def test_a_linked_worktree_counts_as_the_checkout_that_ran(
    tmp_path: Path,
) -> None:
    """Refusing a foreign repository must still accept a linked worktree.

    A linked worktree keeps its history in the repository it was created
    from and carries a ``.git`` file rather than a ``.git`` directory, so a
    root test phrased in terms of that directory rejects it and drops the
    commit from every run started there — and this analysis is developed and
    rerun from worktrees. ``rev-parse --show-toplevel`` names the worktree
    itself, which is the tree whose contents the run actually executed.
    """
    main = tmp_path / "main"
    main.mkdir()
    _init_repo(main)
    linked = tmp_path / "linked"
    subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "worktree", "add", "--quiet", "-b", "side", str(linked)],  # noqa: S607
        cwd=main,
        check=True,
        capture_output=True,
    )

    git = collect_provenance(linked)["git"]

    assert git["commit"] is not None
    assert git["branch"] == "side"


def test_dirty_is_unknown_rather_than_false_when_status_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``dirty`` must never default to the reassuring answer."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    real = subprocess.run

    def fail_on_status(argv: list[str], **kwargs: Any) -> object:
        if "status" in argv:
            raise OSError("git status unavailable")
        return real(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_on_status)

    assert collect_provenance(repo)["git"]["dirty"] is None


def test_the_stack_stamp_is_repeated_into_the_standalone_record(
    tmp_path: Path,
) -> None:
    """config.csv carries it too, but a crashed run never writes one.

    provenance.json is written before the analysis starts, so it is all a
    run that dies in hour three leaves behind — and a record that cannot
    say which xgboost produced it is not much of a record, given that
    version moves 13 of 16 output files (DECISIONS §2.20).
    """
    packages = collect_provenance(tmp_path)["packages"]

    for name in RESULT_AFFECTING_PACKAGES:
        assert f"{name}==" in packages
    assert "python==" in packages
    # These are hard dependencies; "absent" here means the environment the
    # analysis is about to run in is not the one that was pinned.
    assert "xgboost==absent" not in packages
    assert "scikit-learn==absent" not in packages


def test_the_architecture_is_recorded_beside_the_platform(tmp_path: Path) -> None:
    """The field that explains a delta the stack stamp says cannot exist.

    Two machines installing from one lockfile carry identical
    ``package_versions``, so nothing in ``config.csv`` distinguishes an
    arm64 run from an x86_64 one — while their BLAS, and with it their
    rounding, differ. It stays out of the config hash: unlike the stack,
    the architecture is a reason to explain a difference, not to refuse a
    comparison.
    """
    record = collect_provenance(tmp_path)

    assert record["machine"] == platform.machine()
    assert record["machine"]


def test_write_provenance_survives_an_unwritable_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> object:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "open", refuse)

    assert write_provenance(tmp_path) is None


def test_a_second_record_can_be_written_beside_the_first(tmp_path: Path) -> None:
    """A merge describes itself without erasing what computed the folds.

    The stack that assembles finished folds is not the stack that produced
    them, and the produced-by record is the one a reported number has to be
    traceable to. Naming the merge's own record keeps both on disk.
    """
    default = write_provenance(tmp_path)
    assert default is not None
    default.write_text(json.dumps({"git": {"commit": "computed-the-folds"}}))

    merge = write_provenance(tmp_path, filename="provenance_merge.json")

    assert default == tmp_path / "provenance.json"
    assert merge == tmp_path / "provenance_merge.json"
    assert json.loads((tmp_path / "provenance_merge.json").read_text())["packages"]
    assert json.loads(default.read_text())["git"]["commit"] == "computed-the-folds"


def test_pipeline_writes_provenance_before_running(tmp_path: Path) -> None:
    """Written at construction: a run that dies in hour three still counts."""
    PCCMultimodalPipeline(output_dir=tmp_path / "run")

    payload = json.loads((tmp_path / "run" / "provenance.json").read_text())

    assert set(payload) == {
        "git",
        "pcc_analysis_version",
        "platform",
        "machine",
        "packages",
    }
