"""What produced a run: the library stack, and the checkout it ran from.

Two questions, deliberately answered in two places.

*Which stack?* — :func:`package_versions` goes into ``config.csv`` and from
there into the config hash, because the answer decides whether two runs are
comparable at all. DECISIONS §2.20 measured that: xgboost 3.2.0 to 3.4.0
moves 13 of 16 output files. Folds computed against different stacks are
therefore not the same analysis, and failing their merge is the correct
behaviour rather than an inconvenience.

*Which checkout?* — :func:`write_provenance` writes ``provenance.json``
next to the results: commit, branch, whether the working tree was dirty,
and the platform. None of it belongs in the hash. A commit hash cannot make
two runs incomparable — the stack it produced already did that, and hashing
the commit as well would reject a rerun from a checkout whose only change
was a comment. It is recorded so a reported number can be traced back to
the code that produced it, which is a different job from gating a merge.

Nothing here may raise. Provenance describes a run, it is never a
precondition for one: a missing ``git`` binary or a checkout exported
without history has to leave a gap in the record and let the analysis
proceed.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Every library whose numerical output the pipeline consumes. A run directory
# is only comparable to another one if these agree, so they are recorded next
# to the results rather than left to the lockfile of whichever checkout
# happened to produce them.
RESULT_AFFECTING_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "shap",
    "statsmodels",
    "xgboost",
)

_GIT_TIMEOUT_SECONDS: float = 10.0


def package_versions() -> str:
    """Return the runtime stack as a sorted ``name==version`` list.

    Python itself is included; a package that is not installed is recorded as
    ``name==absent`` so the field stays readable instead of raising.
    """
    parts = [f"python=={sys.version.split()[0]}"]
    for package in RESULT_AFFECTING_PACKAGES:
        try:
            parts.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            parts.append(f"{package}==absent")
    return " ".join(parts)


def _git(repo_root: Path, *args: str) -> str | None:
    """Run one read-only git command, or return ``None`` if it cannot."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(repo_root), *args],  # noqa: S607 - git from PATH
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    """Commit, branch and working-tree cleanliness, best effort.

    ``dirty`` is the field that matters most and the one easiest to lose:
    a commit hash alone silently claims that the run used exactly the
    committed code, which is false for every run started from a working
    tree with edits in it. When the status cannot be determined the field
    stays ``None`` rather than defaulting to ``False``.

    The same claim is what makes an enclosing repository dangerous here.
    ``git -C`` searches upward, so a directory that is not a checkout at all
    still answers with whichever repository happens to contain it: a
    non-editable install resolves to a path inside the virtualenv, and a
    virtualenv usually lives inside the working tree it was built from, so
    the record would carry that tree's current commit and dirty flag instead
    of the older code that was installed and executed. Only a ``repo_root``
    that is itself the top of a checkout is believed; anything else leaves
    the same honest gap as a checkout exported without history.
    """
    toplevel = _git(repo_root, "rev-parse", "--show-toplevel")
    commit = None
    if toplevel is not None and Path(toplevel).resolve() == repo_root.resolve():
        commit = _git(repo_root, "rev-parse", "HEAD")
    if commit is None:
        logger.info(
            "No git metadata recorded for %s: git did not confirm it as the "
            "root of a checkout with a readable HEAD. Provenance records the "
            "environment only.",
            repo_root,
        )
        return {"commit": None, "branch": None, "dirty": None}

    status = _git(repo_root, "status", "--porcelain")
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def collect_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Describe the code and environment this process is running.

    Parameters
    ----------
    repo_root
        Directory to ask git about. Defaults to the repository the
        installed package was imported from, which is the checkout that
        produced the run for an editable install and simply has no git
        metadata otherwise.
    """
    if repo_root is None:
        # src/pcc_analysis/_provenance.py -> src/pcc_analysis -> src -> root
        repo_root = Path(__file__).resolve().parents[2]

    try:
        pcc_version: str | None = version("pcc-analysis")
    except PackageNotFoundError:
        pcc_version = None

    return {
        "git": _git_provenance(repo_root),
        "pcc_analysis_version": pcc_version,
        "platform": platform.platform(),
        # The architecture on its own, next to the full platform string.
        # Two runs on x86_64 and arm64 carry identical `package_versions`
        # — the pins are the same — while their BLAS and their rounding are
        # not, so this is the field that explains a delta the stack stamp
        # says should not exist. Not in the config hash: the platform string
        # moves with every kernel patch, and a merge refused on a kernel
        # upgrade would be refusing a sound analysis.
        "machine": platform.machine(),
        # Repeated from config.csv on purpose: provenance.json is written
        # before the run starts and is all a crashed run leaves behind, so
        # it has to be readable on its own.
        "packages": package_versions(),
    }


def write_provenance(
    output_dir: Path,
    repo_root: Path | None = None,
    *,
    filename: str = "provenance.json",
) -> Path | None:
    """Write a provenance record into a run directory.

    Returns the path written, or ``None`` if it could not be written — the
    caller is in the middle of starting a multi-hour analysis and a
    read-only output directory is its problem to hit, not this module's to
    raise about.

    Parameters
    ----------
    filename
        Name of the file to write. A step that only assembles results other
        runs produced — merging folds computed on another machine, against
        another stack — passes a name of its own, so that the record of the
        run which actually computed those folds stays readable beside it
        instead of being replaced by a description of the merge.
    """
    path = output_dir / filename
    try:
        with path.open("w") as handle:
            json.dump(collect_provenance(repo_root), handle, indent=2)
            handle.write("\n")
    except OSError:
        logger.warning("Could not write provenance to %s", path, exc_info=True)
        return None
    return path
