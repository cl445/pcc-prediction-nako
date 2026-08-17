"""Whether two run directories describe the same analysis.

A rerun spread over two machines produces run directories that look
interchangeable — same file names, same columns, same shape — and the
numbers inside them are read against each other constantly: the control
delta against the primary run, the sub-modality panel against the primary
run, the transportability panel across a source and a transfer run, the
clinical-utility panel across two cohorts. Every one of those subtracts a
number produced *here* from a number produced *there*.

``_config_hash`` guards one comparison, the merge of folds computed under
different stacks (DECISIONS §2.21), and that is the comparison a split by
configuration never performs: each run directory is then internally
consistent, and what differs lies between them.

Three fields decide comparability, all recorded in ``config.csv``:

``package_versions``
    The library stack. DECISIONS §2.20 measured that xgboost 3.2.0 to
    3.4.0 moves 13 of 16 output files, so two runs under different stacks
    are not the same analysis and a delta between them is not attributable.

``device``
    ``cpu`` or ``cuda``. XGBoost's histogram construction differs between
    them; a delta across devices measures the hardware, not the change.

``data_fingerprint``
    The analysis data itself. Deliberately *not* compared across run
    directories: two cohorts legitimately differ, so a mismatch between
    ``lean_mri`` and ``lean_non_mri`` is the design, not a defect. It goes
    into the config hash instead, where it guards the one comparison for
    which identical data *is* required — merging folds of a single run.

Architecture, platform and commit are advisory. They are worth printing
next to a delta, but a kernel patch level cannot be made a precondition
for reading two numbers together without rejecting sound comparisons.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas.util import hash_pandas_object

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

#: Fields whose disagreement makes two run directories incomparable.
DECISIVE_FIELDS = ("package_versions", "device")

#: Fields recorded and reported, but never a reason to refuse a comparison.
ADVISORY_FIELDS = ("machine", "platform", "commit")


class IncomparableRunsError(ValueError):
    """Two run directories were produced under stacks that cannot be compared."""


@dataclass(frozen=True)
class RunStack:
    """What a run directory says about the environment that produced it."""

    run_dir: Path
    package_versions: str | None = None
    device: str | None = None
    data_fingerprint: str | None = None
    machine: str | None = None
    platform: str | None = None
    commit: str | None = None
    dirty: bool | None = None
    has_config: bool = True
    unreadable: str | None = None
    """Why ``config.csv`` could not be read, when it exists but is unusable."""

    def field(self, name: str) -> str | None:
        """The comparable value of *name*, or ``None`` if this run has none.

        ``commit`` folds in the working-tree state, because a commit hash on
        its own claims the run used exactly the committed code — which is
        false for every run started from a tree with edits in it, and this
        rerun was prepared from one. Two runs at the same commit, one of
        them dirty, are not known to be the same code and must not read as
        agreement.
        """
        if name == "commit":
            if not self.commit:
                return None
            return f"{self.commit}-dirty" if self.dirty else self.commit
        value = getattr(self, name, None)
        return value if isinstance(value, str) else None


def fingerprint_analysis_data(y: pd.Series, X_dict: dict[str, pd.DataFrame]) -> str:
    """Hash the analysis data a run was given.

    Covers the labels, every modality frame, and the names and order of
    their columns — a feature joined, dropped or reordered upstream changes
    the digest, as does a reprocessed parquet whose values moved.

    The fold merge validates ``y_te`` per fold, which catches a changed
    outcome but not a changed feature matrix: folds computed against
    parquets of different vintages merge as long as the labels agree, and
    the merged model is then fitted on a feature set no fold was scored
    under. Two machines that each process the raw export are the workflow
    that produces such parquets.

    The digest is content-derived rather than taken over the parquet files,
    so regenerating byte-different but value-identical parquets does not
    raise a false alarm.
    """
    digest = hashlib.sha256()
    digest.update(f"y|{len(y)}|{y.name}|".encode())
    digest.update(_frame_digest(y.to_frame()))
    for name in sorted(X_dict):
        digest.update(fingerprint_modality(name, X_dict[name]).encode())
    return digest.hexdigest()[:16]


def fingerprint_modality(name: str, frame: pd.DataFrame) -> str:
    """Digest of one modality frame, including its name and column layout.

    Composed into the run-level digest, and reported per modality by
    ``02_run_pipeline.py --fingerprint-only``: the run-level digest says
    that two machines disagree, this says which parquet they disagree on.
    """
    digest = hashlib.sha256()
    header = f"{name}|{frame.shape[0]}x{frame.shape[1]}|"
    header += ",".join(str(column) for column in frame.columns)
    digest.update(header.encode())
    digest.update(_frame_digest(frame))
    return digest.hexdigest()[:16]


def _frame_digest(frame: pd.DataFrame) -> bytes:
    """Row-wise content hash of one frame, in a fixed byte order.

    ``hash_pandas_object`` combines the columns per row, so the result is
    sensitive to values, dtypes and column order. The explicit
    little-endian cast keeps the digest comparable across machines instead
    of across machines that happen to share a byte order.
    """
    return hash_pandas_object(frame, index=True).to_numpy().astype("<u8").tobytes()


def read_run_stack(run_dir: Path | str) -> RunStack:
    """Read what *run_dir* records about its environment.

    Never raises — this runs at the end of a multi-hour analysis, and over
    directories written by another machine that may still be mid-copy. The
    three states it can report are deliberately distinct:

    *No config.csv.* Not a finished run. ``config.csv`` is the last file a
    run writes, so this is what an interrupted run, an in-flight rsync or a
    fold-subset directory looks like. Advisory: there is nothing to compare,
    which is not the same as a conflict. ``provenance.json`` is written
    first and is still read, so such a directory can at least name its
    machine and commit.

    *Present but unusable.* Truncated, empty or not text. Recorded in
    ``unreadable`` and treated as decisive downstream: a stamp that cannot
    be read must not be allowed to pass as agreement.

    *Readable.* A field the run predates is ``None``, which reads as "cannot
    tell" rather than as agreement.
    """
    run_dir = Path(run_dir)

    provenance: dict[str, object] = {}
    provenance_path = run_dir / "provenance.json"
    if provenance_path.is_file():
        try:
            with provenance_path.open() as handle:
                provenance = json.load(handle)
        except (OSError, ValueError):
            logger.warning("Could not read %s", provenance_path, exc_info=True)

    git = provenance.get("git")
    from_provenance: dict[str, Any] = {
        "machine": _text(provenance.get("machine")),
        "platform": _text(provenance.get("platform")),
        "commit": _text(git.get("commit")) if isinstance(git, dict) else None,
        "dirty": git.get("dirty") if isinstance(git, dict) else None,
    }

    config_path = run_dir / "config.csv"
    if not config_path.is_file():
        return RunStack(run_dir=run_dir, has_config=False, **from_provenance)

    row: dict[str, object] = {}
    try:
        frame = pd.read_csv(config_path)
        if not frame.empty:
            row = dict(frame.iloc[0])
    except (OSError, ValueError, UnicodeDecodeError) as error:
        # ValueError covers pandas' EmptyDataError and ParserError, which is
        # what a config.csv truncated by a full disk or a killed copy looks
        # like. Reported rather than raised, and decisive rather than
        # advisory: see the docstring above.
        logger.warning("Could not read %s: %s", config_path, error)
        return RunStack(
            run_dir=run_dir,
            unreadable=f"{type(error).__name__}: {error}",
            **from_provenance,
        )

    return RunStack(
        run_dir=run_dir,
        package_versions=_text(row.get("package_versions")),
        device=_text(row.get("device")),
        data_fingerprint=_text(row.get("data_fingerprint")),
        **from_provenance,
    )


def _text(value: object) -> str | None:
    """Normalise a CSV/JSON cell to a string, or ``None`` when absent."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def describe_disagreements(stacks: Sequence[RunStack]) -> tuple[list[str], list[str]]:
    """Split the disagreements between *stacks* into decisive and advisory.

    Returns two lists of human-readable lines. A field that only one run
    directory records produces neither: one observation is not a
    disagreement, and reporting it as one would make every comparison
    against a pre-2.17 run directory fail.
    """
    # An unreadable stamp is decisive, not advisory. The alternative — of
    # reporting it like a missing one — lets a truncated config.csv pass as
    # "nothing to compare", which is exactly how a corrupt run directory
    # would slip into a delta.
    decisive: list[str] = [
        f"  {stack.run_dir}: config.csv is unreadable ({stack.unreadable})"
        for stack in stacks
        if stack.unreadable
    ]
    advisory: list[str] = [
        f"  {stack.run_dir}: no config.csv — nothing to compare against"
        for stack in stacks
        if not stack.has_config and not stack.unreadable
    ]

    for field, sink in (
        *((f, decisive) for f in DECISIVE_FIELDS),
        *((f, advisory) for f in ADVISORY_FIELDS),
    ):
        recorded = [(s.run_dir, s.field(field)) for s in stacks if s.field(field)]
        distinct = {value for _, value in recorded}
        if len(distinct) <= 1:
            continue
        lines = "\n".join(f"    {run_dir}: {value}" for run_dir, value in recorded)
        sink.append(f"  {field} differs across the runs:\n{lines}")

    unrecorded = [
        f"  {field} not recorded by {s.run_dir}"
        for field in DECISIVE_FIELDS
        for s in stacks
        if s.has_config and not s.unreadable and not s.field(field)
    ]
    advisory.extend(unrecorded)
    return decisive, advisory


def assert_comparable_runs(run_dirs: Iterable[Path | str], *, comparison: str) -> None:
    """Refuse to read numbers from *run_dirs* against each other if they differ.

    *comparison* names what is about to be computed, so the abort says
    which figure would have been wrong rather than only that something
    mismatched.

    Advisory differences are logged, not raised: a run produced from
    another commit or on another kernel is still comparable, and the person
    reading the delta is better served by knowing than by being stopped.
    """
    stacks = [read_run_stack(run_dir) for run_dir in run_dirs]
    decisive, advisory = describe_disagreements(stacks)
    for line in advisory:
        logger.warning("%s: %s", comparison, line.strip())
    if decisive:
        raise IncomparableRunsError(
            f"{comparison} would compare runs that are not the same analysis.\n"
            + "\n".join(decisive)
            + "\nDECISIONS §2.20 measured that the library stack alone moves 13 of "
            "16 output files, so a delta across these runs is not attributable to "
            "the analysis change it is meant to measure. Re-run the differing "
            "configuration in the environment of the other, or compare only runs "
            "from one machine."
        )
