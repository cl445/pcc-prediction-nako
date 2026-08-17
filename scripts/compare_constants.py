"""Diff two sets of paper constants and report every number that moved.

The rerun's expensive half is not the cross-validation, it is the paper
cascade afterwards: once the feature set changes, the headline metrics, the
calibration, every SHAP share and the whole transfer block move with it,
and each has to be carried into the manuscript by hand. Doing that from two
directories of constants read side by side is how a stale number survives
into a submission — not because anyone is careless, but because nothing
lists what actually changed.

This does. Point it at a copy of the constants directory taken before the
rerun and at the one the rerun wrote, and it walks both directories to
their leaves — the JSON payloads and the ``\\newcommand`` files alike — and
reports what appeared, what vanished, and what changed by how much. The
result is a checklist for the cascade rather than an eyeballing exercise.

``run_analysis.sh`` snapshots the constants before regenerating them and
runs this at the end, so a full rerun produces its own changelist. The
manual invocation below is for comparing against something else — an older
snapshot, another checkout, a run from a different machine.

Relative change is reported where it is defined, because it is what
decides whether a difference matters: PLAN_FULL_RERUN expects the headline
to shift a little and says a jump is to be investigated rather than
adopted. ``--threshold`` hides differences below a relative size so the
floating-point noise of a reordered sum does not crowd out the real
movement.

Usage:
    # Before the rerun
    cp -r results/paper/constants /tmp/constants_before

    # After it
    uv run python scripts/compare_constants.py \\
        --before /tmp/constants_before --after results/paper/constants
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_THRESHOLD = 1e-9
"""Report everything by default; the caller decides what counts as noise."""


class Difference(NamedTuple):
    """One leaf that is not identical across the two payloads."""

    path: str
    before: Any
    after: Any

    @property
    def relative(self) -> float | None:
        """Relative change, or ``None`` where it is not defined.

        Undefined for a non-numeric value, for an appearance or a
        disappearance, and for a move away from zero — where the ratio would
        be infinite rather than large.

        Also undefined when either side is NaN or infinite, which a payload
        carries where a quantity does not apply: ``descriptive_stats.json``
        holds NaN tokens for a chi-square on a table with an empty cell. A
        crossing in either direction — a statistic that is not applicable
        on one side and a number on the other — is one of the larger things
        a rerun can do, and arithmetic on NaN yields NaN, which sits below
        every threshold and above none. Classifying the crossing here keeps
        it out of that float comparison and on the changelist.
        """
        if not isinstance(self.before, int | float) or isinstance(self.before, bool):
            return None
        if not isinstance(self.after, int | float) or isinstance(self.after, bool):
            return None
        if not math.isfinite(self.before) or not math.isfinite(self.after):
            return None
        if self.before == 0:
            return None
        return abs(self.after - self.before) / abs(self.before)


MISSING = object()
"""Sentinel for a key present on only one side.

Distinct from ``None``, which a payload can legitimately carry — the
severity panels write ``null`` for a model the sample could not identify,
and reporting that as an appearance would be wrong.
"""


def _walk(before: Any, after: Any, path: str = "") -> list[Difference]:
    """Recurse to the leaves of two payloads, collecting what differs."""
    if isinstance(before, dict) and isinstance(after, dict):
        out: list[Difference] = []
        for key in sorted(set(before) | set(after)):
            out.extend(
                _walk(
                    before.get(key, MISSING),
                    after.get(key, MISSING),
                    f"{path}.{key}" if path else str(key),
                )
            )
        return out
    if isinstance(before, list) and isinstance(after, list):
        out = []
        for index in range(max(len(before), len(after))):
            out.extend(
                _walk(
                    before[index] if index < len(before) else MISSING,
                    after[index] if index < len(after) else MISSING,
                    f"{path}[{index}]",
                )
            )
        return out
    if _equal(before, after):
        return []
    return [Difference(path, before, after)]


def _equal(before: Any, after: Any) -> bool:
    """Equality that treats two NaNs as the same value.

    ``float("nan") != float("nan")`` would otherwise report every
    not-applicable cell as a change on every comparison.
    """
    if (
        isinstance(before, float)
        and isinstance(after, float)
        and math.isnan(before)
        and math.isnan(after)
    ):
        return True
    return bool(before == after)


def _format(value: Any) -> str:
    if value is MISSING:
        return "[dim]—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def compare_payloads(
    before: dict[str, Any], after: dict[str, Any], threshold: float
) -> list[Difference]:
    """Differences between two parsed payloads, above a relative threshold.

    A difference with no defined relative change — an appearance, a removal,
    a string edit, a move away from zero, a NaN on either side — is always
    reported. The threshold can only hide a numeric change small enough to
    measure.
    """
    return [
        difference
        for difference in _walk(before, after)
        if difference.relative is None or difference.relative >= threshold
    ]


CONSTANT_PATTERNS = ("*.json", "*.tex")
"""Both shapes the constants directory holds.

Most ``.tex`` files mirror a JSON twin, but ``outcome_agreement.tex`` is
written by ``scripts/supplementary/outcome_agreement.py`` and by nothing
else, so the outcome-definition kappas and agreement percentages exist in
no payload. Comparing JSON alone would leave a moved kappa off the
changelist, which is precisely how a stale number reaches a submission.
"""


def _constants_files(directory: Path) -> set[str]:
    """Names of the constants files in a directory, of every shape."""
    return {
        path.name for pattern in CONSTANT_PATTERNS for path in directory.glob(pattern)
    }


_NEWCOMMAND = re.compile(r"^\\newcommand\{\\([A-Za-z@]+)\}\{(.*)\}$")
"""One macro definition per line, the form the generators emit."""


def _as_number(body: str) -> Any:
    """A macro body as a float where it is one, else the body itself.

    Numeric bodies are the common case and the interesting one: read as
    numbers they get the same relative-change treatment as a JSON leaf, so
    ``--threshold`` means the same thing on both sides of the directory.
    Anything else — a placeholder, a unit, a word — compares as a string.
    """
    try:
        return float(body)
    except ValueError:
        return body


def _parse_tex(text: str) -> dict[str, Any]:
    r"""Flatten a constants ``.tex`` file into comparable leaves.

    Two shapes live in the directory. Most files are lists of
    ``\newcommand`` definitions, one macro per constant, and those become
    macro-name keys. ``outcome_agreement.tex`` is a bare table body with no
    macros at all, so its rows are keyed by position and compared as
    strings: a kappa that moves changes its row and lands on the changelist
    with the whole row for context.

    Comment lines are dropped. They head the file with the run that wrote
    it, which differs after every rerun by construction, and reporting that
    on every file would bury the constants underneath it.
    """
    payload: dict[str, Any] = {}
    rows = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        macro = _NEWCOMMAND.match(stripped)
        if macro is None:
            payload[f"row[{rows}]"] = stripped
            rows += 1
            continue
        name, body = macro.groups()
        payload[name] = _as_number(body)
    return payload


def _read_constants(path: Path) -> Any:
    """Parse one constants file into leaves, whichever shape it has.

    The encoding is named rather than taken from the locale. The generated
    ``.tex`` headers carry an em dash, so under a ``LC_ALL=C`` shell — a
    cron entry, a CI runner, a batch job on the cluster — the ambient
    encoding is ASCII and the read raises. ``run_analysis.sh`` invokes this
    with a trailing ``|| true`` so the rerun is not lost to a diff failure,
    which would turn that raise into an empty changelist under a heading
    promising one: the silent omission this whole module exists to prevent.
    """
    if path.suffix == ".tex":
        return _parse_tex(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def compare_directories(
    before_dir: Path, after_dir: Path, threshold: float
) -> dict[str, list[Difference]]:
    """Differences per constants file across two constants directories.

    Files that exist on only one side get one synthetic entry each, so a
    payload the rerun stopped writing is as visible as a number that moved.
    """
    before_files = _constants_files(before_dir)
    after_files = _constants_files(after_dir)

    results: dict[str, list[Difference]] = {}
    for name in sorted(before_files | after_files):
        if name not in after_files:
            results[name] = [Difference("<file>", "present", MISSING)]
            continue
        if name not in before_files:
            results[name] = [Difference("<file>", MISSING, "present")]
            continue
        differences = compare_payloads(
            _read_constants(before_dir / name),
            _read_constants(after_dir / name),
            threshold,
        )
        if differences:
            results[name] = differences
    return results


def _render(name: str, differences: list[Difference]) -> Table:
    table = Table(title=name, show_header=True, title_justify="left")
    table.add_column("Constant", style="bold", overflow="fold")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Rel.", justify="right")
    for difference in differences:
        relative = difference.relative
        table.add_row(
            difference.path,
            _format(difference.before),
            _format(difference.after),
            "[dim]—" if relative is None else f"{relative:.1%}",
        )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before",
        type=Path,
        required=True,
        help="Constants directory as it stood before the rerun.",
    )
    parser.add_argument(
        "--after",
        type=Path,
        required=True,
        help="Constants directory the rerun wrote (results/paper/constants).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="R",
        help=(
            "Hide numeric changes whose relative size is below R "
            f"(default {DEFAULT_THRESHOLD:g}, i.e. report everything). "
            "Appearances, removals and non-numeric edits are always shown."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in (args.before, args.after):
        if not directory.is_dir():
            console.print(f"[red]Not a directory: {directory}")
            sys.exit(1)

    results = compare_directories(args.before, args.after, args.threshold)
    if not results:
        console.print("[green]No differences: every constant is unchanged.")
        return

    for name, differences in results.items():
        console.print(_render(name, differences))

    total = sum(len(differences) for differences in results.values())
    console.print(
        f"\n[bold]{total} constant(s) changed across {len(results)} file(s).[/bold] "
        "Every one of them has to be carried into anything downstream that "
        "quotes it. A document that cites these numbers holds its own edits, "
        "so transfer them value by value rather than replacing files."
    )


if __name__ == "__main__":
    main()
