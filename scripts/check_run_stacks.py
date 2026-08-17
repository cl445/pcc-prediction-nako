"""Check that a set of run directories may be read against each other.

A rerun of this analysis is a set of runs, not one run: the primary result,
its control, four sensitivity variants, four Lean fits and a transfer. Every
number the manuscript reports as a difference is a subtraction across two of
those directories, and each subtraction assumes the two were produced by the
same analysis under the same library stack on the same kind of hardware.

``_config_hash`` covers the folds of a *single* run (DECISIONS §2.21), which
is the comparison a rerun split by configuration never performs: each run
directory is then internally consistent, while the deltas between them can
mix two environments.

Run this once the pieces are collected in one place, and before the numbers
are carried into the manuscript:

    uv run python scripts/check_run_stacks.py --runs-dir results/runs
    uv run python scripts/check_run_stacks.py results/runs/primary results/runs/control

It exits non-zero when the runs disagree on something that makes a delta
unreadable — the library stack or the XGBoost device — and when a run's
stamp exists but cannot be read, since a corrupt record must not pass for
agreement. It reports, without failing, what is worth knowing but is no
reason to reject a comparison: architecture, platform, commit and
working-tree state, a directory that has not finished writing its
``config.csv``, and any field a run directory does not record at all.

``data_fingerprint`` is shown but never compared: the MRI and non-MRI runs
are *meant* to see different data, so only the reader knows whether two runs
should have seen the same input. It guards the one place identical data is
required, the fold merge, from inside the config hash — and before a run
starts, `02_run_pipeline.py --fingerprint-only` prints it for a
configuration without computing anything, which is how two machines check
they hold the same parquets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from pcc_analysis.run_comparability import (
    ADVISORY_FIELDS,
    DECISIVE_FIELDS,
    RunStack,
    describe_disagreements,
    read_run_stack,
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="*",
        help="Run directories to compare (default: every run under --runs-dir)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Compare every immediate subdirectory of DIR that is a run",
    )
    return parser.parse_args()


def collect_run_dirs(args: argparse.Namespace) -> list[Path]:
    """Resolve the directories to compare, in a stable order.

    A directory counts as a run if it holds either of the two records a run
    writes. ``config.csv`` is written last, so an interrupted run, an
    in-flight copy from the other machine and a fold-subset directory have
    only the ``provenance.json`` that ``__init__`` wrote — states to name
    rather than to skip.
    """
    run_dirs = list(args.run_dirs)
    if args.runs_dir is not None and args.runs_dir.is_dir():
        run_dirs += [
            child
            for child in sorted(args.runs_dir.iterdir())
            if (child / "config.csv").is_file() or (child / "provenance.json").is_file()
        ]
    return run_dirs


def render(stacks: list[RunStack]) -> None:
    """Print what each run records, so a mismatch can be read off directly.

    The library stack is eight packages and does not fit a table column
    eleven times over, so each distinct one gets a short tag and is spelled
    out once below the table. Two runs share a tag exactly when they share
    a stack, which is the question the table is read for; the full strings
    are still what the error message quotes.
    """
    tags: dict[str, str] = {}
    for stack in stacks:
        recorded = stack.package_versions
        if recorded and recorded not in tags:
            tags[recorded] = f"stack-{len(tags) + 1}"

    table = Table(title="Run environments")
    table.add_column("run", overflow="fold")
    table.add_column("stack")
    table.add_column("device")
    # Shown, never compared: two cohorts are meant to see different data, so
    # a difference here is only a finding when the two runs should have seen
    # the same input — which the reader knows and this table does not.
    table.add_column("data")
    for field in ADVISORY_FIELDS:
        table.add_column(field, overflow="fold")
    for stack in stacks:
        commit = (stack.commit or "—")[:8]
        if stack.dirty:
            commit += "-dirty"
        table.add_row(
            stack.run_dir.name,
            "unreadable"
            if stack.unreadable
            else tags.get(stack.package_versions or "", "—"),
            stack.device or "—",
            stack.data_fingerprint or "—",
            stack.machine or "—",
            stack.platform or "—",
            commit,
        )
    console.print(table)
    for recorded, tag in tags.items():
        console.print(f"  {tag}: {recorded}")


def main() -> None:
    args = parse_args()
    run_dirs = collect_run_dirs(args)
    if not run_dirs:
        console.print(
            "[yellow]No run directories given, and none found under "
            f"{args.runs_dir}. Nothing to check."
        )
        # Not an error: the assembly step runs this unconditionally, and a
        # machine that has not produced its runs yet has nothing to answer
        # for.
        return

    stacks = [read_run_stack(run_dir) for run_dir in run_dirs]
    render(stacks)

    decisive, advisory = describe_disagreements(stacks)
    for line in advisory:
        console.print(f"[yellow]note:[/yellow] {line.strip()}")
    if decisive:
        console.print("\n[red]These runs are not comparable:[/red]")
        for line in decisive:
            console.print(line)
        console.print(
            "\nA delta across them measures the environment as well as the "
            "analysis change it is meant to measure. Re-run the differing "
            "configuration in the other environment before carrying any "
            "difference into the manuscript."
        )
        sys.exit(1)

    # Count what actually answered, not what was looked at. A directory that
    # recorded nothing agrees with nothing, and folding it into a green
    # "N run(s) agree" is how an interrupted run passes for a checked one.
    answered = [s for s in stacks if all(s.field(field) for field in DECISIVE_FIELDS)]
    fields = " and ".join(DECISIVE_FIELDS)
    if len(answered) == len(stacks):
        console.print(f"[green]{len(answered)} run(s) agree on {fields}.[/green]")
    else:
        console.print(
            f"[yellow]{len(answered)} of {len(stacks)} run(s) agree on {fields}; "
            f"the rest recorded nothing to compare (see the notes above).[/yellow]"
        )


if __name__ == "__main__":
    main()
