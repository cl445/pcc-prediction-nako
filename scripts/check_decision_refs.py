"""Every `§N.M` in the tree must name a section that exists in DECISIONS.md.

The code cites DECISIONS by section number in about two dozen places, in
docstrings, comments and runtime output. A citation is a plain string, so
nothing connects it to the document: renumbering the file leaves every one of
them pointing somewhere else, and a citation that was wrong to begin with looks
exactly like one that is right.

Both have happened here. That is what this check is for.

Usage:
    uv run python scripts/check_decision_refs.py

Exits non-zero and names each unresolved citation.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ROOT / "DECISIONS.md"

# `§2.16`, `§ 2.16`. Anything else is prose about a section rather than a
# citation of one, and guessing at those produces false positives.
CITATION = re.compile(r"§ ?(\d+\.\d+)")
HEADING = re.compile(r"^### (\d+\.\d+) ", re.MULTILINE)


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"],  # noqa: S607 - git from PATH
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / name for name in out.split()]


def main() -> int:
    headings = set(HEADING.findall(DECISIONS.read_text()))
    if not headings:
        print(f"{DECISIONS.name}: no '### N.M' headings found", file=sys.stderr)
        return 2

    dangling: list[str] = []
    for path in _tracked_files():
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; it carries no citations
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in CITATION.finditer(line):
                if match.group(1) not in headings:
                    rel = path.relative_to(ROOT)
                    dangling.append(f"{rel}:{lineno}: §{match.group(1)}")

    if dangling:
        print(
            f"{len(dangling)} citation(s) name a section "
            f"{DECISIONS.name} does not have:",
            file=sys.stderr,
        )
        for entry in dangling:
            print(f"  {entry}", file=sys.stderr)
        return 1

    print(f"All §-citations resolve ({len(headings)} sections).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
