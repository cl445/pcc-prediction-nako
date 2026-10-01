"""Check the numbers typed into the manuscript's tables against the artifacts.

The companion of ``check_paper_constants.py``. Most results tables in the
manuscript hold their numbers as literals rather than macros, so a rerun
moves them without any constants diff noticing. This script reads each
such table out of ``paper/sections/``, matches its rows to the artifact
they were transcribed from, and compares cell by cell at the precision the
table prints.

Per cell the status is the one ``check_paper_constants.py`` uses: ``ok``,
``CHANGED`` (the table still shows the value of the pre-rerun state) or
``MISMATCH`` (it shows neither). Per table it also reports

    MISSING    a row the artifact has and the table does not
    EXTRA      a row in the table with no counterpart in the artifact
    ORDER      the table ranks its rows by a value and the new values
               would rank them differently

Rows that quote macros are left to ``check_paper_constants.py``. Tables
built only from the processed parquets (baseline characteristics,
missingness, MRI selection, inclusion, the MRI positive control) do not
move with a pipeline rerun and are not covered.

Usage:
    uv run python scripts/check_paper_tables.py \\
        --before-runs-dir results/runs_2026-08-14 \\
        --before-constants-dir results/paper/constants_previous
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from check_paper_constants import CODE_ROOT, Artifacts
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_SECTIONS = CODE_ROOT.parent / "paper" / "sections"

MODALITY_LABELS = {
    "Mental Health": "mental_health",
    "Demographics": "demographics",
    "SES": "ses",
    "Medical History": "medical_history",
    "Laboratory": "lab_values",
    "Cardiovascular": "cardiovascular",
    "Lung Function": "lung_function",
    "Physical Activity": "physical_activity",
    "Cognitive": "cognitive",
    "Yeo Networks": "mri_yeo",
    "Destrieux": "mri_destrieux",
    "Subcortical": "mri_subcortical",
    "Julich": "mri_julich",
    "Desikan-Killiany": "mri_desikan",
    "Cerebellar": "mri_cerebellar",
}
LABEL_OF = {modality: label for label, modality in MODALITY_LABELS.items()}
PCT = 100.0


@dataclass
class Expected:
    """What an artifact says a table should hold.

    ``rows`` maps a row label, as the table prints it after normalisation,
    to its cell values in column order. ``None`` stands for a cell the
    table leaves empty. ``ranking`` lists, per group of rows the table
    sorts, the labels in the order the new values imply.
    """

    rows: dict[str, list[float | None]]
    ranking: list[list[str]] = field(default_factory=list)


def _ranked(
    rows: dict[str, list[float | None]], column: int, groups: list[list[str]]
) -> list[list[str]]:
    """Each group's labels, sorted descending by one column."""

    def key(label: str) -> float:
        value = rows[label][column]
        return float("-inf") if value is None else value

    return [sorted(group, key=key, reverse=True) for group in groups]


def _modality_groups(labels: list[str]) -> list[list[str]]:
    mri = [label for label in labels if MODALITY_LABELS[label].startswith("mri_")]
    return [[label for label in labels if label not in mri], mri]


def modality_contributions(run: str, with_p: bool) -> Callable[[Artifacts], Expected]:
    """Standalone ROC/PR mean and SD per modality, SHAP share, Holm p."""

    def build(a: Artifacts) -> Expected:
        scores = a.run_csv(run, "modality_scores_cv.csv").groupby("modality")
        mean = scores[["roc_auc", "pr_auc"]].mean().to_dict()
        sd = scores[["roc_auc", "pr_auc"]].std().to_dict()
        shares = a.shap_shares(run).to_dict()
        p_holm: dict[str, float] = {}
        if with_p:
            nb = a.run_csv(run, "modality_comparisons_nb.csv")
            p_holm = dict(zip(nb["modality"], nb["p_value_holm"], strict=True))
        rows: dict[str, list[float | None]] = {}
        for modality in mean["roc_auc"]:
            row: list[float | None] = [
                float(mean["roc_auc"][modality]),
                float(sd["roc_auc"][modality]),
                float(mean["pr_auc"][modality]),
                float(sd["pr_auc"][modality]),
                float(shares[modality]),
            ]
            if with_p:
                row.append(p_holm.get(modality))
            rows[LABEL_OF[str(modality)]] = row
        return Expected(rows, _ranked(rows, 0, _modality_groups(list(rows))))

    return build


def clinical_utility(a: Artifacts) -> Expected:
    """NPV and F1 at the two operating points; the rest are macros."""
    return Expected(
        {
            "NPV (%)": [
                PCT * a.operating_point("primary", "f1", "npv"),
                PCT * a.operating_point("primary", "spec90", "npv"),
            ],
            "F1 score": [
                a.operating_point("primary", "f1", "f1"),
                a.operating_point("primary", "spec90", "f1"),
            ],
        }
    )


def fold_performance(a: Artifacts) -> Expected:
    folds = a.run_csv("primary", "cv_fold_results.csv")
    rows: dict[str, list[float | None]] = {
        str(int(fold)): [float(pr), float(roc)]
        for fold, pr, roc in zip(
            folds["fold"], folds["pr_auc"], folds["roc_auc"], strict=True
        )
    }
    rows["Mean (SD)"] = [
        float(folds["pr_auc"].mean()),
        float(folds["pr_auc"].std()),
        float(folds["roc_auc"].mean()),
        float(folds["roc_auc"].std()),
    ]
    return Expected(rows)


def modality_ablation(a: Artifacts) -> Expected:
    frame = a.run_csv("primary", "modality_ablation_oos.csv")
    rows: dict[str, list[float | None]] = {
        LABEL_OF[m]: [float(roc), float(pr)]
        for m, roc, pr in zip(
            frame["modality"],
            frame["delta_roc_auc"],
            frame["delta_pr_auc"],
            strict=True,
        )
    }
    return Expected(rows, _ranked(rows, 1, [list(rows)]))


def incremental_performance(a: Artifacts) -> Expected:
    frame = a.run_csv("primary", "incremental_performance.csv")
    rows: dict[str, list[float | None]] = {
        LABEL_OF[m]: [float(augmented), float(delta)]
        for m, augmented, delta in zip(
            frame["modality"],
            frame["pr_auc_augmented"],
            frame["delta_pr_auc"],
            strict=True,
        )
    }
    return Expected(rows, _ranked(rows, 1, [list(rows)]))


M3_SHAP_ROWS = {
    "Mental Health": "mental_health",
    "Demographics": "demographics",
    "SES": "ses",
    "Laboratory": "lab_values",
    "Cardiovascular": "cardiovascular",
    "Medical History": "medical_history",
    "MRI (Desikan-Killiany)": "mri_desikan",
    "Cognitive": "cognitive",
    "MRI (Cerebellar)": "mri_cerebellar",
}


def sensitivity_m3(a: Artifacts) -> Expected:
    """Primary against no_dml: overall metrics, SHAP shares, MRI standalone."""
    runs = ("primary", "no_dml")
    rows: dict[str, list[float | None]] = {
        "ROC-AUC (95 % CI)": [a.metric(r, "roc_auc") for r in runs],
        "PR-AUC (95 % CI)": [a.metric(r, "pr_auc") for r in runs],
        "Calibration slope": [a.metric(r, "calibration_slope") for r in runs],
        "ECE": [a.metric(r, "ece") for r in runs],
    }
    for label, modality in M3_SHAP_ROWS.items():
        rows[label] = [a.shap_share(r, modality) for r in runs]
    for label, modality in MODALITY_LABELS.items():
        if modality.startswith("mri_"):
            rows[label] = [
                a.modality_score(r, modality, "roc_auc", "mean") for r in runs
            ]
    return Expected(rows)


TABLES: dict[str, Callable[[Artifacts], Expected]] = {
    "tab:modality_contributions": modality_contributions("primary", with_p=True),
    "tab:modality_contributions_neurocog": modality_contributions(
        "neurocog", with_p=False
    ),
    "tab:clinical_utility": clinical_utility,
    "tab:fold_performance": fold_performance,
    "tab:modality_ablation": modality_ablation,
    "tab:incremental_performance": incremental_performance,
    "tab:sensitivity_m3": sensitivity_m3,
}


# ---------------------------------------------------------------------------
# Reading the manuscript
# ---------------------------------------------------------------------------


def _normalise_label(cell: str) -> str:
    text = re.sub(r"\\quad|\\enspace|\\textit\{|\\textbf\{", "", cell)
    text = re.sub(r"\\acs?p?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\qty\{([^}]*)\}\{\\percent\}", r"\1 %", text)
    text = re.sub(r"\\textsuperscript\{[^}]*\}|\\tnote\{[^}]*\}", "", text)
    text = text.replace("\\,", " ").replace("\\%", "%").replace("$", "")
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


_TOKEN = re.compile(r"<?-?\d*\.\d+|<?-?\d+")


def _cell_tokens(cell: str) -> list[str]:
    """The numbers one cell prints, as strings, with ``<`` for bounds."""
    text = cell.replace("$<$", "<").replace("\\num{", "").replace("\\;", " ")
    text = re.sub(r"[{}$]", "", text)
    return _TOKEN.findall(text)


@dataclass
class PaperRow:
    label: str
    tokens: list[str]


_INPUT = re.compile(r"^\s*\\input\{([^}]+)\}")


def _expand_inputs(lines: list[str], root: Path) -> list[str]:
    """Replace each ``\\input{path}`` line by the lines of that file.

    Some table bodies live in ``paper/tables/`` and are shared with a second
    manuscript; the section then only holds the ``\\input``.
    """
    out: list[str] = []
    for line in lines:
        match = _INPUT.match(line)
        target = root / match.group(1) if match else None
        if target is not None and target.suffix != ".tex":
            target = target.with_name(target.name + ".tex")
        if target is not None and target.suffix == ".tex" and target.exists():
            out.extend(target.read_text(encoding="utf-8").splitlines())
        else:
            out.append(line)
    return out


def read_table(sections: Path, label: str) -> tuple[Path, list[PaperRow]]:
    """Rows of the tabular under ``\\label{label}``, macros rows left out."""
    for path in sorted(sections.glob("*.tex")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if f"\\label{{{label}}}" not in line:
                continue
            body = []
            for text in _expand_inputs(lines[index + 1 :], sections.parent):
                if "\\end{tabular}" in text:
                    break
                body.append(text.strip())
            return path, _parse_rows(body)
    raise KeyError(f"no \\label{{{label}}} under {sections}")


def _parse_rows(lines: list[str]) -> list[PaperRow]:
    rows = []
    for line in lines:
        if "&" not in line or line.startswith("%"):
            continue
        if "\\res" in line or "\\textbf" in line:
            continue
        cells = line.removesuffix("\\\\").split("&")
        tokens = [t for cell in cells[1:] for t in _cell_tokens(cell)]
        if tokens:
            rows.append(PaperRow(_normalise_label(cells[0]), tokens))
    return rows


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _matches(token: str, value: float | None) -> bool:
    if value is None:
        return False
    if token.startswith("<"):
        return value < float(token[1:])
    decimals = len(token.split(".", 1)[1]) if "." in token else 0
    return float(f"{value:.{decimals}f}") == float(token)


def _format(token: str, value: float | None) -> str:
    if value is None:
        return "—"
    if token.startswith("<"):
        return f"{value:.3g}"
    decimals = len(token.split(".", 1)[1]) if "." in token else 0
    text = f"{value:.{decimals}f}"
    return text.lstrip("0") if token.startswith(".") else text


@dataclass
class Finding:
    table: str
    row: str
    column: int | None
    paper: str
    new: str
    old: str
    status: str


def compare_table(
    label: str,
    paper: list[PaperRow],
    after: Expected,
    before: Expected | None,
) -> list[Finding]:
    findings: list[Finding] = []
    # A table may print "Mental health" where the artifact says "Mental
    # Health"; take the artifact's spelling so the rows still line up.
    spelling = {name.casefold(): name for name in after.rows}
    paper = [
        PaperRow(spelling.get(row.label.casefold(), row.label), row.tokens)
        for row in paper
    ]
    for row in paper:
        if row.label not in after.rows:
            findings.append(Finding(label, row.label, None, "", "", "", "EXTRA"))
            continue
        new_cells = list(after.rows[row.label])
        old_cells = before.rows.get(row.label) if before else None
        # Cells the table leaves empty ("--") carry no token; drop the
        # matching None so the positions line up.
        if len(new_cells) != len(row.tokens):
            new_cells = [v for v in new_cells if v is not None]
            if old_cells is not None:
                old_cells = [v for v in old_cells if v is not None]
        for column, token in enumerate(row.tokens):
            new = new_cells[column] if column < len(new_cells) else None
            old = (
                old_cells[column]
                if old_cells is not None and column < len(old_cells)
                else None
            )
            if _matches(token, new):
                status = "ok"
            elif _matches(token, old):
                status = "CHANGED"
            else:
                status = "MISMATCH"
            findings.append(
                Finding(
                    label,
                    row.label,
                    column + 1,
                    token,
                    _format(token, new),
                    _format(token, old) if old_cells is not None else "",
                    status,
                )
            )
    printed = {row.label for row in paper}
    findings.extend(
        Finding(label, missing, None, "", "", "", "MISSING")
        for missing in after.rows
        if missing not in printed
    )
    order = [row.label for row in paper]
    for group in after.ranking:
        shown = [label_ for label_ in order if label_ in group]
        wanted = [label_ for label_ in group if label_ in shown]
        if shown != wanted:
            findings.append(
                Finding(label, " > ".join(wanted), None, "", "", "", "ORDER")
            )
    return findings


STYLE = {
    "ok": "green",
    "CHANGED": "yellow",
    "MISMATCH": "red",
    "MISSING": "red",
    "EXTRA": "red",
    "ORDER": "yellow",
    "ERROR": "red",
}


def render(findings: list[Finding], show_all: bool, sources: dict[str, Path]) -> None:
    for label in TABLES:
        rows = [f for f in findings if f.table == label]
        shown = [f for f in rows if show_all or f.status != "ok"]
        counts: dict[str, int] = {}
        for finding in rows:
            counts[finding.status] = counts.get(finding.status, 0) + 1
        summary = ", ".join(f"{counts[k]} {k}" for k in STYLE if k in counts)
        source = sources.get(label)
        where = f" ({source.name})" if source else ""
        if not shown:
            console.print(f"[green]{label}{where}: {summary}")
            continue
        table = Table(title=f"{label}{where}: {summary}", title_justify="left")
        table.add_column("Row", overflow="fold")
        table.add_column("Col", justify="right")
        table.add_column("Paper", justify="right")
        table.add_column("New", justify="right")
        table.add_column("Old run", justify="right")
        table.add_column("Status")
        for f in shown:
            table.add_row(
                f.row,
                "" if f.column is None else str(f.column),
                f.paper,
                f.new,
                f.old,
                f"[{STYLE[f.status]}]{f.status}",
            )
        console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--runs-dir", type=Path, default=CODE_ROOT / "results" / "runs")
    parser.add_argument(
        "--constants-dir",
        type=Path,
        default=CODE_ROOT / "results" / "paper" / "constants",
    )
    parser.add_argument("--before-runs-dir", type=Path)
    parser.add_argument("--before-constants-dir", type=Path)
    parser.add_argument("--all", action="store_true", help="Also list matching cells.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    after = Artifacts(args.runs_dir, args.constants_dir, {})
    before = None
    if args.before_runs_dir or args.before_constants_dir:
        if not (args.before_runs_dir and args.before_constants_dir):
            console.print(
                "[red]--before-runs-dir and --before-constants-dir go together."
            )
            sys.exit(2)
        before = Artifacts(args.before_runs_dir, args.before_constants_dir, {})

    findings: list[Finding] = []
    sources: dict[str, Path] = {}
    for label, build in TABLES.items():
        try:
            path, paper = read_table(args.sections, label)
            sources[label] = path
            findings += compare_table(
                label, paper, build(after), build(before) if before else None
            )
        except (KeyError, FileNotFoundError) as exc:
            findings.append(Finding(label, str(exc), None, "", "", "", "ERROR"))

    render(findings, args.all, sources)
    open_items = [f for f in findings if f.status != "ok"]
    console.print(f"\n[bold]{len(findings)} findings, {len(open_items)} open.[/bold]")
    sys.exit(1 if open_items else 0)


if __name__ == "__main__":
    main()
