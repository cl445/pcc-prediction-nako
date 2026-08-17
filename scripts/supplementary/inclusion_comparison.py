"""Who the analytic sample lost, and whether it matters.

Fifty-six per cent of the neuroimaging subsample does not reach the analytic
sample: most of them never reported an infection, the rest have no observable
Corona-2 symptom outcome or fall in the sub-threshold band the clean-controls
design excludes. A reader cannot tell from the participant flow alone whether
that attrition is benign, and a reviewer asked for the comparison directly.

The question this answers is narrow and worth stating precisely: it is not
whether the imaged differ from the unimaged (that is the MRI participation
gradient, reported separately), but whether, *within* the neuroimaging
subsample, those who enter the analysis differ from those who do not.

Output
------
- ``results/paper/constants/inclusion_comparison.json`` — per-variable means,
  standardised mean differences and two-sample tests
- ``results/paper/constants/inclusion_comparison_tex.tex`` — the LaTeX rows
  the supplement transcribes

Usage
-----
    uv run python scripts/supplementary/inclusion_comparison.py
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from scipy import stats

from pcc_analysis.config import get_paper_constants_dir, get_processed_dir
from pcc_analysis.data_manager import NakoDataManager, _mri_participant_ids

console = Console()
logger = logging.getLogger(__name__)

# Baseline variables only. Anything measured at Corona-2 would be unavailable
# for most of the excluded group by construction — that is what excludes them.
CONTINUOUS: dict[str, tuple[str, str]] = {
    "Age, years": ("basis_age", "demographics.parquet"),
    "Education (ISCED)": ("education_isced_level", "socioeconomic_status.parquet"),
    "PHQ-9 baseline (0--27)": ("phq9_sum", "mental_health.parquet"),
    "GAD-7 baseline (0--21)": ("gad7_sum", "mental_health.parquet"),
}


def _analytic_mask(frame: pd.DataFrame) -> pd.Series:
    """The clean-controls filter, spelled out rather than imported.

    Kept explicit here because the comparison is *about* this filter; a reader
    checking the table should be able to see which condition removes whom.
    """
    infected = frame["had_covid"] == 1
    observable = frame["valid_symptoms"] == 1
    sub_threshold = (frame["d_co2_k0"] == 1) & (frame["bahmer_any_pcs"] == 0)
    return infected & observable & ~sub_threshold


def _smd_continuous(a: pd.Series, b: pd.Series) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0


def _smd_binary(p_a: float, p_b: float) -> float:
    pooled = np.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / 2)
    return float((p_a - p_b) / pooled) if pooled else 0.0


def build_comparison() -> dict[str, Any]:
    processed = get_processed_dir()
    mgr = NakoDataManager()
    mri_ids = _mri_participant_ids(mgr)

    frame = pd.read_parquet(processed / "corona2_pcc.parquet")
    frame = frame[frame["ID"].isin(mri_ids)].copy()

    for column, source in CONTINUOUS.values():
        if column in frame.columns:
            continue
        extra = pd.read_parquet(processed / source)
        if column in extra.columns:
            frame = frame.merge(extra[["ID", column]], on="ID", how="left")
    # basis_age already arrived through the loop above; merging it a second
    # time would suffix both copies and lose the column.
    sex = pd.read_parquet(
        processed / "demographics.parquet", columns=["ID", "basis_sex"]
    )
    frame = frame.merge(sex, on="ID", how="left")

    included = _analytic_mask(frame)
    rows: dict[str, Any] = {}

    for label, (column, _) in CONTINUOUS.items():
        if column not in frame.columns:
            logger.warning("column %s absent; row skipped", column)
            continue
        a = frame.loc[included, column].dropna()
        b = frame.loc[~included, column].dropna()
        _, p_value = stats.ttest_ind(a, b, equal_var=False)
        rows[label] = {
            "type": "continuous",
            "included": {
                "n": len(a),
                "mean": float(a.mean()),
                "sd": float(a.std(ddof=1)),
            },
            "excluded": {
                "n": len(b),
                "mean": float(b.mean()),
                "sd": float(b.std(ddof=1)),
            },
            "smd": _smd_continuous(a, b),
            "p_value": float(p_value),
        }

    female = frame["basis_sex"] == 2
    p_in, p_ex = float(female[included].mean()), float(female[~included].mean())
    table = pd.crosstab(included, female)
    chi2 = stats.chi2_contingency(table)
    rows["Female sex"] = {
        "type": "binary",
        "included": {
            "n": int(included.sum()),
            "n_positive": int(female[included].sum()),
            "pct": 100 * p_in,
        },
        "excluded": {
            "n": int((~included).sum()),
            "n_positive": int(female[~included].sum()),
            "pct": 100 * p_ex,
        },
        "smd": _smd_binary(p_in, p_ex),
        "p_value": float(chi2.pvalue),
    }

    return {
        "question": (
            "Within the neuroimaging subsample, do participants entering the "
            "clean-controls analytic sample differ at baseline from those who "
            "do not? Distinct from the MRI participation gradient, which "
            "compares imaged against unimaged participants."
        ),
        "exclusion_reasons": {
            "no_reported_infection": int((frame["had_covid"] != 1).sum()),
            "outcome_not_observable": int(
                ((frame["had_covid"] == 1) & (frame["valid_symptoms"] != 1)).sum()
            ),
            "sub_threshold_symptomatic": int(
                (
                    (frame["had_covid"] == 1)
                    & (frame["valid_symptoms"] == 1)
                    & (frame["d_co2_k0"] == 1)
                    & (frame["bahmer_any_pcs"] == 0)
                ).sum()
            ),
        },
        "n_mri_subsample": len(frame),
        "n_included": int(included.sum()),
        "n_excluded": int((~included).sum()),
        "variables": rows,
    }


def _tex(summary: dict[str, Any]) -> str:
    def fmt_p(value: float) -> str:
        return "{<}.001" if value < 0.001 else f"{value:.3f}".replace("0.", ".")

    lines = [
        "% Generated by scripts/supplementary/inclusion_comparison.py "
        "-- do not edit by hand.",
        "% Included vs. excluded within the neuroimaging subsample.",
        "",
        f"\\newcommand{{\\resInclN}}{{{summary['n_included']}}}",
        f"\\newcommand{{\\resExclN}}{{{summary['n_excluded']}}}",
        f"\\newcommand{{\\resExclPct}}{{{100 * summary['n_excluded'] / summary['n_mri_subsample']:.1f}}}",
        f"\\newcommand{{\\resExclNoInfection}}{{{summary['exclusion_reasons']['no_reported_infection']}}}",
        f"\\newcommand{{\\resExclNoOutcome}}{{{summary['exclusion_reasons']['outcome_not_observable']}}}",
        f"\\newcommand{{\\resExclSubThreshold}}{{{summary['exclusion_reasons']['sub_threshold_symptomatic']}}}",
        "",
        "% Table rows: variable & included & excluded & SMD & p",
    ]
    for label, row in summary["variables"].items():
        if row["type"] == "continuous":
            inc = (
                f"\\num{{{row['included']['mean']:.1f}}} ({row['included']['sd']:.1f})"
            )
            exc = (
                f"\\num{{{row['excluded']['mean']:.1f}}} ({row['excluded']['sd']:.1f})"
            )
        else:
            inc = f"\\num{{{row['included']['n_positive']}}} (\\SI{{{row['included']['pct']:.1f}}}{{\\percent}})"
            exc = f"\\num{{{row['excluded']['n_positive']}}} (\\SI{{{row['excluded']['pct']:.1f}}}{{\\percent}})"
        lines.append(
            f"% {label:24s} & {inc} & {exc} & "
            f"\\num{{{row['smd']:+.2f}}} & {fmt_p(row['p_value'])} \\\\"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]Included vs. excluded within the neuroimaging subsample")

    summary = build_comparison()
    out_dir = get_paper_constants_dir()
    (out_dir / "inclusion_comparison.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "inclusion_comparison_tex.tex").write_text(_tex(summary))

    table = Table(title="Included vs. excluded", show_header=True)
    table.add_column("Variable", style="bold")
    table.add_column("Included", justify="right")
    table.add_column("Excluded", justify="right")
    table.add_column("SMD", justify="right")
    for label, row in summary["variables"].items():
        if row["type"] == "continuous":
            inc = f"{row['included']['mean']:.1f} ({row['included']['sd']:.1f})"
            exc = f"{row['excluded']['mean']:.1f} ({row['excluded']['sd']:.1f})"
        else:
            inc = f"{row['included']['pct']:.1f}%"
            exc = f"{row['excluded']['pct']:.1f}%"
        table.add_row(label, inc, exc, f"{row['smd']:+.2f}")
    console.print(table)
    console.print(
        f"[green]{summary['n_excluded']} of {summary['n_mri_subsample']} excluded "
        f"({100 * summary['n_excluded'] / summary['n_mri_subsample']:.1f} %)"
    )


if __name__ == "__main__":
    main()
