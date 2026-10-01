"""MRI block ablation: paper constants from the primary run's block ablation.

The single-modality ablation masks one atlas at a time. With six correlated
atlases that is the weakest available test of the MRI null, because the other
five keep carrying the same information. The pipeline therefore also removes
all six at once inside every outer fold, in two variants: masked with NaN
under the fold's own meta-learner, and with the meta-learner retrained on the
nine remaining columns (``_block_ablation_oos`` in ``orchestration.py``,
DECISIONS §2.32). The retrained variant is the pre-specified test; the masked
variant is reported next to it as the contrast that shows what masking alone
would have missed.

The verdict is an equivalence test against a margin fixed before the run:
the 95 % paired-bootstrap interval of the held-out delta (full model minus
model without the block) has to lie entirely within ±margin, on ROC-AUC and
on PR-AUC. Both bounds, not only the upper one, so that a model that is
*better* without MRI is also read as "no contribution" rather than as a
pass by default. The margin is the constant below and has no command-line
override: a result well inside it is no reason to tighten it afterwards,
and a result outside it is a finding, not a defect.

Every number this script emits is generated into ``results_constants.tex``
rather than transcribed by hand, so a rerun moves them through the
constants diff and nothing stale compiles quietly.

Outputs:
    results/paper/constants/block_ablation.json
    results/paper/constants/block_ablation_tex.tex

Usage:
    uv run python scripts/supplementary/block_ablation.py \
        --run-dir results/runs/primary
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pcc_analysis.config import get_paper_constants_dir, get_results_dir

console = Console()
logger = logging.getLogger(__name__)

BLOCK = "mri"
"""The block the manuscript asks about; ``block_ablation_oos.csv`` may carry
others, which this script ignores."""

MARGIN = 0.03
"""Pre-specified equivalence margin on ROC-AUC and PR-AUC alike.

Fixed on 2026-09-15, before the run that produces the numbers. The ROC-AUC
value is the manuscript's existing transportability margin
(``\\resTostRocMargin``); the PR-AUC margin is set to the same value because
the manuscript reads PR-AUC as its primary metric and has no margin for it
elsewhere.
"""

VARIANTS: tuple[str, ...] = ("retrained", "masked")
"""Both must be present; ``retrained`` is the test, ``masked`` the contrast."""

TEX_PREFIX = "MriBlock"
"""Macro infix: ``\\resMriBlockDeltaRoc`` and so on. The masked variant gets
``\\resMriBlockMasked...``; the retrained variant, being the test, carries
the unqualified names."""


def _verdict(row: dict[str, Any], margin: float) -> str:
    """Pass when both intervals sit entirely inside ±margin."""
    inside = (
        -margin < float(row["delta_roc_ci_lo"])
        and float(row["delta_roc_ci_hi"]) < margin
        and -margin < float(row["delta_pr_ci_lo"])
        and float(row["delta_pr_ci_hi"]) < margin
    )
    return "Pass" if inside else "Fail"


def build_constants(run_dir: Path, block: str = BLOCK) -> dict[str, Any]:
    csv_path = run_dir / "block_ablation_oos.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. The block ablation is written by a full "
            "(non-distributed) pipeline run; a merged fold run has none."
        )
    table = pd.read_csv(csv_path)
    rows = table[table["block"] == block]
    present = set(rows["variant"])
    if present != set(VARIANTS):
        raise ValueError(
            f"{csv_path} carries variants {sorted(present)} for block "
            f"'{block}', expected {sorted(VARIANTS)}"
        )
    by_variant: dict[str, dict[str, Any]] = {
        v: {str(k): x for k, x in rows[rows["variant"] == v].iloc[0].items()}
        for v in VARIANTS
    }
    n_removed = {int(r["n_removed"]) for r in by_variant.values()}
    if len(n_removed) != 1:
        raise ValueError(f"variants disagree on n_removed: {sorted(n_removed)}")

    variants = {
        name: {
            "roc_variant": float(row["roc_variant"]),
            "delta_roc": float(row["delta_roc"]),
            "delta_roc_ci": [
                float(row["delta_roc_ci_lo"]),
                float(row["delta_roc_ci_hi"]),
            ],
            "pr_variant": float(row["pr_variant"]),
            "delta_pr": float(row["delta_pr"]),
            "delta_pr_ci": [float(row["delta_pr_ci_lo"]), float(row["delta_pr_ci_hi"])],
            "nb_t_roc": float(row["nb_t_roc"]),
            "nb_p_roc": float(row["nb_p_roc"]),
            "nb_t_pr": float(row["nb_t_pr"]),
            "nb_p_pr": float(row["nb_p_pr"]),
            "verdict": _verdict(row, MARGIN),
        }
        for name, row in by_variant.items()
    }
    return {
        "run_dir": str(run_dir),
        "block": block,
        "n_removed": n_removed.pop(),
        "margin": MARGIN,
        "criterion": (
            "95 % paired-bootstrap CI of the held-out delta (full minus "
            "variant) entirely within +/- margin, on ROC-AUC and PR-AUC"
        ),
        "roc_full": float(by_variant["retrained"]["roc_full"]),
        "pr_full": float(by_variant["retrained"]["pr_full"]),
        "variants": variants,
        "verdict": variants["retrained"]["verdict"],
    }


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _tex_lines(constants: dict[str, Any]) -> list[str]:
    lines = [
        "% Generated by scripts/supplementary/block_ablation.py "
        "-- do not edit by hand.",
        f"% Source run: {constants['run_dir']}",
        "",
        f"\\newcommand{{\\res{TEX_PREFIX}NRemoved}}{{{constants['n_removed']}}}",
        f"\\newcommand{{\\res{TEX_PREFIX}Margin}}{{{constants['margin']:.2f}}}",
        f"\\newcommand{{\\res{TEX_PREFIX}Verdict}}{{{constants['verdict']}}}",
        "",
    ]
    for name, entry in constants["variants"].items():
        infix = TEX_PREFIX if name == "retrained" else f"{TEX_PREFIX}Masked"
        lines.extend(
            [
                f"\\newcommand{{\\res{infix}Roc}}{{{entry['roc_variant']:.3f}}}",
                f"\\newcommand{{\\res{infix}Pr}}{{{entry['pr_variant']:.3f}}}",
                f"\\newcommand{{\\res{infix}DeltaRoc}}{{{entry['delta_roc']:.4f}}}",
                f"\\newcommand{{\\res{infix}DeltaRoclo}}"
                f"{{{entry['delta_roc_ci'][0]:.4f}}}",
                f"\\newcommand{{\\res{infix}DeltaRochi}}"
                f"{{{entry['delta_roc_ci'][1]:.4f}}}",
                f"\\newcommand{{\\res{infix}DeltaPr}}{{{entry['delta_pr']:.4f}}}",
                f"\\newcommand{{\\res{infix}DeltaPrlo}}"
                f"{{{entry['delta_pr_ci'][0]:.4f}}}",
                f"\\newcommand{{\\res{infix}DeltaPrhi}}"
                f"{{{entry['delta_pr_ci'][1]:.4f}}}",
                f"\\newcommand{{\\res{infix}NbPRoc}}{{{_p(entry['nb_p_roc'])}}}",
                f"\\newcommand{{\\res{infix}NbPPr}}{{{_p(entry['nb_p_pr'])}}}",
                f"\\newcommand{{\\res{infix}VariantVerdict}}{{{entry['verdict']}}}",
                "",
            ]
        )
    return lines


def _render(constants: dict[str, Any]) -> None:
    console.print(
        f"\n[bold]Block '{constants['block']}'[/bold]  "
        f"{constants['n_removed']} modalities removed, margin "
        f"±{constants['margin']:.2f}, verdict (retrained): "
        f"[bold]{constants['verdict']}[/bold]"
    )
    table = Table(title="Held-out delta, full model minus model without the block")
    table.add_column("Variant")
    table.add_column("ΔROC-AUC [95 % CI]", justify="right")
    table.add_column("ΔPR-AUC [95 % CI]", justify="right")
    table.add_column("NB p (ROC)", justify="right")
    table.add_column("NB p (PR)", justify="right")
    table.add_column("Verdict", justify="right")
    for name, entry in constants["variants"].items():
        table.add_row(
            name,
            f"{entry['delta_roc']:.4f} [{entry['delta_roc_ci'][0]:.4f}, "
            f"{entry['delta_roc_ci'][1]:.4f}]",
            f"{entry['delta_pr']:.4f} [{entry['delta_pr_ci'][0]:.4f}, "
            f"{entry['delta_pr_ci'][1]:.4f}]",
            _p(entry["nb_p_roc"]),
            _p(entry["nb_p_pr"]),
            entry["verdict"],
        )
    console.print(table)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console)],
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=get_results_dir() / "runs" / "primary",
        help="Pipeline run directory carrying block_ablation_oos.csv "
        "(default: the primary run under the configured results directory)",
    )
    args = parser.parse_args()

    constants = build_constants(args.run_dir)
    _render(constants)

    constants_dir = get_paper_constants_dir()
    json_path = constants_dir / "block_ablation.json"
    json_path.write_text(json.dumps(constants, indent=2))
    tex_path = constants_dir / "block_ablation_tex.tex"
    tex_path.write_text("\n".join(_tex_lines(constants)) + "\n")
    console.print(f"\nWrote {json_path}\n      {tex_path}")


if __name__ == "__main__":
    main()
