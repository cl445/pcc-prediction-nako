"""Clinical-utility panel for the Lean Universal Classifier.

Loads the held-out predictions of the two Lean-Stack runs (MRI and
non-MRI, fitted on each cohort separately under cohort='mri'/'non_mri'
+ stack='lean') and produces:

- Calibration: slope, intercept-in-the-large, ECE.
- Threshold operating points: sensitivity / specificity at the
  F1-optimal cut, and PPV / NPV at the high-specificity cut
  (closest to spec >= 0.9, matching the convention used for the
  Full-Stack model in ``evaluation.py``).
- Net benefit: extracted at the F1-optimal threshold (lo) and at
  the high-specificity threshold (hi) so the same operating points
  drive both the threshold-based and decision-curve narratives.
- Combined DCA panel: MRI curve overlaid with non-MRI curve
  (one figure, ``lean_dca_mri_vs_non_mri.pdf``) for visual side-
  by-side inspection.

Outputs:

- ``results/lean_clinical_utility.json`` — full numeric panel
  (Methods/Results-ready constants).
- ``results/lean_dca_mri_vs_non_mri.pdf`` (and .png companion).
- ``results/lean_clinical_utility_tex.tex`` — \\newcommand snippet
  that can be pasted into ``paper/results_constants.tex``.

Usage:
    uv run python scripts/supplementary/lean_clinical_utility.py
    uv run python scripts/supplementary/lean_clinical_utility.py \\
        --mri-run results/runs/lean_mri \\
        --non-mri-run results/runs/lean_non_mri
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pcc_analysis.config import get_paper_constants_dir, get_results_dir
from pcc_analysis.evaluation import (
    EvaluationMetrics,
    compute_decision_curve_analysis,
    plot_decision_curve,
    save_figure_for_latex,
    summarize_thresholds,
)
from pcc_analysis.run_comparability import assert_comparable_runs

console = Console()
logger = logging.getLogger(__name__)

DEFAULT_MRI_RUN = "lean_mri"
DEFAULT_NON_MRI_RUN = "lean_non_mri"


def _runs_dir() -> Path:
    """Where the pipeline writes its run directories.

    From the config rather than a literal ``results``, so a run pointed at
    ``smoke_results/`` by ``PCC_CONFIG`` reads the runs it just wrote instead
    of the real ones next door. A function and not a module constant: reading
    the config at import time makes ``--help`` need one.
    """
    return get_results_dir() / "runs"


def _relative_run_dir(run_dir: Path) -> str:
    """The run directory as written, relative to the project root if possible."""
    root = Path(__file__).resolve().parents[2]
    resolved = run_dir.resolve()
    if resolved.is_relative_to(root):
        return str(resolved.relative_to(root))
    return str(run_dir)


def _named_run(name: str) -> Path:
    """Return the run directory of the given configuration.

    Named directories under ``results/runs/`` rather than a timestamp glob
    over ``results/``: the timestamped directories of superseded runs are
    still on disk, and a most-recently-modified glob will happily pick one of
    them and emit a panel of numbers from the wrong analysis without saying so.
    """
    run_dir = _runs_dir() / name
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"No Lean-Stack run at {run_dir}. "
            "Run scripts/pipeline/02_run_pipeline.py with --cohort=<mri|non_mri> "
            "--stack=lean first."
        )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lean-Stack clinical-utility panel (MRI + non-MRI)",
    )
    parser.add_argument(
        "--mri-run",
        type=Path,
        default=None,
        help=(
            "Lean-Stack MRI run directory (containing predictions.csv). "
            f"Defaults to <results-dir>/runs/{DEFAULT_MRI_RUN}."
        ),
    )
    parser.add_argument(
        "--non-mri-run",
        type=Path,
        default=None,
        help=(
            "Lean-Stack non-MRI run directory. "
            f"Defaults to <results-dir>/runs/{DEFAULT_NON_MRI_RUN}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        # Resolved in main(); see the note in transport_panel.py — evaluating
        # it here breaks `--help` without a config and creates a directory as
        # a side effect of asking for it.
        default=None,
        help="Where to write the combined JSON, DCA PDF and TeX snippet.",
    )
    return parser.parse_args()


def _load_predictions(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.csv missing in {run_dir}")
    df = pd.read_csv(pred_path)
    return df["y_true"].to_numpy(), df["y_pred_proba"].to_numpy()


def _row_value(df: pd.DataFrame, threshold: float, column: str) -> float:
    """Look up ``column`` in the row whose ``threshold`` is closest to
    the requested value.  Returns a Python float regardless of the
    underlying numpy dtype.
    """
    idx = int((df["threshold"] - threshold).abs().argsort().iloc[0])
    return float(df[column].iloc[idx])


def evaluate_cohort(
    run_dir: Path, cohort_label: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute the calibration, threshold and DCA metrics for one cohort.

    Returns the structured summary plus the DCA DataFrame so the caller
    can plot a combined figure without recomputing the curve.
    """
    y_true, y_pred = _load_predictions(run_dir)

    evaluator = EvaluationMetrics(y_true, y_pred)
    metrics = evaluator.compute_all_metrics()

    threshold_summary, threshold_df = summarize_thresholds(y_true, y_pred)
    f1_thr = threshold_summary["f1_optimal_threshold"]
    spec_thr = threshold_summary["spec90_threshold"]

    dca_df = compute_decision_curve_analysis(y_true, y_pred)

    return (
        {
            "cohort": cohort_label,
            # Relative to the project root where possible: this string is
            # transcribed into downstream documents, and an absolute path
            # records whose machine produced the panel rather than which run.
            "run_dir": _relative_run_dir(run_dir),
            "n_total": len(y_true),
            "n_positive": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
            "calibration": {
                "slope": float(metrics["calibration_slope"]),
                "intercept": float(metrics["calibration_intercept"]),
                "ece": float(metrics["ece"]),
                "brier_score": float(metrics["brier_score"]),
            },
            "thresholds": {
                "f1_optimal": {
                    "threshold": float(f1_thr),
                    "f1": float(threshold_summary["f1_optimal_f1"]),
                    "sensitivity": _row_value(threshold_df, f1_thr, "recall"),
                    "specificity": _row_value(threshold_df, f1_thr, "specificity"),
                    "ppv": _row_value(threshold_df, f1_thr, "precision"),
                    "npv": _row_value(threshold_df, f1_thr, "npv"),
                    "net_benefit": _row_value(dca_df, f1_thr, "net_benefit"),
                },
                "high_specificity": {
                    "threshold": float(spec_thr),
                    "sensitivity": _row_value(threshold_df, spec_thr, "recall"),
                    "specificity": _row_value(threshold_df, spec_thr, "specificity"),
                    "ppv": float(threshold_summary["spec90_ppv"]),
                    "npv": float(threshold_summary["spec90_npv"]),
                    "net_benefit": _row_value(dca_df, spec_thr, "net_benefit"),
                },
            },
            "auc": {
                "roc": float(metrics["roc_auc"]),
                "pr": float(metrics["pr_auc"]),
            },
        },
        dca_df,
    )


def plot_combined_dca(
    mri_dca: pd.DataFrame,
    non_mri_dca: pd.DataFrame,
    output_path: Path,
) -> None:
    """Overlay MRI and non-MRI decision curves on a single panel."""
    fig, ax = plt.subplots(figsize=(6.30, 3.2))
    plot_decision_curve(mri_dca, ax=ax, model_label="Lean (MRI)")
    # Replace treat-all/treat-none lines with MRI's already on the axis;
    # Add the non-MRI model line only — keep the reference curves single.
    ax.plot(
        non_mri_dca["threshold"],
        non_mri_dca["net_benefit"],
        label="Lean (non-MRI)",
        linewidth=2,
        linestyle="-",
    )
    ax.set_title("Decision Curve Analysis — Lean Universal Classifier")
    ax.legend(loc="upper right")
    save_figure_for_latex(fig, output_path, formats=["pdf", "png"])
    plt.close(fig)


def _fmt_num(value: float, digits: int = 3) -> str:
    """Format a float with ``digits`` decimals; NaNs become 'NaN'."""
    if math.isnan(value):
        return "NaN"
    return f"{value:.{digits}f}"


def emit_tex_snippet(summary: dict[str, dict[str, Any]], output_path: Path) -> None:
    """Write \\newcommand definitions for paper/results_constants.tex.

    Macro names follow the ``\\resLean<Quantity><Cohort>`` convention so
    they slot into the existing ``resLeanMRT…`` family without
    collisions.  No digits are used in the macro names (per LaTeX hygiene
    rule) — cohort suffixes are spelt out as ``MRT`` / ``NonMRT``.

    Those two suffixes are the one place the German ``MRT`` survives, and
    they stay: they are halves of macro names the manuscript already
    ``\\input``s, so renaming them here would leave every one of those
    commands undefined.
    """
    lines: list[str] = [
        "% Generated by scripts/supplementary/lean_clinical_utility.py — do not edit by hand.",
        "% Lean-Stack clinical-utility panel (MRI and non-MRI).",
        "",
    ]

    cohort_suffix = {"mri": "MRT", "non_mri": "NonMRT"}
    for cohort_key, suffix in cohort_suffix.items():
        s = summary[cohort_key]
        cal = s["calibration"]
        f1 = s["thresholds"]["f1_optimal"]
        hs = s["thresholds"]["high_specificity"]
        nb_lo = f1["net_benefit"]
        nb_hi = hs["net_benefit"]
        lines.extend(
            [
                f"\\newcommand{{\\resLeanDCAlo{suffix}}}{{{_fmt_num(nb_lo, 4)}}}"
                f"  % Lean {suffix} net benefit at F1-optimal threshold.",
                f"\\newcommand{{\\resLeanDCAhi{suffix}}}{{{_fmt_num(nb_hi, 4)}}}"
                f"  % Lean {suffix} net benefit at high-specificity threshold.",
                f"\\newcommand{{\\resLeanCalibSlope{suffix}}}{{{_fmt_num(cal['slope'])}}}"
                f"  % Lean {suffix} calibration slope.",
                f"\\newcommand{{\\resLeanCalibIntercept{suffix}}}{{{_fmt_num(cal['intercept'])}}}"
                f"  % Lean {suffix} calibration intercept-in-the-large.",
                f"\\newcommand{{\\resLeanCalibECE{suffix}}}{{{_fmt_num(cal['ece'])}}}"
                f"  % Lean {suffix} expected calibration error (10 bins).",
                f"\\newcommand{{\\resLeanPPVHighSpec{suffix}}}{{{_fmt_num(hs['ppv'])}}}"
                f"  % Lean {suffix} PPV at high-specificity cut.",
                f"\\newcommand{{\\resLeanNPVHighSpec{suffix}}}{{{_fmt_num(hs['npv'])}}}"
                f"  % Lean {suffix} NPV at high-specificity cut.",
                f"\\newcommand{{\\resLeanSensF{suffix}}}{{{_fmt_num(f1['sensitivity'])}}}"
                f"  % Lean {suffix} sensitivity at F1-optimal cut.",
                f"\\newcommand{{\\resLeanSpecF{suffix}}}{{{_fmt_num(f1['specificity'])}}}"
                f"  % Lean {suffix} specificity at F1-optimal cut.",
                "",
            ]
        )

    output_path.write_text("\n".join(lines))


def _print_panel(summary: dict[str, dict[str, Any]]) -> None:
    tbl = Table(title="Lean-Stack clinical utility (MRI vs. non-MRI)")
    tbl.add_column("Quantity", style="bold")
    tbl.add_column("MRI", justify="right")
    tbl.add_column("non-MRI", justify="right")

    def cell(d: dict[str, Any], dotted: str, digits: int = 3) -> str:
        cur: Any = d
        for key in dotted.split("."):
            cur = cur[key]
        return _fmt_num(float(cur), digits)

    rows = [
        ("N (total)", "n_total", 0),
        ("Prevalence", "prevalence", 3),
        ("ROC-AUC", "auc.roc", 3),
        ("PR-AUC", "auc.pr", 3),
        ("Calib. slope", "calibration.slope", 3),
        ("Calib. intercept", "calibration.intercept", 3),
        ("ECE", "calibration.ece", 3),
        ("Sens (F1)", "thresholds.f1_optimal.sensitivity", 3),
        ("Spec (F1)", "thresholds.f1_optimal.specificity", 3),
        ("Net benefit (F1)", "thresholds.f1_optimal.net_benefit", 4),
        ("PPV (high-spec)", "thresholds.high_specificity.ppv", 3),
        ("NPV (high-spec)", "thresholds.high_specificity.npv", 3),
        ("Net benefit (high-spec)", "thresholds.high_specificity.net_benefit", 4),
    ]
    for label, key, digits in rows:
        tbl.add_row(
            label,
            cell(summary["mri"], key, digits),
            cell(summary["non_mri"], key, digits),
        )
    console.print(tbl)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]Lean-Stack clinical utility (MRI + non-MRI)")

    mri_run = args.mri_run or _named_run(DEFAULT_MRI_RUN)
    non_mri_run = args.non_mri_run or _named_run(DEFAULT_NON_MRI_RUN)
    console.print(f"  MRI run:        {mri_run}")
    console.print(f"  non-MRI run:  {non_mri_run}")
    # The two cohorts are trained separately and read side by side here, so
    # everything except the cohort has to be held equal — including the
    # machine, once a rerun is split across two of them.
    assert_comparable_runs(
        [mri_run, non_mri_run],
        comparison="The Lean-Stack clinical-utility panel",
    )

    mri_summary, mri_dca = evaluate_cohort(mri_run, "mri")
    non_mri_summary, non_mri_dca = evaluate_cohort(non_mri_run, "non_mri")

    summary: dict[str, Any] = {
        "method": (
            "Calibration (slope, intercept, ECE-10), threshold operating "
            "points (F1-optimal and spec>=0.9), and decision-curve net "
            "benefit at both thresholds.  Computed from the held-out "
            "predictions of two Lean-Stack runs trained separately on the "
            "MRI and non-MRI cohorts (cohort='mri'/'non_mri', "
            "stack_variant='lean')."
        ),
        "mri": mri_summary,
        "non_mri": non_mri_summary,
    }

    output_dir = args.output_dir or get_paper_constants_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "lean_clinical_utility.json"
    json_path.write_text(json.dumps(summary, indent=2))

    dca_path = output_dir / "lean_dca_mri_vs_non_mri"
    plot_combined_dca(mri_dca, non_mri_dca, dca_path)

    tex_path = output_dir / "lean_clinical_utility_tex.tex"
    emit_tex_snippet(summary, tex_path)

    _print_panel(summary)
    console.print(f"[green]Wrote {json_path}")
    console.print(f"[green]Wrote {dca_path.with_suffix('.pdf')}")
    console.print(f"[green]Wrote {tex_path}")


if __name__ == "__main__":
    main()
