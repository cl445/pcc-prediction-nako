"""Compact within-study transportability panel.

Reads the source run's ``metrics.csv`` / ``confidence_intervals.json``
and the target run's ``transfer_summary.json`` and emits:

- ``results/tables/table_transport_panel.tex`` — three-row LaTeX table
  (ROC-AUC, Calibration Slope, Calibration Intercept) with source
  point, target point + 95 % CI, pre-specified equivalence band, and
  PASS/FAIL per sub-test.
- ``results/tables/table_transport_panel.json`` — identical content
  as JSON so paper tooling can render a TikZ forest plot later.

Usage
-----
    uv run python scripts/supplementary/transport_panel.py \\
        --source-run results/pcc_pipeline_mri_lean_bahmer_<ts> \\
        --transfer-run results/pcc_transfer_..._<ts>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pcc_analysis.config import get_paper_tables_dir
from pcc_analysis.run_comparability import assert_comparable_runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the within-study transportability panel (LaTeX + JSON) from a "
            "source training run and a transfer-validation run."
        ),
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="Lean-MRI training run directory (contains metrics.csv + "
        "confidence_intervals.json).",
    )
    parser.add_argument(
        "--transfer-run",
        type=Path,
        required=True,
        help="Transfer-validation run directory (contains transfer_summary.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        # Resolved in main() rather than here: get_paper_tables_dir() reads
        # the config and creates the directory, so evaluating it while the
        # parser is built makes `--help` fail on a checkout without a
        # config.toml, and creates a directory as a side effect of asking.
        default=None,
        help="Override output directory (default: results/paper/tables/).",
    )
    return parser.parse_args()


def _verdict_cell(passed: bool) -> str:
    return r"\textsc{Pass}" if passed else r"\textsc{Fail}"


def _fmt_band(band: list[float], precision: int = 3) -> str:
    lo, hi = band
    return (
        f"[{lo:+.{precision}f}, {hi:+.{precision}f}]"
        if min(band) < 0
        else (f"[{lo:.{precision}f}, {hi:.{precision}f}]")
    )


def _fmt_ci(ci: list[float], precision: int = 3) -> str:
    lo, hi = ci
    return (
        f"{lo:+.{precision}f}, {hi:+.{precision}f}"
        if min(ci) < 0
        else (f"{lo:.{precision}f}, {hi:.{precision}f}")
    )


def build_panel(source_run: Path, transfer_run: Path) -> dict[str, Any]:
    source_metrics = pd.read_csv(source_run / "metrics.csv").iloc[0].to_dict()
    with (source_run / "confidence_intervals.json").open() as f:
        source_ci = json.load(f)
    with (transfer_run / "transfer_summary.json").open() as f:
        transfer = json.load(f)
    eq = transfer["performance_equivalence"]

    return {
        "source_run": str(source_run),
        "transfer_run": str(transfer_run),
        "n_source": int(source_metrics.get("n_effective", 0))
        if "n_effective" in source_metrics
        else None,
        "n_target": int(transfer.get("n_target", 0)),
        "prevalence_source": None,
        "prevalence_target": transfer.get("prevalence_target"),
        "overall_pass": bool(eq["all_pass"]),
        "rows": [
            {
                "metric": "ROC-AUC",
                "source_point": float(source_metrics["roc_auc"]),
                "source_ci": list(source_ci.get("roc_auc_ci", [None, None])),
                "target_point": eq["delta_roc_auc"]["target"],
                "target_ci": eq["delta_roc_auc"]["target_ci"],
                "equivalence_band": eq["delta_roc_auc"]["equivalence_band"],
                "delta": eq["delta_roc_auc"]["delta_point"],
                "margin": eq["delta_roc_auc"]["margin"],
                "pass": bool(eq["delta_roc_auc"]["pass"]),
            },
            {
                "metric": "Calibration slope",
                "source_point": float(source_metrics["calibration_slope"]),
                "source_ci": [None, None],
                "target_point": eq["calibration_slope"]["target"],
                "target_ci": eq["calibration_slope"]["target_ci"],
                "equivalence_band": eq["calibration_slope"]["equivalence_band"],
                "delta": None,
                "margin": None,
                "pass": bool(eq["calibration_slope"]["pass"]),
            },
            {
                "metric": "Calibration intercept",
                "source_point": float(source_metrics["calibration_intercept"]),
                "source_ci": [None, None],
                "target_point": eq["calibration_intercept"]["target"],
                "target_ci": eq["calibration_intercept"]["target_ci"],
                "equivalence_band": eq["calibration_intercept"]["equivalence_band"],
                "delta": None,
                "margin": None,
                "pass": bool(eq["calibration_intercept"]["pass"]),
            },
        ],
    }


def render_latex(panel: dict[str, Any]) -> str:
    rows = panel["rows"]
    overall = "\\textsc{Pass}" if panel["overall_pass"] else "\\textsc{Fail}"
    header = (
        r"\begin{table}[H]"
        "\n"
        r"\centering"
        "\n"
        r"\caption{Within-study transportability performance-equivalence panel.  Target-cohort 95\,\% CI must lie entirely within the pre-specified equivalence band for PASS.  Bands follow literature-standard clinical-prediction-model transportability thresholds (Van Calster et al.\ 2016).}"
        "\n"
        r"\label{tab:transport_panel}"
        "\n"
        r"\begin{tabular}{lcccc}"
        "\n"
        r"\toprule"
        "\n"
        r"Metric & Source point & Target (95\,\% CI) & Equivalence band & Verdict \\"
        "\n"
        r"\midrule"
        "\n"
    )
    body: list[str] = []
    for r in rows:
        if r["metric"] == "ROC-AUC":
            src_cell = f"{r['source_point']:.3f}"
            tgt_cell = f"{r['target_point']:.3f} ({_fmt_ci(r['target_ci'])})"
            band_cell = _fmt_band(r["equivalence_band"])
        elif r["metric"] == "Calibration slope":
            src_cell = f"{r['source_point']:.3f}"
            tgt_cell = f"{r['target_point']:.3f} ({_fmt_ci(r['target_ci'])})"
            band_cell = _fmt_band(r["equivalence_band"], precision=2)
        else:  # Calibration intercept
            src_cell = f"{r['source_point']:+.3f}"
            tgt_cell = f"{r['target_point']:+.3f} ({_fmt_ci(r['target_ci'])})"
            band_cell = _fmt_band(r["equivalence_band"], precision=2)
        body.append(
            f"{r['metric']} & {src_cell} & {tgt_cell} & {band_cell} & {_verdict_cell(r['pass'])} \\\\"
        )
    footer = (
        r"\midrule"
        "\n"
        f"\\multicolumn{{4}}{{r}}{{\\textbf{{Overall transportability:}}}} & {overall} \\\\\n"
        r"\bottomrule"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\end{table}"
        "\n"
    )
    return header + "\n".join(body) + "\n" + footer


def main() -> None:
    args = _parse_args()
    # The equivalence verdict is a subtraction across two run directories,
    # so it is only a transportability result if both were produced by the
    # same analysis. Split a rerun across two machines and this panel is
    # where the difference would surface as a failed TOST.
    assert_comparable_runs(
        [args.source_run, args.transfer_run],
        comparison="The transportability panel",
    )
    panel = build_panel(args.source_run, args.transfer_run)

    tables_dir = args.output_dir or get_paper_tables_dir()
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = tables_dir / "table_transport_panel.json"
    with json_path.open("w") as f:
        json.dump(panel, f, indent=2, default=float)
    print(f"Saved: {json_path}")

    tex_path = tables_dir / "table_transport_panel.tex"
    tex_path.write_text(render_latex(panel))
    print(f"Saved: {tex_path}")

    overall = "PASS" if panel["overall_pass"] else "FAIL"
    print(f"\nTransportability overall: {overall}")
    for r in panel["rows"]:
        verdict = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['metric']:25s} {verdict}")


if __name__ == "__main__":
    main()
