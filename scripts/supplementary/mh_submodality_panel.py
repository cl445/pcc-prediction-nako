"""Mental-health sub-modality decomposition: paper constants from a split-MH run.

The split-MH configuration replaces the monolithic ``mental_health``
modality with five instrument-specific sub-modalities (PHQ-9, GAD-7, MINI,
PHQ-Panic, PHQ-Stress) at load time. Its interpretive value is not the
aggregate performance -- five correlated base learners make the meta-learner
slightly over-confident, so the aggregate is reported only to show it does
not diverge from the primary run -- but the ranking of the sub-scales
against each other and against the other modalities.

Every number this script emits is generated into
``paper/results_constants.tex`` rather than transcribed by hand. Hand-copied
constants from a run directory nothing in the code base references are the
arrangement in which a rerun leaves stale numbers compiling quietly, because
no generator ever contradicts them.

The run has one failure mode worth naming. A split-MH run is
indistinguishable from a primary run by its config: the flag contributes no
field to ``config.csv`` and no tag to the auto-generated directory name. The
only evidence that the split actually happened is the presence of the five
``mh_*`` entries in the modality artifacts, so this script refuses to emit
anything if they are not all there. A 2026-04-25 run silently trained with
no mental-health input at all and lost 0.06 ROC-AUC without failing.

Outputs:
    results/paper/constants/mh_submodality_panel.json
    results/paper/constants/mh_submodality_panel_tex.tex

Usage:
    uv run python scripts/supplementary/mh_submodality_panel.py \
        --run-dir results/runs/split_mh
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

from pcc_analysis.config import get_paper_constants_dir
from pcc_analysis.run_comparability import assert_comparable_runs

console = Console()
logger = logging.getLogger(__name__)

# Sub-modality key -> the TeX macro infix used in results_constants.tex.
# The paper spells digits as words in macro names (PHQNine, GADSeven),
# because TeX control sequences cannot contain them.
SUBMODALITY_TEX_NAMES: dict[str, str] = {
    "mh_phq9": "PHQNine",
    "mh_stress": "PHQStress",
    "mh_gad7": "GADSeven",
    "mh_mini": "MINI",
    "mh_panic": "PHQPanic",
}


REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "metrics.csv",
    "confidence_intervals.json",
)
"""Aggregate artifacts every constant below is read from.

Checked up front so an incomplete run directory is reported as one, rather
than as a bare ``FileNotFoundError`` from whichever ``read_csv`` happens to
run first. ``modality_scores_cv.csv`` is not in this tuple because
:func:`_require_split_run` checks it first on its own: the sub-modality
verdict outranks a missing aggregate file.
"""


def _require_split_run(run_dir: Path) -> None:
    """Refuse to emit constants unless the split actually took effect.

    The sub-modality check runs first and the remaining artifacts after it: a
    run that lost its mental-health input is the diagnosis this script exists
    to deliver, and reporting an incidentally missing ``metrics.csv`` ahead of
    it would bury the more informative answer.
    """
    scores_path = run_dir / "modality_scores_cv.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} not found; is this a pipeline run?")

    present = set(pd.read_csv(scores_path)["modality"].unique())
    missing = sorted(set(SUBMODALITY_TEX_NAMES) - present)
    if missing:
        raise ValueError(
            f"{run_dir} is missing the sub-modalities {missing}. Either the run "
            "was not started with --split-mh-submodalities, or the sub-modality "
            "columns were dropped at load time. Constants are not emitted from "
            "a run whose mental-health input cannot be verified."
        )

    absent = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
    if absent:
        raise FileNotFoundError(
            f"{run_dir} carries the sub-modalities but is missing {absent}; "
            "is this a completed pipeline run?"
        )


def build_panel(run_dir: Path, primary_dir: Path | None) -> dict[str, Any]:
    _require_split_run(run_dir)

    metrics = pd.read_csv(run_dir / "metrics.csv").iloc[0].to_dict()
    with (run_dir / "confidence_intervals.json").open() as f:
        ci = json.load(f)

    scores = pd.read_csv(run_dir / "modality_scores_cv.csv")
    per_fold = scores[scores["modality"].isin(list(SUBMODALITY_TEX_NAMES))]
    standalone = (
        per_fold.groupby("modality")["roc_auc"]
        .agg(["mean", "std", "count"])
        .to_dict("index")
    )

    # Optional for the same reason the SHAP file below is. The pipeline writes
    # this one only ``if fold_perm_importances``, so it is absent when the
    # per-fold permutation step produced nothing -- every fold failed, or the
    # run covered a fold subset. The ranking still rests on the standalone ROC,
    # and a bare FileNotFoundError here would say nothing about which of those
    # happened.
    perm_path = run_dir / "feature_importance_permutation.csv"
    perm: dict[str, float] = {}
    if perm_path.exists():
        importances = pd.read_csv(perm_path).set_index("feature")["importance_mean"]
        perm = {str(feature): float(value) for feature, value in importances.items()}
    else:
        logger.warning(
            "%s not found; permutation importances omitted. The per-fold "
            "permutation step produced no results during the run.",
            perm_path,
        )

    shap_path = run_dir / "shap_modality_importance.csv"
    shap_share: dict[str, float | None] = dict.fromkeys(SUBMODALITY_TEX_NAMES)
    if shap_path.exists():
        shap = pd.read_csv(shap_path)
        total = float(shap["mean_abs_shap"].sum())
        if total > 0:
            by_mod = shap.set_index("modality")["mean_abs_shap"]
            shap_share = {
                key: float(by_mod[key]) / total * 100 if key in by_mod.index else None
                for key in SUBMODALITY_TEX_NAMES
            }
    else:
        # _compute_shap catches its own exceptions and returns None, so a
        # missing file means the SHAP step failed rather than that it was
        # skipped. Not fatal here -- the ranking rests on the standalone
        # ROC and the permutation importance -- but it must be visible.
        logger.warning(
            "%s not found; SHAP shares omitted. The SHAP step failed silently "
            "during the run.",
            shap_path,
        )

    delta_roc: float | None = None
    if primary_dir is not None and (primary_dir / "metrics.csv").exists():
        primary = pd.read_csv(primary_dir / "metrics.csv").iloc[0].to_dict()
        delta_roc = float(metrics["roc_auc"]) - float(primary["roc_auc"])

    submodalities = {
        key: {
            "tex_name": tex_name,
            "standalone_roc_auc": float(standalone[key]["mean"]),
            "standalone_roc_auc_sd": float(standalone[key]["std"]),
            "n_folds": int(standalone[key]["count"]),
            "permutation_importance": (float(perm[key]) if key in perm else None),
            "shap_share_pct": shap_share[key],
        }
        for key, tex_name in SUBMODALITY_TEX_NAMES.items()
    }

    return {
        "run_dir": str(run_dir),
        "primary_run_dir": str(primary_dir) if primary_dir else None,
        "aggregate": {
            "roc_auc": float(metrics["roc_auc"]),
            "roc_auc_ci": list(ci["roc_auc_ci"]),
            "pr_auc": float(metrics["pr_auc"]),
            "pr_auc_ci": list(ci["pr_auc_ci"]),
            "calibration_slope": float(metrics["calibration_slope"]),
            "ece": float(metrics["ece"]),
            "delta_roc_auc_vs_primary": delta_roc,
        },
        "submodalities": submodalities,
    }


def _tex_lines(panel: dict[str, Any]) -> list[str]:
    agg = panel["aggregate"]
    lines = [
        "% Generated by scripts/supplementary/mh_submodality_panel.py "
        "-- do not edit by hand.",
        f"% Source run: {panel['run_dir']}",
        "",
        f"\\newcommand{{\\resSplitMHRoc}}{{{agg['roc_auc']:.3f}}}",
        f"\\newcommand{{\\resSplitMHRoclo}}{{{agg['roc_auc_ci'][0]:.3f}}}",
        f"\\newcommand{{\\resSplitMHRochi}}{{{agg['roc_auc_ci'][1]:.3f}}}",
        f"\\newcommand{{\\resSplitMHPr}}{{{agg['pr_auc']:.3f}}}",
        f"\\newcommand{{\\resSplitMHPrlo}}{{{agg['pr_auc_ci'][0]:.3f}}}",
        f"\\newcommand{{\\resSplitMHPrhi}}{{{agg['pr_auc_ci'][1]:.3f}}}",
        f"\\newcommand{{\\resSplitMHSlope}}{{{agg['calibration_slope']:.2f}}}",
        f"\\newcommand{{\\resSplitMHECE}}{{{agg['ece']:.3f}}}",
    ]
    if agg["delta_roc_auc_vs_primary"] is not None:
        lines.append(
            f"\\newcommand{{\\resSplitMHDeltaRoc}}"
            f"{{{agg['delta_roc_auc_vs_primary']:+.3f}}}"
        )
    lines.append("")

    for entry in panel["submodalities"].values():
        suffix = entry["tex_name"]
        lines.append(
            f"\\newcommand{{\\resStandaloneRoc{suffix}}}"
            f"{{{entry['standalone_roc_auc']:.3f}}}"
        )
        lines.append(
            f"\\newcommand{{\\resStandaloneRoc{suffix}SD}}"
            f"{{{entry['standalone_roc_auc_sd']:.3f}}}"
        )
    lines.append("")

    for entry in panel["submodalities"].values():
        if entry["permutation_importance"] is None:
            continue
        lines.append(
            f"\\newcommand{{\\resPermImp{entry['tex_name']}}}"
            f"{{{entry['permutation_importance']:.3f}}}"
        )
    lines.append("")

    for entry in panel["submodalities"].values():
        if entry["shap_share_pct"] is None:
            continue
        lines.append(
            f"\\newcommand{{\\resShap{entry['tex_name']}}}"
            f"{{{entry['shap_share_pct']:.1f}}}"
        )
    return lines


def _standalone_roc_auc(item: tuple[str, dict[str, Any]]) -> float:
    """Sort key: the standalone ROC-AUC of one ``submodalities`` entry."""
    return float(item[1]["standalone_roc_auc"])


def _render(panel: dict[str, Any]) -> None:
    agg = panel["aggregate"]
    console.print(
        f"\n[bold]Aggregate[/bold]  ROC-AUC {agg['roc_auc']:.3f} "
        f"[{agg['roc_auc_ci'][0]:.3f}, {agg['roc_auc_ci'][1]:.3f}]  "
        f"slope {agg['calibration_slope']:.2f}  ECE {agg['ece']:.3f}"
    )
    table = Table(title="Mental-health sub-modalities, ranked by standalone ROC-AUC")
    table.add_column("Sub-modality")
    table.add_column("Standalone ROC-AUC", justify="right")
    table.add_column("Permutation importance", justify="right")
    table.add_column("SHAP share", justify="right")
    ranked = sorted(
        panel["submodalities"].items(),
        key=_standalone_roc_auc,
        reverse=True,
    )
    for key, entry in ranked:
        perm = entry["permutation_importance"]
        share = entry["shap_share_pct"]
        table.add_row(
            key,
            f"{entry['standalone_roc_auc']:.3f} ± {entry['standalone_roc_auc_sd']:.3f}",
            "n/a" if perm is None else f"{perm:.3f}",
            "n/a" if share is None else f"{share:.1f} %",
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
        required=True,
        help="Pipeline run directory produced with --split-mh-submodalities",
    )
    parser.add_argument(
        "--primary-run",
        type=Path,
        default=None,
        help="Primary run directory, for the ROC-AUC delta against it",
    )
    args = parser.parse_args()

    if args.primary_run is not None:
        # The panel's headline is a ROC-AUC delta between these two runs.
        # A delta is only about the sub-modality split if nothing else
        # separates the runs that produced it.
        assert_comparable_runs(
            [args.run_dir, args.primary_run],
            comparison="The mental-health sub-modality panel",
        )

    panel = build_panel(args.run_dir, args.primary_run)
    _render(panel)

    constants_dir = get_paper_constants_dir()
    json_path = constants_dir / "mh_submodality_panel.json"
    json_path.write_text(json.dumps(panel, indent=2))
    tex_path = constants_dir / "mh_submodality_panel_tex.tex"
    tex_path.write_text("\n".join(_tex_lines(panel)) + "\n")
    console.print(f"\nWrote {json_path}\n      {tex_path}")


if __name__ == "__main__":
    main()
