"""Check every hand-carried constant in the manuscript against the run artifacts.

``compare_constants.py`` diffs two directories of generated constants. That
covers the numbers a script writes, but most of what the manuscript quotes
is typed into ``paper/results_constants.tex`` by hand: the headline metrics,
the operating points, the SHAP shares, the transfer block. After a rerun
those have to be found and carried over one by one, and nothing lists which
of them moved.

This script does. Every ``\\newcommand`` in the constants file is mapped to
the artifact and field it was transcribed from. The script reads the value
there, rounds it to the number of decimals the manuscript prints, and
compares:

    ok         the manuscript already shows the rounded artifact value
    CHANGED    the manuscript shows the old value; carry the new one over
    MISMATCH   the manuscript matches neither side; the mapping or the
               manuscript is wrong and needs a look
    no source  an external figure (NAKO recruitment) or a composed macro

A macro whose name also appears in one of the generated ``*_tex.tex`` files
takes its value from there, so the generator stays the single definition.
A generated word, such as a Pass/Fail verdict, is compared as text.

``--before-runs-dir`` and ``--before-constants-dir`` point the same mapping
at the state before the rerun. A macro the manuscript quotes at its old
value is then CHANGED rather than MISMATCH, and every ``ok`` against the
old state is evidence that the mapping reads the right field.

It also checks the one caption that carries sample arithmetic in prose: the
baseline comparison table's analytic samples minus the modality drops it
names must equal the sample sizes the runs trained on.

The exit status is non-zero while anything is CHANGED or MISMATCH, or the
caption does not add up, so the last run after the cascade doubles as its
check.

Usage:
    uv run python scripts/check_paper_constants.py \\
        --before-runs-dir results/runs_2026-08-14 \\
        --before-constants-dir results/paper/constants_previous
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from pcc_analysis.evaluation import summarize_thresholds

console = Console()

CODE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAPER_CONSTANTS = CODE_ROOT.parent / "paper" / "results_constants.tex"

_NEWCOMMAND = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\s*(?:%.*)?$")
_BLOCK_TITLE = re.compile(r"^%\s*---\s*(.+?)\s*-{2,}\s*$")
_RULE = re.compile(r"^%\s*-{20,}\s*$")


# ---------------------------------------------------------------------------
# Artifact access
# ---------------------------------------------------------------------------


class Artifacts:
    """Read-only view of one state of the results: run directories plus constants.

    Every accessor is cached, so a mapping that touches the same file forty
    times reads it once.
    """

    def __init__(self, runs_dir: Path, constants_dir: Path, paper: dict[str, str]):
        self.runs_dir = runs_dir
        self.constants_dir = constants_dir
        self._paper = paper
        self._memo: dict[tuple[Any, ...], Any] = {}

    def _cached(self, key: tuple[Any, ...], compute: Callable[[], Any]) -> Any:
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    # -- raw files -----------------------------------------------------------

    def _metrics(self, run: str) -> dict[str, float]:
        def read() -> dict[str, float]:
            with (self.runs_dir / run / "metrics.csv").open() as handle:
                row = next(csv.DictReader(handle))
            return {key: float(value) for key, value in row.items()}

        return self._cached(("metrics", run), read)

    def _run_json(self, run: str, name: str) -> Any:
        path = self.runs_dir / run / name
        return self._cached(("json", path), lambda: json.loads(path.read_text("utf-8")))

    def _run_csv(self, run: str, name: str) -> pd.DataFrame:
        path = self.runs_dir / run / name
        return self._cached(("csv", path), lambda: pd.read_csv(path))

    def _constants_json(self, name: str) -> Any:
        path = self.constants_dir / name
        return self._cached(("json", path), lambda: json.loads(path.read_text("utf-8")))

    def generated(self) -> dict[str, str]:
        """Every macro the generators wrote, by name."""

        def read() -> dict[str, str]:
            macros: dict[str, str] = {}
            for path in sorted(self.constants_dir.glob("*_tex.tex")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    match = _NEWCOMMAND.match(line.strip())
                    if match:
                        macros[match.group(1)] = match.group(2)
            return macros

        return self._cached(("generated",), read)

    def _predictions(self, run: str) -> tuple[np.ndarray, np.ndarray]:
        frame = self._run_csv(run, "predictions.csv")
        return frame["y_true"].to_numpy(), frame["y_pred_proba"].to_numpy()

    def _thresholds(self, run: str) -> tuple[dict[str, float], pd.DataFrame]:
        return self._cached(
            ("thresholds", run),
            lambda: summarize_thresholds(*self._predictions(run)),
        )

    # -- accessors used by the mapping -------------------------------------

    def paper(self, macro: str) -> float:
        """A constant the manuscript defines without a source, for derivations."""
        return float(self._paper[macro])

    def metric(self, run: str, key: str) -> float:
        return self._metrics(run)[key]

    def ci(self, run: str, key: str, bound: int) -> float:
        return float(self._run_json(run, "confidence_intervals.json")[key][bound])

    def run_csv(self, run: str, name: str) -> pd.DataFrame:
        return self._run_csv(run, name)

    def run_json(self, run: str, name: str, *path: str | int) -> Any:
        return _dig(self._run_json(run, name), path)

    def const(self, name: str, *path: str | int) -> Any:
        return _dig(self._constants_json(name), path)

    def transfer(self, *path: str | int) -> Any:
        return self.run_json(
            "transfer_lean_mri_to_non_mri", "transfer_summary.json", *path
        )

    def config(self, key: str) -> float:
        with (CODE_ROOT / "config.toml").open("rb") as handle:
            return _find_key(tomllib.load(handle), key)

    def n(self, run: str) -> int:
        return len(self._predictions(run)[0])

    def n_positive(self, run: str) -> int:
        return int(self._predictions(run)[0].sum())

    def prevalence_pct(self, run: str) -> float:
        return 100 * float(self._predictions(run)[0].mean())

    def shap_shares(self, run: str) -> pd.Series:
        """Each modality's share of the meta-learner's mean |SHAP|, in percent."""
        frame = self._run_csv(run, "shap_modality_importance.csv")
        series = frame.set_index("modality")["mean_abs_shap"]
        return 100 * series / series.sum()

    def shap_share(self, run: str, modality: str) -> float:
        return float(self.shap_shares(run).to_dict()[modality])

    def mri_shap(self, run: str, how: str) -> float:
        shares = self.shap_shares(run)
        mri = shares[shares.index.str.startswith("mri_")]
        return float(getattr(mri, how)())

    def modality_score(self, run: str, modality: str, metric: str, stat: str) -> float:
        """Mean or SD across outer folds of one modality's standalone score."""
        frame = self._run_csv(run, "modality_scores_cv.csv")
        values = frame.loc[frame["modality"] == modality, metric]
        return float(getattr(values, stat)())

    def mri_standalone(self, run: str, how: str) -> float:
        """Smallest or largest fold-mean ROC-AUC across the six atlases."""
        frame = self._run_csv(run, "modality_scores_cv.csv")
        mri = frame[frame["modality"].str.startswith("mri_")]
        return float(getattr(mri.groupby("modality")["roc_auc"].mean(), how)())

    def incremental(self, run: str, modality: str, column: str) -> float:
        frame = self._run_csv(run, "incremental_performance.csv")
        return float(frame.loc[frame["modality"] == modality, column].iloc[0])

    def operating_point(self, run: str, point: str, column: str) -> float:
        """Sensitivity, specificity or PPV at the F1-optimal or 90 %-spec cut."""
        summary, grid = self._thresholds(run)
        key = {"f1": "f1_optimal_threshold", "spec90": "spec90_threshold"}[point]
        row = grid.loc[np.isclose(grid["threshold"], summary[key])].iloc[0]
        return float(row[column])

    def threshold(self, run: str, point: str) -> float:
        key = {"f1": "f1_optimal_threshold", "spec90": "spec90_threshold"}[point]
        return self._thresholds(run)[0][key]

    def dca_band(self, run: str, bound: str) -> float:
        """Edge of the band where net benefit beats treat-all and treat-none.

        Same grid and rule as ``regenerate_decision_curve.report_useful_range``,
        which prints this band next to the figure; returned in percent.
        """
        y_true, y_score = self._predictions(run)
        grid = np.round(np.arange(0.001, 0.999, 0.001), 3)
        n = len(y_true)
        prevalence = float(y_true.mean())
        above = []
        for t in grid:
            predicted = y_score >= t
            tp = int((predicted & (y_true == 1)).sum())
            fp = int((predicted & (y_true == 0)).sum())
            net_benefit = tp / n - fp / n * t / (1 - t)
            treat_all = prevalence - (1 - prevalence) * t / (1 - t)
            above.append(net_benefit > max(treat_all, 0.0))
        inside = grid[np.array(above)]
        return 100 * float(inside.min() if bound == "lo" else inside.max())

    def _agreement_rows(self) -> list[list[str]]:
        text = (self.constants_dir / "outcome_agreement.tex").read_text("utf-8")
        rows = []
        for line in text.splitlines():
            if "&" in line and not line.lstrip().startswith("%"):
                cells = [cell.strip() for cell in line.rstrip("\\ ").split("&")]
                rows.append([re.sub(r"\\num\{(.*?)\}", r"\1", c) for c in cells])
        return rows

    def agreement(self, row: int, column: int) -> float:
        return float(self._agreement_rows()[row][column])


def _dig(payload: Any, path: tuple[str | int, ...]) -> Any:
    for key in path:
        payload = payload[key]
    return payload


def _find_key(table: dict[str, Any], key: str) -> Any:
    if key in table:
        return table[key]
    for value in table.values():
        if isinstance(value, dict):
            try:
                return _find_key(value, key)
            except KeyError:
                continue
    raise KeyError(key)


def _per_modality(entries: list[dict[str, Any]], modality: str) -> dict[str, Any]:
    return next(entry for entry in entries if entry["modality"] == modality)


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------

Source = Callable[[Artifacts], float]

EXTERNAL = {"resNrecruitTarget", "resNbaseline"}
"""NAKO design figures from the literature; no artifact carries them."""

PCT = 100.0


def _mh_share_ci(side: str, bound: int | None) -> Source:
    def read(a: Artifacts) -> float:
        if bound is None:
            entry = _per_modality(
                a.transfer("shap_share_ci_descriptive", "per_modality"),
                "mental_health",
            )
            return PCT * entry[f"{side}_share_mean"]
        key = f"baseline_mh_{side}_ci"
        return PCT * a.transfer("shap_share_ci_descriptive", key, bound)

    return read


def _sex_roc(how: str) -> Source:
    def read(a: Artifacts) -> float:
        results = a.const("mri_positive_control.json", "results")
        values = [v["sex"]["roc_auc"] for k, v in results.items() if k != "combined"]
        return min(values) if how == "min" else max(values)

    return read


def _pooled_demographics(stat: str) -> Source:
    """Whole-sample age and sex, pooled exactly from the per-group Table 1 rows.

    Same arithmetic as ``paper/jama/scripts/pooled_demographics.py``, which
    wrote these constants: count-weighted mean, SD from within- plus
    between-group sums of squares (ddof=1).
    """

    def read(a: Artifacts) -> float:
        table = a.const("descriptive_stats.json", "table1_baseline")
        age = [table["Age, years"][g] for g in ("pcc_neg", "pcc_pos")]
        n = sum(g["n"] for g in age)
        mean = sum(g["n"] * g["mean"] for g in age) / n
        if stat == "age_mean":
            return mean
        if stat == "age_sd":
            ss = sum(
                (g["n"] - 1) * g["std"] ** 2 + g["n"] * (g["mean"] - mean) ** 2
                for g in age
            )
            return float(np.sqrt(ss / (n - 1)))
        sex = table["Female sex"]
        n_female = sex["pcc_neg"]["n_positive"] + sex["pcc_pos"]["n_positive"]
        return n_female if stat == "n_female" else PCT * n_female / n

    return read


MAPPING: dict[str, Source] = {
    # --- CV topology --------------------------------------------------------
    "resNouter": lambda a: a.config("n_outer_folds"),
    "resNinner": lambda a: a.config("n_inner_folds"),
    # --- Sample funnel -------------------------------------------------------
    "resNcoronaTwo": lambda a: a.const(
        "descriptive_stats.json", "outcome_distribution", "total_n"
    ),
    "resNexNoCoronaTwo": lambda a: (
        a.paper("resNbaseline")
        - a.const("descriptive_stats.json", "outcome_distribution", "total_n")
    ),
    "resNmri": lambda a: a.const("inclusion_comparison.json", "n_mri_subsample"),
    "resNnoMRI": lambda a: (
        a.const("descriptive_stats.json", "outcome_distribution", "total_n")
        - a.const("inclusion_comparison.json", "n_mri_subsample")
    ),
    "resNinfMRI": lambda a: a.const(
        "robustness_analyses.json", "mixed_controls", "mixed_controls", "n"
    ),
    "resNexInfMRI": lambda a: (
        a.const("inclusion_comparison.json", "n_mri_subsample")
        - a.const("robustness_analyses.json", "mixed_controls", "mixed_controls", "n")
    ),
    "resNsub": lambda a: a.const(
        "inclusion_comparison.json", "exclusion_reasons", "sub_threshold_symptomatic"
    ),
    "resN": lambda a: a.const("inclusion_comparison.json", "n_included"),
    "resNcv": lambda a: a.n("primary"),
    "resNpos": lambda a: a.n_positive("primary"),
    "resNneg": lambda a: a.n("primary") - a.n_positive("primary"),
    "resNposAll": lambda a: a.const(
        "descriptive_stats.json", "table1_baseline", "Age, years", "pcc_pos", "n"
    ),
    "resNnegAll": lambda a: a.const(
        "descriptive_stats.json", "table1_baseline", "Age, years", "pcc_neg", "n"
    ),
    "resAgeMeanAll": _pooled_demographics("age_mean"),
    "resAgeSDAll": _pooled_demographics("age_sd"),
    "resNfemaleAll": _pooled_demographics("n_female"),
    "resPctFemaleAll": _pooled_demographics("pct_female"),
    "resPrevPos": lambda a: a.prevalence_pct("primary"),
    "resPrevNeg": lambda a: PCT - a.prevalence_pct("primary"),
    # --- Overall performance (primary) --------------------------------------
    "resROCauc": lambda a: a.metric("primary", "roc_auc"),
    "resROClo": lambda a: a.ci("primary", "roc_auc_ci", 0),
    "resROChi": lambda a: a.ci("primary", "roc_auc_ci", 1),
    "resPRauc": lambda a: a.metric("primary", "pr_auc"),
    "resPRlo": lambda a: a.ci("primary", "pr_auc_ci", 0),
    "resPRhi": lambda a: a.ci("primary", "pr_auc_ci", 1),
    "resECE": lambda a: a.metric("primary", "ece"),
    "resCalibSlope": lambda a: a.metric("primary", "calibration_slope"),
    "resCalibSlopeRounded": lambda a: a.metric("primary", "calibration_slope"),
    "resCalibIntercept": lambda a: a.metric("primary", "calibration_intercept"),
    "resBrier": lambda a: a.metric("primary", "brier_score"),
    "resPermNullMean": lambda a: a.run_json(
        "primary", "meta_permutation_test.json", "perm_mean"
    ),
    # --- Mental health, demographics, other modalities (primary) -----------
    "resMHroc": lambda a: a.modality_score(
        "primary", "mental_health", "roc_auc", "mean"
    ),
    "resMHrocSD": lambda a: a.modality_score(
        "primary", "mental_health", "roc_auc", "std"
    ),
    "resMHpr": lambda a: a.modality_score("primary", "mental_health", "pr_auc", "mean"),
    "resMHshap": lambda a: a.shap_share("primary", "mental_health"),
    "resMHincrement": lambda a: a.incremental(
        "primary", "mental_health", "delta_pr_auc"
    ),
    "resMHincrementFrom": lambda a: a.incremental(
        "primary", "mental_health", "pr_auc_baseline"
    ),
    "resMHincrementTo": lambda a: a.incremental(
        "primary", "mental_health", "pr_auc_augmented"
    ),
    "resDemoROC": lambda a: a.modality_score(
        "primary", "demographics", "roc_auc", "mean"
    ),
    "resDemoROCsd": lambda a: a.modality_score(
        "primary", "demographics", "roc_auc", "std"
    ),
    "resDemoShap": lambda a: a.shap_share("primary", "demographics"),
    "resSesShap": lambda a: a.shap_share("primary", "ses"),
    "resMRIshapMax": lambda a: a.mri_shap("primary", "max"),
    "resMRIshapSum": lambda a: a.mri_shap("primary", "sum"),
    # --- Neurocognitive subtype ---------------------------------------------
    "resNeuroN": lambda a: a.n("neurocog"),
    "resNeuroPrev": lambda a: a.prevalence_pct("neurocog"),
    "resNeuroROC": lambda a: a.metric("neurocog", "roc_auc"),
    "resNeuroROClo": lambda a: a.ci("neurocog", "roc_auc_ci", 0),
    "resNeuroROChi": lambda a: a.ci("neurocog", "roc_auc_ci", 1),
    "resNeuroPR": lambda a: a.metric("neurocog", "pr_auc"),
    "resNeuroPRlo": lambda a: a.ci("neurocog", "pr_auc_ci", 0),
    "resNeuroPRhi": lambda a: a.ci("neurocog", "pr_auc_ci", 1),
    "resNeuroMHroc": lambda a: a.modality_score(
        "neurocog", "mental_health", "roc_auc", "mean"
    ),
    "resNeuroMHshap": lambda a: a.shap_share("neurocog", "mental_health"),
    "resNeuroDemoROC": lambda a: a.modality_score(
        "neurocog", "demographics", "roc_auc", "mean"
    ),
    "resNeuroDemoShap": lambda a: a.shap_share("neurocog", "demographics"),
    # --- Clinical utility (primary) -----------------------------------------
    "resThrF": lambda a: a.threshold("primary", "f1"),
    "resThrHS": lambda a: a.threshold("primary", "spec90"),
    "resSensF": lambda a: PCT * a.operating_point("primary", "f1", "recall"),
    "resSpecF": lambda a: PCT * a.operating_point("primary", "f1", "specificity"),
    "resPPVF": lambda a: PCT * a.operating_point("primary", "f1", "precision"),
    "resSensHS": lambda a: PCT * a.operating_point("primary", "spec90", "recall"),
    "resSpecHS": lambda a: PCT * a.operating_point("primary", "spec90", "specificity"),
    "resPPVHS": lambda a: PCT * a.operating_point("primary", "spec90", "precision"),
    "resPPVenrich": lambda a: (
        PCT
        * (
            PCT
            * a.operating_point("primary", "spec90", "precision")
            / a.prevalence_pct("primary")
            - 1
        )
    ),
    "resDCAlo": lambda a: a.dca_band("primary", "lo"),
    "resDCAhi": lambda a: a.dca_band("primary", "hi"),
    # --- Outcome agreement ---------------------------------------------------
    "resKappaNeuro": lambda a: a.agreement(1, 3),
    "resAgreeBahmerN": lambda a: a.agreement(0, 1),
    "resAgreeBahmerPct": lambda a: a.agreement(0, 2),
    "resAgreeNeuroN": lambda a: a.agreement(1, 1),
    "resAgreeNeuroPct": lambda a: a.agreement(1, 2),
    "resAgreeNeuroKappa": lambda a: a.agreement(1, 3),
    "resAgreeSevereN": lambda a: a.agreement(2, 1),
    "resAgreeSeverePct": lambda a: a.agreement(2, 2),
    "resAgreeSevereKappa": lambda a: a.agreement(2, 3),
    # --- Transfer: sample, Lean-MRI source, target --------------------------
    "resNonMRTn": lambda a: a.transfer("n_target"),
    "resNonMRTprev": lambda a: PCT * a.transfer("prevalence_target"),
    "resLeanMRTroc": lambda a: a.metric("lean_mri", "roc_auc"),
    "resLeanMRTroclo": lambda a: a.ci("lean_mri", "roc_auc_ci", 0),
    "resLeanMRTrochi": lambda a: a.ci("lean_mri", "roc_auc_ci", 1),
    "resLeanMRTpr": lambda a: a.metric("lean_mri", "pr_auc"),
    "resLeanMRTslope": lambda a: a.metric("lean_mri", "calibration_slope"),
    "resLeanMRTintercept": lambda a: a.metric("lean_mri", "calibration_intercept"),
    "resTransferRocTarget": lambda a: a.transfer("metrics", "roc_auc"),
    "resTransferRoclo": lambda a: a.transfer("confidence_intervals", "roc_auc_ci", 0),
    "resTransferRochi": lambda a: a.transfer("confidence_intervals", "roc_auc_ci", 1),
    "resTransferRocDelta": lambda a: a.transfer(
        "performance_equivalence", "delta_roc_auc", "delta_point"
    ),
    "resTransferPrTarget": lambda a: a.transfer("metrics", "pr_auc"),
    "resTransferSlope": lambda a: a.transfer("metrics", "calibration_slope"),
    "resTransferSlopelo": lambda a: a.transfer(
        "performance_equivalence", "calibration_slope", "target_ci", 0
    ),
    "resTransferSlopehi": lambda a: a.transfer(
        "performance_equivalence", "calibration_slope", "target_ci", 1
    ),
    "resTransferIntercept": lambda a: a.transfer("metrics", "calibration_intercept"),
    "resTransferInterceptlo": lambda a: a.transfer(
        "performance_equivalence", "calibration_intercept", "target_ci", 0
    ),
    "resTransferIntercepthi": lambda a: a.transfer(
        "performance_equivalence", "calibration_intercept", "target_ci", 1
    ),
    "resTransferBrier": lambda a: a.transfer("metrics", "brier_score"),
    "resTostRocMargin": lambda a: a.transfer(
        "performance_equivalence", "delta_roc_auc", "margin"
    ),
    "resTostSlopeLo": lambda a: a.transfer(
        "performance_equivalence", "calibration_slope", "equivalence_band", 0
    ),
    "resTostSlopeHi": lambda a: a.transfer(
        "performance_equivalence", "calibration_slope", "equivalence_band", 1
    ),
    "resTostInterceptLo": lambda a: a.transfer(
        "performance_equivalence", "calibration_intercept", "equivalence_band", 0
    ),
    "resTostInterceptHi": lambda a: a.transfer(
        "performance_equivalence", "calibration_intercept", "equivalence_band", 1
    ),
    "resMHshareSource": _mh_share_ci("source", None),
    "resMHshareSourcelo": _mh_share_ci("source", 0),
    "resMHshareSourcehi": _mh_share_ci("source", 1),
    "resMHshareTarget": _mh_share_ci("target", None),
    "resMHshareTargetlo": _mh_share_ci("target", 0),
    "resMHshareTargethi": _mh_share_ci("target", 1),
    # --- E-value -------------------------------------------------------------
    "resMHpositiveShare": lambda a: (
        PCT
        * a.const("e_value.json", "sample", "n_exposed_mh_positive")
        / a.const("e_value.json", "sample", "n_total")
    ),
    "resPCCifMHpos": lambda a: (
        PCT * a.const("e_value.json", "sample", "prevalence_exposed")
    ),
    "resPCCifMHneg": lambda a: (
        PCT * a.const("e_value.json", "sample", "prevalence_unexposed")
    ),
    "resBaselineMHor": lambda a: a.const(
        "e_value.json", "odds_ratio", "adjusted_age_sex_center", "point"
    ),
    "resBaselineMHorLo": lambda a: a.const(
        "e_value.json", "odds_ratio", "adjusted_age_sex_center", "ci_95", 0
    ),
    "resBaselineMHorHi": lambda a: a.const(
        "e_value.json", "odds_ratio", "adjusted_age_sex_center", "ci_95", 1
    ),
    "resEvaluePoint": lambda a: a.const("e_value.json", "e_value", "point"),
    "resEvalueCI": lambda a: a.const("e_value.json", "e_value", "ci_lower"),
    # --- Sex interaction and demographics decomposition --------------------
    "resSexIntOR": lambda a: a.const(
        "sex_interaction.json",
        "interaction_model",
        "interaction_or_female_vs_male",
        "point",
    ),
    "resSexIntORlo": lambda a: a.const(
        "sex_interaction.json",
        "interaction_model",
        "interaction_or_female_vs_male",
        "ci_95",
        0,
    ),
    "resSexIntORhi": lambda a: a.const(
        "sex_interaction.json",
        "interaction_model",
        "interaction_or_female_vs_male",
        "ci_95",
        1,
    ),
    "resSexIntP": lambda a: a.const(
        "sex_interaction.json",
        "interaction_model",
        "interaction_or_female_vs_male",
        "wald_p",
    ),
    "resMHorFemale": lambda a: a.const(
        "sex_interaction.json", "stratified_models", "female", "odds_ratio", "point"
    ),
    "resMHorFemaleLo": lambda a: a.const(
        "sex_interaction.json", "stratified_models", "female", "odds_ratio", "ci_95", 0
    ),
    "resMHorFemaleHi": lambda a: a.const(
        "sex_interaction.json", "stratified_models", "female", "odds_ratio", "ci_95", 1
    ),
    "resMHorMale": lambda a: a.const(
        "sex_interaction.json", "stratified_models", "male", "odds_ratio", "point"
    ),
    "resMHorMaleLo": lambda a: a.const(
        "sex_interaction.json", "stratified_models", "male", "odds_ratio", "ci_95", 0
    ),
    "resMHorMaleHi": lambda a: a.const(
        "sex_interaction.json", "stratified_models", "male", "odds_ratio", "ci_95", 1
    ),
    "resSexAuc": lambda a: a.const(
        "sex_interaction.json",
        "demographics_decomposition",
        "sex_female",
        "standalone_auc",
    ),
    "resSexUniOr": lambda a: a.const(
        "sex_interaction.json",
        "demographics_decomposition",
        "sex_female",
        "univariable_or_female_vs_male",
        "point",
    ),
    "resSexUniOrLo": lambda a: a.const(
        "sex_interaction.json",
        "demographics_decomposition",
        "sex_female",
        "univariable_or_female_vs_male",
        "ci_lo",
    ),
    "resSexUniOrHi": lambda a: a.const(
        "sex_interaction.json",
        "demographics_decomposition",
        "sex_female",
        "univariable_or_female_vs_male",
        "ci_hi",
    ),
    "resAgeAuc": lambda a: a.const(
        "sex_interaction.json", "demographics_decomposition", "age", "standalone_auc"
    ),
    # --- Persistence adjustment ----------------------------------------------
    "resBaselineMHorAdj": lambda a: a.const(
        "persistence_adjustment.json", "model_b_persistence_adjusted", "or_baseline_mh"
    ),
    "resBaselineMHorAdjLo": lambda a: a.const(
        "persistence_adjustment.json",
        "model_b_persistence_adjusted",
        "ci_95_baseline_mh",
        0,
    ),
    "resBaselineMHorAdjHi": lambda a: a.const(
        "persistence_adjustment.json",
        "model_b_persistence_adjusted",
        "ci_95_baseline_mh",
        1,
    ),
    "resCurrentMHor": lambda a: a.const(
        "persistence_adjustment.json", "model_b_persistence_adjusted", "or_current_mh"
    ),
    "resCurrentMHorLo": lambda a: a.const(
        "persistence_adjustment.json",
        "model_b_persistence_adjusted",
        "ci_95_current_mh",
        0,
    ),
    "resCurrentMHorHi": lambda a: a.const(
        "persistence_adjustment.json",
        "model_b_persistence_adjusted",
        "ci_95_current_mh",
        1,
    ),
    "resAdjCoeffShrink": lambda a: (
        PCT
        * a.const(
            "persistence_adjustment.json", "coefficient_shrinkage", "shrink_ratio"
        )
    ),
    "resPersistenceN": lambda a: a.const(
        "persistence_adjustment.json", "model_b_persistence_adjusted", "n"
    ),
    # --- MH trajectory adjustment ------------------------------------------
    "resBaselineMHorTrajAdj": lambda a: a.const(
        "mh_trajectory_adjustment.json", "model_c_t0_t1_t2_trajectory", "or_t0"
    ),
    "resBaselineMHorTrajAdjLo": lambda a: a.const(
        "mh_trajectory_adjustment.json", "model_c_t0_t1_t2_trajectory", "ci_95_t0", 0
    ),
    "resBaselineMHorTrajAdjHi": lambda a: a.const(
        "mh_trajectory_adjustment.json", "model_c_t0_t1_t2_trajectory", "ci_95_t0", 1
    ),
    "resCoronaOneMHor": lambda a: a.const(
        "mh_trajectory_adjustment.json", "model_c_t0_t1_t2_trajectory", "or_t1"
    ),
    "resCoronaOneMHorLo": lambda a: a.const(
        "mh_trajectory_adjustment.json", "model_c_t0_t1_t2_trajectory", "ci_95_t1", 0
    ),
    "resCoronaOneMHorHi": lambda a: a.const(
        "mh_trajectory_adjustment.json", "model_c_t0_t1_t2_trajectory", "ci_95_t1", 1
    ),
    "resCoronaTwoMHorTraj": lambda a: a.const(
        "mh_trajectory_adjustment.json", "model_c_t0_t1_t2_trajectory", "or_t2"
    ),
    "resCoronaTwoMHorTrajLo": lambda a: a.const(
        "mh_trajectory_adjustment.json", "model_c_t0_t1_t2_trajectory", "ci_95_t2", 0
    ),
    "resCoronaTwoMHorTrajHi": lambda a: a.const(
        "mh_trajectory_adjustment.json", "model_c_t0_t1_t2_trajectory", "ci_95_t2", 1
    ),
    "resTrajShrinkagePct": lambda a: (
        PCT
        * a.const(
            "mh_trajectory_adjustment.json",
            "trajectory_shrinkage_baseline",
            "shrink_ratio",
        )
    ),
    "resTrajN": lambda a: a.const(
        "mh_trajectory_adjustment.json", "model_c_t0_t1_t2_trajectory", "n"
    ),
    # --- Non-MRI Lean re-fit ----------------------------------------------
    "resNonMRTreFitRoc": lambda a: a.metric("lean_non_mri", "roc_auc"),
    "resNonMRTreFitRoclo": lambda a: a.ci("lean_non_mri", "roc_auc_ci", 0),
    "resNonMRTreFitRochi": lambda a: a.ci("lean_non_mri", "roc_auc_ci", 1),
    "resNonMRTreFitPr": lambda a: a.metric("lean_non_mri", "pr_auc"),
    "resNonMRTreFitPrlo": lambda a: a.ci("lean_non_mri", "pr_auc_ci", 0),
    "resNonMRTreFitPrhi": lambda a: a.ci("lean_non_mri", "pr_auc_ci", 1),
    "resNonMRTreFitSlope": lambda a: a.metric("lean_non_mri", "calibration_slope"),
    "resNonMRTreFitIntercept": lambda a: a.metric(
        "lean_non_mri", "calibration_intercept"
    ),
    "resNonMRTreFitECE": lambda a: a.metric("lean_non_mri", "ece"),
    "resTransferReFitDeltaRoc": lambda a: (
        a.metric("lean_non_mri", "roc_auc") - a.transfer("metrics", "roc_auc")
    ),
    # --- Symptom decomposition -----------------------------------------------
    "resOverlapPrev": lambda a: (
        PCT
        * a.const(
            "symptom_decomposition.json", "sample", "prevalence_any_overlap_symptom"
        )
    ),
    "resHardPCCprev": lambda a: (
        PCT
        * a.const(
            "symptom_decomposition.json", "sample", "prevalence_any_hard_pcc_symptom"
        )
    ),
    "resOverlapOr": lambda a: a.const(
        "symptom_decomposition.json",
        "odds_ratios_binary_outcome",
        "any_overlap_symptom",
        "or",
    ),
    "resOverlapOrLo": lambda a: a.const(
        "symptom_decomposition.json",
        "odds_ratios_binary_outcome",
        "any_overlap_symptom",
        "ci_95",
        0,
    ),
    "resOverlapOrHi": lambda a: a.const(
        "symptom_decomposition.json",
        "odds_ratios_binary_outcome",
        "any_overlap_symptom",
        "ci_95",
        1,
    ),
    "resHardPCCor": lambda a: a.const(
        "symptom_decomposition.json",
        "odds_ratios_binary_outcome",
        "any_hard_pcc_symptom",
        "or",
    ),
    "resHardPCCorLo": lambda a: a.const(
        "symptom_decomposition.json",
        "odds_ratios_binary_outcome",
        "any_hard_pcc_symptom",
        "ci_95",
        0,
    ),
    "resHardPCCorHi": lambda a: a.const(
        "symptom_decomposition.json",
        "odds_ratios_binary_outcome",
        "any_hard_pcc_symptom",
        "ci_95",
        1,
    ),
    "resOverlapIrr": lambda a: a.const(
        "symptom_decomposition.json",
        "incidence_rate_ratios_counts",
        "overlap_count",
        "irr",
    ),
    "resHardPCCIrr": lambda a: a.const(
        "symptom_decomposition.json",
        "incidence_rate_ratios_counts",
        "hard_pcc_count",
        "irr",
    ),
    # --- Pooled Lean ---------------------------------------------------------
    "resPoolLeanN": lambda a: a.n("lean_pooled"),
    "resPoolLeanPrev": lambda a: a.prevalence_pct("lean_pooled"),
    "resPoolLeanRoc": lambda a: a.metric("lean_pooled", "roc_auc"),
    "resPoolLeanRoclo": lambda a: a.ci("lean_pooled", "roc_auc_ci", 0),
    "resPoolLeanRochi": lambda a: a.ci("lean_pooled", "roc_auc_ci", 1),
    "resPoolLeanPr": lambda a: a.metric("lean_pooled", "pr_auc"),
    "resPoolLeanPrlo": lambda a: a.ci("lean_pooled", "pr_auc_ci", 0),
    "resPoolLeanPrhi": lambda a: a.ci("lean_pooled", "pr_auc_ci", 1),
    "resPoolLeanSlope": lambda a: a.metric("lean_pooled", "calibration_slope"),
    "resPoolLeanIntercept": lambda a: a.metric("lean_pooled", "calibration_intercept"),
    "resPoolLeanECE": lambda a: a.metric("lean_pooled", "ece"),
    "resPoolLeanBrier": lambda a: a.metric("lean_pooled", "brier_score"),
    "resPoolLeanDeltaRocVsRefit": lambda a: (
        a.metric("lean_pooled", "roc_auc") - a.metric("lean_non_mri", "roc_auc")
    ),
    "resPoolLeanMHshap": lambda a: a.shap_share("lean_pooled", "mental_health"),
    "resPoolLeanDemoShap": lambda a: a.shap_share("lean_pooled", "demographics"),
    "resPoolLeanMedHistShap": lambda a: a.shap_share("lean_pooled", "medical_history"),
    "resPoolLeanSesShap": lambda a: a.shap_share("lean_pooled", "ses"),
    # --- MRI positive control and standalone atlas ranges ------------------
    "resMriPosSexRocMin": _sex_roc("min"),
    "resMriPosSexRocMax": _sex_roc("max"),
    "resMriRocLo": lambda a: a.mri_standalone("primary", "min"),
    "resMriRocHi": lambda a: a.mri_standalone("primary", "max"),
    "resMriStandaloneNoDMLlo": lambda a: a.mri_standalone("no_dml", "min"),
    "resMriStandaloneNoDMLhi": lambda a: a.mri_standalone("no_dml", "max"),
    "resNeuroMriRocLo": lambda a: a.mri_standalone("neurocog", "min"),
    "resNeuroMriRocHi": lambda a: a.mri_standalone("neurocog", "max"),
    # --- Olfactory anchor, fields the generator does not emit -------------
    "resOlfObjPostOrSelfReport": lambda a: a.const(
        "olfactory_objective_anchor.json",
        "panels",
        "post_infection",
        "self_reported_smell_loss",
        "odds_ratio",
    ),
    "resOlfObjPostOrSelfReportlo": lambda a: a.const(
        "olfactory_objective_anchor.json",
        "panels",
        "post_infection",
        "self_reported_smell_loss",
        "ci",
        0,
    ),
    "resOlfObjPostOrSelfReporthi": lambda a: a.const(
        "olfactory_objective_anchor.json",
        "panels",
        "post_infection",
        "self_reported_smell_loss",
        "ci",
        1,
    ),
    "resOlfObjPostNposObjective": lambda a: a.const(
        "olfactory_objective_anchor.json",
        "panels",
        "post_infection",
        "objective_hyposmia",
        "n_events",
    ),
    "resOlfObjPostNposSelfReport": lambda a: a.const(
        "olfactory_objective_anchor.json",
        "panels",
        "post_infection",
        "self_reported_smell_loss",
        "n_events",
    ),
    "resOlfObjNeverInfN": lambda a: a.const(
        "olfactory_objective_anchor.json", "panels", "never_infected", "n"
    ),
}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class Constant:
    name: str
    body: str
    block: str


def read_paper_constants(path: Path) -> list[Constant]:
    """Every ``\\newcommand`` in the file, tagged with the block it sits in.

    A block is named by a ``% --- Title ---`` line, or by the first comment
    line inside a pair of full-width rules. Rules come in pairs around a
    header, so only the opening one of a pair announces a title.
    """
    constants: list[Constant] = []
    block = ""
    header_open = False
    expect_title = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if _RULE.match(stripped):
            header_open = not header_open
            expect_title = header_open
            continue
        title = _BLOCK_TITLE.match(stripped)
        if title:
            block, expect_title = title.group(1), False
            continue
        if expect_title and stripped.startswith("%"):
            text = stripped.lstrip("% ").strip()
            if text:
                block, expect_title = text, False
            continue
        match = _NEWCOMMAND.match(stripped)
        if match:
            constants.append(Constant(match.group(1), match.group(2), block))
    return constants


def _decimals(body: str) -> int:
    return len(body.split(".", 1)[1]) if "." in body else 0


def _is_number(body: str) -> bool:
    try:
        float(body)
    except ValueError:
        return False
    return True


def _rounded(value: float, decimals: int, signed: bool = False) -> str:
    return f"{value:{'+' if signed else ''}.{decimals}f}"


def _signed(body: str) -> bool:
    """Whether the manuscript prints this value with an explicit sign.

    A signed delta that rounds to zero still says which way it points, so
    ``-0.000`` against a new ``+0.000`` is a change even though the two
    are equal as numbers.
    """
    return body.startswith("+") or (body.startswith("-") and float(body) == 0)


def _same(body: str, value: float) -> bool:
    if _signed(body):
        return _rounded(value, _decimals(body), signed=True) == body
    return float(_rounded(value, _decimals(body))) == float(body)


def resolve(constant: Constant, artifacts: Artifacts) -> float | None:
    """The artifact value behind one constant, or ``None`` if it has none."""
    generated = artifacts.generated().get(constant.name)
    if generated is not None and _is_number(generated):
        return float(generated)
    source = MAPPING.get(constant.name)
    if source is None:
        return None
    return float(source(artifacts))


@dataclass
class Row:
    constant: Constant
    after: float | None
    before: float | None
    status: str
    error: str = ""


def compare(
    constants: list[Constant], after: Artifacts, before: Artifacts | None
) -> list[Row]:
    rows = []
    for constant in constants:
        generated = after.generated().get(constant.name)
        if not _is_number(constant.body) and generated is not None:
            # A generated word such as a Pass/Fail verdict: compared as text.
            old_word = before.generated().get(constant.name) if before else None
            if constant.body == generated:
                status = "ok"
            elif old_word is not None and constant.body == old_word:
                status = "CHANGED"
            else:
                status = "MISMATCH"
            rows.append(Row(constant, None, None, status))
            continue
        if not _is_number(constant.body) or constant.name in EXTERNAL:
            rows.append(Row(constant, None, None, "no source"))
            continue
        try:
            new = resolve(constant, after)
            old = resolve(constant, before) if before else None
        except (KeyError, IndexError, FileNotFoundError, StopIteration) as exc:
            rows.append(
                Row(constant, None, None, "ERROR", f"{type(exc).__name__}: {exc}")
            )
            continue
        if new is None:
            rows.append(Row(constant, None, None, "UNMAPPED"))
            continue
        if _same(constant.body, new):
            status = "ok"
        elif old is not None and _same(constant.body, old):
            status = "CHANGED"
        else:
            status = "MISMATCH"
        rows.append(Row(constant, new, old, status))
    return rows


# The baseline comparison table prints the descriptive analytic samples and
# names, in its caption, the participants dropped on the way to the
# cross-validation samples. Its JSON carries both, and the remainder has to
# equal what the runs trained on, or the caption no longer adds up.
CV_SAMPLES: dict[str, tuple[str, Source]] = {
    "mri": ("n_mri", lambda a: a.n("primary")),
    "non_mri": ("n_non_mri", lambda a: a.transfer("n_target")),
}


def check_cv_sample_drops(artifacts: Artifacts) -> list[str]:
    """Where analytic sample minus caption drops misses the run's sample size."""
    path = artifacts.constants_dir.parent / "tables" / "table1_baseline_comparison.json"
    comparison = json.loads(path.read_text("utf-8"))
    problems = []
    for cohort, (key, source) in CV_SAMPLES.items():
        drops = comparison["cv_sample_drops"][cohort]
        expected = comparison[key] - sum(drops.values())
        actual = int(source(artifacts))
        if expected != actual:
            problems.append(
                f"{cohort}: {comparison[key]} analytic - {sum(drops.values())} "
                f"dropped {drops} = {expected}, but the run has N = {actual}"
            )
    return problems


STYLE = {
    "ok": "green",
    "CHANGED": "yellow",
    "MISMATCH": "red",
    "UNMAPPED": "red",
    "ERROR": "red",
    "no source": "dim",
}


def render(rows: list[Row], show_all: bool, with_before: bool) -> None:
    table = Table(show_header=True, show_lines=False)
    table.add_column("Block", overflow="fold", max_width=28)
    table.add_column("Constant", style="bold")
    table.add_column("Paper", justify="right")
    table.add_column("New", justify="right")
    if with_before:
        table.add_column("Old run", justify="right")
    table.add_column("New raw", justify="right", style="dim")
    table.add_column("Status")
    previous_block = None
    for row in rows:
        if not show_all and row.status in ("ok", "no source"):
            continue
        decimals = _decimals(row.constant.body)
        signed = _signed(row.constant.body)
        cells = [
            row.constant.block if row.constant.block != previous_block else "",
            row.constant.name,
            row.constant.body,
            "" if row.after is None else _rounded(row.after, decimals, signed),
        ]
        if with_before:
            cells.append(
                "" if row.before is None else _rounded(row.before, decimals, signed)
            )
        cells += [
            "" if row.after is None else f"{row.after:.6g}",
            f"[{STYLE[row.status]}]{row.status}[/] {row.error}".rstrip(),
        ]
        table.add_row(*cells)
        previous_block = row.constant.block
    console.print(table)


def write_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["block", "macro", "paper", "new_rounded", "new_raw", "old_raw", "status"]
        )
        for row in rows:
            decimals = _decimals(row.constant.body)
            signed = _signed(row.constant.body)
            writer.writerow(
                [
                    row.constant.block,
                    row.constant.name,
                    row.constant.body,
                    "" if row.after is None else _rounded(row.after, decimals, signed),
                    "" if row.after is None else repr(row.after),
                    "" if row.before is None else repr(row.before),
                    row.status,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--paper-constants", type=Path, default=DEFAULT_PAPER_CONSTANTS)
    parser.add_argument("--runs-dir", type=Path, default=CODE_ROOT / "results" / "runs")
    parser.add_argument(
        "--constants-dir",
        type=Path,
        default=CODE_ROOT / "results" / "paper" / "constants",
    )
    parser.add_argument(
        "--before-runs-dir",
        type=Path,
        help="Run directories as they stood before the rerun.",
    )
    parser.add_argument(
        "--before-constants-dir",
        type=Path,
        help="Generated constants as they stood before the rerun.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Also list unchanged constants."
    )
    parser.add_argument(
        "--csv", type=Path, help="Write the full comparison to this file."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constants = read_paper_constants(args.paper_constants)
    paper = {c.name: c.body for c in constants}

    after = Artifacts(args.runs_dir, args.constants_dir, paper)
    before = None
    if args.before_runs_dir or args.before_constants_dir:
        if not (args.before_runs_dir and args.before_constants_dir):
            console.print(
                "[red]--before-runs-dir and --before-constants-dir go together."
            )
            sys.exit(2)
        before = Artifacts(args.before_runs_dir, args.before_constants_dir, paper)

    rows = compare(constants, after, before)
    render(rows, args.all, before is not None)
    if args.csv:
        write_csv(rows, args.csv)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    summary = ", ".join(f"{counts[k]} {k}" for k in STYLE if k in counts)
    console.print(f"\n[bold]{len(rows)} constants:[/bold] {summary}")
    open_items = sum(
        counts.get(k, 0) for k in ("CHANGED", "MISMATCH", "UNMAPPED", "ERROR")
    )

    try:
        drop_problems = check_cv_sample_drops(after)
    except (KeyError, FileNotFoundError) as exc:
        drop_problems = [f"{type(exc).__name__}: {exc}"]
    if drop_problems:
        console.print(
            "\n[red]Baseline comparison caption: modality drops do not add up "
            "(update CV_SAMPLE_DROPS in cohort_comparison.py and rerun it):"
        )
        for problem in drop_problems:
            console.print(f"  {problem}")
    else:
        console.print("[green]Baseline comparison caption: modality drops add up.")
    sys.exit(1 if open_items or drop_problems else 0)


if __name__ == "__main__":
    main()
