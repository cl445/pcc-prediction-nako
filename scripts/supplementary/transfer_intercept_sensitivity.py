"""How narrowly does the transfer's calibration intercept meet its band?

The within-study transfer is judged on three pre-specified equivalence
criteria. The calibration-intercept criterion is the tight one: its 95 %
interval comes from a bootstrap that resamples stratified by outcome, which
holds the prevalence fixed in every draw. Calibration-in-the-large is the
quantity most sensitive to prevalence, so that scheme gives it the
narrowest interval of the reasonable choices. DECISIONS §2.31 stops
stratifying the PR-AUC interval for the same reason.

This script keeps the pre-specified verdict and reports the post hoc
sensitivity next to it. On the transfer's target predictions it computes the
intercept interval three ways:

    stratified     the pipeline's own scheme and seed; must reproduce the
                   interval in transfer_summary.json, or the script stops
    unstratified   the same bootstrap without stratification, at the
                   pipeline seed and over a sweep of further seeds, because
                   the upper limit of a 1000-draw percentile interval moves
                   with the seed by about as much as the margin at stake
    Wald           the model-based interval of the same recalibration fit

Pointed at an earlier transfer run, the same computation reproduces that
run's numbers, which is how the supplement shows that a change in the
verdict comes from the model and not from the method.

Outputs:
    results/paper/constants/transfer_intercept_sensitivity.json
    results/paper/constants/transfer_intercept_sensitivity_tex.tex

Usage:
    uv run python scripts/supplementary/transfer_intercept_sensitivity.py \\
        --source-run results/runs/lean_mri \\
        --transfer-run results/runs/transfer_lean_mri_to_non_mri
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression

from pcc_analysis.config import get_paper_constants_dir

console = Console()

N_BOOTSTRAP = 1000
"""Matches CALIB_BOOTSTRAP_ITERATIONS in 03_apply_transfer.py."""

SEED_OFFSET = 2
"""03_apply_transfer.py seeds the calibration bootstrap at random_state + 2."""

N_SWEEP_SEEDS = 20
REPRODUCTION_TOLERANCE = 1e-9


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="Lean-MRI training run (its config.csv carries the random_state).",
    )
    parser.add_argument(
        "--transfer-run",
        type=Path,
        required=True,
        help="Transfer run (predictions.csv and transfer_summary.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the paper constants directory.",
    )
    return parser.parse_args()


def _logits(y_pred: np.ndarray) -> np.ndarray:
    proba = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return np.log(proba / (1 - proba))


def _intercept(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Recalibration intercept, fitted as in _bootstrap_target_calibration."""
    model = LogisticRegression(C=1e10, max_iter=1000)
    model.fit(logits.reshape(-1, 1), y_true)
    return float(model.intercept_[0])


def bootstrap_interval(
    logits: np.ndarray, y_true: np.ndarray, seed: int, stratify: bool
) -> tuple[float, float]:
    """Percentile 95 % interval of the intercept over N_BOOTSTRAP draws.

    The stratified branch draws positives before negatives on one
    generator, as _bootstrap_target_calibration does, so the same seed
    gives the same draws.
    """
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y_true == 1)
    negatives = np.flatnonzero(y_true == 0)
    estimates = []
    for _ in range(N_BOOTSTRAP):
        if stratify:
            index = np.concatenate(
                [
                    rng.choice(positives, size=len(positives), replace=True),
                    rng.choice(negatives, size=len(negatives), replace=True),
                ]
            )
        else:
            index = rng.choice(len(y_true), size=len(y_true), replace=True)
        estimates.append(_intercept(logits[index], y_true[index]))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def wald_interval(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    fit = sm.Logit(y_true, sm.add_constant(logits)).fit(disp=0)
    low, high = np.asarray(fit.conf_int())[0]
    return float(low), float(high)


def main() -> None:
    args = _parse_args()
    summary_path = args.transfer_run / "transfer_summary.json"
    transfer: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    criterion = transfer["performance_equivalence"]["calibration_intercept"]
    band_low, band_high = (float(v) for v in criterion["equivalence_band"])

    source_config = pd.read_csv(args.source_run / "config.csv").iloc[0]
    seed = int(source_config["random_state"]) + SEED_OFFSET

    predictions = pd.read_csv(args.transfer_run / "predictions.csv")
    y_true = predictions["y_true"].to_numpy().astype(np.int8)
    logits = _logits(predictions["y_pred_proba"].to_numpy())

    point = _intercept(logits, y_true)
    console.log(f"Stratified bootstrap at seed {seed} ...")
    stratified = bootstrap_interval(logits, y_true, seed, stratify=True)
    reported = tuple(float(v) for v in criterion["target_ci"])
    if max(abs(a - b) for a, b in zip(stratified, reported, strict=True)) > (
        REPRODUCTION_TOLERANCE
    ):
        raise SystemExit(
            f"Stratified interval {stratified} does not reproduce the one in "
            f"{summary_path} ({reported}); the sensitivity would not be about "
            "the reported interval."
        )

    console.log("Unstratified bootstrap and Wald ...")
    unstratified = bootstrap_interval(logits, y_true, seed, stratify=False)
    wald = wald_interval(logits, y_true)
    console.log(f"Seed sweep over {N_SWEEP_SEEDS} seeds ...")
    sweep_upper = [
        bootstrap_interval(logits, y_true, s, stratify=False)[1]
        for s in range(N_SWEEP_SEEDS)
    ]

    result = {
        "transfer_run": str(args.transfer_run),
        "n_target": len(y_true),
        "band": [band_low, band_high],
        "n_bootstrap": N_BOOTSTRAP,
        "pipeline_seed": seed,
        "point": point,
        "stratified": {
            "ci": list(stratified),
            "pass": band_high >= stratified[1] and band_low <= stratified[0],
        },
        "unstratified": {
            "ci": list(unstratified),
            "pass": band_high >= unstratified[1] and band_low <= unstratified[0],
        },
        "wald": {
            "ci": list(wald),
            "pass": band_high >= wald[1] and band_low <= wald[0],
        },
        "unstratified_seed_sweep": {
            "seeds": list(range(N_SWEEP_SEEDS)),
            "upper": sweep_upper,
            "upper_mean": float(np.mean(sweep_upper)),
            "upper_sd": float(np.std(sweep_upper, ddof=1)),
            "n_above_band": int(sum(u > band_high for u in sweep_upper)),
        },
    }

    output_dir = args.output_dir or get_paper_constants_dir()
    json_path = output_dir / "transfer_intercept_sensitivity.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    sweep = result["unstratified_seed_sweep"]
    macros = {
        "resTransferInterceptUnstratHi": f"{unstratified[1]:.3f}",
        "resTransferInterceptSweepHiMean": f"{sweep['upper_mean']:.3f}",
        "resTransferInterceptSweepHiSD": f"{sweep['upper_sd']:.3f}",
        "resTransferInterceptSweepSeeds": f"{N_SWEEP_SEEDS}",
        "resTransferInterceptSweepAbove": f"{sweep['n_above_band']}",
        "resTransferInterceptWaldHi": f"{wald[1]:.3f}",
    }
    tex_lines = [
        "% Generated by scripts/supplementary/transfer_intercept_sensitivity.py; "
        "do not edit by hand.",
        "% Post hoc sensitivity of the transfer calibration-intercept interval.",
        *(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()),
    ]
    tex_path = output_dir / "transfer_intercept_sensitivity_tex.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    table = Table(title=f"Transfer calibration intercept (point {point:+.4f})")
    table.add_column("Scheme", style="bold")
    table.add_column("95 % interval", justify="right")
    table.add_column(f"Inside [{band_low}, {band_high}]")
    for name, (low, high) in (
        ("stratified (reported)", stratified),
        ("unstratified", unstratified),
        ("Wald", wald),
    ):
        inside = band_low <= low and high <= band_high
        table.add_row(name, f"[{low:+.4f}, {high:+.4f}]", "yes" if inside else "no")
    console.print(table)
    console.print(
        f"Unstratified upper limit over {N_SWEEP_SEEDS} seeds: mean "
        f"{sweep['upper_mean']:.4f}, SD {sweep['upper_sd']:.4f}, "
        f"{sweep['n_above_band']} above {band_high}"
    )
    console.print(f"[green]Wrote {json_path}")
    console.print(f"[green]Wrote {tex_path}")


if __name__ == "__main__":
    main()
