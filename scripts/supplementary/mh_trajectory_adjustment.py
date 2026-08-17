"""Sharpened persistence sensitivity using the full T0/T1/T2 trajectory.

The persistence adjustment (``persistence_adjustment.py``) adjusts the
baseline-MH effect for current MH at Corona-2 only. With the
``mh_longitudinal.parquet`` (built by the longitudinal MH extractor) we can
adjust for the full mental-health trajectory across all three NAKO
waves jointly: the baseline phenotype (T0, 2014..2019), the Corona-1
mail-in questionnaire (T1, 2020), and the Corona-2 follow-up
(T2, 2022..2023).

This refines the persistence-adjustment finding: the baseline-MH first-
hit retains predictive power for PCC even after partialing out the
intermediate (T1) and concurrent (T2) mental-health states. If the
T0 odds-ratio collapses once T1+T2 are entered, the baseline signal
is mostly persistent-state; if it survives, the pre-pandemic
phenotype carries information that neither pandemic-era wave
captures.

Exposure definitions (consistent with the persistence and E-value analyses):
- T0 baseline-MH-positive: composite PHQ-9 sum >= 10 OR GAD-7 sum
  >= 10 OR MINI major depression == 1 (from mental_health.parquet —
  carries MINI which is a baseline-only instrument).
- T1 / T2 MH-positive: composite PHQ-9 sum >= 10 OR GAD-7 sum >= 10
  (from mh_longitudinal.parquet, wave=T1 or T2 — MINI not collected
  at follow-up waves).

Confounders: age (baseline), sex, study centre — same as the persistence and E-value analyses.

Usage:
    uv run python scripts/supplementary/mh_trajectory_adjustment.py
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pcc_analysis.config import get_paper_constants_dir, get_processed_dir
from pcc_analysis.data_manager import load_pipeline_data
from pcc_analysis.sensitivity import (
    build_confounder_design,
    composite_mh_positive,
    fit_logit,
    is_estimable,
    odds_ratios_from_fit,
)

if TYPE_CHECKING:
    from pcc_analysis.sensitivity import OddsRatio

console = Console()
logger = logging.getLogger(__name__)


def _try_odds_ratios(
    model: str, y: pd.Series, design: pd.DataFrame, terms: list[str]
) -> dict[str, OddsRatio] | None:
    """Odds ratios for one model of the trajectory, or ``None`` if none exist.

    The three models are three designs on three sets of complete rows, and
    rank is a property of the design: the T0-only model can identify on a
    sample where the full trajectory model does not, so each is guarded on
    its own rather than the panel as a whole. Inside one model there is
    nothing partial to salvage — every term is read off the same fit, so the
    T0, T1 and T2 coefficients stand or fall together.

    Three ways to fail, and each arrives differently. A design the sample
    cannot identify makes the Hessian singular and its inversion throw, which
    :func:`fit_logit` reports as ``None``. Waves that do not overlap — no
    participant carrying both the Corona-1 and the Corona-2 answer on top of
    the confounders, which a longitudinal extract covering a different
    delivery than the pipeline data produces — leave no complete row at all
    once ``missing="drop"`` has run, and an empty sample raises out of the
    estimator's constructor before a coefficient is ever computed. Perfect
    separation and a stalled optimiser, finally, merely warn and hand back a
    fit whose coefficients sit at the optimiser's boundary with standard
    errors overflowed to infinity; an odds ratio read off that looks like a
    number and means nothing, so :func:`is_estimable` is consulted alongside
    the missing fit.

    All three are reported rather than raised on, because one panel that the
    sample cannot carry must not take down the supplementary steps that run
    after this one, and the JSON says so in the same ``estimable`` field the
    severity interaction writes rather than leaving a NaN behind.
    """
    try:
        result = fit_logit(y, design)
    except ValueError as error:
        logger.warning(
            "Model %s has no sample to fit on: %s. No odds ratios for %s, "
            "reported as not estimable.",
            model,
            error,
            ", ".join(terms),
        )
        return None
    if result is None or not is_estimable(result):
        logger.warning(
            "Model %s does not identify on this sample: no odds ratios for "
            "%s, reported as not estimable.",
            model,
            ", ".join(terms),
        )
        return None
    return odds_ratios_from_fit(result, terms)


def _shrinkage(
    baseline: dict[str, OddsRatio] | None, adjusted: dict[str, OddsRatio] | None
) -> float | None:
    """Fractional shrinkage of the T0 log-odds coefficient between two models.

    Computed on the log-odds scale rather than from the odds ratios, since a
    ratio of exponentiated estimates is not the shrinkage of the coefficient.
    ``None`` where either model is missing: a shrinkage is a contrast between
    two fits, so one unestimable end leaves no half of it to report.
    """
    if baseline is None or adjusted is None:
        return None
    coef_baseline = baseline["mh_t0_pos"].log_odds
    coef_adjusted = adjusted["mh_t0_pos"].log_odds
    if coef_baseline == 0:
        return float("nan")
    return (coef_baseline - coef_adjusted) / coef_baseline


def _or_cell(
    ratios: dict[str, OddsRatio] | None, term: str, *, interval: bool = True
) -> str:
    """One term's odds ratio, or the marker that its model has none.

    The T2 term of model B is reported as a point estimate, which is what the
    JSON carries for it, so the interval is a switch here rather than a
    second formatter.
    """
    if ratios is None:
        return "[yellow]not estimable"
    ratio = ratios[term]
    if not interval:
        return f"{ratio.point:.2f}"
    return f"{ratio.point:.2f} [{ratio.ci_lo:.2f}, {ratio.ci_hi:.2f}]"


def _n_cell(ratios: dict[str, OddsRatio] | None) -> str:
    """Rows the model was fitted on, which an unfitted model does not have.

    ``missing="drop"`` decides the sample inside the fit, so the count is a
    property of the fit and not of the design handed to it.
    """
    return "—" if ratios is None else f"{ratios['mh_t0_pos'].n}"


def _shrink_cell(shrinkage: float | None) -> str:
    """Shrinkage as a percentage, or the marker that one end of it is missing."""
    return "[yellow]not estimable" if shrinkage is None else f"{shrinkage * 100:.1f}%"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]Baseline-MH OR adjusted for the T0/T1/T2 trajectory")

    y, X, conf = load_pipeline_data(
        target="bahmer", clean_controls=True, cohort="mri", stack_variant="full"
    )
    mh_baseline = X["mental_health"]

    t0_pos = composite_mh_positive(mh_baseline).rename("mh_t0_pos")

    long_path = get_processed_dir() / "mh_longitudinal.parquet"
    long_df = pd.read_parquet(long_path)
    # Pivot to wide form: ID -> mh_positive at T1, T2
    t1_pos = (
        long_df[long_df["wave"] == "T1"]
        .set_index("ID")["mh_positive"]
        .reindex(y.index)
        .astype("Float64")
        .rename("mh_t1_pos")
    )
    t2_pos = (
        long_df[long_df["wave"] == "T2"]
        .set_index("ID")["mh_positive"]
        .reindex(y.index)
        .astype("Float64")
        .rename("mh_t2_pos")
    )

    conf_frame = build_confounder_design(conf)

    # --- Model A: T0 only (= persistence baseline-only) ----------------
    X_a = pd.concat([t0_pos.astype(float).rename("mh_t0_pos"), conf_frame], axis=1)
    fit_a = _try_odds_ratios("A (T0 only)", y, X_a, ["mh_t0_pos"])

    # --- Model B: T0 + T2 (= persistence-adjusted) ---------------------
    X_b = pd.concat(
        [
            t0_pos.astype(float).rename("mh_t0_pos"),
            t2_pos.astype(float).rename("mh_t2_pos"),
            conf_frame,
        ],
        axis=1,
    )
    fit_b = _try_odds_ratios("B (T0 + T2)", y, X_b, ["mh_t0_pos", "mh_t2_pos"])

    # --- Model C: T0 + T1 + T2 (full trajectory adjustment) ------------
    X_c = pd.concat(
        [
            t0_pos.astype(float).rename("mh_t0_pos"),
            t1_pos.astype(float).rename("mh_t1_pos"),
            t2_pos.astype(float).rename("mh_t2_pos"),
            conf_frame,
        ],
        axis=1,
    )
    fit_c = _try_odds_ratios(
        "C (T0 + T1 + T2 trajectory)",
        y,
        X_c,
        ["mh_t0_pos", "mh_t1_pos", "mh_t2_pos"],
    )

    shrink_persistence = _shrinkage(fit_a, fit_b)
    shrink_traj = _shrinkage(fit_a, fit_c)

    model_a: dict[str, Any] = {"estimable": fit_a is not None}
    if fit_a is not None:
        t0_a = fit_a["mh_t0_pos"]
        model_a |= {
            "or_t0": t0_a.point,
            "ci_95_t0": [t0_a.ci_lo, t0_a.ci_hi],
            "p_t0": t0_a.p_value,
            "n": t0_a.n,
        }

    model_b: dict[str, Any] = {"estimable": fit_b is not None}
    if fit_b is not None:
        t0_b, t2_b = fit_b["mh_t0_pos"], fit_b["mh_t2_pos"]
        model_b |= {
            "or_t0": t0_b.point,
            "ci_95_t0": [t0_b.ci_lo, t0_b.ci_hi],
            "p_t0": t0_b.p_value,
            "or_t2": t2_b.point,
            "n": t0_b.n,
        }

    model_c: dict[str, Any] = {"estimable": fit_c is not None}
    if fit_c is not None:
        t0_c, t1_c, t2_c = fit_c["mh_t0_pos"], fit_c["mh_t1_pos"], fit_c["mh_t2_pos"]
        model_c |= {
            "or_t0": t0_c.point,
            "ci_95_t0": [t0_c.ci_lo, t0_c.ci_hi],
            "p_t0": t0_c.p_value,
            "or_t1": t1_c.point,
            "ci_95_t1": [t1_c.ci_lo, t1_c.ci_hi],
            "or_t2": t2_c.point,
            "ci_95_t2": [t2_c.ci_lo, t2_c.ci_hi],
            "n": t0_c.n,
        }

    # The headline of this analysis is a contrast between models A and C, so
    # it carries its own marker: either end going unestimable leaves the
    # shrinkage undefined even where the other model is fine.
    shrinkage: dict[str, Any] = {"estimable": shrink_traj is not None}
    if fit_a is not None and fit_c is not None:
        shrinkage |= {
            "logit_coef_t0_only": fit_a["mh_t0_pos"].log_odds,
            "logit_coef_t0_trajectory_adjusted": fit_c["mh_t0_pos"].log_odds,
            "shrink_ratio": shrink_traj,
        }

    summary: dict[str, Any] = {
        "exposure_definitions": {
            "mh_t0_pos": "PHQ-9 >= 10 OR GAD-7 >= 10 OR MINI major depression (baseline 2014..2019)",
            "mh_t1_pos": "PHQ-9 >= 10 OR GAD-7 >= 10 (Corona-1 questionnaire 2020, derived from raw items)",
            "mh_t2_pos": "PHQ-9 >= 10 OR GAD-7 >= 10 (Corona-2 follow-up 2022..2023)",
        },
        "model_a_t0_only": model_a,
        "model_b_t0_t2": model_b,
        "model_c_t0_t1_t2_trajectory": model_c,
        "trajectory_shrinkage_baseline": shrinkage,
    }

    out_path = get_paper_constants_dir() / "mh_trajectory_adjustment.json"
    out_path.write_text(json.dumps(summary, indent=2))

    tbl = Table(
        title="Baseline-MH OR under T0-only / +T2 / full T0+T1+T2 trajectory",
        show_header=True,
    )
    tbl.add_column("Model", style="bold")
    tbl.add_column("N", justify="right")
    tbl.add_column("T0 OR [95% CI]", justify="right")
    tbl.add_column("T1 OR [95% CI]", justify="right")
    tbl.add_column("T2 OR [95% CI]", justify="right")
    tbl.add_column("Shrink", justify="right")
    # A dash marks a term the model does not contain; the not-estimable
    # marker marks one it contains but the sample cannot determine.
    tbl.add_row(
        "A: T0 only",
        _n_cell(fit_a),
        _or_cell(fit_a, "mh_t0_pos"),
        "—",
        "—",
        "—",
    )
    tbl.add_row(
        "B: T0 + T2",
        _n_cell(fit_b),
        _or_cell(fit_b, "mh_t0_pos"),
        "—",
        _or_cell(fit_b, "mh_t2_pos", interval=False),
        _shrink_cell(shrink_persistence),
    )
    tbl.add_row(
        "C: T0 + T1 + T2 (trajectory)",
        _n_cell(fit_c),
        _or_cell(fit_c, "mh_t0_pos"),
        _or_cell(fit_c, "mh_t1_pos"),
        _or_cell(fit_c, "mh_t2_pos"),
        _shrink_cell(shrink_traj),
    )
    console.print(tbl)
    console.print(f"[green]Wrote {out_path}")


if __name__ == "__main__":
    main()
