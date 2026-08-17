"""Construct-persistence sensitivity: adjust baseline-MH OR for current depression.

The concern is that the baseline-mental-health → PCC association in the
primary analysis may in part reflect participants whose baseline psychiatric
phenotype persisted into the Corona-2 assessment, so that "baseline MH"
is partially a proxy for "currently depressed at Corona-2", the latter of
which could inflate symptom reporting (reporter-bias path) rather than
reflect a true pre-infection first-hit. To bound this, we refit the
baseline-MH logistic regression with a second MH indicator constructed
from the PHQ-9 and GAD-7 items administered at Corona-2.

Operationalisations, aligned with the E-value script (evalue.py):
- Baseline MH-positive: PHQ-9 sum >= 10 OR GAD-7 sum >= 10 OR MINI major
  depression == 1 (from the baseline mental-health parquet).
- Current MH-positive: PHQ-9 (d_co2_phq9_d1..d9) sum >= 10 OR GAD-7
  (d_co2_gad7_d10..d16) sum >= 10. Raw items are coded 1..4 in the NAKO
  export; we subtract 1 to obtain the canonical 0..3 scoring before
  summing, and treat the sentinel 8888 as missing.
- Confounders: age, sex, study centre (identical to the E-value analysis).

Usage:
    uv run python scripts/supplementary/persistence_adjustment.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pcc_analysis.config import get_paper_constants_dir, load_config
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

PHQ9_ITEMS = [f"d_co2_phq9_d{i}" for i in range(1, 10)]
GAD7_ITEMS = [f"d_co2_gad7_d{i}" for i in range(10, 17)]
MISSING_SENTINEL = 8888

NOT_ESTIMABLE = "[yellow]not estimable"
"""What the table prints where a quantity has no value to print.

Kept out of the numeric columns deliberately: a dash reads as a quantity
that does not apply — as it does in the columns Model A has no term for —
while a model that could not be fitted is a quantity that is missing.
"""


def load_current_mh_positive() -> pd.Series:
    """Return Corona-2 MH-positive indicator (PHQ9>=10 or GAD7>=10) by ID."""
    cfg = load_config()
    paths = cfg["paths"]
    csv_path = (
        Path(str(paths["nako_data_dir"])).expanduser() / paths["files"]["corona2"]
    )
    cols = ["ID", *PHQ9_ITEMS, *GAD7_ITEMS]
    raw = pd.read_csv(csv_path, sep=";", usecols=cols, low_memory=False)
    raw = raw.set_index("ID")

    def score(items: list[str]) -> pd.Series:
        sub = raw[items].replace(MISSING_SENTINEL, np.nan)
        sub = sub - 1  # NAKO codes 1..4, canonical PHQ/GAD are 0..3
        # If any item is missing, return NaN (conservative — don't impute).
        return sub.sum(axis=1, skipna=False)

    phq_sum = score(PHQ9_ITEMS)
    gad_sum = score(GAD7_ITEMS)
    # Participants with both scales missing: leave NaN.
    missing_all = phq_sum.isna() & gad_sum.isna()
    current_pos = ((phq_sum >= 10) | (gad_sum >= 10)).astype("Int64").mask(missing_all)
    return current_pos.rename("current_mh_pos")


def _try_odds_ratios(
    model: str, y: pd.Series, design: pd.DataFrame, terms: list[str]
) -> dict[str, OddsRatio] | None:
    """Fit, or report that this sample cannot carry the model.

    Both models here adjust for study centre, so a sample thin enough to
    leave a centre holding a handful of participants makes the design
    rank-deficient and the coefficients unidentified. That is a statement
    about the achievable precision and is reported as one. Dropping the
    centre adjustment to make the fit succeed would quietly answer a
    different question than the E-value model this is meant to be compared
    with, and letting the failure propagate would take out the supplementary
    steps queued behind this one.

    Three ways to arrive at no usable estimate, and they surface
    differently. Rank deficiency throws out of the Hessian inversion, which
    :func:`fit_logit` turns into ``None``; separation and non-convergence
    hand back a fitted object all the same, one whose standard errors have
    overflowed or whose optimiser never settled, and only
    :func:`is_estimable` tells those apart from a result; and a design with
    no complete rows raises out of the estimator before either applies. The
    last is reachable here rather than hypothetical, because the current-MH
    indicator is read from the raw Corona-2 export and merged on ID: an
    export that no longer overlaps the analytic index — stale, or
    re-pseudonymised since — leaves Model B with nothing to fit. The silent
    modes are the dangerous ones, since an odds ratio read off such a fit
    looks like a number and carries no information.
    """
    try:
        result = fit_logit(y, design)
    except ValueError as error:
        logger.warning("%s cannot be fitted: %s", model, error)
        return None
    if result is None or not is_estimable(result):
        logger.warning(
            "%s does not identify on the %d participants offered to it; its "
            "odds ratios are reported as not estimable rather than as numbers.",
            model,
            len(design),
        )
        return None
    return odds_ratios_from_fit(result, terms)


def _odds_ratio_cell(model: OddsRatio | None) -> str:
    """One odds ratio with its interval, or the marker where none exists."""
    if model is None:
        return NOT_ESTIMABLE
    return f"{model.point:.2f} [{model.ci_lo:.2f}, {model.ci_hi:.2f}]"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]Construct-persistence adjustment for baseline-MH → PCC")

    y, X, conf = load_pipeline_data(
        target="bahmer", clean_controls=True, cohort="mri", stack_variant="full"
    )
    mh = X["mental_health"]
    baseline_pos = composite_mh_positive(mh).rename("baseline_mh_pos")

    current_pos = load_current_mh_positive()
    current_aligned = current_pos.reindex(baseline_pos.index).astype("Float64")

    conf_frame = build_confounder_design(conf)

    # --- Model A: baseline-only (the E-value primary model) --------------
    X_a = pd.concat(
        [baseline_pos.astype(float).rename("baseline_mh_pos"), conf_frame], axis=1
    )
    fit_a = _try_odds_ratios("Model A (baseline-only)", y, X_a, ["baseline_mh_pos"])
    model_a = None if fit_a is None else fit_a["baseline_mh_pos"]

    # --- Model B: + current-MH adjustment --------------------------------
    X_b = pd.concat(
        [
            baseline_pos.astype(float).rename("baseline_mh_pos"),
            current_aligned.astype(float).rename("current_mh_pos"),
            conf_frame,
        ],
        axis=1,
    )
    fit_b = _try_odds_ratios(
        "Model B (persistence-adjusted)",
        y,
        X_b,
        ["baseline_mh_pos", "current_mh_pos"],
    )
    base_b = None if fit_b is None else fit_b["baseline_mh_pos"]
    cur_b = None if fit_b is None else fit_b["current_mh_pos"]

    # Coefficient shrinkage on the log-odds scale ------------------------
    #
    # The adjustment is a comparison of two coefficients, so it exists only
    # where both models do: with one side missing there is nothing to shrink
    # against, and the surviving coefficient on its own answers a different
    # question. A baseline coefficient of exactly zero leaves the ratio
    # undefined rather than merely imprecise, which is again a missing number
    # and not a zero one.
    shrinkage: dict[str, Any] = {
        "estimable": model_a is not None and base_b is not None
    }
    shrink_ratio: float | None = None
    if model_a is not None and base_b is not None:
        coef_a, coef_b = model_a.log_odds, base_b.log_odds
        shrink_ratio = (coef_a - coef_b) / coef_a if coef_a != 0 else None
        shrinkage |= {
            "logit_coef_baseline_only": coef_a,
            "logit_coef_persistence_adjusted": coef_b,
            "shrink_ratio": shrink_ratio,
        }

    # Diagnostics on the joint sample (Model B drops rows missing current-MH)
    n_a = None if model_a is None else model_a.n
    n_b = None if base_b is None else base_b.n
    n_current_missing = int(current_aligned.isna().sum())

    summary = {
        "sample": {
            "n_baseline_model": n_a,
            "n_persistence_model": n_b,
            "n_current_mh_missing": n_current_missing,
        },
        "exposure_definitions": {
            "baseline_mh_pos": "PHQ-9 sum >= 10 OR GAD-7 sum >= 10 OR MINI major depression == 1 (baseline 2014--2019).",
            "current_mh_pos": (
                "Corona-2 PHQ-9 sum >= 10 OR GAD-7 sum >= 10; items coded 1..4 in NAKO "
                "export, rescaled by -1 to obtain 0..3 scoring; 8888 treated as missing."
            ),
        },
        "model_a_baseline_only": {
            "estimable": model_a is not None,
            **(
                {
                    "or": model_a.point,
                    "ci_95": [model_a.ci_lo, model_a.ci_hi],
                    "p_value": model_a.p_value,
                    "n": model_a.n,
                }
                if model_a is not None
                else {}
            ),
        },
        # Both terms come off the same fit, so the model is estimable or it
        # is not; there is no case where one of the two coefficients exists.
        "model_b_persistence_adjusted": {
            "estimable": base_b is not None,
            **(
                {
                    "or_baseline_mh": base_b.point,
                    "ci_95_baseline_mh": [base_b.ci_lo, base_b.ci_hi],
                    "p_value_baseline_mh": base_b.p_value,
                    "or_current_mh": cur_b.point,
                    "ci_95_current_mh": [cur_b.ci_lo, cur_b.ci_hi],
                    "p_value_current_mh": cur_b.p_value,
                    "n": base_b.n,
                }
                if base_b is not None and cur_b is not None
                else {}
            ),
        },
        "coefficient_shrinkage": shrinkage,
    }

    out_path = get_paper_constants_dir() / "persistence_adjustment.json"
    out_path.write_text(json.dumps(summary, indent=2))

    tbl = Table(title="Persistence adjustment for baseline-MH → PCC", show_header=True)
    tbl.add_column("Model", style="bold")
    tbl.add_column("N", justify="right")
    tbl.add_column("Baseline-MH OR [95% CI]", justify="right")
    tbl.add_column("Current-MH OR [95% CI]", justify="right")
    tbl.add_column("Log-odds shrinkage", justify="right")
    tbl.add_row(
        "A: baseline-only (+ age/sex/centre)",
        f"{n_a}" if n_a is not None else "—",
        _odds_ratio_cell(model_a),
        "—",
        "—",
    )
    tbl.add_row(
        "B: + current-MH adjustment",
        f"{n_b}" if n_b is not None else "—",
        _odds_ratio_cell(base_b),
        _odds_ratio_cell(cur_b),
        f"{shrink_ratio * 100:.1f}%" if shrink_ratio is not None else NOT_ESTIMABLE,
    )
    console.print(tbl)
    console.print(f"[green]Wrote {out_path}")


if __name__ == "__main__":
    main()
