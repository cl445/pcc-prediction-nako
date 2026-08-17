"""Compute an E-value for the Baseline-Mental-Health → PCC association.

Fits a logistic regression of the Bahmer PCC outcome on a dichotomous
baseline mental-health indicator (MH-positive = PHQ-9 sum >= 10 OR
GAD-7 sum >= 10 OR MINI major depression == 1), adjusted for age, sex,
and study center, on the clean-controls MRI analytic sample (the same
sample reported as the paper's primary analysis).

The VanderWeele-Ding E-value quantifies the minimum strength of
association (on the risk-ratio scale) that an unmeasured confounder
would need to have, jointly with both the exposure and the outcome,
to fully explain away the observed effect. For binary outcomes we
first approximate the risk ratio from the adjusted odds ratio using
the Zhang-Yu common-outcome correction (preferred over the minimax
square-root approximation of VanderWeele 2020 because p0 is observed
in our sample):

    RR_approx = OR / (1 - p0 + p0 * OR)           (Zhang & Yu 1998)

where ``p0`` is the outcome probability in the unexposed. Then

    E-value = RR + sqrt(RR * (RR - 1))            (VanderWeele & Ding 2017)

Every quantity here descends from one of two fits, so a sample
that cannot identify them leaves nothing to publish: the constants file
then carries an explicit not-estimable marker in place of the numbers,
never a placeholder that reads like one. A bound derived from a boundary
coefficient would state that no confounder at all is needed to explain the
association away, which is the most reassuring way of reporting nothing.

Usage:
    uv run python scripts/supplementary/evalue.py
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pcc_analysis.config import get_paper_constants_dir
from pcc_analysis.data_manager import load_pipeline_data
from pcc_analysis.sensitivity import (
    build_confounder_design,
    composite_mh_positive,
    e_value_from_rr,
    fit_logit,
    is_estimable,
    odds_ratios_from_fit,
    or_to_rr,
)

if TYPE_CHECKING:
    from pcc_analysis.sensitivity import OddsRatio

console = Console()
logger = logging.getLogger(__name__)


EXPOSURE = "mh_positive"


def _try_fit(y: pd.Series, design: pd.DataFrame, model: str) -> Any | None:
    """Fit, or report that this sample cannot carry the model.

    Two ways a logistic fit fails to identify, and only one of them raises.
    A design the sample does not determine — study-center dummies holding a
    single participant each, say — makes the Hessian singular, and its
    inversion throws; :func:`fit_logit` intercepts that and has no fit to
    hand back. Perfect separation and a stalled optimiser are the quiet ones:
    statsmodels returns a fitted object whose coefficients have run to the
    optimiser's boundary and whose standard errors have overflowed, so the
    odds ratio read off it prints as ``0.00 [0.00, inf]`` and means nothing.
    That second kind is what :func:`is_estimable` is for, and a fit is
    returned only where both judgements agree one exists.

    Reported rather than raised on, because this script is one link in a
    chain of supplementary analyses and an unestimable panel must not take
    down the ones that run after it.
    """
    result = fit_logit(y, design)
    if result is not None and is_estimable(result):
        return result
    logger.warning(
        "The %s model does not identify on %d participants (%d PCC cases); "
        "it is reported as not estimable rather than as a number.",
        model,
        len(y),
        int(y.sum()),
    )
    return None


def _unadjusted_odds_ratio(fit: Any) -> dict[str, Any]:
    """Odds ratio and interval read off the unadjusted sanity check.

    Read off the fitted object rather than through
    :func:`odds_ratios_from_fit`, whose intervals use a rounded 1.96 where
    ``conf_int`` uses 1.959964. The two differ in the fifth decimal, well
    below anything this file reports, and the constants written here follow
    the statsmodels convention.
    """
    ci = np.exp(fit.conf_int().loc[EXPOSURE]).tolist()
    return {
        "point": float(np.exp(fit.params[EXPOSURE])),
        "ci_95": [float(ci[0]), float(ci[1])],
    }


def _bound_confounding(adjusted: OddsRatio, p_unexposed: float) -> dict[str, Any]:
    """Approximate risk ratios and the E-values they imply.

    Both are functions of the same two numbers — the adjusted odds ratio and
    its lower confidence limit — so they exist exactly where that one fit
    does, and are derived in one place rather than separately per payload
    block.
    """
    rr_point = or_to_rr(adjusted.point, p_unexposed)
    rr_lo = or_to_rr(adjusted.ci_lo, p_unexposed)
    return {
        "risk_ratio": {"point": rr_point, "ci_lower": rr_lo},
        "e_value": {
            "point": e_value_from_rr(rr_point),
            "ci_lower": e_value_from_rr(rr_lo),
        },
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]E-value for baseline MH → PCC")

    y, X, conf = load_pipeline_data(
        target="bahmer", clean_controls=True, cohort="mri", stack_variant="full"
    )
    mh = X["mental_health"]

    exposure = composite_mh_positive(mh)
    n_total = len(y)
    n_exposed = int(exposure.sum())
    n_outcome = int(y.sum())
    prevalence_overall = float(y.mean())

    # Stratum-specific outcome probabilities (for narrative and for p0 in RR approximation).
    y_exposed = y[exposure == 1]
    y_unexposed = y[exposure == 0]
    p_exposed = float(y_exposed.mean())
    p_unexposed = float(y_unexposed.mean())

    # --- Unadjusted OR (sanity check) ------------------------------------
    unadjusted_fit = _try_fit(
        y, pd.DataFrame({EXPOSURE: exposure.astype(float)}), "unadjusted"
    )
    unadjusted = (
        None if unadjusted_fit is None else _unadjusted_odds_ratio(unadjusted_fit)
    )

    # --- Adjusted OR -----------------------------------------------------
    design = pd.concat(
        [exposure.astype(float).rename(EXPOSURE), build_confounder_design(conf)],
        axis=1,
    )
    adjusted_fit = _try_fit(y, design, "age-, sex- and center-adjusted")
    adjusted = (
        None
        if adjusted_fit is None
        else odds_ratios_from_fit(adjusted_fit, [EXPOSURE])[EXPOSURE]
    )

    # --- RR approximation + E-value --------------------------------------
    # The whole bounding argument rests on the adjusted fit: with no odds
    # ratio there is no risk ratio to approximate and no E-value to state.
    bounded = None if adjusted is None else _bound_confounding(adjusted, p_unexposed)

    # --- Emit --------------------------------------------------------------
    summary = {
        "sample": {
            "cohort": "mri",
            "clean_controls": True,
            "n_total": n_total,
            "n_exposed_mh_positive": n_exposed,
            "n_outcome_pcc": n_outcome,
            "prevalence_overall": prevalence_overall,
            "prevalence_exposed": p_exposed,
            "prevalence_unexposed": p_unexposed,
        },
        "exposure_definition": (
            "Composite baseline mental-health positivity: PHQ-9 sum >= 10 "
            "OR GAD-7 sum >= 10 OR MINI major depression == 1 "
            "(missing components treated as non-positive)."
        ),
        "odds_ratio": {
            "unadjusted": {
                "estimable": unadjusted is not None,
                **(unadjusted if unadjusted is not None else {}),
            },
            "adjusted_age_sex_center": {
                "estimable": adjusted is not None,
                **(
                    {
                        "point": adjusted.point,
                        "ci_95": [adjusted.ci_lo, adjusted.ci_hi],
                        "p_value": adjusted.p_value,
                    }
                    if adjusted is not None
                    else {}
                ),
            },
        },
        "risk_ratio_approx": {
            "method": "Zhang & Yu 1998 common-outcome OR→RR: RR = OR / (1 - p0 + p0*OR)",
            "p0_unexposed_outcome_probability": p_unexposed,
            "estimable": bounded is not None,
            **(bounded["risk_ratio"] if bounded is not None else {}),
        },
        "e_value": {
            "method": "VanderWeele & Ding 2017: E = RR + sqrt(RR * (RR - 1))",
            "estimable": bounded is not None,
            **(bounded["e_value"] if bounded is not None else {}),
        },
    }

    out_path = get_paper_constants_dir() / "e_value.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Pretty-print
    tbl = Table(
        title="Baseline-MH → PCC, adjusted for age/sex/center", show_header=True
    )
    tbl.add_column("Quantity", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("N (analytic sample)", f"{n_total}")
    tbl.add_row(
        "MH-positive (exposed)", f"{n_exposed} ({100 * n_exposed / n_total:.1f}%)"
    )
    tbl.add_row("PCC prevalence (overall)", f"{prevalence_overall:.3f}")
    tbl.add_row("PCC | MH-positive", f"{p_exposed:.3f}")
    tbl.add_row("PCC | MH-negative", f"{p_unexposed:.3f}")
    tbl.add_row(
        "OR unadjusted",
        f"{unadjusted['point']:.2f} "
        f"[{unadjusted['ci_95'][0]:.2f}, {unadjusted['ci_95'][1]:.2f}]"
        if unadjusted is not None
        else "[yellow]not estimable",
    )
    tbl.add_row(
        "OR adjusted",
        f"{adjusted.point:.2f} [{adjusted.ci_lo:.2f}, {adjusted.ci_hi:.2f}]  "
        f"(p={adjusted.p_value:.2g})"
        if adjusted is not None
        else "[yellow]not estimable",
    )
    if bounded is not None:
        risk_ratio = bounded["risk_ratio"]
        e_value = bounded["e_value"]
        tbl.add_row(
            "Approx RR",
            f"{risk_ratio['point']:.2f} [lower={risk_ratio['ci_lower']:.2f}]",
        )
        tbl.add_row("E-value (point)", f"{e_value['point']:.2f}")
        tbl.add_row("E-value (CI lower)", f"{e_value['ci_lower']:.2f}")
    else:
        tbl.add_row("Approx RR", "[yellow]not estimable")
        tbl.add_row("E-value (point)", "[yellow]not estimable")
        tbl.add_row("E-value (CI lower)", "[yellow]not estimable")
    console.print(tbl)
    console.print(f"[green]Wrote {out_path}")


if __name__ == "__main__":
    main()
