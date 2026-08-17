"""Decompose the Bahmer PCC symptom set into hard-PCC vs MH-overlap items.

The construct-overlap concern (Greissel 2024) is that
the baseline-mental-health → PCC association may be partly an artefact of
symptom overlap: PHQ-9 and GAD-7 include fatigue, sleep disturbance, and
concentration difficulty, which are also prominent PCC symptoms. A
baseline-depressed participant may endorse the same somatic items again
at Corona-2 follow-up not because of COVID-19 sequelae but because their
depressive/anxious phenotype generates those items directly.

Design:
  - OVERLAP symptoms (map onto DSM-5 MDD/GAD somatic criteria):
        fatigue, concentration_problems, memory_problems, sleep_problems,
        loss_of_appetite.
  - HARD-PCC symptoms (mechanism-aligned with COVID pathophysiology,
    low construct overlap with MDD/GAD):
        loss_of_smell, loss_of_taste, fever, runny_nose, cough,
        breathing_problems, chest_tightness, heart_problems,
        circulation_problems, nerve_problems, hair_loss,
        joint_muscle_pain, gastrointestinal_problems, sweating,
        reduced_physical_capacity, headache.

For each symptom set we build two secondary binary outcomes on the
clean-controls MRI analytic sample:
  - ``hard_pcc_any`` = 1 iff >=1 hard-PCC symptom endorsed.
  - ``overlap_any`` = 1 iff >=1 overlap symptom endorsed.

We then fit logistic regressions of each outcome on baseline-MH-positive
(same composite as A7: PHQ-9 sum >= 10 OR GAD-7 sum >= 10 OR MINI major
depression), adjusted for age, sex, and study centre. If the baseline-MH
odds ratio is substantially larger for OVERLAP than for HARD-PCC, the
construct-overlap concern is partly supported; if the two odds ratios
are of similar magnitude, the baseline-MH → PCC effect cannot be
reduced to construct overlap.

Usage:
    uv run python scripts/supplementary/symptom_decomposition.py
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pcc_analysis.config import get_paper_constants_dir, get_processed_dir
from pcc_analysis.data_manager import load_pipeline_data
from pcc_analysis.sensitivity import (
    OddsRatio,
    build_confounder_design,
    composite_mh_positive,
    fit_logit,
    is_estimable,
    odds_ratios_from_fit,
)

console = Console()
logger = logging.getLogger(__name__)

EXPOSURE = "baseline_mh_pos"

OVERLAP_SYMPTOMS = [
    "symptom_fatigue",
    "symptom_concentration_problems",
    "symptom_memory_problems",
    "symptom_sleep_problems",
    "symptom_loss_of_appetite",
]
HARD_PCC_SYMPTOMS = [
    "symptom_loss_of_smell",
    "symptom_loss_of_taste",
    "symptom_fever",
    "symptom_runny_nose",
    "symptom_cough",
    "symptom_breathing_problems",
    "symptom_chest_tightness",
    "symptom_heart_problems",
    "symptom_circulation_problems",
    "symptom_nerve_problems",
    "symptom_hair_loss",
    "symptom_joint_muscle_pain",
    "symptom_gastrointestinal_problems",
    "symptom_sweating",
    "symptom_reduced_physical_capacity",
    "symptom_headache",
]


def _as_dict(model: OddsRatio | None) -> dict[str, Any]:
    """One term of a fitted model, or the marker that no fit exists.

    The estimates are absent rather than null where the model does not
    identify, so a consumer reading ``["or"]`` off it raises instead of
    carrying a missing number into a table as if it were one.
    """
    if model is None:
        return {"estimable": False}
    return {
        "estimable": True,
        "or": model.point,
        "ci_95": [model.ci_lo, model.ci_hi],
        "p_value": model.p_value,
        "n": model.n,
    }


def mh_odds_ratio(
    outcome: str, y: pd.Series, exposure: pd.Series, conf_frame: pd.DataFrame
) -> OddsRatio | None:
    """Baseline-MH odds ratio for one decomposed outcome, age/sex/centre-adjusted.

    ``None`` where the sample does not identify the design. The adjustment
    set carries a dummy for every study centre but one, so a sample thin
    enough to leave centres holding a handful of participants each is
    rank-deficient and its Hessian singular. That is a statement about the
    precision this sample can reach, and it is reported as one: an unfittable
    outcome must not take down the supplementary steps queued behind this
    script, and it must not reach the constants file as a number either.

    The decision is taken per outcome. The three are different outcomes on
    the same participants — each endorsed by around a quarter of them — so
    one failing to identify says nothing about the other two, and a hard-PCC
    estimate is still worth having when the overlap one cannot be had.

    Two failure modes, and only one of them raises. Rank deficiency throws
    out of the Hessian inversion, and :func:`fit_logit` hands back nothing at
    all; perfect separation merely warns and hands back a fit whose standard
    errors have overflowed to infinity. The silent one is the dangerous one,
    since an odds ratio read off it looks like a number and means nothing, so
    :func:`is_estimable` is consulted alongside the missing fit.
    """
    design = pd.concat([exposure.astype(float).rename(EXPOSURE), conf_frame], axis=1)
    result = fit_logit(y, design)
    if result is None or not is_estimable(result):
        logger.warning(
            "No baseline-MH odds ratio for %s: the model did not identify on "
            "%d participants (%d events).",
            outcome,
            len(y),
            int(y.sum()),
        )
        return None
    return odds_ratios_from_fit(result, [EXPOSURE])[EXPOSURE]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]Symptom decomposition: hard-PCC vs MH-overlap")

    y_bahmer, X, conf = load_pipeline_data(
        target="bahmer", clean_controls=True, cohort="mri", stack_variant="full"
    )
    sample_ids = y_bahmer.index

    # Load raw symptom indicators ----------------------------------------
    pcc = pd.read_parquet(get_processed_dir() / "corona2_pcc.parquet").set_index("ID")
    symptoms = pcc.loc[sample_ids, OVERLAP_SYMPTOMS + HARD_PCC_SYMPTOMS].astype(float)

    overlap_count = symptoms[OVERLAP_SYMPTOMS].sum(axis=1)
    hard_count = symptoms[HARD_PCC_SYMPTOMS].sum(axis=1)
    overlap_any = (overlap_count >= 1).astype(int)
    hard_any = (hard_count >= 1).astype(int)

    exposure = composite_mh_positive(X["mental_health"]).rename(EXPOSURE)
    conf_frame = build_confounder_design(conf)

    # Prevalence summary --------------------------------------------------
    prev_overlap = float(overlap_any.mean())
    prev_hard = float(hard_any.mean())
    prev_bahmer = float(y_bahmer.mean())

    # Per-symptom prevalence in MH-positive vs MH-negative ---------------
    per_symptom = []
    for sym in OVERLAP_SYMPTOMS + HARD_PCC_SYMPTOMS:
        s = symptoms[sym]
        per_symptom.append(
            {
                "symptom": sym,
                "category": "overlap" if sym in OVERLAP_SYMPTOMS else "hard_pcc",
                "prev_overall": float(s.mean()),
                "prev_mh_pos": float(s[exposure == 1].mean()),
                "prev_mh_neg": float(s[exposure == 0].mean()),
            }
        )

    # Logit ORs for each decomposed outcome ------------------------------
    or_overlap = mh_odds_ratio("any overlap symptom", overlap_any, exposure, conf_frame)
    or_hard = mh_odds_ratio("any hard-PCC symptom", hard_any, exposure, conf_frame)
    or_bahmer = mh_odds_ratio("Bahmer PCS", y_bahmer, exposure, conf_frame)

    # Count-based complement: is the MH-effect ALSO visible on symptom count,
    # not only on the "at least one" binary? Use a Negative Binomial GLM on the
    # count to guard against over-dispersion in the hard-PCC set (16 items).
    def nb_effect(outcome: str, count: pd.Series) -> dict[str, Any]:
        X_ = pd.concat([exposure.astype(float).rename(EXPOSURE), conf_frame], axis=1)
        X_ = sm.add_constant(X_, has_constant="add")
        try:
            fit = sm.GLM(
                count.astype(float),
                X_,
                family=sm.families.NegativeBinomial(alpha=1.0),
                missing="drop",
            ).fit()
        # The same design that leaves the logistic fits unidentified reaches
        # the count models too, and the iteratively reweighted least squares
        # behind the GLM has more ways to give up on it than the two the
        # logistic fit raises. The exception goes to the log, where it can be
        # read, rather than into a constants file, where it would sit among
        # the numbers.
        except Exception as error:  # pragma: no cover - guarded fallback
            logger.warning(
                "No incidence rate ratio for the %s count: the negative "
                "binomial model did not fit on %d participants: %s",
                outcome,
                len(count),
                error,
            )
            return {"estimable": False}

        coef = float(fit.params[EXPOSURE])
        lo, hi = fit.conf_int().loc[EXPOSURE]
        # The GLM does not raise on a design its sample cannot identify. It
        # reaches the coefficients through a pseudo-inverse and hands them
        # back finite, with standard errors that run to five figures, so
        # is_estimable finds nothing wrong with them; the interval is what
        # gives it away. A log-odds bound that wide overflows when it is
        # exponentiated, and a rate ratio whose interval runs to infinity
        # says nothing about the count. The overflow is thus the diagnosis
        # rather than an accident, which is why numpy's warning about it is
        # silenced and the finiteness is judged instead.
        with np.errstate(over="ignore"):
            irr = float(np.exp(coef))
            interval = [float(np.exp(lo)), float(np.exp(hi))]
        if not is_estimable(fit) or not bool(np.isfinite([irr, *interval]).all()):
            logger.warning(
                "No incidence rate ratio for the %s count: the negative "
                "binomial model did not identify on %d participants, and its "
                "interval carries no information.",
                outcome,
                int(fit.nobs),
            )
            return {"estimable": False}
        return {
            "estimable": True,
            "irr": irr,
            "ci_95": interval,
            "p_value": float(fit.pvalues[EXPOSURE]),
            "n": int(fit.nobs),
        }

    irr_overlap = nb_effect("overlap symptom", overlap_count)
    irr_hard = nb_effect("hard-PCC symptom", hard_count)

    summary = {
        "sample": {
            "n": len(sample_ids),
            "prevalence_bahmer": prev_bahmer,
            "prevalence_any_overlap_symptom": prev_overlap,
            "prevalence_any_hard_pcc_symptom": prev_hard,
            "mh_positive_share": float(exposure.mean()),
        },
        "symptom_definitions": {
            "overlap": OVERLAP_SYMPTOMS,
            "hard_pcc": HARD_PCC_SYMPTOMS,
        },
        "odds_ratios_binary_outcome": {
            name: _as_dict(fitted)
            for name, fitted in (
                ("any_overlap_symptom", or_overlap),
                ("any_hard_pcc_symptom", or_hard),
                ("bahmer_pcc_reference", or_bahmer),
            )
        },
        "incidence_rate_ratios_counts": {
            "overlap_count": irr_overlap,
            "hard_pcc_count": irr_hard,
        },
        "per_symptom": per_symptom,
    }

    out_path = get_paper_constants_dir() / "symptom_decomposition.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Pretty print --------------------------------------------------------
    tbl = Table(
        title="Baseline-MH → decomposed PCC outcomes (adjusted for age/sex/centre)",
        show_header=True,
    )
    tbl.add_column("Outcome", style="bold")
    tbl.add_column("Prev.", justify="right")
    tbl.add_column("OR [95% CI]", justify="right")
    tbl.add_column("p", justify="right")
    tbl.add_column("Count IRR [95% CI]", justify="right")

    def fmt_irr(entry: dict[str, Any]) -> str:
        if not entry["estimable"]:
            return "[yellow]not estimable"
        return f"{entry['irr']:.2f} [{entry['ci_95'][0]:.2f}, {entry['ci_95'][1]:.2f}]"

    def fmt_or(fitted: OddsRatio | None) -> str:
        if fitted is None:
            return "[yellow]not estimable"
        return f"{fitted.point:.2f} [{fitted.ci_lo:.2f}, {fitted.ci_hi:.2f}]"

    def fmt_p(fitted: OddsRatio | None) -> str:
        # The odds-ratio cell beside this one says why it is empty, so the
        # dash here only has to avoid printing a p-value for a model that
        # was never fitted.
        return "—" if fitted is None else f"{fitted.p_value:.2g}"

    tbl.add_row(
        "Any overlap symptom (5 items)",
        f"{prev_overlap:.3f}",
        fmt_or(or_overlap),
        fmt_p(or_overlap),
        fmt_irr(irr_overlap),
    )
    tbl.add_row(
        "Any hard-PCC symptom (16 items)",
        f"{prev_hard:.3f}",
        fmt_or(or_hard),
        fmt_p(or_hard),
        fmt_irr(irr_hard),
    )
    tbl.add_row(
        "Bahmer PCS (primary)",
        f"{prev_bahmer:.3f}",
        fmt_or(or_bahmer),
        fmt_p(or_bahmer),
        "(reference)",
    )
    console.print(tbl)
    console.print(f"[green]Wrote {out_path}")


if __name__ == "__main__":
    main()
