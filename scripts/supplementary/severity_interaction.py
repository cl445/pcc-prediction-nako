"""Does the acute infection modify the baseline-mental-health effect?

The second-hit framing says a pre-infection vulnerability needs an
infection to become post-COVID condition. The obvious follow-up — whether a
more severe infection makes the vulnerability matter more — needs a measure
of the acute course, which ``acute_infection.parquet`` supplies. This
script runs the interaction on the same composite baseline-MH exposure and
the same adjustment set as every other sensitivity here, without a pipeline
re-fit.

Two severity axes, because they fail differently:

``acute_hospitalised`` is the clinically meaningful one and the thin one.
Across the whole Corona-2 respondent set 458 participants were hospitalised
and 95 of those went to intensive care; in the MRI analytic sample the
expected count is around fifty. An interaction estimated on that is wide
whatever it shows, which is a statement about the data and not a reason to
skip it — but it is a reason to report the event counts next to every
estimate and to run the pooled cohort as the primary sample.

``reinfected`` (two or more reported infections) is the better-powered
dose axis: roughly 6,500 participants report more than one infection. It
measures repeated rather than severe exposure, so it answers a related but
distinct question, and the two are reported side by side rather than
pooled into one "severity" claim.

``omicron_era`` is not a severity axis at all but the question a reader
asks next: is this an early-pandemic phenomenon that no longer applies?
The infection month predicts PCC strongly — 33.6 % against 26.8 % across
the Omicron boundary — while being almost unrelated to baseline mental
health (r = -0.02, 78.1 % against 80.3 % infected in the later era). A
variable associated with the outcome but not the exposure cannot confound
the association, which is why it is stratified on rather than adjusted
for, and why the primary analysis is left alone. Adjusting for it would
also mean conditioning on a post-exposure variable, the same objection the
Discussion already makes about the T2 adjustment.

Both are measured after the exposure, so neither may enter the prediction
stack; see ``data_processing/acute_infection.py``. This is effect
modification on an established association, not prediction.

Each axis is reported on both scales and, separately, adjusted for the time
between infection and the survey. The two scales are there because an
interaction is scale-dependent and the obvious objection to a negative one
— that PCC is common enough after a severe infection for the odds ratio to
be compressed by the baseline risk — predicts a specific pattern, namely a
multiplicative interaction with no additive counterpart. Reporting both
turns that objection into something a reader can check. The timing
adjustment is there because the survey window is fixed while the infections
it asks about span three years, so participants differ in how long they had
to recover before being asked.

Usage:
    uv run python scripts/supplementary/severity_interaction.py
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd
from numpy.linalg import LinAlgError
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from pcc_analysis.config import get_paper_constants_dir
from pcc_analysis.data_manager import (
    CohortName,
    NakoDataManager,
    StackVariant,
    load_pipeline_data,
    prepare_index,
)
from pcc_analysis.sensitivity import (
    build_confounder_design,
    composite_mh_positive,
    fit_linear_probability,
    fit_logit,
    is_estimable,
    likelihood_ratio_test,
    odds_ratios_from_fit,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pcc_analysis.sensitivity import OddsRatio

console = Console()
logger = logging.getLogger(__name__)

EXPOSURE = "baseline_mh"
INTERACTION = "mh_x_severity"
TIMING = "months_since_first_infection"

MIN_EVENTS_FOR_A_STABLE_FIT = 30
"""Below this many outcome events in a stratum, say so next to the estimate.

Not a threshold that suppresses anything — a stratum with eight events
still gets fitted and reported. It marks the estimates whose confidence
intervals are wide enough that reading a direction off them is
over-interpretation, so a reader does not have to work that out from the
sample sizes.
"""

OMICRON_ERA_START_MONTHS: int = 25
"""January 2022, counted from the December 2019 reference month.

The month in which the Omicron variant became dominant in Germany. It is a
cut-off rather than a measurement, and the underlying dates are
month-precision, so the boundary is coarse by construction: a participant
infected in January 2022 may well have met Delta. The stratification is
therefore an era contrast, not a variant assignment, and it is reported as
such.
"""

SEVERITY_AXES: dict[str, str] = {
    "acute_hospitalised": (
        "Hospitalised during the acute infection (ward or intensive care), "
        "self-reported in the Corona-2 survey."
    ),
    "reinfected": ("Two or more reported SARS-CoV-2 infections, against exactly one."),
    "omicron_era": (
        "First infection in or after January 2022, when Omicron became "
        "dominant in Germany, against an earlier first infection. A coarse "
        "era contrast rather than a variant assignment."
    ),
}


def load_severity(index: pd.Index, processed_dir: Path | None = None) -> pd.DataFrame:
    """Acute-course markers aligned to an analytic sample.

    Participants absent from the parquet stay NA rather than being filled:
    an unanswered acute-care block is not a report of no hospitalisation.
    """
    mgr = NakoDataManager(processed_dir)
    acute = prepare_index(mgr.load_acute_infection(), "Acute infection")
    aligned = acute.reindex(index)

    hospitalised = aligned["acute_hospitalised"]
    n_infections = pd.to_numeric(aligned["n_infections"], errors="coerce")
    months = (
        pd.to_numeric(aligned["first_infection_months"], errors="coerce")
        .astype("Float64")
        .astype(float)
    )
    return pd.DataFrame(
        {
            "acute_hospitalised": hospitalised.astype("Float64").astype(float),
            "reinfected": (n_infections >= 2)
            .astype("Float64")
            .where(n_infections.notna())
            .astype(float),
            "omicron_era": pd.Series(months >= OMICRON_ERA_START_MONTHS, index=index)
            .astype(float)
            .where(months.notna()),
            # An axis in its own right above, and a covariate here — see
            # ``fit_axis`` for why the same variable appears twice.
            TIMING: months,
        },
        index=index,
    )


def _as_dict(model: OddsRatio) -> dict[str, Any]:
    return {
        "or": model.point,
        "ci_95": [model.ci_lo, model.ci_hi],
        "p_value": model.p_value,
        "n": model.n,
    }


def _try_fit(
    fit: Callable[[pd.Series, pd.DataFrame], Any],
    y: pd.Series,
    design: pd.DataFrame,
) -> Any | None:
    """Fit, or report that this sample cannot carry the model.

    Hospitalisation is rare, and inside the MRI cohort the hospitalised
    stratum has around fifty participants spread over eleven study centres —
    the centre dummies are then rank-deficient and the fit is singular. That
    is a real finding about the achievable precision, so it is reported as
    such. Dropping the centre adjustment to make the fit succeed would
    quietly answer a different question than every neighbouring model, and
    letting the failure propagate would take out the eleven supplementary
    steps that run after this one.

    Every fit in this file goes through here, least squares as much as the
    logistic ones, because the thin sample is not the only way to arrive at a
    model that cannot be estimated: a severity parquet that no longer
    overlaps the analytic index — a stale delivery, or one re-pseudonymised
    since — leaves no complete rows at all, and an empty sample raises out of
    both estimators alike.

    Three failure modes, and only one of them raises. Rank deficiency throws
    out of the Hessian inversion; perfect separation merely warns and hands
    back a logistic fit whose standard errors have overflowed to infinity;
    and a degenerate least-squares fit returns finite coefficients with zero
    standard errors and NaN p-values. The two that stay silent are the
    dangerous ones, since an estimate read off either is a number that means
    nothing, so :func:`is_estimable` is consulted alongside the exceptions.

    The fitted object is handed back rather than its coefficients because a
    likelihood-ratio test needs the log-likelihood, which no odds ratio
    carries, and refitting an identical model to recover it is the cost of
    the whole design over again.
    """
    try:
        result = fit(y, design)
    except (LinAlgError, PerfectSeparationError, ValueError) as error:
        logger.warning(
            "Model not estimable on %d participants (%d events): %s",
            len(y),
            int(y.sum()),
            error,
        )
        return None
    # A fitter that absorbed the failure itself has already said so, and has
    # no object for the estimability check to read.
    if result is None:
        return None
    if not is_estimable(result):
        logger.warning(
            "Model did not identify on %d participants (%d events): "
            "separation or non-convergence.",
            len(y),
            int(y.sum()),
        )
        return None
    return result


def _try_odds_ratios(
    y: pd.Series, design: pd.DataFrame, terms: list[str]
) -> dict[str, OddsRatio] | None:
    """Guarded logistic fit for the callers that need only its coefficients."""
    result = _try_fit(fit_logit, y, design)
    return None if result is None else odds_ratios_from_fit(result, terms)


def _linear_terms(result: Any, terms: list[str]) -> dict[str, dict[str, Any]]:
    """Read risk differences off a fitted linear probability model."""
    confidence = result.conf_int()
    return {
        term: {
            "risk_difference": float(result.params[term]),
            "ci_95": [
                float(confidence.loc[term, 0]),
                float(confidence.loc[term, 1]),
            ],
            "p_value": float(result.pvalues[term]),
            "n": int(result.nobs),
        }
        for term in terms
    }


def fit_axis(
    y: pd.Series,
    exposure: pd.Series,
    confounders: pd.DataFrame,
    severity: pd.Series,
    timing: pd.Series | None = None,
) -> dict[str, Any]:
    """Interaction model plus the two stratified models, on identical rows."""
    modifier = severity.rename("severity")
    complete = modifier.notna() & exposure.notna()
    y_sub = y[complete]
    design = pd.concat(
        [
            exposure.rename(EXPOSURE).astype(float),
            modifier.astype(float),
            confounders,
        ],
        axis=1,
    ).loc[complete]

    with_interaction = design.assign(
        **{INTERACTION: design[EXPOSURE] * design["severity"]}
    )
    full = _try_fit(fit_logit, y_sub, with_interaction)
    terms = (
        None
        if full is None
        else odds_ratios_from_fit(full, [EXPOSURE, "severity", INTERACTION])
    )

    interaction: dict[str, Any] = {
        "specification": (
            "logit P(PCC) ~ MH + severity + MH:severity + age + sex + centre"
        ),
        "estimable": terms is not None,
    }
    if terms is not None:
        interaction |= {
            "interaction_or": _as_dict(terms[INTERACTION]),
            "mh_main_effect_or_unexposed": _as_dict(terms[EXPOSURE]),
            "severity_main_effect_or": _as_dict(terms["severity"]),
        }
        # The wider model is the one the odds ratios above are read off, and
        # the narrower one is fitted on the identical rows, so the likelihood
        # ratio is a test of the interaction and not of a change of sample.
        # The narrower model can fail on its own — it is a different design,
        # and rank is a property of the design — in which case the odds
        # ratios stand and only the test statistic is withheld.
        restricted = _try_fit(fit_logit, y_sub, design)
        if restricted is not None:
            chi2, p_value = likelihood_ratio_test(restricted, full)
            interaction["likelihood_ratio_test"] = {
                "chi2_df1": chi2,
                "p_value": p_value,
            }

    # The same contrast on the risk-difference scale. An interaction term is
    # scale-dependent, and the saturation reading of a negative one — the
    # outcome already common in the severe stratum, leaving less room for the
    # odds to move — is exactly the case where the two scales disagree. With
    # both reported, that reading is checkable rather than merely admitted.
    #
    # Guarded like the logistic fit, and for a third failure mode: least
    # squares on a degenerate design returns finite coefficients with zero
    # standard errors and NaN p-values, which print as an exact null with a
    # zero-width interval — the most confident-looking way to report nothing.
    additive = _try_fit(fit_linear_probability, y_sub, with_interaction)
    interaction["additive_scale"] = {
        "specification": (
            "P(PCC) ~ MH + severity + MH:severity + age + sex + centre, "
            "identity link, HC1 robust standard errors"
        ),
        "estimable": additive is not None,
        **(
            _linear_terms(additive, [INTERACTION, EXPOSURE, "severity"])
            if additive is not None
            else {}
        ),
    }

    # Time between infection and the Corona-2 symptom assessment varies by up
    # to three years across the sample, because the survey window is fixed
    # and the infections are not. It is a plausible confounder of both the
    # severity marker and the outcome — early-pandemic infections were both
    # more severe and observed after longer recovery — so the interaction is
    # refitted with it. Reported next to the primary model rather than
    # replacing it: the date is missing for some participants, and moving the
    # headline onto a smaller sample would confound the adjustment with the
    # change of sample.
    if timing is not None:
        timed = complete & timing.notna()
        adjusted = _try_odds_ratios(
            y[timed],
            with_interaction.loc[timed[complete]].assign(
                **{TIMING: timing[timed].astype(float)}
            ),
            [INTERACTION, EXPOSURE, TIMING],
        )
        interaction["timing_adjusted"] = {
            "specification": (
                "As above, plus months from December 2019 to the first "
                "reported infection"
            ),
            "n": int(timed.sum()),
            "n_without_a_date": int(complete.sum() - timed.sum()),
            "estimable": adjusted is not None,
            **(
                {
                    "interaction_or": _as_dict(adjusted[INTERACTION]),
                    "mh_main_effect_or_unexposed": _as_dict(adjusted[EXPOSURE]),
                    "months_since_infection_or": _as_dict(adjusted[TIMING]),
                }
                if adjusted is not None
                else {}
            ),
        }

    strata: dict[str, dict[str, Any]] = {}
    for label, mask in (
        ("exposed_to_severity", modifier == 1),
        ("not_exposed_to_severity", modifier == 0),
    ):
        in_stratum = (mask & complete).fillna(False)
        y_stratum = y[in_stratum]
        events = int(y_stratum.sum())
        stratum_design = pd.concat(
            [exposure.rename(EXPOSURE).astype(float), confounders], axis=1
        ).loc[in_stratum]
        fitted = _try_odds_ratios(y_stratum, stratum_design, [EXPOSURE])
        n_stratum = int(in_stratum.sum())
        strata[label] = {
            "n": n_stratum,
            "n_events": events,
            # Reported next to the odds ratio because it carries the first
            # alternative reading of any interaction found here: where the
            # outcome is already common, there is less room left for an
            # additional risk factor to move the odds, and an attenuated
            # effect in the severe stratum can be saturation rather than
            # absence of effect modification.
            "pcc_prevalence": events / n_stratum if n_stratum else None,
            "underpowered": events < MIN_EVENTS_FOR_A_STABLE_FIT,
            "estimable": fitted is not None,
            "mh_odds_ratio": (
                _as_dict(fitted[EXPOSURE]) if fitted is not None else None
            ),
        }

    return {
        "n_modelled": int(complete.sum()),
        "n_severity_positive": int((modifier == 1).sum()),
        "n_severity_missing": int(modifier.isna().sum()),
        "interaction_model": interaction,
        "stratified_models": {
            "specification": (
                "logit P(PCC) ~ MH + age + sex + centre, within each stratum"
            ),
            **strata,
        },
    }


def build_panel(cohort: CohortName, stack_variant: StackVariant) -> dict[str, Any]:
    y, X, conf = load_pipeline_data(
        target="bahmer",
        clean_controls=True,
        cohort=cohort,
        stack_variant=stack_variant,
    )
    exposure = composite_mh_positive(X["mental_health"])
    confounders = build_confounder_design(conf)
    severity = load_severity(y.index)
    timing = severity[TIMING]

    return {
        "cohort": cohort,
        "stack_variant": stack_variant,
        "n_analytic": len(y),
        "n_events": int(y.sum()),
        "n_with_an_infection_date": int(timing.notna().sum()),
        "axes": {
            # The era axis gets no timing adjustment: it is a coarsening of
            # that very variable, so the two would be collinear and the
            # "adjusted" estimate would be an artefact of the overlap rather
            # than a robustness check.
            axis: fit_axis(
                y,
                exposure,
                confounders,
                severity[axis],
                None if axis == "omicron_era" else timing,
            )
            for axis in SEVERITY_AXES
        },
    }


def _render(name: str, panel: dict[str, Any]) -> Table:
    table = Table(title=f"Baseline-MH x acute course — {name}", show_header=True)
    table.add_column("Quantity", style="bold")
    table.add_column("Value", justify="right")
    for axis, result in panel["axes"].items():
        model = result["interaction_model"]
        table.add_row(f"[cyan]{axis}", "")
        table.add_row(
            "  N modelled / severity-positive",
            f"{result['n_modelled']:,} / {result['n_severity_positive']:,}",
        )
        if model["estimable"]:
            interaction = model["interaction_or"]
            table.add_row(
                "  Interaction OR",
                f"{interaction['or']:.2f} "
                f"[{interaction['ci_95'][0]:.2f}, {interaction['ci_95'][1]:.2f}] "
                f"(Wald p={interaction['p_value']:.2g})",
            )
            # The model the interaction is tested against has its own rank,
            # so the Wald estimate can stand while the test statistic does not.
            lrt = model.get("likelihood_ratio_test")
            table.add_row(
                "  LR test (df=1)",
                f"chi2={lrt['chi2_df1']:.2f}, p={lrt['p_value']:.2g}"
                if lrt is not None
                else "[yellow]not estimable",
            )
        else:
            table.add_row("  Interaction OR", "[yellow]not estimable")
        if model["additive_scale"]["estimable"]:
            additive = model["additive_scale"][INTERACTION]
            table.add_row(
                "  Interaction, risk difference",
                f"{additive['risk_difference']:+.3f} "
                f"[{additive['ci_95'][0]:+.3f}, {additive['ci_95'][1]:+.3f}] "
                f"(p={additive['p_value']:.2g})",
            )
        else:
            table.add_row("  Interaction, risk difference", "[yellow]not estimable")
        timing_model = model.get("timing_adjusted")
        if timing_model and timing_model["estimable"]:
            timed = timing_model["interaction_or"]
            table.add_row(
                f"  Interaction OR | + timing (n={timing_model['n']:,})",
                f"{timed['or']:.2f} [{timed['ci_95'][0]:.2f}, {timed['ci_95'][1]:.2f}]",
            )
            months = timing_model["months_since_infection_or"]
            table.add_row(
                "  OR per month since first infection",
                f"{months['or']:.3f} "
                f"[{months['ci_95'][0]:.3f}, {months['ci_95'][1]:.3f}]",
            )
        elif timing_model:
            table.add_row("  Interaction OR | + timing", "[yellow]not estimable")
        for label, stratum in result["stratified_models"].items():
            if label == "specification":
                continue
            # A stratum can be empty — no participant in a cohort reports the
            # severity marker at all — and then there is no prevalence to
            # print. Rare on real data, routine on the synthetic fixtures.
            prevalence = stratum["pcc_prevalence"]
            counts = f"n={stratum['n']:,}, {stratum['n_events']} events"
            if prevalence is not None:
                counts += f", {prevalence:.0%} PCC"
            if not stratum["estimable"]:
                table.add_row(f"  MH OR | {label} ({counts})", "[yellow]not estimable")
                continue
            odds = stratum["mh_odds_ratio"]
            flag = "  (underpowered)" if stratum["underpowered"] else ""
            table.add_row(
                f"  MH OR | {label} ({counts})",
                f"{odds['or']:.2f} "
                f"[{odds['ci_95'][0]:.2f}, {odds['ci_95'][1]:.2f}]{flag}",
            )
        table.add_section()
    return table


# TeX macro stems per severity axis. Digits are spelled out and underscores
# dropped, matching the naming rule the constants file already follows.
_TEX_AXIS_NAMES: dict[str, str] = {
    "acute_hospitalised": "Hosp",
    "reinfected": "Reinf",
    "omicron_era": "Omicron",
}


def _tex_constants(panels: dict[str, dict[str, Any]]) -> str:
    """Emit the \\newcommand block the supplement transcribes.

    Only the pooled panel, which is the primary one, plus the single MRI-cohort
    figure the supplement quotes to show why that panel cannot carry the
    hospitalisation axis on its own.
    """
    lines = [
        "% Generated by scripts/supplementary/severity_interaction.py "
        "-- do not edit by hand.",
        "% Acute-course modification of the baseline-MH -> PCC association.",
        "",
    ]

    def macro(name: str, value: str) -> None:
        lines.append(f"\\newcommand{{\\res{name}}}{{{value}}}")

    pooled = panels["pooled_lean"]
    macro("SevN", f"{pooled['n_analytic']}")
    macro("SevEvents", f"{pooled['n_events']}")
    lines.append("")

    for axis, stem in _TEX_AXIS_NAMES.items():
        entry = pooled["axes"][axis]
        interaction = entry["interaction_model"]
        macro(f"Sev{stem}N", f"{entry['n_severity_positive']}")
        # A stratum that does not identify carries no odds ratio at all: the
        # key is absent rather than None, so every read here goes through
        # ``get``. Its macros are then simply not defined, and the manuscript
        # fails loudly on the undefined command instead of printing a number
        # that was never estimated.
        odds = interaction.get("interaction_or")
        if odds is not None:
            macro(f"Sev{stem}Or", f"{odds['or']:.2f}")
            macro(f"Sev{stem}OrLo", f"{odds['ci_95'][0]:.2f}")
            macro(f"Sev{stem}OrHi", f"{odds['ci_95'][1]:.2f}")
            macro(f"Sev{stem}P", _format_p(odds["p_value"]))
        risk = interaction["additive_scale"].get("mh_x_severity")
        if risk is not None:
            macro(f"Sev{stem}Rd", f"{risk['risk_difference']:+.3f}")
            macro(f"Sev{stem}RdLo", f"{risk['ci_95'][0]:+.3f}")
            macro(f"Sev{stem}RdHi", f"{risk['ci_95'][1]:+.3f}")
            macro(f"Sev{stem}RdP", _format_p(risk["p_value"]))
        timing = interaction.get("timing_adjusted", {}).get("interaction_or")
        if timing is not None:
            macro(f"Sev{stem}OrTiming", f"{timing['or']:.2f}")
            macro(f"Sev{stem}OrTimingP", _format_p(timing["p_value"]))
        for label, suffix in (
            ("exposed_to_severity", "Exposed"),
            ("not_exposed_to_severity", "Unexposed"),
        ):
            stratum = entry["stratified_models"][label]
            # A stratum nobody falls into has no prevalence to report, so the
            # macro is left undefined rather than emitted as a division that
            # never happened.
            prevalence = stratum["pcc_prevalence"]
            if prevalence is not None:
                macro(f"Sev{stem}Prev{suffix}", f"{100 * prevalence:.1f}")
            stratum_odds = stratum["mh_odds_ratio"]
            if stratum_odds is not None:
                macro(f"Sev{stem}MHor{suffix}", f"{stratum_odds['or']:.2f}")
                macro(f"Sev{stem}MHor{suffix}Lo", f"{stratum_odds['ci_95'][0]:.2f}")
                macro(f"Sev{stem}MHor{suffix}Hi", f"{stratum_odds['ci_95'][1]:.2f}")
        lines.append("")

    # The MRI-cohort hospitalisation estimate, quoted to show that the interval
    # there is uninformative rather than reassuring.
    mri = panels["mri_full"]["axes"]["acute_hospitalised"]
    mri_odds = mri["interaction_model"].get("interaction_or")
    macro("SevMriHospN", f"{mri['n_severity_positive']}")
    if mri_odds is not None:
        macro("SevMriHospOr", f"{mri_odds['or']:.2f}")
        macro("SevMriHospOrLo", f"{mri_odds['ci_95'][0]:.2f}")
        macro("SevMriHospOrHi", f"{mri_odds['ci_95'][1]:.2f}")

    return "\n".join(lines) + "\n"


def _format_p(p_value: float) -> str:
    """Leading-dot p, matching the manuscript's convention."""
    if p_value < 0.001:
        return "{<}.001"
    return f"{p_value:.3f}".replace("0.", ".")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]Acute-course modification of baseline MH -> PCC")

    # The pooled cohort is primary here, against the MRI-cohort convention
    # of the other sensitivities: hospitalisation is rare enough that the
    # MRI sample alone cannot carry the interaction. The MRI panel is kept
    # so the estimate can be read next to the rest of the analysis, with its
    # event counts stated rather than implied.
    panels = {
        "pooled_lean": build_panel("mri_plus_non_mri", "lean"),
        "mri_full": build_panel("mri", "full"),
    }

    summary = {
        "exposure_definition": (
            "Composite baseline mental-health positivity: PHQ-9 sum >= 10 "
            "OR GAD-7 sum >= 10 OR MINI major depression == 1 "
            "(missing components treated as non-positive)."
        ),
        "severity_definitions": SEVERITY_AXES,
        "caveat": (
            "The acute course is measured after the exposure and is never a "
            "predictor in the pipeline; these are effect-modification models "
            "on an established association. Self-reported, so hospitalisation "
            "is subject to recall as much as the outcome is. The competing "
            "reading of a negative interaction is saturation: PCC is far more "
            "common after a severe infection, so an odds ratio there has less "
            "room to move regardless of effect modification. That reading is "
            "testable rather than merely admitted, which is why every axis "
            "reports both scales and the stratum-wise prevalences. A "
            "multiplicative interaction that vanishes on the risk-difference "
            "scale is the signature of saturation; one that survives on both "
            "is not."
        ),
        "primary_panel": "pooled_lean",
        "panels": panels,
    }

    out_dir = get_paper_constants_dir()
    out_path = out_dir / "severity_interaction.json"
    out_path.write_text(json.dumps(summary, indent=2))

    tex_path = out_dir / "severity_interaction_tex.tex"
    tex_path.write_text(_tex_constants(panels))

    for name, panel in panels.items():
        console.print(_render(name, panel))
    console.print(f"[green]Wrote {out_path} and {tex_path}")


if __name__ == "__main__":
    main()
