"""Smoking sensitivity: does tobacco exposure account for the baseline-MH effect?

Smoking is associated with depressive symptoms and with poor general health,
so it is a candidate explanation for the baseline-mental-health to PCC
association rather than merely another predictor. This script refits the
primary composite-exposure logistic regression with tobacco covariates added
and reports how far the odds ratio moves.

It is deliberately a sensitivity analysis and not a change to the primary
model. The headline odds ratio and the E-value built on it stay defined by
the pre-specified age, sex and centre adjustment; what this adds is a
quantified answer to "how much of it is smoking?".

Three models, in the order they should be read:

``A``   baseline MH + age/sex/centre, on the full analytic sample. This is
        the primary model and reproduces the odds ratio reported in the
        Discussion.
``A'``  the same model refitted on the participants who have complete
        tobacco data. Without it the comparison below would confound the
        effect of adjusting for smoking with the effect of dropping the
        roughly one per cent of participants who lack it, almost all of
        them because NAKO records their smoking status as unknown.
``B``   A' plus smoking status (never as reference, former and current as
        indicators) and pack-years.

The attributable shrinkage is therefore A' to B, not A to B. Pack-years
enters as a continuous dose term on top of the status indicators; never
smokers carry a structural zero, so the term is read as dose among those who
ever smoked.

Model B also answers the other half of the question a reader will ask, which
is whether smoking predicts PCC at all in this sample: its own odds ratios
are reported alongside.

A model the sample cannot identify is reported as not estimable rather than
fitted. The tobacco adjustment adds three columns to a design that already
carries the centre dummies, and it fits them on the subsample with complete
tobacco data, so whether the sample determines every coefficient is a
property of the delivery rather than something this script may assume. Where
it does not, the constants that would have been read off that model are
withheld from both outputs — no number, and no NaN standing in for one —
while the models that do identify are reported as usual.

Outputs:
    results/paper/constants/smoking_adjustment.json
    results/paper/constants/smoking_adjustment_tex.tex

Usage:
    uv run python scripts/supplementary/smoking_adjustment.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pcc_analysis.config import get_paper_constants_dir, get_processed_dir
from pcc_analysis.data_manager import FULL_STACK_JOINS, load_pipeline_data
from pcc_analysis.sensitivity import (
    OddsRatio,
    build_confounder_design,
    composite_mh_positive,
    e_value_from_rr,
    fit_logit,
    is_estimable,
    odds_ratios_from_fit,
    or_to_rr,
)

console = Console()
logger = logging.getLogger(__name__)

EXPOSURE = "baseline_mh_pos"

SMOKING_COLUMNS: list[str] = ["smoking_status", "pack_years"]
"""Tobacco covariates entering the adjustment.

Kept to the two variables an epidemiological reader expects: the exposure
category and the cumulative dose. The remaining columns of the tobacco block
(years smoked, current cigarettes per day, age at cessation) are either
collinear with pack-years or defined for one status group only, which would
make the model harder to read without adding an adjustment axis.
"""


class TobaccoBlockUnavailableError(ValueError):
    """The tobacco columns never reached the medical-history modality.

    A ``ValueError`` subclass because that is what callers passing a frame in
    directly have always caught, and because for them it *is* an error: a Lean
    Stack frame has no tobacco and a smoking-adjusted odds ratio from a model
    with no smoking in it would be worse than no answer.

    ``main`` treats it as a skip instead. Every layer below it — the data
    orchestrator's ``if supplementary_csv.exists()``, ``_attach``'s missing-
    parquet and no-overlap warnings — already treats an absent tobacco
    delivery as a supported configuration rather than a fault.
    """


def build_smoking_design(medical_history: pd.DataFrame) -> pd.DataFrame:
    """Smoking status indicators plus pack-years, never smokers as reference.

    Status is expanded into indicators rather than entered as its 1/2/3
    numeric code: that code happens to order the groups by current exposure,
    but nothing guarantees the risk is linear in it, and the never/former
    contrast is the one a reader wants to see separately.
    """
    missing = [c for c in SMOKING_COLUMNS if c not in medical_history.columns]
    if missing:
        raise TobaccoBlockUnavailableError(
            f"Tobacco columns absent from the medical-history modality: {missing}. "
            "They are joined only in the Full Stack, and only from a "
            "smoking.parquet whose IDs overlap the modality — check the "
            "stack_variant, that the parquet exists, and the loader's join "
            "warnings for a reported ID-space mismatch."
        )

    status = pd.to_numeric(medical_history["smoking_status"], errors="coerce")
    known = status.notna()
    # Never (code 1) is the reference level. Unknown status stays NA rather
    # than collapsing into the reference, so those participants drop out of
    # the fit instead of being silently counted as never-smokers.
    return pd.DataFrame(
        {
            "former_smoker": (status == 2).astype(float).where(known),
            "current_smoker": (status == 3).astype(float).where(known),
            "pack_years": pd.to_numeric(
                medical_history["pack_years"], errors="coerce"
            ).astype(float),
        },
        index=medical_history.index,
    )


def _try_odds_ratios(
    y: pd.Series, design: pd.DataFrame, terms: list[str], model: str
) -> dict[str, OddsRatio] | None:
    """Odds ratios for one model, or the report that it does not identify.

    Three ways a sample fails to determine a design, and only one of them
    raises here. Rank deficiency throws out of the Hessian inversion, which
    :func:`fit_logit` answers with ``None``: no coefficients exist to be
    read. Perfect separation merely warns and hands back a fit whose standard
    errors have overflowed to infinity, and a stalled optimiser hands back one
    that stopped wherever it gave up; both look like estimates and are none,
    so :func:`is_estimable` is consulted alongside. An empty sample raises
    ``ValueError`` out of the estimator instead — the case where no
    participant has a complete row, as a tobacco delivery whose ID space no
    longer meets the modality produces.

    Reported rather than raised on, because this analysis is step five of
    ``run_analysis.sh`` and an unestimable tobacco model would otherwise take
    the remaining supplementary steps down with it. ``model`` names the fit in
    the warning: all three models here are variations of one design, and a log
    line that only counts participants would not say which of them is missing
    from the output.
    """
    try:
        result = fit_logit(y, design)
    except ValueError as error:
        logger.warning(
            "Model %s not estimable on %d participants (%d events): %s",
            model,
            len(y),
            int(y.sum()),
            error,
        )
        return None
    if result is None or not is_estimable(result):
        logger.warning(
            "Model %s did not identify on %d participants (%d events): the "
            "sample does not determine every coefficient. Its constants are "
            "withheld.",
            model,
            len(y),
            int(y.sum()),
        )
        return None
    return odds_ratios_from_fit(result, terms)


def _try_odds_ratio(
    y: pd.Series, design: pd.DataFrame, term: str, model: str
) -> OddsRatio | None:
    """Single-term convenience wrapper around :func:`_try_odds_ratios`."""
    fitted = _try_odds_ratios(y, design, [term], model)
    return None if fitted is None else fitted[term]


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


def _shrinkage(
    reference: OddsRatio | None, adjusted: OddsRatio | None
) -> dict[str, Any]:
    """How far the exposure coefficient moves from A' to B.

    A contrast between two fits on the log-odds scale, so it exists only
    where both of them do. A missing side leaves it undefined rather than
    zero, and zero is not a neutral fallback here: it reads as "tobacco
    accounts for none of the effect", which is the conclusion this script
    exists to reach and must not fall out of a model that was never fitted.

    Undefined equally where the reference coefficient is zero. The ratio is
    then a division by zero rather than a shrinkage of nothing, since a
    baseline association of exactly none leaves the adjusted coefficient
    nothing to shrink from.
    """
    block: dict[str, Any] = {"reference": "model_a_restricted"}
    if reference is None or adjusted is None or reference.log_odds == 0.0:
        return block | {"estimable": False}
    return block | {
        "estimable": True,
        "logit_coef_reference": reference.log_odds,
        "logit_coef_adjusted": adjusted.log_odds,
        "shrink_ratio": (reference.log_odds - adjusted.log_odds) / reference.log_odds,
    }


def build_panel() -> dict[str, Any]:
    y, X, conf = load_pipeline_data(
        target="bahmer", clean_controls=True, cohort="mri", stack_variant="full"
    )
    exposure = composite_mh_positive(X["mental_health"]).rename(EXPOSURE).astype(float)
    confounders = build_confounder_design(conf)
    medical_history = X["medical_history"]
    smoking = build_smoking_design(medical_history)

    complete = smoking.notna().all(axis=1)
    n_full = len(y)
    n_complete = int(complete.sum())

    design_a = pd.concat([exposure, confounders], axis=1)
    model_a = _try_odds_ratio(y, design_a, EXPOSURE, "A, on the full analytic sample")

    y_sub = y[complete]
    design_a_restricted = design_a.loc[complete]
    model_a_restricted = _try_odds_ratio(
        y_sub, design_a_restricted, EXPOSURE, "A', on the tobacco-complete subsample"
    )

    design_b = pd.concat([design_a, smoking], axis=1).loc[complete]
    terms_b = [EXPOSURE, "former_smoker", "current_smoker", "pack_years"]
    fitted_b = _try_odds_ratios(y_sub, design_b, terms_b, "B, tobacco-adjusted")
    model_b = None if fitted_b is None else fitted_b[EXPOSURE]

    p_unexposed = float(y_sub[design_a_restricted[EXPOSURE] == 0].mean())
    status = pd.to_numeric(medical_history["smoking_status"], errors="coerce")

    return {
        "sample": {
            "n_analytic": n_full,
            "n_smoking_complete": n_complete,
            "n_smoking_incomplete": n_full - n_complete,
            "pct_smoking_incomplete": (
                (n_full - n_complete) / n_full * 100 if n_full else 0.0
            ),
        },
        "smoking_distribution": {
            "never": int((status == 1).sum()),
            "former": int((status == 2).sum()),
            "current": int((status == 3).sum()),
            "unknown": int(status.isna().sum()),
        },
        "model_a_primary": {
            "description": "Baseline MH + age/sex/centre, full analytic sample.",
            **_as_dict(model_a),
        },
        "model_a_restricted": {
            "description": "Model A refitted on participants with complete tobacco data.",
            **_as_dict(model_a_restricted),
        },
        "model_b_smoking_adjusted": {
            "description": "Model A' plus smoking status indicators and pack-years.",
            **_as_dict(model_b),
        },
        "smoking_effects": {
            term: _as_dict(None if fitted_b is None else fitted_b[term])
            for term in ("former_smoker", "current_smoker", "pack_years")
        },
        # Against the restricted reference, so the shrinkage reflects the
        # adjustment rather than the change of sample.
        "coefficient_shrinkage": _shrinkage(model_a_restricted, model_b),
        "e_value_smoking_adjusted": {
            "method": "Zhang & Yu 1998 OR->RR, then VanderWeele & Ding 2017.",
            "estimable": model_b is not None,
            # A transformation of model B's odds ratio, so it exists exactly
            # where B does. The observed p0 is reported next to it because the
            # conversion is only interpretable against the baseline risk it
            # used, and it is written only alongside a fit, which is what
            # guarantees the stratum it averages over is non-empty.
            **(
                {
                    "p0_unexposed_outcome_probability": p_unexposed,
                    "point": e_value_from_rr(or_to_rr(model_b.point, p_unexposed)),
                    "ci_lower": e_value_from_rr(or_to_rr(model_b.ci_lo, p_unexposed)),
                }
                if model_b is not None
                else {}
            ),
        },
    }


def _macro(name: str, value: float | None, digits: int = 2) -> str:
    r"""One ``\newcommand`` definition, or the marker that it has no value.

    A constant with no fit behind it is written as a comment naming the macro
    that is absent, rather than as a macro carrying a placeholder. Every one
    of these is consumed inside ``\num{}`` in the manuscript, where a word or
    a NaN would either typeset as a result or stop the build several hundred
    pages away from the model that failed.

    The comment is what a reader of the constants file sees, since these are
    transcribed into ``paper/results_constants.tex`` by hand. What the
    transcription works from is the changelist of
    ``scripts/compare_constants.py``, which drops comment lines and keys the
    rest by macro name, so a withheld constant reaches it as a disappearance —
    reported unconditionally, the same way any constant that stops being
    written is.
    """
    if value is None:
        return f"% \\{name}: not estimable; see the log for the model that failed."
    return f"\\newcommand{{\\{name}}}{{{value:.{digits}f}}}"


def _tex_lines(panel: dict[str, Any]) -> list[str]:
    b = panel["model_b_smoking_adjusted"]
    ar = panel["model_a_restricted"]
    eff = panel["smoking_effects"]
    ev = panel["e_value_smoking_adjusted"]
    shrink = panel["coefficient_shrinkage"]
    former = eff["former_smoker"]
    current = eff["current_smoker"]
    dose = eff["pack_years"]
    adjusted = b["estimable"]
    return [
        "% Generated by scripts/supplementary/smoking_adjustment.py "
        "-- do not edit by hand.",
        "",
        _macro("resSmokAdjN", b["n"] if adjusted else None, digits=0),
        _macro("resSmokAdjOrRef", ar["or"] if ar["estimable"] else None),
        _macro("resSmokAdjOr", b["or"] if adjusted else None),
        _macro("resSmokAdjOrLo", b["ci_95"][0] if adjusted else None),
        _macro("resSmokAdjOrHi", b["ci_95"][1] if adjusted else None),
        _macro(
            "resSmokAdjShrinkPct",
            shrink["shrink_ratio"] * 100 if shrink["estimable"] else None,
            digits=1,
        ),
        _macro("resSmokAdjEvalue", ev["point"] if ev["estimable"] else None),
        "",
        _macro("resSmokFormerOr", former["or"] if former["estimable"] else None),
        _macro(
            "resSmokFormerOrLo", former["ci_95"][0] if former["estimable"] else None
        ),
        _macro(
            "resSmokFormerOrHi", former["ci_95"][1] if former["estimable"] else None
        ),
        _macro("resSmokCurrentOr", current["or"] if current["estimable"] else None),
        _macro(
            "resSmokCurrentOrLo", current["ci_95"][0] if current["estimable"] else None
        ),
        _macro(
            "resSmokCurrentOrHi", current["ci_95"][1] if current["estimable"] else None
        ),
        _macro(
            "resSmokPackYearOr", dose["or"] if dose["estimable"] else None, digits=3
        ),
        _macro(
            "resSmokIncompletePct",
            panel["sample"]["pct_smoking_incomplete"],
            digits=1,
        ),
    ]


def _render(panel: dict[str, Any]) -> None:
    s = panel["sample"]
    d = panel["smoking_distribution"]
    console.print(
        f"\nAnalytic sample N = {s['n_analytic']:,}; tobacco data complete for "
        f"{s['n_smoking_complete']:,} ({100 - s['pct_smoking_incomplete']:.1f} %). "
        f"Never {d['never']:,} / former {d['former']:,} / current {d['current']:,}."
    )

    tbl = Table(title="Baseline-MH odds ratio, before and after tobacco adjustment")
    tbl.add_column("Model", style="bold")
    tbl.add_column("N", justify="right")
    tbl.add_column("OR [95% CI]", justify="right")
    for key in ("model_a_primary", "model_a_restricted", "model_b_smoking_adjusted"):
        m = panel[key]
        if not m["estimable"]:
            tbl.add_row(m["description"], "[dim]—", "[yellow]not estimable")
            continue
        tbl.add_row(
            m["description"],
            f"{m['n']:,}",
            f"{m['or']:.2f} [{m['ci_95'][0]:.2f}, {m['ci_95'][1]:.2f}]",
        )
    console.print(tbl)
    shrink = panel["coefficient_shrinkage"]
    console.print(
        "Log-odds shrinkage attributable to tobacco adjustment: "
        + (
            f"{shrink['shrink_ratio'] * 100:.1f} %"
            if shrink["estimable"]
            else "[yellow]not estimable"
        )
    )

    eff = Table(title="Tobacco covariates in the adjusted model")
    eff.add_column("Term", style="bold")
    eff.add_column("OR [95% CI]", justify="right")
    eff.add_column("p", justify="right")
    for term, res in panel["smoking_effects"].items():
        if not res["estimable"]:
            eff.add_row(term, "[yellow]not estimable", "[dim]—")
            continue
        eff.add_row(
            term,
            f"{res['or']:.3f} [{res['ci_95'][0]:.3f}, {res['ci_95'][1]:.3f}]",
            f"{res['p_value']:.3g}",
        )
    console.print(eff)


def tobacco_parquet_path() -> Path:
    """Where the processed tobacco block would be, if the delivery exists."""
    return get_processed_dir() / f"{FULL_STACK_JOINS['medical_history']}.parquet"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]Tobacco sensitivity for baseline-MH -> PCC")

    # A checkout configured against the primary delivery alone is a supported
    # configuration, so this analysis skips rather than aborting
    # run_analysis.sh in step 5 -- after the eight CV runs and before every
    # remaining supplementary script. Both routes to an absent tobacco block
    # are covered: no parquet at all, and a parquet whose ID space does not
    # meet the modality (which ``_attach`` reports and skips, so the columns
    # are missing downstream just the same).
    try:
        panel = build_panel()
    except TobaccoBlockUnavailableError as exc:
        parquet = tobacco_parquet_path()
        cause = (
            f"{parquet} not found — the supplementary tobacco delivery was "
            "never processed"
            if not parquet.exists()
            else f"{parquet} exists but its columns did not reach the "
            "medical-history modality; check the loader's join warnings for "
            "an ID-space mismatch"
        )
        console.print(
            f"[yellow]{cause}. Tobacco sensitivity skipped; no constants written."
        )
        logger.warning("Tobacco sensitivity skipped: %s", exc)
        return
    _render(panel)

    constants_dir = get_paper_constants_dir()
    json_path = constants_dir / "smoking_adjustment.json"
    json_path.write_text(json.dumps(panel, indent=2))
    tex_path = constants_dir / "smoking_adjustment_tex.tex"
    tex_path.write_text("\n".join(_tex_lines(panel)) + "\n")
    console.print(f"\nWrote {json_path}\n      {tex_path}")


if __name__ == "__main__":
    main()
