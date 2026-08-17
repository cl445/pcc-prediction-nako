"""Shared building blocks for the baseline-mental-health sensitivity analyses.

Half a dozen supplementary scripts fit the same logistic regression of the
PCC outcome on the same composite baseline-mental-health exposure, adjusted
for the same three confounders, and then vary one thing each: an added
covariate, a different outcome, a stratification, a bounding calculation.
Those shared parts live here once rather than as a copy per script. The one
exposure definition the whole Discussion section rests on is a poor
candidate for duplication: a drift in any single copy would make the odds
ratios incomparable without anything failing.

All three parts of the logistic model belong here: the exposure
(:func:`composite_mh_positive`), the adjustment set
(:func:`build_confounder_design`) and the fit itself
(:func:`logit_odds_ratios`). Two fits that are not logistic read
``conf_int()`` directly — the negative-binomial IRRs in
``symptom_decomposition`` and the unadjusted sanity check in ``evalue`` —
so a JSON payload can carry both conventions.

One arithmetic detail is worth naming: the confidence intervals here use a
fixed 1.96 multiplier, where reading ``conf_int()`` off the statsmodels
result implies 1.959964. Measured on the analytic sample the bound moves by
at most 1.4e-05 (1.428e-05 across the nine models in
``mh_trajectory_adjustment``, 1.365e-05 in ``persistence_adjustment``,
5.612e-06 in ``symptom_decomposition``) — the fifth decimal of an odds ratio,
and none of the two- or three-decimal constants the paper reports. Point
estimates, p-values and N are identical under either convention. Using one
convention everywhere is worth more than matching payloads written under the
other one digit for digit.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from scipy import stats
from statsmodels.tools.sm_exceptions import PerfectSeparationError

logger = logging.getLogger(__name__)

__all__ = [
    "MH_POSITIVE_CUTOFF",
    "OddsRatio",
    "build_confounder_design",
    "composite_mh_positive",
    "e_value_from_rr",
    "fit_linear_probability",
    "fit_logit",
    "is_estimable",
    "likelihood_ratio_test",
    "logit_odds_ratio",
    "logit_odds_ratios",
    "odds_ratios_from_fit",
    "or_to_rr",
]


class OddsRatio(NamedTuple):
    """One term of a fitted logistic regression.

    ``log_odds`` is carried alongside the exponentiated point estimate
    because coefficient shrinkage between two models has to be computed on
    the log-odds scale, and re-taking the log of a rounded odds ratio is a
    reliable way to lose the last digit.
    """

    point: float
    ci_lo: float
    ci_hi: float
    p_value: float
    log_odds: float
    n: int


MH_POSITIVE_CUTOFF: int = 10
"""Standard clinical cut-off for both the PHQ-9 and the GAD-7 sum score."""


def composite_mh_positive(frame: pd.DataFrame) -> pd.Series:
    """Composite binary baseline mental-health positivity.

    Positive if any of: PHQ-9 sum at or above the cut-off, GAD-7 sum at or
    above the cut-off, MINI major depression. A missing component counts as
    not positive, which is the conservative direction: it can only move the
    exposed group toward the unexposed one and thus attenuate the estimated
    association, never inflate it.

    The MINI column is optional because the instrument is administered at
    baseline only, so frames built from a follow-up wave do not carry it.
    """
    phq = (frame["phq9_sum"] >= MH_POSITIVE_CUTOFF).fillna(False)
    gad = (frame["gad7_sum"] >= MH_POSITIVE_CUTOFF).fillna(False)
    mini_raw = frame.get("mini_major_depression")
    mini = (
        mini_raw.fillna(0).astype(bool)
        if mini_raw is not None
        else pd.Series(False, index=frame.index)
    )
    return (phq | gad | mini).astype(int)


def build_confounder_design(conf: pd.DataFrame) -> pd.DataFrame:
    """Age, sex and study centre as a numeric design block.

    Centre is one-hot encoded with the first level dropped, so the design
    stays full rank once an intercept is added. This is the same adjustment
    set the DML orthogonalisation uses in the prediction pipeline, which is
    what makes the two sets of results comparable.
    """
    out = pd.DataFrame(
        {
            "age": conf["basis_age"].astype(float),
            "sex": conf["basis_sex"].astype("category").cat.codes.astype(float),
        },
        index=conf.index,
    )
    centre = pd.get_dummies(
        conf["basis_uort"].astype("category"),
        prefix="center",
        drop_first=True,
        dtype=float,
    )
    return pd.concat([out, centre], axis=1)


def fit_logit(y: pd.Series, design: pd.DataFrame) -> Any | None:
    """Fit the logistic regression the sensitivity analyses share.

    Exposed separately from :func:`logit_odds_ratios` for the callers that
    need the fitted object rather than its coefficients — a likelihood-ratio
    test needs the log-likelihood, which no odds ratio carries. Going
    through here keeps those callers on the same intercept handling and the
    same missing-data rule instead of growing another copy of the fit.

    Rows with any missing value in the design are dropped, so a model with an
    extra covariate is generally fitted on fewer participants than one
    without it. Where that matters, fit the reference model on the same
    reduced sample rather than comparing across sample sizes.

    ``None`` means no fit exists: a design the sample cannot identify makes
    the Hessian singular and its inversion raise, so there is no object to
    read anything off. Every other way a fit fails to identify leaves an
    object behind, which is returned with a warning for the caller to judge
    through :func:`is_estimable`. Both outcomes are reported rather than
    raised on, so a single unestimable panel cannot take down the analyses
    that run after it, and neither can be mistaken for a result: ``None`` has
    no coefficients to misread, and the warning names the fit that has
    unusable ones.
    """
    X = sm.add_constant(design.astype(float), has_constant="add")
    try:
        result = sm.Logit(y.astype(float), X, missing="drop").fit(disp=False)
    except (LinAlgError, PerfectSeparationError):
        logger.warning(
            "Logistic fit on a %d x %d design did not identify: the sample "
            "does not determine every coefficient. No estimates exist.",
            *X.shape,
        )
        return None
    if not is_estimable(result):
        logger.warning(
            "Logistic fit on %d observations did not identify: separation or "
            "non-convergence. Estimates read off it are not usable.",
            int(result.nobs),
        )
    return result


def fit_linear_probability(y: pd.Series, design: pd.DataFrame) -> Any:
    """Fit the same model on the risk-difference scale.

    An interaction is scale-dependent: a product term that is significant on
    the odds scale can be absent on the risk scale and the other way round,
    and neither answer is wrong — they are answers to different questions.
    Where the multiplicative and additive verdicts disagree, the usual
    explanation is that the outcome is common enough in one stratum for the
    odds ratio to be compressed by the baseline risk rather than by the
    exposure. Reporting only the odds ratio leaves that unresolved.

    The identity link is used rather than RERI computed from odds ratios,
    because that approximation needs a rare outcome and PCC is not rare here
    — it runs near 28 % in the unexposed stratum and near 58 % in the
    hospitalised one. Standard errors are heteroskedasticity-robust, which
    is what makes the linear probability model's variance estimate usable
    for a binary outcome; its fitted values can still fall outside [0, 1],
    which is why it is used for a contrast and not for prediction.
    """
    X = sm.add_constant(design.astype(float), has_constant="add")
    return sm.OLS(y.astype(float), X, missing="drop").fit(cov_type="HC1")


def likelihood_ratio_test(
    restricted: Any, full: Any, df: int = 1
) -> tuple[float, float]:
    """Chi-squared statistic and p-value for two nested logistic fits.

    The caller is responsible for the part that is easy to get wrong: both
    models have to be fitted on the identical rows. ``missing="drop"``
    silently changes the sample when the wider design has an extra column
    with its own missingness, and a likelihood ratio taken across two
    different samples is not a test of anything.
    """
    chi2 = float(2.0 * (full.llf - restricted.llf))
    return chi2, float(stats.chi2.sf(chi2, df))


def is_estimable(result: Any) -> bool:
    """Whether a fitted model carries usable estimates.

    A sample too thin to identify the design does not raise. Perfect
    separation is a warning in statsmodels, not an error, and the returned
    logistic fit then has coefficients that ran off to the optimiser's
    boundary and standard errors that overflow to infinity — an odds ratio
    read off it looks like a number and means nothing. A degenerate
    least-squares fit fails a second way: finite coefficients, zero standard
    errors and NaN p-values, which print as an exact null with a zero-width
    interval. Rank deficiency is the one mode that leaves nothing to inspect
    — the Hessian inversion raises — so :func:`fit_logit` absorbs it and
    hands back ``None``, and this function is never asked about it.

    Hence three conditions rather than one, and no assumption that the fit
    is logistic: the convergence flag is only consulted where the estimator
    has one, so this applies to :func:`fit_linear_probability` as well.

    :func:`fit_logit` logs when this returns ``False`` but still returns the
    fit, so no caller silently loses a result. A caller that can do
    something better than report a bad number — as the severity interaction
    can, by declaring a stratum not estimable — checks this explicitly.
    """
    converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
    finite = all(
        bool(np.isfinite(np.asarray(values, dtype=float)).all())
        for values in (result.bse, result.params, result.pvalues)
    )
    return converged and finite


def odds_ratios_from_fit(result: Any, terms: Sequence[str]) -> dict[str, OddsRatio]:
    """Read several terms off an already-fitted logistic model."""
    out: dict[str, OddsRatio] = {}
    for term in terms:
        coef = float(result.params[term])
        se = float(result.bse[term])
        out[term] = OddsRatio(
            point=float(np.exp(coef)),
            ci_lo=float(np.exp(coef - 1.96 * se)),
            ci_hi=float(np.exp(coef + 1.96 * se)),
            p_value=float(result.pvalues[term]),
            log_odds=coef,
            n=int(result.nobs),
        )
    return out


def logit_odds_ratios(
    y: pd.Series, design: pd.DataFrame, terms: Sequence[str]
) -> dict[str, OddsRatio] | None:
    """Fit one logistic regression and read several terms off it.

    ``None`` where the design does not identify and :func:`fit_logit` has no
    fit to hand back.
    """
    result = fit_logit(y, design)
    return None if result is None else odds_ratios_from_fit(result, terms)


def logit_odds_ratio(y: pd.Series, design: pd.DataFrame, term: str) -> OddsRatio | None:
    """Single-term convenience wrapper around :func:`logit_odds_ratios`."""
    ratios = logit_odds_ratios(y, design, [term])
    return None if ratios is None else ratios[term]


def or_to_rr(odds_ratio: float, p0: float) -> float:
    """Zhang and Yu (1998) common-outcome odds-ratio to risk-ratio conversion.

    ``p0`` is the observed outcome probability in the unexposed. Preferred
    over the minimax square-root approximation because p0 is observed here
    rather than assumed.
    """
    return odds_ratio / (1.0 - p0 + p0 * odds_ratio)


def e_value_from_rr(rr: float) -> float:
    """VanderWeele and Ding (2017) E-value from an approximate risk ratio.

    The minimum association, on the risk-ratio scale, that an unmeasured
    confounder would need with both exposure and outcome to explain the
    observed effect away. Defined for ``rr >= 1``; below that the effect is
    protective and the E-value is computed on the inverse, so a caller that
    can produce such an estimate should invert it first.
    """
    if rr <= 1.0:
        return 1.0
    return rr + float(np.sqrt(rr * (rr - 1.0)))
