"""Tests for the acute-course effect-modification analysis.

Two risks here, and neither announces itself. A stratum too thin to carry
the centre adjustment makes the logistic fit singular — inside the MRI
cohort the hospitalised stratum really is that thin — and an exception
there would take down the eleven supplementary steps that run after this
one. And the likelihood-ratio test compares two fits that must be on the
identical rows; ``missing="drop"`` makes it easy for them not to be, and a
ratio taken across two samples tests nothing while looking fine.

A singular design arrives here as an absent fit rather than as an
exception, since :func:`pcc_analysis.sensitivity.fit_logit` absorbs the
error the Hessian inversion raises. Both shapes are pinned below, because a
guard that only catches exceptions hands the absence straight to code that
expects a fitted object, and the run then dies where the unguarded error
would have killed it.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pcc_analysis.sensitivity import build_confounder_design

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "supplementary"
    / "severity_interaction.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("severity_interaction_cli", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_module()
fit_axis = _mod.fit_axis
render = _mod._render
try_fit = _mod._try_fit
load_severity = _mod.load_severity
SEVERITY_AXES = _mod.SEVERITY_AXES
OMICRON_ERA_START_MONTHS = _mod.OMICRON_ERA_START_MONTHS
INTERACTION = _mod.INTERACTION


def _record_logistic_designs(designs: list[list[str]]) -> Any:
    """Wrap the module's logistic fit so each design it sees is recorded.

    The column list is what distinguishes the models: the interaction model
    carries the product term, the model it is tested against does not. Both
    are fitted on the same rows, so counting fits alone would not say which
    one ran twice.
    """
    original = _mod.fit_logit

    def recording(y: pd.Series, design: pd.DataFrame) -> Any:
        designs.append(list(design.columns))
        return original(y, design)

    return recording


def _sample(n: int = 4000, seed: int = 0) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Outcome, exposure and adjustment set with a known interaction.

    The exposure doubles the odds among the unexposed-to-severity and does
    nothing among the severe, which is the shape the analysis has to be able
    to recover.
    """
    rng = np.random.default_rng(seed)
    exposure = pd.Series(rng.binomial(1, 0.2, n).astype(float))
    severe = pd.Series(rng.binomial(1, 0.1, n).astype(float))
    logit = -1.5 + 0.7 * exposure + 1.2 * severe - 0.7 * exposure * severe
    probability = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, probability).astype(float))
    confounders = pd.DataFrame(
        {
            "age": rng.normal(55, 8, n),
            "sex": rng.binomial(1, 0.5, n).astype(float),
        }
    )
    return y, exposure, confounders


def test_recovers_a_known_interaction() -> None:
    y, exposure, confounders = _sample()
    severity = pd.Series(np.where(np.arange(len(y)) % 10 == 0, 1.0, 0.0))

    result = fit_axis(y, exposure, confounders, severity)

    assert result["interaction_model"]["estimable"] is True
    assert result["n_modelled"] == len(y)
    assert result["n_severity_positive"] == int(severity.sum())
    lrt = result["interaction_model"]["likelihood_ratio_test"]
    assert lrt["chi2_df1"] >= 0.0
    assert 0.0 <= lrt["p_value"] <= 1.0


def test_a_stratum_too_thin_to_fit_is_reported_not_raised() -> None:
    """The MRI cohort's hospitalised stratum is genuinely this thin.

    Reporting it keeps the rest of the run alive and states the limit
    instead of implying the question was never asked.
    """
    y, exposure, confounders = _sample()
    # A stratum in which the outcome never varies: no logistic fit exists.
    severity = pd.Series(0.0, index=y.index)
    severity.iloc[:6] = 1.0
    y = y.copy()
    y.iloc[:6] = 0.0

    result = fit_axis(y, exposure, confounders, severity)

    exposed = result["stratified_models"]["exposed_to_severity"]
    assert exposed["n"] == 6
    assert exposed["n_events"] == 0
    assert exposed["underpowered"] is True
    assert exposed["estimable"] is False
    assert exposed["mh_odds_ratio"] is None
    # The other stratum is unaffected and still reported.
    assert result["stratified_models"]["not_exposed_to_severity"]["estimable"] is True


def test_participants_without_a_severity_answer_are_excluded_not_assumed() -> None:
    """An unanswered acute-care block is not a report of no hospitalisation."""
    y, exposure, confounders = _sample()
    severity = pd.Series(0.0, index=y.index)
    severity.iloc[:100] = 1.0
    severity.iloc[100:400] = np.nan

    result = fit_axis(y, exposure, confounders, severity)

    assert result["n_severity_missing"] == 300
    assert result["n_modelled"] == len(y) - 300
    strata = result["stratified_models"]
    assert (
        strata["exposed_to_severity"]["n"] + strata["not_exposed_to_severity"]["n"]
        == result["n_modelled"]
    )


def test_reports_the_interaction_on_both_scales() -> None:
    """Saturation predicts a multiplicative interaction with no additive one.

    Without the second scale that objection can only be admitted, not
    tested, and it is the first one a reader will raise about a negative
    interaction in a stratum where the outcome is twice as common.
    """
    y, exposure, confounders = _sample()
    severity = pd.Series(np.where(np.arange(len(y)) % 10 == 0, 1.0, 0.0))

    model = fit_axis(y, exposure, confounders, severity)["interaction_model"]

    assert model["additive_scale"]["estimable"] is True
    additive = model["additive_scale"]["mh_x_severity"]
    assert set(additive) == {"risk_difference", "ci_95", "p_value", "n"}
    assert additive["ci_95"][0] <= additive["risk_difference"]
    assert additive["risk_difference"] <= additive["ci_95"][1]
    # A risk difference is a probability contrast, not an odds ratio.
    assert -1.0 <= additive["risk_difference"] <= 1.0


def test_the_additive_scale_survives_a_stratum_that_cannot_be_fitted() -> None:
    """The linear model identifies where the logistic one separates.

    Reporting nothing at all for an axis because the logistic fit failed
    would throw away the one estimate that is still available.
    """
    y, exposure, confounders = _sample()
    severity = pd.Series(0.0, index=y.index)
    severity.iloc[:6] = 1.0
    y = y.copy()
    y.iloc[:6] = 0.0

    model = fit_axis(y, exposure, confounders, severity)["interaction_model"]

    assert model["additive_scale"]["estimable"] is True


def test_a_degenerate_linear_fit_is_not_reported_as_an_exact_null() -> None:
    """Least squares fails a third way: zero standard errors, NaN p-values.

    That prints as +0.000 [+0.000, +0.000] — the most confident-looking way
    to report nothing at all.
    """
    y, exposure, confounders = _sample(n=40)
    # No variation in the modifier: the interaction column is all zero.
    severity = pd.Series(0.0, index=y.index)

    model = fit_axis(y, exposure, confounders, severity)["interaction_model"]

    assert model["additive_scale"]["estimable"] is False
    assert "mh_x_severity" not in model["additive_scale"]


def test_timing_adjustment_is_reported_beside_the_primary_model() -> None:
    """Not replacing it: the date is missing for some participants.

    Moving the headline onto the smaller sample would confound the
    adjustment with the change of sample.
    """
    y, exposure, confounders = _sample()
    severity = pd.Series(np.where(np.arange(len(y)) % 10 == 0, 1.0, 0.0))
    rng = np.random.default_rng(1)
    timing = pd.Series(rng.uniform(0, 36, len(y)))
    timing.iloc[:50] = np.nan

    model = fit_axis(y, exposure, confounders, severity, timing)["interaction_model"]

    adjusted = model["timing_adjusted"]
    assert adjusted["n"] == len(y) - 50
    assert adjusted["n_without_a_date"] == 50
    assert adjusted["estimable"] is True
    assert "months_since_infection_or" in adjusted
    # The primary model is untouched and still on the full sample.
    assert model["mh_main_effect_or_unexposed"]["n"] == len(y)


def test_the_era_axis_splits_at_the_documented_month() -> None:
    """A cut-off is a setting, so it has to be pinned rather than inferred.

    January 2022 counted from the December 2019 reference month, which is
    when Omicron became dominant in Germany.
    """
    assert OMICRON_ERA_START_MONTHS == 25
    assert "omicron_era" in SEVERITY_AXES


def test_the_era_axis_is_derived_from_the_infection_month() -> None:
    index = pd.Index(range(4), name="ID")
    acute = pd.DataFrame(
        {
            "ID": list(index),
            "acute_hospitalised": [False, False, False, False],
            "n_infections": [1, 1, 1, 1],
            # Just before, exactly on, and after the boundary; then no date.
            "first_infection_months": [24.0, 25.0, 30.0, np.nan],
        }
    )

    class _Manager:
        def load_acute_infection(self) -> pd.DataFrame:
            return acute

    def _manager(_processed_dir: Path | None = None) -> _Manager:
        return _Manager()

    original = _mod.NakoDataManager
    _mod.NakoDataManager = _manager
    try:
        severity = load_severity(index)
    finally:
        _mod.NakoDataManager = original

    assert severity["omicron_era"].tolist()[:3] == [0.0, 1.0, 1.0]
    # No date is not a report of an early infection.
    assert pd.isna(severity["omicron_era"].iloc[3])


def test_no_timing_series_means_no_timing_block() -> None:
    y, exposure, confounders = _sample()
    severity = pd.Series(np.where(np.arange(len(y)) % 10 == 0, 1.0, 0.0))

    model = fit_axis(y, exposure, confounders, severity)["interaction_model"]

    assert "timing_adjusted" not in model


def test_an_empty_stratum_still_renders() -> None:
    """No participant reports the marker, so there is no prevalence to print.

    Routine on the synthetic fixtures and possible on a small cohort; a
    formatting crash here would take out the rest of the analysis run for a
    stratum that has nothing to say.
    """
    y, exposure, confounders = _sample()
    severity = pd.Series(0.0, index=y.index)

    result = fit_axis(y, exposure, confounders, severity)
    empty = result["stratified_models"]["exposed_to_severity"]
    assert empty["n"] == 0
    assert empty["pcc_prevalence"] is None

    table = render("empty", {"axes": {"acute_hospitalised": result}})
    assert table.row_count > 0


def test_prevalence_is_reported_next_to_every_stratified_estimate() -> None:
    """The saturation reading of an interaction has to be checkable."""
    y, exposure, confounders = _sample()
    severity = pd.Series(np.where(np.arange(len(y)) % 10 == 0, 1.0, 0.0))

    strata = fit_axis(y, exposure, confounders, severity)["stratified_models"]

    for label in ("exposed_to_severity", "not_exposed_to_severity"):
        stratum = strata[label]
        assert stratum["pcc_prevalence"] == stratum["n_events"] / stratum["n"]


def test_an_axis_nobody_answered_is_reported_not_raised() -> None:
    """A severity parquet that does not overlap the analytic index at all.

    A stale or re-pseudonymised delivery leaves every participant NA, so the
    modelled sample is empty and no fit of any kind exists. Every model here
    has to report that rather than raise: this script runs under ``set -e``
    with the rest of the supplementary steps queued behind it.
    """
    y, exposure, confounders = _sample()
    severity = pd.Series(np.nan, index=y.index)
    timing = pd.Series(np.nan, index=y.index)

    result = fit_axis(y, exposure, confounders, severity, timing)

    assert result["n_modelled"] == 0
    model = result["interaction_model"]
    assert model["estimable"] is False
    assert model["additive_scale"]["estimable"] is False
    assert INTERACTION not in model["additive_scale"]
    assert model["timing_adjusted"]["estimable"] is False
    for label in ("exposed_to_severity", "not_exposed_to_severity"):
        assert result["stratified_models"][label]["estimable"] is False
    # The panel still prints, as any unestimable panel does.
    assert render("empty", {"axes": {"acute_hospitalised": result}}).row_count > 0


def test_the_interaction_model_is_fitted_once() -> None:
    """The likelihood-ratio test reuses the fit the odds ratios came from.

    The full design is ~50k rows against a dozen centre dummies, run for
    three axes across two panels, so fitting it twice per axis doubles the
    cost of the whole script for a number that is already in hand.
    """
    y, exposure, confounders = _sample()
    severity = pd.Series(np.where(np.arange(len(y)) % 10 == 0, 1.0, 0.0))
    designs: list[list[str]] = []

    original = _mod.fit_logit
    _mod.fit_logit = _record_logistic_designs(designs)
    try:
        model = fit_axis(y, exposure, confounders, severity)["interaction_model"]
    finally:
        _mod.fit_logit = original

    assert model["estimable"] is True
    assert sum(INTERACTION in columns for columns in designs) == 1
    # One fit is the right count only if the likelihood-ratio test is intact
    # alongside it: the model the interaction is tested against is fitted too,
    # on the same rows, and the ratio is reported.
    assert sum(INTERACTION not in columns for columns in designs) >= 1
    assert "likelihood_ratio_test" in model


def test_a_failed_restricted_fit_is_reported_not_raised() -> None:
    """The model the interaction is tested against can fail on its own.

    Rank is a property of the design, and dropping the product term changes
    the design. An estimator that raises here — an empty sample raises out
    of least squares and the logistic fit alike — would abort the run just
    as surely as the interaction model would. The odds ratios survive it,
    since they are read off a fit that succeeded, so only the test statistic
    is withheld.
    """
    y, exposure, confounders = _sample()
    severity = pd.Series(np.where(np.arange(len(y)) % 10 == 0, 1.0, 0.0))
    original = _mod.fit_logit

    def failing_without_the_product_term(y: pd.Series, design: pd.DataFrame) -> Any:
        if INTERACTION not in design.columns:
            raise np.linalg.LinAlgError("Singular matrix")
        return original(y, design)

    _mod.fit_logit = failing_without_the_product_term
    try:
        result = fit_axis(y, exposure, confounders, severity)
    finally:
        _mod.fit_logit = original

    model = result["interaction_model"]
    assert model["estimable"] is True
    assert "interaction_or" in model
    assert "likelihood_ratio_test" not in model
    # The panel still prints without the test statistic.
    assert render("singular", {"axes": {"acute_hospitalised": result}}).row_count > 0


def test_a_restricted_fit_that_does_not_exist_only_costs_the_test_statistic() -> None:
    """The same withholding when the failure arrives as an absent fit.

    A rank-deficient restricted design does not raise out to this file: the
    shared fit catches the error and returns nothing. Handing that to the
    likelihood ratio would read a log-likelihood off it, so the absence has
    to be recognised where the exception is, and cost exactly as much — the
    test statistic, not the odds ratios.
    """
    y, exposure, confounders = _sample()
    severity = pd.Series(np.where(np.arange(len(y)) % 10 == 0, 1.0, 0.0))
    original = _mod.fit_logit

    def no_fit_without_the_product_term(y: pd.Series, design: pd.DataFrame) -> Any:
        return None if INTERACTION not in design.columns else original(y, design)

    _mod.fit_logit = no_fit_without_the_product_term
    try:
        result = fit_axis(y, exposure, confounders, severity)
    finally:
        _mod.fit_logit = original

    model = result["interaction_model"]
    assert model["estimable"] is True
    assert "interaction_or" in model
    assert "likelihood_ratio_test" not in model


def test_an_axis_with_no_logistic_fit_at_all_is_reported_not_raised() -> None:
    """Every logistic model on the axis is singular, and the panel survives.

    This is what the thin hospitalised stratum does to the whole axis inside
    the MRI cohort. Nothing is fitted, so nothing can be read: the panel
    marks each model not estimable and carries no odds ratio rather than a
    number with no model behind it. The additive scale is a separate
    estimator on a different design and is reported wherever it identifies,
    which is the reason the axis is worth printing at all.
    """
    y, exposure, confounders = _sample()
    severity = pd.Series(np.where(np.arange(len(y)) % 10 == 0, 1.0, 0.0))
    original = _mod.fit_logit

    def no_fit(y: pd.Series, design: pd.DataFrame) -> Any:
        return None

    _mod.fit_logit = no_fit
    try:
        result = fit_axis(y, exposure, confounders, severity)
    finally:
        _mod.fit_logit = original

    model = result["interaction_model"]
    assert model["estimable"] is False
    assert "interaction_or" not in model
    assert "likelihood_ratio_test" not in model
    for label in ("exposed_to_severity", "not_exposed_to_severity"):
        stratum = result["stratified_models"][label]
        assert stratum["estimable"] is False
        assert stratum["mh_odds_ratio"] is None
    assert model["additive_scale"]["estimable"] is True
    # The panel still prints, as any unestimable panel does.
    assert render("singular", {"axes": {"acute_hospitalised": result}}).row_count > 0


def _a_stratum_no_fit_can_identify() -> tuple[pd.Series, pd.DataFrame]:
    """Outcome and design for the stratified model that cannot be estimated.

    Built the way this script builds one: the confounder block covers the
    whole sample and the stratum is sliced out of it, so the centres nobody
    in the stratum belongs to survive the slice as columns of zeros. A
    coefficient for a centre with no participants rests on no observation,
    which makes the information matrix singular for any coefficient value
    and the inversion inside statsmodels raise.
    """
    rng = np.random.default_rng(11)
    n = 400
    centre = rng.integers(0, 12, n)
    conf = pd.DataFrame(
        {
            "basis_age": rng.normal(55, 8, n),
            "basis_sex": rng.integers(1, 3, n),
            "basis_uort": centre,
        }
    )
    design = build_confounder_design(conf)
    y = pd.Series(rng.binomial(1, 0.4, n).astype(float))
    in_stratum = pd.Series(centre < 2)
    return y[in_stratum], design.loc[in_stratum]


def test_a_singular_design_costs_a_fit_and_exactly_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The guard on a design that really is singular, not a stand-in for one.

    Nothing propagates out, so the steps queued behind this script keep
    running, and the absence is announced once. The shared fit already names
    the design it could not estimate; a second message here would report one
    unestimable model as two, and a reader counting warnings would over-state
    how much of the panel is missing.
    """
    y, design = _a_stratum_no_fit_can_identify()

    with caplog.at_level(logging.WARNING):
        assert try_fit(_mod.fit_logit, y, design) is None

    assert len(caplog.records) == 1
    assert "did not identify" in caplog.text


def test_the_constants_block_survives_an_axis_that_does_not_identify() -> None:
    """The unestimable panel reaches the LaTeX emitter, not just the console.

    ``fit_axis`` leaves ``interaction_or`` absent rather than None when
    nothing was fitted, and the MRI hospitalised axis really does come back
    that way. The emitter therefore has to read every optional term through
    ``get``: a plain subscript raises here, and it raises at the very end of
    the run, after the fits that took the time.

    What the axis does not estimate it does not define. The manuscript then
    fails on an undefined command, which is the loud half of this contract —
    far better than a macro that resolves to a number nobody estimated.
    """
    y, exposure, confounders = _sample()
    severity = pd.Series(np.where(np.arange(len(y)) % 10 == 0, 1.0, 0.0))
    original = _mod.fit_logit

    def no_fit(y: pd.Series, design: pd.DataFrame) -> Any:
        return None

    _mod.fit_logit = no_fit
    try:
        axis = fit_axis(y, exposure, confounders, severity)
    finally:
        _mod.fit_logit = original

    assert axis["interaction_model"]["estimable"] is False
    assert "interaction_or" not in axis["interaction_model"]

    panel = {
        "n_analytic": 100,
        "n_events": 20,
        "axes": dict.fromkeys(_mod._TEX_AXIS_NAMES, axis),
    }
    tex = _mod._tex_constants({"pooled_lean": panel, "mri_full": panel})

    # The counts that need no model are still emitted...
    assert "\\resSevHospN" in tex
    assert "\\resSevMriHospN" in tex
    # ...and every macro that would have needed one is simply absent.
    for stem in _mod._TEX_AXIS_NAMES.values():
        assert f"\\resSev{stem}Or}}" not in tex
    assert "\\resSevMriHospOr}" not in tex
