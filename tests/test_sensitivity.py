"""Tests for the shared sensitivity-analysis building blocks.

The supplementary scripts share these functions rather than each carrying its
own copy, so the composite exposure has exactly one definition. The tests
below pin the parts of that definition a future edit could quietly get wrong:
which cut-off applies, which direction missing values are resolved in, and
that the MINI column is genuinely optional rather than accidentally required.

They also pin what a fit hands back when it does not identify, which is two
different things and must stay two: an object whose estimates are unusable,
warned about and left to the caller to judge, and no object at all where the
design is singular and nothing was ever estimated. A script that cannot tell
the two apart either publishes a meaningless number or crashes on the
absence of one.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pcc_analysis.sensitivity import (
    MH_POSITIVE_CUTOFF,
    build_confounder_design,
    composite_mh_positive,
    e_value_from_rr,
    fit_linear_probability,
    fit_logit,
    is_estimable,
    logit_odds_ratio,
    logit_odds_ratios,
    or_to_rr,
)


def _mh_frame(**cols: list[object]) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_any_component_at_the_cutoff_makes_a_participant_positive() -> None:
    frame = _mh_frame(
        phq9_sum=[10, 0, 0, 0],
        gad7_sum=[0, 10, 0, 0],
        mini_major_depression=[0, 0, 1, 0],
    )
    assert composite_mh_positive(frame).tolist() == [1, 1, 1, 0]


def test_the_cutoff_is_inclusive() -> None:
    """Ten counts as positive, nine does not."""
    frame = _mh_frame(
        phq9_sum=[MH_POSITIVE_CUTOFF, MH_POSITIVE_CUTOFF - 1],
        gad7_sum=[0, 0],
        mini_major_depression=[0, 0],
    )
    assert composite_mh_positive(frame).tolist() == [1, 0]


def test_missing_components_resolve_to_not_positive() -> None:
    """The conservative direction: missingness can only attenuate.

    Resolving the other way would let an unmeasured participant enter the
    exposed group and inflate the association.
    """
    frame = _mh_frame(
        phq9_sum=[pd.NA, pd.NA, 12],
        gad7_sum=[pd.NA, 3, pd.NA],
        mini_major_depression=[pd.NA, pd.NA, pd.NA],
    )
    assert composite_mh_positive(frame).tolist() == [0, 0, 1]


def test_the_mini_column_is_optional() -> None:
    """Follow-up waves carry no MINI; it is administered at baseline only."""
    frame = _mh_frame(phq9_sum=[12, 2], gad7_sum=[1, 1])
    assert composite_mh_positive(frame).tolist() == [1, 0]


def test_confounder_design_drops_one_centre_level() -> None:
    """One level has to go, or the design is singular once intercepted."""
    conf = pd.DataFrame(
        {
            "basis_age": [40, 50, 60, 70],
            "basis_sex": [1, 2, 1, 2],
            "basis_uort": [1, 2, 3, 1],
        }
    )
    design = build_confounder_design(conf)

    centre_cols = [c for c in design.columns if c.startswith("center_")]
    assert len(centre_cols) == 2, design.columns.tolist()
    assert list(design.columns[:2]) == ["age", "sex"]
    assert design.notna().all().all()


def test_one_fit_serves_several_terms() -> None:
    """logit_odds_ratios must fit once, not once per term."""
    rng = np.random.default_rng(0)
    n = 400
    a = rng.integers(0, 2, n).astype(float)
    b = rng.integers(0, 2, n).astype(float)
    logit = -1.0 + 1.5 * a + 0.5 * b
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(float))
    design = pd.DataFrame({"a": a, "b": b})

    both = logit_odds_ratios(y, design, ["a", "b"])
    single = logit_odds_ratio(y, design, "a")

    # A design this sample identifies has a fit, so both wrappers hand back
    # coefficients; ``None`` is reserved for the design that has none.
    assert both is not None
    assert single is not None
    assert set(both) == {"a", "b"}
    assert both["a"].n == both["b"].n == n
    # Each term read off the joint fit must match the single-term wrapper.
    assert single.point == pytest.approx(both["a"].point)
    # The recovered direction should follow the simulated coefficients.
    assert both["a"].point > both["b"].point > 1.0


def test_a_fit_that_did_not_identify_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Separation warns in statsmodels and returns overflowing estimates.

    Every odds ratio this repository reports comes off ``fit_logit``.
    A model that never converged must not be able to produce one without
    the run saying so somewhere, even though the fit is still returned —
    the caller decides what to do with it.

    The scenario is the one that actually occurs: a stratum in which the
    outcome never varies. Separation severe enough to raise inside
    statsmodels leaves no fit behind at all and is a different case; this is
    the path that hands one back whose numbers mean nothing.
    """
    rng = np.random.default_rng(2)
    y = pd.Series([0.0] * 8)
    design = pd.DataFrame({"x": rng.normal(size=8), "z": rng.normal(size=8)})

    with caplog.at_level(logging.WARNING, logger="pcc_analysis.sensitivity"):
        result = fit_logit(y, design)

    # An object to judge, which is what separates this failure from the one
    # below: ``is_estimable`` has something to be asked about.
    assert result is not None
    assert not is_estimable(result)
    assert "did not identify" in caplog.text


def _design_the_sample_cannot_identify() -> tuple[pd.Series, pd.DataFrame]:
    """A subgroup taken out of a design built on the whole sample.

    Every stratified model in these analyses is fitted this way: the
    confounder block is built once over all participants, and a subgroup is
    then sliced out of it. The centres nobody in the subgroup belongs to
    keep their columns, holding nothing, and a coefficient for a centre with
    no participants is determined by no observation at all. The information
    matrix is then singular whatever the coefficients are, so the fit does
    not wander off to a boundary the way separation does — the inversion
    raises immediately, which is what makes this the failure mode with no
    fitted object behind it. The hospitalised stratum inside the MRI cohort
    is fitted from exactly this slice. A sample carrying more centre levels
    than it has participants to fill them, as a fixture does that profiles
    the study centre as a free integer, arrives at the same singularity by a
    slower route: the coefficients of the centres holding one participant
    run away, the weight those rows carry vanishes, and the matrix collapses
    to the same rank.
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
    # Two of the twelve centres: the ten dummies for the others survive the
    # slice as columns of zeros.
    in_stratum = pd.Series(centre < 2)
    return y[in_stratum], design.loc[in_stratum]


def test_a_design_the_sample_cannot_identify_has_no_fit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rank deficiency raises where separation merely warns.

    Nothing is returned because nothing was estimated, and the run says so.
    A supplementary script reaching this reports an unestimable panel rather
    than aborting the eleven steps queued behind it.
    """
    y, design = _design_the_sample_cannot_identify()

    with caplog.at_level(logging.WARNING, logger="pcc_analysis.sensitivity"):
        result = fit_logit(y, design)

    assert result is None
    assert "did not identify" in caplog.text
    # Named apart from the warning above, because the two ask different
    # things of the caller: there are no estimates here, not unusable ones.
    assert "No estimates exist" in caplog.text


def test_no_fit_leaves_no_odds_ratios_to_read() -> None:
    """The absence travels out through both wrappers rather than being filled.

    Every odds ratio here is read off a fitted object, so where there is no
    fit there is no coefficient to hand back and no defensible number to
    invent. How a panel says ``not estimable`` is a property of the file it
    is written to, so that decision stays with the caller.
    """
    y, design = _design_the_sample_cannot_identify()

    assert logit_odds_ratios(y, design, ["age"]) is None
    assert logit_odds_ratio(y, design, "age") is None


def test_a_healthy_fit_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    """Otherwise the warning would be noise nobody reads."""
    rng = np.random.default_rng(3)
    n = 500
    x = rng.integers(0, 2, n).astype(float)
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 0.8 * x)))).astype(float))

    with caplog.at_level(logging.WARNING, logger="pcc_analysis.sensitivity"):
        result = fit_logit(y, pd.DataFrame({"x": x}))

    assert is_estimable(result)
    assert caplog.text == ""


def test_the_linear_probability_model_returns_a_risk_difference() -> None:
    """The additive-scale counterpart of the logistic fit.

    Simulated with a known risk difference, because the point of the second
    scale is that it answers a different question from the odds ratio and
    has to be checkable against the quantity it claims to estimate.
    """
    rng = np.random.default_rng(4)
    n = 20_000
    x = rng.integers(0, 2, n).astype(float)
    y = pd.Series(rng.binomial(1, 0.3 + 0.2 * x).astype(float))

    result = fit_linear_probability(y, pd.DataFrame({"x": x}))

    assert float(result.params["x"]) == pytest.approx(0.2, abs=0.02)
    # Robust standard errors are what make the variance usable here.
    assert result.cov_type == "HC1"


def test_odds_ratio_carries_the_log_odds_for_shrinkage_maths() -> None:
    """Shrinkage is computed on the log-odds scale, so it must not be re-derived
    from a rounded odds ratio."""
    rng = np.random.default_rng(1)
    n = 300
    x = rng.integers(0, 2, n).astype(float)
    y = pd.Series(rng.binomial(1, 0.3, n).astype(float))
    fitted = logit_odds_ratio(y, pd.DataFrame({"x": x}), "x")

    assert fitted is not None
    assert fitted.log_odds == pytest.approx(float(np.log(fitted.point)))


def test_or_to_rr_shrinks_toward_one_as_the_outcome_gets_common() -> None:
    """The whole point of the Zhang-Yu correction: with a common outcome the
    odds ratio overstates the risk ratio."""
    assert or_to_rr(2.0, 0.0) == pytest.approx(2.0)
    assert 1.0 < or_to_rr(2.0, 0.3) < 2.0
    assert or_to_rr(2.0, 0.5) < or_to_rr(2.0, 0.3)


def test_e_value_is_one_for_a_null_or_protective_association() -> None:
    assert e_value_from_rr(1.0) == 1.0
    assert e_value_from_rr(0.5) == 1.0


def test_e_value_matches_the_published_formula() -> None:
    """E = RR + sqrt(RR * (RR - 1)); RR = 2 gives 2 + sqrt(2)."""
    assert e_value_from_rr(2.0) == pytest.approx(2.0 + np.sqrt(2.0))
    assert e_value_from_rr(3.0) > e_value_from_rr(2.0)
