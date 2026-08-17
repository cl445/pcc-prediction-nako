"""Tests for the difference-in-association analysis.

The risk this covers is one that leaves no trace in the output: the reported
interval belonging to a different estimate than the point estimate it is
printed next to. Both are floats, both are plausible on their own, and the
JSON schema is satisfied either way — only the arithmetic relation between
them gives it away.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "supplementary"
    / "reporting_style_robustness.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "reporting_style_robustness_cli", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_module()
analysis_difference_in_association = _mod.analysis_difference_in_association


def _frame(n: int = 4000, seed: int = 20260816) -> pd.DataFrame:
    """Synthetic cohort with a baseline-MH x infection interaction built in.

    The generating log-odds are -2.0 + 1.0*baseline + 0.3*infected +
    0.8*baseline*infected, so the interaction odds ratio sits near exp(0.8)
    while the two stratum-specific odds ratios sit near exp(1.0) and
    exp(1.8) — far enough apart that an interval taken from the wrong
    estimate cannot pass for the right one.
    """
    rng = np.random.default_rng(seed)
    infected = rng.integers(0, 2, size=n)
    baseline_mh = rng.binomial(1, 0.25, size=n)
    logit = -2.0 + 1.0 * baseline_mh + 0.3 * infected + 0.8 * baseline_mh * infected
    current = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "had_covid": infected,
            # Baseline exposure, expressed through the PHQ-9 sum so that
            # composite_mh_positive picks it up at its own cut-off.
            "phq9_sum": np.where(baseline_mh == 1, 15.0, 2.0),
            "gad7_sum": 2.0,
            "mini_major_depression": 0,
            # Current (Corona-2) outcome, same encoding through PHQ-9.
            "phq9_total": np.where(current == 1, 15.0, 2.0),
            "gad7_total": 2.0,
            "basis_age": rng.normal(55, 10, size=n),
            "basis_sex": rng.integers(1, 3, size=n),
            "basis_uort": rng.integers(1, 4, size=n),
        }
    )


def test_the_interaction_interval_belongs_to_the_interaction_estimate() -> None:
    dia = analysis_difference_in_association(_frame())

    lo, hi = dia["interaction_ci"]
    assert lo < dia["interaction_or"] < hi


def test_the_interaction_interval_is_not_a_stratum_interval() -> None:
    """The stratified fits run after the interaction fit and must not leak into it."""
    dia = analysis_difference_in_association(_frame())

    lo, hi = dia["interaction_ci"]
    for stratum in dia["stratified"].values():
        assert stratum["ci"] != [lo, hi]
        assert not lo <= stratum["or"] <= hi
