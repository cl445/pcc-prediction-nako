"""Tests for the sex modification of the baseline-mental-health association.

The interaction model adjusts for the study centre with one dummy per
centre. On a sample too thin for that, the logistic fit is singular or
separates, and an exception there would take down every supplementary step
that runs after this one. The smoke data are such a sample, so the guard is
pinned here: an unidentifiable model is reported as not estimable, and an
identifiable one still yields its odds ratios.
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
    / "sex_interaction.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("sex_interaction_cli", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_module()
fit_interaction = _mod.fit_interaction
fit_stratum = _mod.fit_stratum


def _sample(
    n: int, n_centres: int, seed: int = 0
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.RangeIndex(n)
    exposure = pd.Series((rng.random(n) < 0.2).astype(int), index=index)
    conf = pd.DataFrame(
        {
            "basis_age": rng.integers(20, 70, n).astype(float),
            "basis_sex": rng.integers(1, 3, n),
            "basis_uort": rng.integers(0, n_centres, n),
        },
        index=index,
    )
    logit = -1.0 + 0.7 * exposure
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), index=index)
    return y, exposure, conf


def test_a_sample_too_thin_for_the_centre_dummies_is_not_estimable() -> None:
    y, exposure, conf = _sample(n=30, n_centres=20)

    result = fit_interaction(y, exposure, conf)

    assert result["interaction_or_female_vs_male"]["point"] is None
    assert result["likelihood_ratio_test"]["p_value"] is None
    stratum = fit_stratum(y, exposure, conf, conf["basis_sex"] == 2)
    assert stratum["odds_ratio"]["point"] is None


def test_an_identifiable_sample_still_yields_odds_ratios() -> None:
    y, exposure, conf = _sample(n=4000, n_centres=5)

    result = fit_interaction(y, exposure, conf)

    interaction = result["interaction_or_female_vs_male"]
    assert interaction["point"] is not None
    assert interaction["ci_95"][0] < interaction["point"] < interaction["ci_95"][1]
    male = fit_stratum(y, exposure, conf, conf["basis_sex"] == 1)
    assert male["odds_ratio"]["point"] > 1.0
