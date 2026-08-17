"""Tests for the tobacco design matrix of the smoking sensitivity analysis.

The risk this covers is the one that would not show up as an error: folding
participants whose smoking status NAKO records as unknown into the
never-smoker reference group. That would silently enlarge the reference
category with people who may well have smoked, biasing the adjustment toward
finding no effect, and nothing about the fit would look wrong.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "supplementary"
    / "smoking_adjustment.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("smoking_adjustment_cli", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_module()
build_smoking_design = _mod.build_smoking_design
SMOKING_COLUMNS = _mod.SMOKING_COLUMNS


def _frame() -> pd.DataFrame:
    """One participant per smoking status, in the order never/former/current/unknown."""
    return pd.DataFrame(
        {
            "smoking_status": pd.array([1, 2, 3, None], dtype="UInt8"),
            "pack_years": pd.array([0.0, 12.5, 30.0, None], dtype="Float32"),
            "some_other_medical_history_column": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_never_smokers_are_the_reference_level() -> None:
    design = build_smoking_design(_frame())

    assert design["former_smoker"].tolist()[:3] == [0.0, 1.0, 0.0]
    assert design["current_smoker"].tolist()[:3] == [0.0, 0.0, 1.0]


def test_unknown_status_stays_missing_and_never_joins_the_reference() -> None:
    """An unknown-status participant must drop out of the fit, not become a
    never-smoker."""
    design = build_smoking_design(_frame())

    assert pd.isna(design["former_smoker"].iloc[3])
    assert pd.isna(design["current_smoker"].iloc[3])
    assert not design.notna().all(axis=1).iloc[3]


def test_only_the_two_intended_covariates_enter() -> None:
    """The rest of the medical-history modality must not leak into the design."""
    design = build_smoking_design(_frame())

    assert list(design.columns) == ["former_smoker", "current_smoker", "pack_years"]


def test_pack_years_survives_as_a_continuous_dose_term() -> None:
    design = build_smoking_design(_frame())

    assert design["pack_years"].iloc[0] == 0.0
    assert design["pack_years"].iloc[2] == pytest.approx(30.0)


def test_a_lean_stack_frame_fails_loudly() -> None:
    """Tobacco is joined in the Full Stack only, so a Lean frame lacks it.

    Failing here is the point: a silent skip would report a smoking-adjusted
    odds ratio from a model with no smoking in it.
    """
    lean = _frame().drop(columns=SMOKING_COLUMNS)
    with pytest.raises(ValueError, match="Tobacco columns absent"):
        build_smoking_design(lean)
