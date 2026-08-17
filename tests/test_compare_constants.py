"""Tests for the constants diff that drives the paper cascade.

What this has to get right is what it does *not* report. A change it
misses is a stale number left in the manuscript, so the failure mode worth
testing is silence: a key that vanished, a value that went to null, a
nested constant several levels down.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_constants.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("compare_constants_cli", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_module()
compare_payloads = _mod.compare_payloads
compare_directories = _mod.compare_directories
MISSING = _mod.MISSING


def _paths(differences: list[Any]) -> set[str]:
    return {difference.path for difference in differences}


def test_finds_a_number_that_moved_however_deep() -> None:
    before = {"transfer": {"tost": {"roc_auc": {"delta": 0.002}}}}
    after = {"transfer": {"tost": {"roc_auc": {"delta": 0.031}}}}

    differences = compare_payloads(before, after, 0.0)

    assert _paths(differences) == {"transfer.tost.roc_auc.delta"}
    assert differences[0].relative == abs(0.031 - 0.002) / 0.002


def test_reports_a_constant_that_disappeared() -> None:
    """A removal is the change most likely to survive into a submission."""
    differences = compare_payloads({"e_value": 2.66}, {}, 0.0)

    assert _paths(differences) == {"e_value"}
    assert differences[0].after is MISSING


def test_null_is_a_value_not_an_absence() -> None:
    """The severity panels write null for a model the sample cannot fit."""
    differences = compare_payloads(
        {"mh_odds_ratio": {"or": 2.06}}, {"mh_odds_ratio": None}, 0.0
    )

    assert _paths(differences) == {"mh_odds_ratio"}
    assert differences[0].after is None
    assert differences[0].after is not MISSING


def test_list_elements_are_compared_positionally() -> None:
    differences = compare_payloads(
        {"ci_95": [1.80, 2.28]}, {"ci_95": [1.81, 2.28]}, 0.0
    )

    assert _paths(differences) == {"ci_95[0]"}


def test_identical_payloads_report_nothing() -> None:
    payload = {"a": 1, "b": [1, 2, {"c": "x"}], "d": None}

    assert compare_payloads(payload, dict(payload), 0.0) == []


def test_two_nans_are_the_same_value() -> None:
    """Otherwise every not-applicable cell reads as a change every time."""
    nan = float("nan")

    assert compare_payloads({"shrinkage": nan}, {"shrinkage": nan}, 0.0) == []


def test_a_constant_that_stops_being_not_applicable_is_reported() -> None:
    """A model the old sample could not identify and the new one can.

    The relative change of a NaN is a NaN, and a NaN is below every
    threshold as readily as it is above one, so this transition has to be
    classified rather than measured — it is a larger move than any number
    on the changelist beside it.
    """
    differences = compare_payloads(
        {"shrinkage": float("nan")}, {"shrinkage": 0.42}, 1e-9
    )

    assert _paths(differences) == {"shrinkage"}
    assert differences[0].relative is None


def test_a_constant_that_becomes_not_applicable_is_reported() -> None:
    differences = compare_payloads(
        {"shrinkage": 0.42}, {"shrinkage": float("nan")}, 1e-9
    )

    assert _paths(differences) == {"shrinkage"}
    assert differences[0].relative is None


def test_an_infinite_constant_is_reported_like_a_not_applicable_one() -> None:
    """A separating odds ratio: infinite arithmetic collapses to NaN too."""
    differences = compare_payloads(
        {"odds_ratio": float("inf")}, {"odds_ratio": 2.06}, 1e-9
    )

    assert _paths(differences) == {"odds_ratio"}
    assert differences[0].relative is None


def test_threshold_hides_noise_but_never_a_structural_change() -> None:
    before = {"roc_auc": 0.700, "label": "primary", "dropped": 1}
    after = {"roc_auc": 0.7000000001, "label": "control"}

    differences = compare_payloads(before, after, 1e-3)

    # The 1e-10 relative move is below the threshold; the string edit and
    # the removal have no relative size and must survive it.
    assert _paths(differences) == {"label", "dropped"}


def test_a_file_present_on_only_one_side_is_reported(tmp_path: Path) -> None:
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    (before_dir / "e_value.json").write_text(json.dumps({"e_value": 2.66}))
    (before_dir / "gone.json").write_text(json.dumps({"x": 1}))
    (before_dir / "gone.tex").write_text("\\newcommand{\\resGone}{1.0}\n")
    (after_dir / "e_value.json").write_text(json.dumps({"e_value": 2.66}))
    (after_dir / "new.json").write_text(json.dumps({"y": 2}))

    results = compare_directories(before_dir, after_dir, 0.0)

    assert set(results) == {"gone.json", "gone.tex", "new.json"}
    assert results["gone.json"][0].after is MISSING
    assert results["gone.tex"][0].after is MISSING
    assert results["new.json"][0].before is MISSING


SMOKING_TEX = r"""% Generated by scripts/supplementary/smoking_adjustment.py -- do not edit.
% Source run: {run}

\newcommand{{\resSmokAdjN}}{{8375}}
\newcommand{{\resSmokAdjOr}}{{{odds_ratio}}}
\newcommand{{\resSmokAdjShrinkPct}}{{2.2}}
"""
"""The macro shape, as scripts/supplementary/smoking_adjustment.py writes it."""


def test_tex_macros_are_compared_as_numbers(tmp_path: Path) -> None:
    """Four of the .tex files hold constants the manuscript reads directly.

    Parsed as numbers they answer to ``--threshold`` like a JSON leaf: the
    header naming the source run differs after every rerun and says nothing
    about the manuscript, so it stays off the changelist.
    """
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    (before_dir / "smoking_adjustment_tex.tex").write_text(
        SMOKING_TEX.format(run="pcc_pipeline_earlier", odds_ratio="2.02")
    )
    (after_dir / "smoking_adjustment_tex.tex").write_text(
        SMOKING_TEX.format(run="pcc_pipeline_later", odds_ratio="2.31")
    )

    results = compare_directories(before_dir, after_dir, 1e-3)

    differences = results["smoking_adjustment_tex.tex"]
    assert _paths(differences) == {"resSmokAdjOr"}
    assert differences[0].before == 2.02
    assert differences[0].relative == abs(2.31 - 2.02) / 2.02


AGREEMENT_TEX = r"""Bahmer weighted PCS (>10.75)             & \num{{2302}} & \num{{27.2}} & --- \\
Neurocognitive subtype ($\geq 2$ of 4)   & \num{{1648}} & \num{{19.5}} & \num{{{kappa}}} \\
Bahmer severity (>26.25)                 & \num{{1219}} & \num{{14.4}} & \num{{0.621}} \\
"""
"""The table-body shape, as scripts/supplementary/outcome_agreement.py writes it."""


def test_a_tex_table_body_is_compared_row_by_row(tmp_path: Path) -> None:
    """outcome_agreement.tex has no JSON twin.

    Its kappas and agreement percentages live in this file and nowhere
    else, so skipping it would let the rerun move the outcome-definition
    agreement without a word on the changelist.
    """
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    (before_dir / "outcome_agreement.tex").write_text(
        AGREEMENT_TEX.format(kappa="0.786")
    )
    (after_dir / "outcome_agreement.tex").write_text(
        AGREEMENT_TEX.format(kappa="0.731")
    )

    results = compare_directories(before_dir, after_dir, 0.0)

    differences = results["outcome_agreement.tex"]
    assert _paths(differences) == {"row[1]"}
    assert "0.786" in differences[0].before
    assert "0.731" in differences[0].after


ROBUSTNESS_TEX = (
    "% Generated by scripts/supplementary/reporting_style_robustness.py — "
    "do not edit by hand.\n"
    "\\newcommand{{\\resMixedOr}}{{{odds_ratio}}}\n"
)
"""The generated header, em dash and all, as the generators write it."""


def test_a_generated_header_is_read_whatever_the_locale_says(tmp_path: Path) -> None:
    """Two of the .tex files head themselves with an em dash.

    Reading them at the locale's encoding makes the changelist depend on
    the shell that launched the rerun: under ``LC_ALL=C`` — a cron entry, a
    CI runner, a cluster batch job — that encoding is ASCII and the read
    raises. ``run_analysis.sh`` trails the invocation with ``|| true`` so a
    diff failure cannot lose the rerun, which means the raise surfaces as
    nothing at all underneath the "Constants changed by this run" heading.
    The JSON payloads cannot catch this: they are ASCII by construction,
    because ``json.dumps`` escapes as it writes.
    """
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    (before_dir / "robustness_analyses_tex.tex").write_text(
        ROBUSTNESS_TEX.format(odds_ratio="2.07"), encoding="utf-8"
    )
    (after_dir / "robustness_analyses_tex.tex").write_text(
        ROBUSTNESS_TEX.format(odds_ratio="2.44"), encoding="utf-8"
    )

    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            str(SCRIPT),
            "--before",
            str(before_dir),
            "--after",
            str(after_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            # Force the interpreter onto the locale's encoding: PEP 540 UTF-8
            # mode and PEP 538 C-locale coercion would each override LC_ALL
            # and hide what a plain POSIX shell actually does.
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONUTF8": "0",
            "PYTHONCOERCECLOCALE": "0",
            # The file read is under test, not the rendering of it.
            "PYTHONIOENCODING": "utf-8",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert "resMixedOr" in completed.stdout
