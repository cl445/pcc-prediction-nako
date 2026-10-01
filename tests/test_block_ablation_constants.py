"""Tests for the MRI block-ablation constants generator.

The generator turns ``block_ablation_oos.csv`` into paper macros and an
equivalence verdict against a margin fixed before the run. What it has to get
right is the verdict rule (both bounds of both intervals inside the margin)
and that it refuses a run whose block ablation is incomplete rather than
emitting half a table.
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
    / "block_ablation.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("block_ablation_cli", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_module()


def _row(variant: str, **overrides: float) -> dict[str, Any]:
    base: dict[str, Any] = {
        "block": "mri",
        "variant": variant,
        "n_removed": 6,
        "roc_full": 0.6643,
        "roc_variant": 0.6630,
        "delta_roc": 0.0013,
        "delta_roc_ci_lo": -0.0020,
        "delta_roc_ci_hi": 0.0046,
        "pr_full": 0.4128,
        "pr_variant": 0.4100,
        "delta_pr": 0.0028,
        "delta_pr_ci_lo": -0.0030,
        "delta_pr_ci_hi": 0.0090,
        "nb_t_roc": 0.8,
        "nb_p_roc": 0.44,
        "nb_t_pr": 1.1,
        "nb_p_pr": 0.30,
    }
    base.update(overrides)
    return base


def _write_run(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    run_dir = tmp_path / "primary"
    run_dir.mkdir()
    pd.DataFrame(rows).to_csv(run_dir / "block_ablation_oos.csv", index=False)
    return run_dir


def test_both_intervals_inside_the_margin_pass(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, [_row("masked"), _row("retrained")])
    constants = _mod.build_constants(run_dir)

    assert constants["verdict"] == "Pass"
    assert constants["variants"]["masked"]["verdict"] == "Pass"
    assert constants["n_removed"] == 6
    assert constants["margin"] == _mod.MARGIN
    assert constants["roc_full"] == pytest.approx(0.6643)


def test_an_interval_crossing_the_margin_fails_on_either_metric(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(
        tmp_path,
        [_row("masked"), _row("retrained", delta_pr_ci_hi=_mod.MARGIN + 0.001)],
    )
    constants = _mod.build_constants(run_dir)

    assert constants["verdict"] == "Fail"
    assert constants["variants"]["masked"]["verdict"] == "Pass"


def test_a_better_model_without_the_block_still_needs_the_lower_bound(
    tmp_path: Path,
) -> None:
    """Equivalence, not non-inferiority: a large negative delta is not a pass."""
    run_dir = _write_run(
        tmp_path,
        [
            _row("masked"),
            _row(
                "retrained",
                delta_roc=-0.04,
                delta_roc_ci_lo=-0.05,
                delta_roc_ci_hi=-0.03,
            ),
        ],
    )
    assert _mod.build_constants(run_dir)["verdict"] == "Fail"


def test_the_verdict_is_read_from_the_retrained_variant(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        [_row("masked", delta_roc_ci_hi=0.05), _row("retrained")],
    )
    constants = _mod.build_constants(run_dir)

    assert constants["verdict"] == "Pass"
    assert constants["variants"]["masked"]["verdict"] == "Fail"


def test_a_missing_variant_is_refused(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, [_row("retrained")])
    with pytest.raises(ValueError, match="masked"):
        _mod.build_constants(run_dir)


def test_a_missing_file_names_the_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"block_ablation_oos\.csv"):
        _mod.build_constants(tmp_path / "merged_run")


def test_other_blocks_in_the_file_are_ignored(tmp_path: Path) -> None:
    other = _row("retrained", delta_roc_ci_hi=0.2)
    other["block"] = "somatic"
    run_dir = _write_run(tmp_path, [_row("masked"), _row("retrained"), other])
    assert _mod.build_constants(run_dir)["verdict"] == "Pass"


def test_tex_macros_carry_both_variants_and_the_margin(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, [_row("masked"), _row("retrained")])
    tex = "\n".join(_mod._tex_lines(_mod.build_constants(run_dir)))

    for macro in (
        "\\resMriBlockNRemoved}{6}",
        "\\resMriBlockMargin}{0.03}",
        "\\resMriBlockVerdict}{Pass}",
        "\\resMriBlockDeltaRoc}{0.0013}",
        "\\resMriBlockDeltaRoclo}{-0.0020}",
        "\\resMriBlockDeltaPrhi}{0.0090}",
        "\\resMriBlockNbPRoc}{0.440}",
        "\\resMriBlockMaskedDeltaPr}{0.0028}",
        "\\resMriBlockMaskedNbPPr}{0.300}",
        "\\resMriBlockRoc}{0.663}",
    ):
        assert macro in tex, macro
    assert "\\resMriBlockMaskedVariantVerdict}{Pass}" in tex


def test_small_p_values_are_reported_as_a_bound() -> None:
    assert _mod._p(0.0004) == "<0.001"
    assert _mod._p(0.0499) == "0.050"
