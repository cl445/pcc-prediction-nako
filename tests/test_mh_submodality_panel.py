"""Tests for the mental-health sub-modality constants generator.

The generator's job is to stop the split-MH constants from being
hand-transcribed, so the risk it has to cover is a run that looks fine but
carries no mental-health input. A split-MH run is indistinguishable from a
primary run by its config -- the flag leaves no trace in ``config.csv`` or
in the auto-generated directory name -- so the sub-modality rows in the
artifacts are the only evidence the split happened.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PANEL_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "supplementary"
    / "mh_submodality_panel.py"
)


def _load_panel_module() -> Any:
    """Load the CLI script as a module so we can call its helpers."""
    spec = importlib.util.spec_from_file_location("mh_panel_cli", PANEL_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_panel = _load_panel_module()
SUBMODALITY_TEX_NAMES = _panel.SUBMODALITY_TEX_NAMES
build_panel = _panel.build_panel
_tex_lines = _panel._tex_lines


def _write_run(run_dir: Path, modalities: list[str], *, with_shap: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"fold": fold, "modality": mod, "roc_auc": 0.6, "pr_auc": 0.4}
            for fold in range(1, 11)
            for mod in modalities
        ]
    ).to_csv(run_dir / "modality_scores_cv.csv", index=False)
    pd.DataFrame(
        [
            {"feature": mod, "importance_mean": 0.02, "importance_std": 0.01}
            for mod in modalities
        ]
    ).to_csv(run_dir / "feature_importance_permutation.csv", index=False)
    pd.DataFrame(
        [
            {
                "roc_auc": 0.655,
                "pr_auc": 0.407,
                "calibration_slope": 1.26,
                "ece": 0.023,
            }
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "confidence_intervals.json").write_text(
        json.dumps({"roc_auc_ci": [0.642, 0.668], "pr_auc_ci": [0.391, 0.424]})
    )
    if with_shap:
        pd.DataFrame(
            [{"modality": mod, "mean_abs_shap": 0.1} for mod in modalities]
        ).to_csv(run_dir / "shap_modality_importance.csv", index=False)


ALL_MODALITIES = ["demographics", *SUBMODALITY_TEX_NAMES]


def test_panel_covers_every_submodality(tmp_path: Path) -> None:
    _write_run(tmp_path / "split_mh", ALL_MODALITIES)
    panel = build_panel(tmp_path / "split_mh", None)

    assert set(panel["submodalities"]) == set(SUBMODALITY_TEX_NAMES)
    assert panel["aggregate"]["roc_auc"] == pytest.approx(0.655)
    # Six modalities sharing the SHAP total: each is one sixth.
    assert panel["submodalities"]["mh_phq9"]["shap_share_pct"] == pytest.approx(100 / 6)


def test_a_run_without_the_split_is_refused(tmp_path: Path) -> None:
    """The decisive guard: a primary run must not yield split-MH constants."""
    _write_run(tmp_path / "primary", ["demographics", "mental_health"])
    with pytest.raises(ValueError, match="missing the sub-modalities"):
        build_panel(tmp_path / "primary", None)


def test_a_partial_split_is_refused(tmp_path: Path) -> None:
    """Three of five sub-modalities is the silent-drop signature."""
    _write_run(tmp_path / "partial", ["demographics", "mh_phq9", "mh_gad7"])
    with pytest.raises(ValueError, match="missing the sub-modalities"):
        build_panel(tmp_path / "partial", None)


def test_missing_shap_is_tolerated_but_omitted(tmp_path: Path) -> None:
    """A swallowed SHAP failure must not fabricate shares of zero."""
    _write_run(tmp_path / "no_shap", ALL_MODALITIES, with_shap=False)
    panel = build_panel(tmp_path / "no_shap", None)

    assert all(
        entry["shap_share_pct"] is None for entry in panel["submodalities"].values()
    )
    assert not any(
        line.startswith("\\newcommand{\\resShap") for line in _tex_lines(panel)
    )


def test_missing_permutation_csv_is_tolerated_but_omitted(tmp_path: Path) -> None:
    """The pipeline writes that file conditionally, so its absence is a state.

    ``feature_importance_permutation.csv`` is written only ``if
    fold_perm_importances``. Reading it unguarded turned "the per-fold step
    produced nothing" into a bare ``FileNotFoundError`` that pre-empted the
    sub-modality diagnosis this script exists to report.
    """
    run_dir = tmp_path / "no_perm"
    _write_run(run_dir, ALL_MODALITIES)
    (run_dir / "feature_importance_permutation.csv").unlink()

    panel = build_panel(run_dir, None)

    assert all(
        entry["permutation_importance"] is None
        for entry in panel["submodalities"].values()
    )
    assert not any(
        line.startswith("\\newcommand{\\resPermImp") for line in _tex_lines(panel)
    )


def test_an_incomplete_run_directory_names_what_is_missing(tmp_path: Path) -> None:
    """Required artifacts are checked up front, not by whichever read runs first."""
    run_dir = tmp_path / "incomplete"
    _write_run(run_dir, ALL_MODALITIES)
    (run_dir / "metrics.csv").unlink()

    with pytest.raises(FileNotFoundError, match=r"metrics\.csv"):
        build_panel(run_dir, None)


def test_tex_macro_names_avoid_digits(tmp_path: Path) -> None:
    """TeX control sequences cannot contain digits — hence PHQNine, GADSeven."""
    _write_run(tmp_path / "split_mh", ALL_MODALITIES)
    lines = _tex_lines(build_panel(tmp_path / "split_mh", None))

    macros = [
        ln.split("}")[0].removeprefix("\\newcommand{\\")
        for ln in lines
        if ln.startswith("\\newcommand")
    ]
    assert macros, "generator emitted no macros"
    assert not any(any(ch.isdigit() for ch in m) for m in macros), macros
