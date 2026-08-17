"""Tests for configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcc_analysis.config import (
    NAKOPaths,
    get_pipeline_config,
    get_processed_dir,
    get_results_dir,
    load_config,
)


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Write a minimal config.toml and return its path."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""\
[paths]
nako_data_dir = "{tmp_path / "nako"}"
nako_baseline_subdir = "BL"
nako_insurance_subdir = "KV"

[paths.files]
mri1 = "mri1.csv"
corona2 = "corona2.csv"

[pipeline]
n_outer_folds = 10
random_state = 99

[output]
processed_dir = "out/processed"
results_dir = "out/results"
"""
    )
    return cfg


# -- load_config -------------------------------------------------------------


def test_load_config_valid(sample_config: Path) -> None:
    cfg = load_config(sample_config)
    assert "paths" in cfg
    assert "pipeline" in cfg


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "nonexistent.toml")


# -- NAKOPaths ---------------------------------------------------------------


def test_nako_paths_resolution(sample_config: Path, tmp_path: Path) -> None:
    cfg = load_config(sample_config)
    paths = NAKOPaths(cfg)
    assert paths.base_dir == tmp_path / "nako"
    assert paths.baseline_dir == tmp_path / "nako" / "BL"
    assert paths.health_insurance_dir == tmp_path / "nako" / "KV"
    assert paths.mri1_csv == tmp_path / "nako" / "BL" / "mri1.csv"
    assert paths.corona2_csv == tmp_path / "nako" / "corona2.csv"


def test_nako_paths_defaults(tmp_path: Path) -> None:
    """With no files section, default file names are used."""
    cfg = {"paths": {"nako_data_dir": str(tmp_path)}}
    paths = NAKOPaths(cfg)
    assert paths.mri1_csv.name == "export_baseline_mrt1.csv"
    assert paths.sex_age_csv.name == "export_baseline_sex_age.csv"


def test_an_unset_delivery_root_does_not_resolve_against_the_cwd(
    tmp_path: Path,
) -> None:
    """An optional delivery that the config does not name must read as absent.

    The supplementary and amendment deliveries are optional, and the
    orchestrator and the processors decide that with ``.exists()``. Falling
    back to ``Path("")`` made those checks test a bare filename against the
    process working directory, so the answer depended on where the run was
    started from.
    """
    paths = NAKOPaths({"paths": {"nako_data_dir": str(tmp_path)}})

    assert paths.amendment_baseline_csv != Path("export_baseline.csv")
    assert not paths.amendment_baseline_csv.exists()
    assert not paths.supplementary_baseline_csv.exists()


def test_smoke_config_declares_every_delivery_path() -> None:
    """config_smoke.toml must carry every path key config.toml.example does.

    A key added to the example alone resolves to nothing under ``--smoke``,
    which is how ``nako_amendment_dir`` came to point at the working
    directory for a full release cycle.
    """
    repo_root = Path(__file__).resolve().parent.parent
    example = load_config(repo_root / "config.toml.example")["paths"]
    smoke = load_config(repo_root / "config_smoke.toml")["paths"]

    assert not set(example) - set(smoke)
    assert not set(example["files"]) - set(smoke["files"])


def test_nako_paths_validate(sample_config: Path) -> None:
    cfg = load_config(sample_config)
    paths = NAKOPaths(cfg)
    result = paths.validate()
    # None of the files exist
    assert all(v is False for v in result.values())
    assert "mri1" in result
    assert "corona2" in result


# -- get_pipeline_config -----------------------------------------------------


def test_pipeline_config_defaults(sample_config: Path) -> None:
    cfg = load_config(sample_config)
    pipe_cfg = get_pipeline_config(cfg)
    # Overridden values
    assert pipe_cfg["n_outer_folds"] == 10
    assert pipe_cfg["random_state"] == 99
    # Default values
    assert pipe_cfg["n_inner_folds"] == 5
    assert pipe_cfg["meta_model"] == "gbdt"
    assert pipe_cfg["orthogonalize"] is True


def test_pipeline_config_no_pipeline_section(tmp_path: Path) -> None:
    """Config without [pipeline] section returns all defaults."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[paths]\nnako_data_dir = "/tmp"\n')
    cfg = load_config(cfg_path)
    pipe_cfg = get_pipeline_config(cfg)
    assert pipe_cfg["n_outer_folds"] == 10
    assert pipe_cfg["n_jobs"] == -1


# -- get_processed_dir / get_results_dir -------------------------------------


def test_get_processed_dir(sample_config: Path) -> None:
    cfg = load_config(sample_config)
    d = get_processed_dir(cfg)
    assert d.parts[-2:] == ("out", "processed")


def test_get_results_dir(sample_config: Path) -> None:
    cfg = load_config(sample_config)
    d = get_results_dir(cfg)
    assert d.parts[-2:] == ("out", "results")


def test_get_dirs_defaults(tmp_path: Path) -> None:
    """Without [output] section, use default paths."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[paths]\nnako_data_dir = "/tmp"\n')
    cfg = load_config(cfg_path)
    assert get_processed_dir(cfg).name == "processed"
    assert get_results_dir(cfg).name == "results"
