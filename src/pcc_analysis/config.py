"""Configuration management for PCC prediction analysis.

Loads pipeline settings and NAKO data paths from config.toml.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from config.toml.

    Parameters
    ----------
    config_path : Path, optional
        Explicit path to a config file. When omitted, the ``PCC_CONFIG``
        environment variable is consulted (relative paths are resolved
        against the project root); otherwise ``config.toml`` at the project
        root is used. ``PCC_CONFIG`` lets every script — including those
        without a ``--config`` flag — share one config in a full-analysis
        run (e.g. ``run_analysis.sh --smoke``).

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        env_config = os.environ.get("PCC_CONFIG")
        if env_config:
            config_path = Path(env_config)
            if not config_path.is_absolute():
                config_path = _PROJECT_ROOT / config_path
        else:
            config_path = _PROJECT_ROOT / "config.toml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please create config.toml from config.toml.example."
        )

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    logger.info("Loaded config from %s", config_path)
    return config


_UNCONFIGURED_ROOT = _PROJECT_ROOT / ".unconfigured-delivery"
"""Stand-in for a delivery root the config does not name.

Never exists, which is the whole point: the supplementary and amendment
deliveries are optional, and their absence is decided by ``.exists()`` gates
in the orchestrator and the processors. ``Path("")`` — the previous
fallback — resolves to ``PosixPath('.')``, so those gates would have tested
a bare filename against the process working directory instead of against a
delivery root, and could have answered "present" for a file that has nothing
to do with NAKO.
"""


def _optional_data_root(paths: dict[str, Any], key: str) -> Path:
    """Resolve an optional NAKO delivery root, or a path that never exists."""
    raw = str(paths.get(key, "")).strip()
    if not raw:
        logger.warning(
            "%s is not set in the config; that delivery is treated as absent.", key
        )
        return _UNCONFIGURED_ROOT
    return Path(raw).expanduser()


class NAKOPaths:
    """NAKO data paths resolved from config.toml."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()

        paths = config["paths"]
        self.base_dir = Path(str(paths["nako_data_dir"])).expanduser()
        self._supplementary_dir = _optional_data_root(paths, "nako_supplementary_dir")
        self._amendment_dir = _optional_data_root(paths, "nako_amendment_dir")
        self._baseline_subdir: str = str(paths.get("nako_baseline_subdir", "Baseline"))
        self._insurance_subdir: str = str(
            paths.get("nako_insurance_subdir", "Krankenkassendaten")
        )
        files_raw = paths.get("files", {})
        self._files: dict[str, str] = (
            {str(k): str(v) for k, v in files_raw.items()}
            if isinstance(files_raw, dict)
            else {}
        )

        if not self.base_dir.exists():
            logger.warning("NAKO data directory not found: %s", self.base_dir)

    @property
    def baseline_dir(self) -> Path:
        return self.base_dir / self._baseline_subdir

    @property
    def health_insurance_dir(self) -> Path:
        return self.base_dir / self._insurance_subdir

    @property
    def mri1_csv(self) -> Path:
        return self.baseline_dir / self._files.get("mri1", "export_baseline_mrt1.csv")

    @property
    def mri2_csv(self) -> Path:
        return self.baseline_dir / self._files.get("mri2", "export_baseline_mrt2.csv")

    @property
    def mri3_csv(self) -> Path:
        return self.baseline_dir / self._files.get("mri3", "export_baseline_mrt3.csv")

    @property
    def kvad_csv(self) -> Path:
        return self.health_insurance_dir / self._files.get(
            "kvad", "exportfile_kv_amb_diag_mapped.csv"
        )

    @property
    def sex_age_csv(self) -> Path:
        return self.baseline_dir / self._files.get(
            "sex_age", "export_baseline_sex_age.csv"
        )

    @property
    def corona1_csv(self) -> Path:
        """Path to the Corona-1 mail-in questionnaire export.

        Corona-1 carries the cov87s acute-symptom block and the per-wave
        PHQ-9/GAD-7 items used in :mod:`mh_longitudinal` and
        :mod:`pcs_corona1_acute`.
        """
        return self.base_dir / self._files.get("corona1", "export_corona.csv")

    @property
    def corona2_csv(self) -> Path:
        return self.base_dir / self._files.get("corona2", "export_corona2.csv")

    @property
    def supplementary_dir(self) -> Path:
        return self._supplementary_dir

    @property
    def amendment_dir(self) -> Path:
        """Directory of the NAM-45 amendment delivery (2026-08-06).

        Carries the nine baseline PHQ-9 items and ``a_gpaq_ptotalmet``,
        neither of which is present in the primary or supplementary
        delivery.
        """
        return self._amendment_dir

    @property
    def amendment_baseline_csv(self) -> Path:
        return self.amendment_dir / self._files.get(
            "amendment_baseline", "export_baseline.csv"
        )

    @property
    def supplementary_baseline_csv(self) -> Path:
        """Baseline export of the supplementary delivery (2026-04-10).

        A partial export of 38 columns: NAKO's derived mental-health block,
        the six tobacco variables, the OGTT block and a few visit-timing
        fields. The primary delivery carries none of the tobacco variables,
        so this is the only source for :mod:`~pcc_analysis.data_processing.smoking`.
        """
        return self.supplementary_dir / self._files.get(
            "mental_health", "export_baseline.csv"
        )

    @property
    def mental_health_csv(self) -> Path:
        """Alias for :attr:`supplementary_baseline_csv`.

        The supplementary delivery carries several blocks; this name reaches
        it through the one the processors and ``validate()`` are wired to.
        """
        return self.supplementary_baseline_csv

    @property
    def followup1_csv(self) -> Path:
        """Path to the clinical follow-up 1 (FU1) export."""
        return self.supplementary_dir / self._files.get(
            "followup1", "export_followup1.csv"
        )

    @property
    def gefu1_csv(self) -> Path:
        """Path to the general follow-up 1 (GEFU1) questionnaire export.

        GEFU1 ("Generelle Follow-up 1") is the Corona-1 mail-in
        questionnaire. It is administered ~2-3 years after the baseline
        visit and is the only export in our data access that carries a
        per-participant ``basis_age`` post-baseline. It is therefore used
        as a timing proxy for FU1 in
        ``process_followup1_visit_meta``; see that function's docstring
        for the limitations of this proxy.
        """
        return self.base_dir / self._files.get("gefu1", "export_gefu1.csv")

    def validate(self) -> dict[str, bool]:
        """Check existence of all expected files."""
        files = {
            "mri1": self.mri1_csv,
            "mri2": self.mri2_csv,
            "mri3": self.mri3_csv,
            "kvad": self.kvad_csv,
            "sex_age": self.sex_age_csv,
            "corona2": self.corona2_csv,
            "mental_health": self.mental_health_csv,
            "amendment_baseline": self.amendment_baseline_csv,
        }
        results = {}
        for name, path in files.items():
            exists = path.exists()
            results[name] = exists
            if not exists:
                logger.warning("File not found: %s at %s", name, path)
        return results


def get_pipeline_config(
    config: dict[str, Any] | None = None,
) -> dict[str, int | str | bool]:
    """Return pipeline section of config with defaults."""
    if config is None:
        config = load_config()
    defaults: dict[str, int | str | bool] = {
        "n_outer_folds": 10,
        "n_inner_folds": 5,
        "meta_model": "gbdt",
        "orthogonalize": True,
        "random_state": 42,
        "n_jobs": -1,
        "meta_permutation_iterations": 1000,
        "n_subsamples": 100,
        "n_bootstrap": 1000,
    }
    defaults.update(config.get("pipeline", {}))
    return defaults


def get_processed_dir(config: dict[str, Any] | None = None) -> Path:
    """Return path to processed data directory."""
    if config is None:
        config = load_config()
    rel = str(config.get("output", {}).get("processed_dir", "data/processed"))
    return _PROJECT_ROOT / rel


def get_results_dir(config: dict[str, Any] | None = None) -> Path:
    """Return path to results directory."""
    if config is None:
        config = load_config()
    rel = str(config.get("output", {}).get("results_dir", "results"))
    return _PROJECT_ROOT / rel


def get_paper_dir(config: dict[str, Any] | None = None) -> Path:
    """Return ``<results>/paper`` — the collected paper-bound artifacts.

    All numbers and tables that appear in the manuscript are written under
    this directory so a reader can find every reported value in one place:
    ``constants/`` holds the LaTeX ``\\newcommand`` files and JSON payloads,
    ``tables/`` holds the generated LaTeX tables.
    """
    return get_results_dir(config) / "paper"


def get_paper_constants_dir(config: dict[str, Any] | None = None) -> Path:
    """Return ``<results>/paper/constants`` (created if missing)."""
    d = get_paper_dir(config) / "constants"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_paper_tables_dir(config: dict[str, Any] | None = None) -> Path:
    """Return ``<results>/paper/tables`` (created if missing)."""
    d = get_paper_dir(config) / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d
