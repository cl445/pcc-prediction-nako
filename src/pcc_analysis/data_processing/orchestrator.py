"""Top-level orchestrator: ``process_all`` runs every NAKO data processor."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pandas import DataFrame

from pcc_analysis.config import NAKOPaths, get_processed_dir
from pcc_analysis.data_processing._common import _load_nako_csv
from pcc_analysis.data_processing.acute_infection import process_acute_infection
from pcc_analysis.data_processing.cardiovascular import process_cardiovascular
from pcc_analysis.data_processing.cognitive import process_cognitive_tests
from pcc_analysis.data_processing.corona import process_corona2, process_kvad
from pcc_analysis.data_processing.demographics import process_demographics
from pcc_analysis.data_processing.followup1 import (
    process_followup1_feno,
    process_followup1_lab_values,
    process_followup1_markers_misc,
    process_followup1_olfactometry,
    process_followup1_sleep_objective,
    process_followup1_visit_meta,
)
from pcc_analysis.data_processing.lab import process_lab_values
from pcc_analysis.data_processing.lung import process_lung_function
from pcc_analysis.data_processing.medical_history import process_medical_history
from pcc_analysis.data_processing.mental_health import (
    process_mental_health,
    process_psychometric,
)
from pcc_analysis.data_processing.mh_longitudinal import process_mh_longitudinal
from pcc_analysis.data_processing.mri import process_mri1, process_mri2, process_mri3
from pcc_analysis.data_processing.pcs_baseline_proxy import (
    process_kvad_pcs_proxy_all_variants,
)
from pcc_analysis.data_processing.pcs_corona1_acute import process_pcs_corona1_acute
from pcc_analysis.data_processing.phq9_items import process_phq9_items
from pcc_analysis.data_processing.physical_activity import (
    process_gpaq_activity,
    process_physical_activity,
)
from pcc_analysis.data_processing.ses import process_ses
from pcc_analysis.data_processing.smoking import process_smoking

logger = logging.getLogger(__name__)


def process_all(config: dict[str, Any] | None = None) -> dict[str, Path | list[Path]]:
    """Run every processor and write parquet files to ``data/processed/``.

    Parameters
    ----------
    config : dict, optional
        Pre-loaded config dict.  When *None*, ``load_config()`` is called
        automatically via ``NAKOPaths`` / ``get_processed_dir``.

    Returns
    -------
    dict[str, Path | list[Path]]
        Mapping from processor name to the path(s) of the output files.
    """
    nako = NAKOPaths(config)
    out_dir = get_processed_dir(config)

    logger.info("=" * 80)
    logger.info("NAKO data processing -- all modalities")
    logger.info("Output directory: %s", out_dir)
    logger.info("=" * 80)

    # Load baseline CSV once for all baseline-derived modalities
    baseline_csv = nako.baseline_dir / "export_baseline.csv"
    logger.info("Loading shared baseline CSV from %s", baseline_csv)
    baseline_df = _load_nako_csv(baseline_csv)

    # Load follow-up 1 (FU1) CSV once for all FU1-derived modalities.
    # Absent on smoke configurations without FU1 data -- skip gracefully.
    followup1_df: DataFrame | None
    try:
        fu1_csv = nako.followup1_csv
        if fu1_csv.exists():
            logger.info("Loading shared FU1 CSV from %s", fu1_csv)
            followup1_df = _load_nako_csv(fu1_csv)
        else:
            logger.warning(
                "FU1 CSV not found at %s; FU1 modalities will be skipped.",
                fu1_csv,
            )
            followup1_df = None
    except (AttributeError, KeyError, FileNotFoundError):
        followup1_df = None

    # Load GEFU1 CSV (Corona-1 questionnaire) — carries a per-participant
    # post-baseline basis_age used as timing proxy for FU1
    # (see process_followup1_visit_meta).
    gefu1_df: DataFrame | None
    try:
        gefu1_csv = nako.gefu1_csv
        if gefu1_csv.exists():
            logger.info("Loading shared GEFU1 CSV from %s", gefu1_csv)
            gefu1_df = _load_nako_csv(gefu1_csv)
        else:
            logger.warning(
                "GEFU1 CSV not found at %s; FU1 visit-meta will be skipped.",
                gefu1_csv,
            )
            gefu1_df = None
    except (AttributeError, KeyError, FileNotFoundError):
        gefu1_df = None

    outputs: dict[str, Path | list[Path]] = {}

    processors: list[str] = [
        "demographics",
        "ses",
        "cardiovascular",
        "cognitive_tests",
        "corona2",
        "acute_infection",
        "kvad",
        "lab_values",
        "lung_function",
        "medical_history",
        "mental_health",
        "physical_activity",
        "psychometric",
        "mri1",
        "mri2",
        "mri3",
    ]
    if followup1_df is not None:
        processors.extend(
            [
                "followup1_olfactometry",
                "followup1_lab_values",
                "followup1_sleep_objective",
                "followup1_feno",
                "followup1_markers_misc",
            ]
        )
        if gefu1_df is not None:
            processors.append("followup1_visit_meta")
    processors.append("mh_longitudinal")
    processors.append("kvad_pcs_proxy")
    processors.append("pcs_corona1_acute")

    # Supplementary delivery (2026-04-10): the six tobacco variables. Absent
    # on smoke configurations and on any checkout configured against the
    # primary delivery alone, which carries no tobacco variable at all.
    supplementary_csv = nako.supplementary_baseline_csv
    if supplementary_csv.exists():
        processors.append("smoking")
    else:
        logger.warning(
            "Supplementary baseline CSV not found at %s; smoking skipped.",
            supplementary_csv,
        )

    # Amendment delivery (NAM-45): baseline PHQ-9 items and the GPAQ total.
    # Absent on smoke configurations and on any checkout configured against
    # the primary delivery alone -- skip gracefully rather than fail the whole
    # run. Loaded once here and passed to all three amendment-derived
    # processors, the same way baseline_df is shared above: the export is
    # ~117,000 rows wide-loaded with nullable dtypes, and phq9_items,
    # gpaq_activity and the GPAQ join inside physical_activity would otherwise
    # each parse their own copy.
    #
    # The read is guarded the way the FU1 one above is. Sharing the frame
    # moves it out of the per-processor try/except in the dispatch loop, so an
    # unreadable optional delivery would otherwise take down all thirty
    # modalities instead of the three that need it.
    amendment_df: DataFrame | None = None
    amendment_csv = nako.amendment_baseline_csv
    if amendment_csv.exists():
        logger.info("Loading shared amendment CSV from %s", amendment_csv)
        try:
            amendment_df = _load_nako_csv(amendment_csv)
        except Exception:
            logger.exception(
                "Amendment baseline CSV at %s could not be read; PHQ-9 items "
                "and GPAQ activity skipped.",
                amendment_csv,
            )
    else:
        logger.warning(
            "Amendment baseline CSV not found at %s; PHQ-9 items and GPAQ "
            "activity skipped.",
            amendment_csv,
        )
    if amendment_df is not None:
        processors.append("phq9_items")
        processors.append("gpaq_activity")

    dispatch: dict[str, Callable[[], Path | list[Path]]] = {
        "demographics": lambda: process_demographics(
            nako, out_dir, baseline_df=baseline_df
        ),
        "ses": lambda: process_ses(nako, out_dir, baseline_df=baseline_df),
        "cardiovascular": lambda: process_cardiovascular(
            nako, out_dir, baseline_df=baseline_df
        ),
        "cognitive_tests": lambda: process_cognitive_tests(
            nako, out_dir, baseline_df=baseline_df
        ),
        "corona2": lambda: process_corona2(nako, out_dir),
        "acute_infection": lambda: process_acute_infection(nako, out_dir),
        "kvad": lambda: process_kvad(nako, out_dir),
        "lab_values": lambda: process_lab_values(
            nako, out_dir, baseline_df=baseline_df
        ),
        "lung_function": lambda: process_lung_function(
            nako, out_dir, baseline_df=baseline_df
        ),
        "medical_history": lambda: process_medical_history(
            nako, out_dir, baseline_df=baseline_df
        ),
        "mental_health": lambda: process_mental_health(nako, out_dir),
        "physical_activity": lambda: process_physical_activity(
            nako, out_dir, baseline_df=baseline_df, amendment_df=amendment_df
        ),
        "psychometric": lambda: process_psychometric(nako, out_dir),
        "smoking": lambda: process_smoking(nako, out_dir),
        "phq9_items": lambda: process_phq9_items(
            nako, out_dir, amendment_df=amendment_df
        ),
        "gpaq_activity": lambda: process_gpaq_activity(
            nako, out_dir, amendment_df=amendment_df
        ),
        "mri1": lambda: process_mri1(nako, out_dir),
        "mri2": lambda: process_mri2(nako, out_dir),
        "mri3": lambda: process_mri3(nako, out_dir),
        "followup1_olfactometry": lambda: process_followup1_olfactometry(
            nako, out_dir, followup1_df=followup1_df
        ),
        "followup1_lab_values": lambda: process_followup1_lab_values(
            nako, out_dir, followup1_df=followup1_df
        ),
        "followup1_sleep_objective": lambda: process_followup1_sleep_objective(
            nako, out_dir, followup1_df=followup1_df
        ),
        "followup1_feno": lambda: process_followup1_feno(
            nako, out_dir, followup1_df=followup1_df
        ),
        "followup1_markers_misc": lambda: process_followup1_markers_misc(
            nako, out_dir, followup1_df=followup1_df
        ),
        "followup1_visit_meta": lambda: process_followup1_visit_meta(
            nako,
            out_dir,
            baseline_df=baseline_df,
            gefu1_df=gefu1_df,
            followup1_df=followup1_df,
        ),
        "mh_longitudinal": lambda: process_mh_longitudinal(nako, out_dir),
        "kvad_pcs_proxy": lambda: list(
            process_kvad_pcs_proxy_all_variants(nako, out_dir).values()
        ),
        "pcs_corona1_acute": lambda: process_pcs_corona1_acute(nako, out_dir),
    }

    for name in processors:
        logger.info("-" * 60)
        try:
            outputs[name] = dispatch[name]()
        except Exception:
            logger.exception("Failed to process %s", name)

    logger.info("=" * 80)
    logger.info(
        "Processing complete. %d/%d modalities succeeded.",
        len(outputs),
        len(processors),
    )
    logger.info("=" * 80)

    return outputs
