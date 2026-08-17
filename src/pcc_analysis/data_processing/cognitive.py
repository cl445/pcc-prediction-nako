"""Cognitive tests: word-list recall, verbal fluency, Stroop, digit span."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis.data_processing._common import _load_nako_csv, _save_parquet

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


def _extract_cognitive_tests(df: DataFrame) -> DataFrame:
    """Extract neuropsychological test battery variables."""
    r = pd.DataFrame()
    r["ID"] = df["ID"]

    # Word List Recall (episodic memory)
    r["word_list_recall1"] = df["a_npsy_sumts2_1"]
    r["word_list_recall2"] = df["a_npsy_sumts2_2"]
    r["word_list_delayed"] = df["a_npsy_sumts2_3"]
    r["word_list_learning"] = df["a_npsy_difts2_21"]
    r["word_list_forgetting"] = df["a_npsy_difts2_23"]

    # Verbal Fluency (executive function)
    r["verbal_fluency_animals"] = df["a_npsy_sumts1"]

    # Stroop Task (selective attention)
    r["stroop_colors_time"] = df["a_npsy_sumts3_2"]
    r["stroop_interference_time"] = df["a_npsy_sumts3_3"]
    r["stroop_interference_effect"] = df["a_npsy_difts3_32"]

    # Digit Span Backwards (working memory)
    r["digit_span_backwards"] = df["a_npsy_sumts4"]

    # Number Series (numerical reasoning, Level 2)
    r["number_series"] = df["a_npsy_sumts5"]

    # Composite score (NAKO-provided weighted combination)
    r["cognitive_composite_score"] = df["a_npsy_sumts6"]

    for col in r.columns:
        if col != "ID":
            r[col] = r[col].replace([7777, 8888, 9999], pd.NA)
    return r


def process_cognitive_tests(
    paths: NAKOPaths, output_dir: Path, *, baseline_df: DataFrame | None = None
) -> Path:
    """Process cognitive tests -> ``cognitive_tests.parquet``."""
    logger.info("Processing cognitive tests ...")
    if baseline_df is not None:
        df = baseline_df
    else:
        csv_path = paths.baseline_dir / "export_baseline.csv"
        df = _load_nako_csv(csv_path)
    result = _extract_cognitive_tests(df)
    return _save_parquet(result, "cognitive_tests", output_dir)
