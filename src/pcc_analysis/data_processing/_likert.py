"""Generic Likert-scale questionnaire scoring (PHQ-9 / GAD-7 etc.)."""

from __future__ import annotations

import logging

import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


# NAKO official missing codes for psychometric items
_PSYCHOMETRIC_MISSING = [7775, 7776, 7777, 8886, 8888, 8889, 9999]


def _score_likert_questionnaire(
    df: DataFrame,
    items: list[str],
    name: str,
    max_missing: int = 0,
) -> pd.Series:
    """Score a Likert-scale questionnaire from item-level data.

    Handles NAKO missing codes and auto-detects 1-based coding (recodes to 0-based).

    Parameters
    ----------
    df : DataFrame
        Source data containing item columns.
    items : list[str]
        Column names of the questionnaire items.
    name : str
        Questionnaire name for logging.
    max_missing : int
        Maximum number of missing items allowed (default 0, i.e. complete-case
        scoring). This is how NAKO derives its own PHQ-9 and GAD-7 sums —
        "for all participants without missing values on the respective items
        following the manual" (Streit et al. 2023, World J Biol Psychiatry,
        doi:10.1080/15622975.2021.2014152). Any tolerance above 0 has to be
        passed in deliberately per call site, so a scale that needs one cannot
        hand it to the next caller by default.
    """
    available = [c for c in items if c in df.columns]
    if not available:
        logger.warning("%s: no item columns found", name)
        return pd.Series(pd.NA, index=df.index, dtype="Float64")

    data = df[available].copy()
    data = data.replace(_PSYCHOMETRIC_MISSING, pd.NA)

    # Auto-detect 1-based coding and recode to 0-based
    valid = data.dropna()
    if len(valid) > 0 and valid.min().min() >= 1:
        logger.info("%s: recoding 1-based → 0-based", name)
        data = data - 1

    # No prorating: scores are simple sums, not adjusted for missing items.
    # These are used for descriptive Table 1 only, not as model predictors.
    n_missing = data.isna().sum(axis=1)
    score = data.sum(axis=1, skipna=True).astype("Float64")
    score = score.mask(n_missing > max_missing)
    return score
