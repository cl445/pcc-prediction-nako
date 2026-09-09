"""Unit tests for the FU1 visit-timing reconstruction.

Three things carry the risk here, and none of them is caught by the
shared quality harness. The parquet now holds two age anchors that look
alike and mean different things: ``gefu1_age_proxy`` dates the Corona-1
questionnaire, ``fu1_age`` dates the clinical FU1 examination, and only
the second one may enter the reconstructed visit year. The second is
coverage: the FU1 age comes from the primary delivery, which reaches
fewer participants than the supplementary one every measurement block is
read from, so participants outside it must end up missing on the derived
columns rather than dropped from the frame or silently dated from the
proxy. The third is that both anchors are optional, because a smoke
configuration has neither, and their absence has to leave the proxy
columns intact instead of failing the run.
"""

from __future__ import annotations

import pandas as pd

from pcc_analysis._plausibility import FOLLOWUP1_VISIT_META_RANGES
from pcc_analysis.data_processing.followup1 import _extract_followup1_visit_meta

# Four participants. Only 1 and 2 appear in the primary FU1 delivery, so
# only they can carry a reconstructed year; 3 is FU1 but primary-absent,
# and 4 never reached FU1 at all.
_BASELINE = pd.DataFrame({"ID": [1, 2, 3, 4], "basis_age": [50, 40, 60, 30]})
_GEFU1 = pd.DataFrame({"ID": [1, 2, 3, 4], "basis_age": [52, 42, 62, 32]})
_FU1_AGE = pd.DataFrame({"ID": [1, 2], "basis_age": [54, 45]})
_BASELINE_YEAR = pd.DataFrame(
    {"ID": [1, 2, 3, 4], "basis_year": [2015, 2016, 2014, 2017]}
)
_FU1_IDS = pd.Index([1, 2, 3])


def _extract(
    fu1_age_df: pd.DataFrame | None = None,
    baseline_year_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = _extract_followup1_visit_meta(
        _BASELINE,
        _GEFU1,
        _FU1_IDS,
        fu1_age_df=fu1_age_df,
        baseline_year_df=baseline_year_df,
    )
    return frame.set_index("ID")


def test_visit_year_comes_from_the_fu1_age_not_the_questionnaire_proxy() -> None:
    """basis_year + (fu1_age - baseline_age), never the GEFU1 delta."""
    r = _extract(fu1_age_df=_FU1_AGE, baseline_year_df=_BASELINE_YEAR)
    # Participant 1: 2015 + (54 - 50) = 2019. The GEFU1 proxy would say
    # 2015 + (52 - 50) = 2017, so a mix-up is visible in the value.
    assert r.loc[1, "years_baseline_to_fu1"] == 4
    assert r.loc[1, "fu1_year_reconstructed"] == 2019
    assert r.loc[2, "years_baseline_to_fu1"] == 5
    assert r.loc[2, "fu1_year_reconstructed"] == 2021
    # The proxy stays what it was and keeps its own, different delta.
    assert r.loc[1, "years_since_baseline_proxy"] == 2


def test_participants_outside_the_primary_delivery_are_missing_not_dropped() -> None:
    """Narrower FU1 coverage must not shrink the frame."""
    r = _extract(fu1_age_df=_FU1_AGE, baseline_year_df=_BASELINE_YEAR)
    assert set(r.index) == {1, 2, 3}  # 4 is not an FU1 participant
    assert pd.isna(r.loc[3, "fu1_age"])
    assert pd.isna(r.loc[3, "years_baseline_to_fu1"])
    assert pd.isna(r.loc[3, "fu1_year_reconstructed"])
    # ... while the proxy still reaches that participant.
    assert r.loc[3, "years_since_baseline_proxy"] == 2


def test_absent_anchors_leave_the_proxy_columns_intact() -> None:
    """Smoke configurations have neither anchor; the run must survive."""
    r = _extract()
    assert "fu1_year_reconstructed" not in r.columns
    assert "years_baseline_to_fu1" not in r.columns
    assert r.loc[1, "years_since_baseline_proxy"] == 2

    # An FU1 age without a baseline year gives the delta but no year.
    partial = _extract(fu1_age_df=_FU1_AGE)
    assert partial.loc[1, "years_baseline_to_fu1"] == 4
    assert "fu1_year_reconstructed" not in partial.columns


def test_the_join_column_does_not_leak_into_the_parquet() -> None:
    """basis_year is an input, not an output of this modality."""
    r = _extract(fu1_age_df=_FU1_AGE, baseline_year_df=_BASELINE_YEAR)
    assert "basis_year" not in r.columns


def test_derived_timing_columns_have_plausibility_ranges() -> None:
    """A derived column without a range is invisible to the harness."""
    for column in ("fu1_age", "years_baseline_to_fu1", "fu1_year_reconstructed"):
        assert column in FOLLOWUP1_VISIT_META_RANGES
