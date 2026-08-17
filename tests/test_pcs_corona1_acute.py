"""Unit tests for ``pcs_corona1_acute`` (Corona-1 cov87s → T1 PCS proxy)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pcc_analysis.data_processing._likert import _PSYCHOMETRIC_MISSING
from pcc_analysis.data_processing.pcs_corona1_acute import (
    _aggregate_cov87s,
    _read_corona1_block,
    process_pcs_corona1_acute,
)


def _corona1_frame(
    rows: list[dict[str, object]],
    *,
    extra_cols: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Build a Corona-1 raw frame with all 13 cov87s columns by default.

    All cov87s columns are coerced to nullable ``Int64`` so the
    sentinel-replace path (which mirrors the real ``_load_nako_csv``
    pipeline that returns nullable dtypes) does not hit the pandas-3.0
    block-internal pop bug seen on plain object/int8 frames.
    """
    base: dict[str, object] = {f"d_co1_cov87s_{i}": 0 for i in range(1, 14)}
    if extra_cols:
        base.update(extra_cols)
    df = pd.DataFrame([{**base, **r} for r in rows])
    cov_cols = [f"d_co1_cov87s_{i}" for i in range(1, 14)]
    df[cov_cols] = df[cov_cols].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# _read_corona1_block
# ---------------------------------------------------------------------------


def test_read_corona1_block_replaces_psychometric_sentinels() -> None:
    raw = _corona1_frame(
        [
            {"ID": 1, "d_co1_cov87s_1": 1, "d_co1_cov87s_2": _PSYCHOMETRIC_MISSING[0]},
        ]
    )
    out = _read_corona1_block(raw)
    assert int(out["d_co1_cov87s_1"].iloc[0]) == 1
    assert pd.isna(out["d_co1_cov87s_2"].iloc[0])


def test_read_corona1_block_raises_when_no_cov87s_columns() -> None:
    raw = pd.DataFrame({"ID": [1], "unrelated": [0]})
    with pytest.raises(ValueError, match="d_co1_cov87s"):
        _read_corona1_block(raw)


# ---------------------------------------------------------------------------
# _aggregate_cov87s
# ---------------------------------------------------------------------------


def test_aggregate_renames_to_bahmer_aligned_names() -> None:
    raw = _corona1_frame([{"ID": 1, "d_co1_cov87s_1": 1, "d_co1_cov87s_8": 1}])
    block = _read_corona1_block(raw)
    out = _aggregate_cov87s(block)
    # cov87s_1 = Abgeschlagenheit → fatigue; cov87s_8 = Husten → cough.
    assert int(out["pcs_t1_acute_fatigue"].iloc[0]) == 1
    assert int(out["pcs_t1_acute_cough"].iloc[0]) == 1


def test_aggregate_other_flag_preserved_and_excluded_from_count() -> None:
    """cov87s_13 ('Andere') is renamed to ``pcs_t1_acute_other_flag`` but
    must NOT contribute to ``pcs_t1_acute_count`` (Bahmer-mapped only)."""
    raw = _corona1_frame([{"ID": 1, "d_co1_cov87s_13": 1}])  # only Andere
    block = _read_corona1_block(raw)
    out = _aggregate_cov87s(block)
    assert int(out["pcs_t1_acute_other_flag"].iloc[0]) == 1
    assert int(out["pcs_t1_acute_count"].iloc[0]) == 0


def test_aggregate_count_tolerates_two_missing_items() -> None:
    """Up to 2 missing items still produce a numeric count; beyond that the
    row's count becomes NA.

    This tolerance is a design decision for the acute-symptom count, not an
    inherited scoring convention, and specifically not one borrowed from
    PHQ-9/GAD-7: those sums are scored complete-case (see
    ``mh_longitudinal.MH_LONG_MAX_MISSING_PHQ9``). A symptom count degrades
    gracefully under a missing item in a way a summed severity scale does
    not, which is why the two instruments differ here.
    """
    # 2 missing → still a count.
    raw_2na = _corona1_frame(
        [
            {
                "ID": 1,
                "d_co1_cov87s_1": 1,
                "d_co1_cov87s_2": _PSYCHOMETRIC_MISSING[0],
                "d_co1_cov87s_3": _PSYCHOMETRIC_MISSING[0],
            }
        ]
    )
    block = _read_corona1_block(raw_2na)
    out = _aggregate_cov87s(block)
    assert int(out["pcs_t1_acute_count"].iloc[0]) == 1

    # 3 missing → NA count.
    raw_3na = _corona1_frame(
        [
            {
                "ID": 2,
                "d_co1_cov87s_1": 1,
                "d_co1_cov87s_2": _PSYCHOMETRIC_MISSING[0],
                "d_co1_cov87s_3": _PSYCHOMETRIC_MISSING[0],
                "d_co1_cov87s_4": _PSYCHOMETRIC_MISSING[0],
            }
        ]
    )
    block = _read_corona1_block(raw_3na)
    out = _aggregate_cov87s(block)
    assert pd.isna(out["pcs_t1_acute_count"].iloc[0])


def test_aggregate_n_answered_counts_observed_items() -> None:
    raw = _corona1_frame(
        [
            {
                "ID": 1,
                "d_co1_cov87s_1": 1,
                "d_co1_cov87s_2": _PSYCHOMETRIC_MISSING[0],
            }
        ]
    )
    block = _read_corona1_block(raw)
    out = _aggregate_cov87s(block)
    # 12 Bahmer-mapped items minus 1 NA = 11 answered.
    assert int(out["pcs_t1_acute_n_answered"].iloc[0]) == 11


# ---------------------------------------------------------------------------
# process_pcs_corona1_acute (end-to-end with synthetic input)
# ---------------------------------------------------------------------------


def test_process_pcs_corona1_acute_writes_parquet(tmp_path: Path) -> None:
    raw = _corona1_frame(
        [
            {"ID": 1, "d_co1_cov87s_1": 1, "d_co1_cov87s_5": 1},
            {"ID": 2, "d_co1_cov87s_8": 1},
        ]
    )
    paths = pytest.importorskip("pcc_analysis.config").NAKOPaths.__new__(
        pytest.importorskip("pcc_analysis.config").NAKOPaths
    )
    out_path = process_pcs_corona1_acute(paths, tmp_path, corona1_df=raw)
    assert out_path.exists()
    parquet = pd.read_parquet(out_path)
    assert sorted(parquet["ID"].tolist()) == [1, 2]
    assert "pcs_t1_acute_count" in parquet.columns
    assert "pcs_t1_acute_other_flag" in parquet.columns
