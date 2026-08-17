"""Quality-harness for the processed-parquet fleet.

The harness enforces three invariants on every parquet under
``data/processed/``:

1. **Missing-ness is captured**: no residual NAKO sentinel codes survive,
   no sentinel-flagged integer-like values in numeric columns, and no
   ``object``-dtype columns sneaking through as implicit missing markers.

2. **Dtypes are optimal**: integer columns use nullable
   ``UInt8/16/32`` or ``Int8/16/32``; floating columns are at most
   ``Float32`` unless the observed range genuinely requires Float64
   (NAKO measurements never do). ``ID`` is exempt.

3. **No outliers**: columns listed in
   :mod:`pcc_analysis._plausibility` lie inside their declared range.

Tests are skipped when ``data/processed/`` is not populated (e.g.\\ on a
fresh clone before ``scripts/pipeline/01_process_data.py`` has run); see
``_require_processed``.

Run via::

    uv run pytest tests/test_data_processing_quality.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.core.arrays.integer import IntegerDtype

from pcc_analysis._plausibility import PARQUET_RANGE_CATALOGUES
from pcc_analysis.config import get_processed_dir
from pcc_analysis.data_processing import NAKO_SENTINEL_CODES

# ---------------------------------------------------------------------------
# Allow-lists for legitimate exceptions
# ---------------------------------------------------------------------------

# Parquets whose row granularity is legitimately not 1-row-per-participant.
# Duplicate IDs are expected and the ID-uniqueness check is skipped.
LONG_FORMAT_PARQUETS: set[str] = {"kvad", "mh_longitudinal"}

# Parquets that do not key on participant ID and exist as flat-per-row
# look-ups. The ID-column check is skipped for them; they must be joined
# positionally (see their respective extractors for the convention).
NO_ID_PARQUETS: set[str] = {"baseline_sex_age"}

# Per-(parquet, column) allow-list of sentinel-numeric values that carry
# domain meaning rather than "missing". Example: augmentation-index
# legitimately takes the value -5 for certain waveform conditions, and
# the d_co2_k0 routing gate uses 7775 ("filter not shown") as a
# structural not-applicable marker rather than an error. These
# exemptions are approved by the extractor authors; the harness treats
# listed values as valid observations rather than sentinels.
SENTINEL_EXEMPTIONS: dict[tuple[str, str], set[int]] = {
    ("cardiovascular", "augmentation_index"): {-5},
    ("cognitive_tests", "word_list_learning"): {-5},
    ("cognitive_tests", "word_list_forgetting"): {-5},
    ("cognitive_tests", "stroop_interference_effect"): {-5},
    ("corona2_pcc", "d_co2_k0"): {7775, 8888},
    # NAKO case-identifier columns may legitimately contain integers that
    # resemble sentinels (sequence numbers reaching 99999/999999, plus the
    # lab sentinels 111111/444444/555555).
    ("kvad", "kvad_amb_case_id"): {
        99999,
        111111,
        444444,
        555555,
        666666,
        999999,
    },
    # Audit-trail row count: how many KVAD ambulatory diagnoses each
    # participant has in the pre-pandemic window after the certainty
    # filter. Heavy users of the public-health system genuinely reach
    # the low-thousands range (observed max ≈ 5 000 on production data),
    # so any of the small-magnitude sentinels (-9, 999, 9999, ...) can
    # surface as a real count. The column carries no missingness — a
    # missing diagnosis history is encoded by the participant's absence
    # from the parquet, not by a sentinel value. All three sensitivity
    # variants (default / chronic / strict) emit this identical count
    # column, so the exemption applies to each variant's parquet stem.
    ("kvad_pcs_proxy", "pcs_t0_proxy_n_diagnoses"): set(NAKO_SENTINEL_CODES),
    ("kvad_pcs_proxy_chronic", "pcs_t0_proxy_n_diagnoses"): set(NAKO_SENTINEL_CODES),
    ("kvad_pcs_proxy_strict", "pcs_t0_proxy_n_diagnoses"): set(NAKO_SENTINEL_CODES),
}

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_processed_dir() -> Path | None:
    """The configured processed-data directory, or None if unconfigured.

    This runs at import time because the checks below are parametrized over
    the parquets it contains. A checkout without a config.toml — a fresh
    clone, or CI, which has no NAKO access by design — has to skip these
    checks. Letting the exception escape here fails collection and takes
    the entire test suite down with it, rather than the one module that
    actually needs the data.
    """
    try:
        return get_processed_dir()
    except FileNotFoundError:
        return None


PROCESSED_DIR = _discover_processed_dir()
PARQUET_FILES: list[Path] = (
    sorted(PROCESSED_DIR.glob("*.parquet")) if PROCESSED_DIR else []
)


def _require_processed() -> None:
    if PROCESSED_DIR is None:
        pytest.skip(
            "No config.toml; copy config.toml.example and set the data paths "
            "to run the data-processing quality checks."
        )
    if not PARQUET_FILES:
        pytest.skip(
            f"No parquet files under {PROCESSED_DIR}; "
            "run `uv run python scripts/pipeline/01_process_data.py` first."
        )


# ---------------------------------------------------------------------------
# Missing-ness invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parquet",
    PARQUET_FILES,
    ids=[p.stem for p in PARQUET_FILES] or ["<none>"],
)
def test_no_residual_sentinel_codes(parquet: Path) -> None:
    """No NAKO sentinel code survives in any numeric column.

    Values approved in :data:`SENTINEL_EXEMPTIONS` carry domain meaning
    and are treated as valid observations.
    """
    _require_processed()
    df = pd.read_parquet(parquet)
    offenders: dict[str, list[int]] = {}
    for col in df.columns:
        if col == "ID" or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        exempt = SENTINEL_EXEMPTIONS.get((parquet.stem, col), set())
        hits = [
            c for c in NAKO_SENTINEL_CODES if c not in exempt and (df[col] == c).any()
        ]
        if hits:
            offenders[col] = hits
    assert not offenders, (
        f"{parquet.name}: residual NAKO sentinel codes detected — "
        f"{offenders}. Update the extractor to call _replace_nako_missing() "
        "or approve the value in SENTINEL_EXEMPTIONS with a domain rationale."
    )


@pytest.mark.parametrize(
    "parquet",
    PARQUET_FILES,
    ids=[p.stem for p in PARQUET_FILES] or ["<none>"],
)
def test_no_object_dtype_columns(parquet: Path) -> None:
    """Object dtype is forbidden outside explicit categorical columns.

    Allow-list: well-known string columns (``basis_sex``, ``sex``, ``wave``).
    Everything else must be numeric / nullable-integer / categorical.
    """
    _require_processed()
    df = pd.read_parquet(parquet)
    allowed_object_cols = {"basis_sex", "sex", "wave", "ID"}
    bad = [
        col
        for col in df.columns
        if pd.api.types.is_object_dtype(df[col]) and col not in allowed_object_cols
    ]
    assert not bad, (
        f"{parquet.name}: object-dtype columns disallowed — {bad}. "
        "Cast to a concrete numeric / nullable-integer / Categorical dtype."
    )


# ---------------------------------------------------------------------------
# Dtype-optimality invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parquet",
    PARQUET_FILES,
    ids=[p.stem for p in PARQUET_FILES] or ["<none>"],
)
def test_integer_columns_use_narrow_dtypes(parquet: Path) -> None:
    """Integer columns should use the narrowest nullable dtype that fits."""
    _require_processed()
    df = pd.read_parquet(parquet)
    wasteful: list[str] = []
    for col in df.columns:
        if col == "ID" or not isinstance(df[col].dtype, IntegerDtype):
            continue
        if df[col].isna().all():
            continue
        mi, ma = df[col].min(), df[col].max()
        dtype = str(df[col].dtype)
        # We only flag clearly-wasteful cases: an Int64/UInt64 that fits in 32 bits.
        if dtype in {"Int64", "UInt64"} and mi >= -(2**31) and ma < 2**31:
            wasteful.append(f"{col} ({dtype}, range [{mi}, {ma}])")
    assert not wasteful, (
        f"{parquet.name}: oversized integer dtypes — {wasteful}. "
        "Run _optimize_dtypes() before saving."
    )


@pytest.mark.parametrize(
    "parquet",
    PARQUET_FILES,
    ids=[p.stem for p in PARQUET_FILES] or ["<none>"],
)
def test_float_columns_are_float32_when_possible(parquet: Path) -> None:
    """Float64 columns whose values fit Float32 envelope should be downcast."""
    _require_processed()
    df = pd.read_parquet(parquet)
    wasteful: list[str] = []
    float32_max = float(np.finfo(np.float32).max)
    for col in df.columns:
        if col == "ID" or str(df[col].dtype) != "Float64":
            continue
        finite = df[col].dropna()
        if finite.empty:
            continue
        if float(finite.abs().max()) <= float32_max:
            wasteful.append(f"{col} (max |x|={finite.abs().max():.3g})")
    assert not wasteful, (
        f"{parquet.name}: Float64 columns fit Float32 and should be downcast — "
        f"{wasteful}. Run _optimize_dtypes() before saving."
    )


# ---------------------------------------------------------------------------
# Plausibility invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parquet",
    PARQUET_FILES,
    ids=[p.stem for p in PARQUET_FILES] or ["<none>"],
)
def test_plausibility_ranges(parquet: Path) -> None:
    """Columns listed in the plausibility catalogue must lie inside range."""
    _require_processed()
    catalogue = PARQUET_RANGE_CATALOGUES.get(parquet.stem)
    if catalogue is None:
        pytest.skip(f"No plausibility catalogue declared for {parquet.stem}")
    df = pd.read_parquet(parquet)
    violations: dict[str, dict[str, Any]] = {}
    for col, (lo, hi) in catalogue.items():
        if col not in df.columns:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        finite = series.dropna()
        if finite.empty:
            continue
        n_low = int((finite < lo).sum())
        n_high = int((finite > hi).sum())
        if n_low or n_high:
            violations[col] = {
                "range_allowed": [lo, hi],
                "observed_min": float(finite.min()),
                "observed_max": float(finite.max()),
                "n_below": n_low,
                "n_above": n_high,
            }
    assert not violations, (
        f"{parquet.name}: plausibility violations — {violations}. "
        "Either widen the range in pcc_analysis._plausibility (if clinically "
        "justified) or call _enforce_plausibility() in the extractor."
    )


# ---------------------------------------------------------------------------
# Structural invariants (quick smoke checks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parquet",
    PARQUET_FILES,
    ids=[p.stem for p in PARQUET_FILES] or ["<none>"],
)
def test_has_id_column(parquet: Path) -> None:
    """Every parquet under data/processed must expose a unique ID column.

    Long-format parquets (listed in :data:`LONG_FORMAT_PARQUETS`) are
    exempt from the uniqueness check; no-ID parquets (listed in
    :data:`NO_ID_PARQUETS`) are exempt entirely.
    """
    _require_processed()
    if parquet.stem in NO_ID_PARQUETS:
        pytest.skip(
            f"{parquet.stem} is a flat-per-row lookup without participant ID "
            "by design (see NO_ID_PARQUETS)."
        )
    df = pd.read_parquet(parquet)
    assert "ID" in df.columns, (
        f"{parquet.name}: missing ID column. Every processed parquet must "
        "key on participant ID (or opt into NO_ID_PARQUETS with rationale)."
    )
    if parquet.stem in LONG_FORMAT_PARQUETS:
        return
    assert df["ID"].is_unique, (
        f"{parquet.name}: duplicate IDs detected — each participant must "
        "appear at most once (long-format parquets must be declared in "
        "LONG_FORMAT_PARQUETS)."
    )
