"""Schema of the smoke-test profile JSON.

``profiler.py`` writes this structure from real parquet files and
``generators.py`` reads it back to synthesise data with the same shape. The
two modules only agree on a file format, so the format is spelled out here
rather than in either of them.

Column profiles form a union discriminated on ``dtype``, which is the tag
``generators._generate_column`` branches on: each branch sees exactly the
fields its own dtype carries.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class _ColumnProfileCommon(TypedDict):
    """Fields every column profile carries, whatever its dtype."""

    missing_frac: float
    # Set by ``profiler.extract_profile`` for columns the pipeline derives
    # from others, so the generator can recompute them instead of drawing
    # them independently.
    derived: NotRequired[bool]
    in_derivation_group: NotRequired[bool]


class _NumericStats(_ColumnProfileCommon):
    """Distribution summary shared by the numeric and integer dtypes."""

    mean: float
    std: float
    q01: float
    q25: float
    q50: float
    q75: float
    q99: float


class NumericColumnProfile(_NumericStats):
    """Floating-point column: drawn from a normal, clipped to [q01, q99]."""

    dtype: Literal["numeric"]


class IntegerColumnProfile(_NumericStats):
    """Integer column: as numeric, but rounded and restored to its dtype."""

    dtype: Literal["integer"]
    pandas_dtype: str


class BooleanColumnProfile(_ColumnProfileCommon):
    """Boolean or 0/1 integer column, summarised by its rate of ones."""

    dtype: Literal["boolean"]
    true_frac: float


class CategoricalColumnProfile(_ColumnProfileCommon):
    """Categorical column, summarised by its category frequencies."""

    dtype: Literal["categorical"]
    categories: dict[str, float]


class TextColumnProfile(_ColumnProfileCommon):
    """String or date column. Only the missing rate survives profiling —
    the analysis pipeline reads no values out of either."""

    dtype: Literal["string", "datetime"]


class UnknownColumnProfile(_ColumnProfileCommon):
    """Column of a dtype the profiler has no summary for; generated as NaN."""

    dtype: Literal["unknown"]


ColumnProfile = (
    NumericColumnProfile
    | IntegerColumnProfile
    | BooleanColumnProfile
    | CategoricalColumnProfile
    | TextColumnProfile
    | UnknownColumnProfile
)

COLUMN_DTYPES: frozenset[str] = frozenset(
    {"numeric", "integer", "boolean", "categorical", "string", "datetime", "unknown"}
)


class DerivationGroupProfile(TypedDict):
    """One set of mutually exclusive indicator columns and their rates."""

    columns: list[str]
    proportions: dict[str, float]


class ModalityProfile(TypedDict):
    """Everything needed to synthesise one modality's parquet file.

    No row or column count: the generator sizes its output from
    ``--n-subjects`` and derives the width from ``columns``, so the counts of
    the delivery this was profiled from are not needed to synthesise anything.
    They describe a particular NAKO delivery rather than a distribution, which
    is the one thing this file is meant to hold.
    """

    systematic_missing_frac: float
    columns: dict[str, ColumnProfile]
    derivation_groups: NotRequired[dict[str, DerivationGroupProfile]]


class ProfileMeta(TypedDict):
    """What the profile covers.

    Carries no generation timestamp and no source path. The profile is a
    checked-in artifact whose only job is to shape synthetic fixtures; a
    stamp of when and on whose machine it was derived is provenance for a
    directory nobody else can reach, and it churns the file on every
    regeneration without changing what it describes.
    """

    n_modalities: int


class Profile(TypedDict):
    """Top level of the profile JSON."""

    meta: ProfileMeta
    modalities: dict[str, ModalityProfile]
