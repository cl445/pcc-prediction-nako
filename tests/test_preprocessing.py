"""Smoke tests for ModalityPreprocessor."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from pcc_analysis.preprocessing import (
    FeatureToConfounder,
    LogTransformer,
    MissingnessThreshold,
    ModalityPreprocessor,
    create_cognitive_preprocessor,
    create_demographics_preprocessor,
    create_lab_values_preprocessor,
    create_lung_function_preprocessor,
    create_medical_history_preprocessor,
    create_physical_activity_preprocessor,
    create_preprocessor_for_modality,
    create_ses_preprocessor,
)


def test_ordinal_encoding() -> None:
    df = pd.DataFrame({"severity": ["low", "medium", "high", "low", "high"]})
    prep = ModalityPreprocessor(ordinal_columns={"severity": ["low", "medium", "high"]})
    prep.fit(df)
    out = prep.transform(df)
    assert "severity" in out.columns
    # Values should be numeric (0, 1, 2)
    assert out["severity"].dtype == np.float64
    assert set(out["severity"].unique()) == {0.0, 1.0, 2.0}


def test_date_extraction() -> None:
    df = pd.DataFrame(
        {
            "visit_date": ["2023-01-15", "2023-06-20", "2024-03-10"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    prep = ModalityPreprocessor(date_columns=["visit_date"])
    prep.fit(df)
    out = prep.transform(df)
    assert "visit_date_year" in out.columns
    assert "visit_date_month" in out.columns
    assert "visit_date" not in out.columns
    assert out["visit_date_year"].tolist() == [2023.0, 2023.0, 2024.0]
    assert out["visit_date_month"].tolist() == [1.0, 6.0, 3.0]


def test_drop_columns() -> None:
    df = pd.DataFrame(
        {
            "keep": [1.0, 2.0, 3.0],
            "drop_me": [4.0, 5.0, 6.0],
        }
    )
    prep = ModalityPreprocessor(drop_columns=["drop_me"])
    prep.fit(df)
    out = prep.transform(df)
    assert "drop_me" not in out.columns
    assert "keep" in out.columns


def test_passthrough_numeric() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )
    prep = ModalityPreprocessor()
    prep.fit(df)
    out = prep.transform(df)
    pd.testing.assert_frame_equal(out, df)


def test_requires_dataframe() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    prep = ModalityPreprocessor()
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        # Passing the wrong type is the point of the test.
        prep.fit(arr)  # pyrefly: ignore[bad-argument-type]


def test_feature_to_confounder_extracts_and_drops() -> None:
    """FeatureToConfounder extracts named column and makes it available."""
    df = pd.DataFrame(
        {
            "region_a": [100.0, 200.0, 300.0],
            "region_b": [50.0, 100.0, 150.0],
            "etiv": [1000.0, 2000.0, 3000.0],
        }
    )
    ftc = FeatureToConfounder(columns="etiv")
    result = ftc.fit_transform(df)
    assert "etiv" not in result.columns
    assert list(result.columns) == ["region_a", "region_b"]
    extra = ftc.get_extra_confounders()
    assert extra is not None
    assert "etiv" in extra.columns
    np.testing.assert_allclose(
        np.asarray(extra["etiv"].values, dtype=float), [1000.0, 2000.0, 3000.0]
    )


def test_feature_to_confounder_passthrough_without_column() -> None:
    """FeatureToConfounder passes through data if named column is absent."""
    df = pd.DataFrame({"region_a": [1.0, 2.0], "region_b": [3.0, 4.0]})
    ftc = FeatureToConfounder(columns="etiv")
    result = ftc.fit_transform(df)
    pd.testing.assert_frame_equal(result, df)
    assert ftc.get_extra_confounders() is None


def test_feature_to_confounder_handles_nan() -> None:
    """NaN values in extracted column are preserved in extra confounders."""
    df = pd.DataFrame(
        {
            "region_a": [100.0, 200.0],
            "etiv": [1000.0, np.nan],
        }
    )
    ftc = FeatureToConfounder(columns="etiv")
    result = ftc.fit_transform(df)
    assert "etiv" not in result.columns
    extra = ftc.get_extra_confounders()
    assert extra is not None
    assert np.isnan(extra["etiv"].iloc[1])


def test_feature_to_confounder_multiple_columns() -> None:
    """FeatureToConfounder can extract multiple columns at once."""
    df = pd.DataFrame(
        {
            "region_a": [1.0, 2.0],
            "etiv": [1000.0, 2000.0],
            "head_size": [50.0, 60.0],
        }
    )
    ftc = FeatureToConfounder(columns=["etiv", "head_size"])
    result = ftc.fit_transform(df)
    assert list(result.columns) == ["region_a"]
    extra = ftc.get_extra_confounders()
    assert extra is not None
    assert list(extra.columns) == ["etiv", "head_size"]


# -- MissingnessThreshold -----------------------------------------------------


def test_missingness_threshold_drops_high_missing() -> None:
    df = pd.DataFrame(
        {
            "ok": [1.0, 2.0, 3.0, 4.0],
            "bad": [np.nan, np.nan, np.nan, 1.0],  # 75% missing
        }
    )
    mt = MissingnessThreshold(threshold=0.5)
    mt.fit(df)
    out = mt.transform(df)
    assert "ok" in out.columns
    assert "bad" not in out.columns


def test_missingness_threshold_keeps_below_threshold() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],  # 25% missing
            "b": [1.0, 2.0, 3.0, 4.0],  # 0% missing
        }
    )
    mt = MissingnessThreshold(threshold=0.5)
    mt.fit(df)
    out = mt.transform(df)
    assert list(out.columns) == ["a", "b"]


def test_missingness_threshold_get_feature_names_out() -> None:
    df = pd.DataFrame({"keep": [1.0, 2.0], "drop": [np.nan, np.nan]})
    mt = MissingnessThreshold(threshold=0.5)
    mt.fit(df)
    names = mt.get_feature_names_out(["keep", "drop"])
    assert list(names) == ["keep"]


def test_missingness_threshold_requires_dataframe() -> None:
    mt = MissingnessThreshold()
    with pytest.raises(TypeError, match="pandas DataFrame"):
        # Passing the wrong type is the point of the test.
        mt.fit(np.array([[1, 2]]))  # pyrefly: ignore[bad-argument-type]


# -- Preprocessor factory functions -------------------------------------------


@pytest.mark.parametrize(
    "factory_fn",
    [
        create_demographics_preprocessor,
        create_ses_preprocessor,
        create_cognitive_preprocessor,
        create_physical_activity_preprocessor,
        create_medical_history_preprocessor,
        create_lab_values_preprocessor,
        create_lung_function_preprocessor,
    ],
)
def test_factory_returns_modality_preprocessor(
    factory_fn: Callable[[], ModalityPreprocessor],
) -> None:
    prep = factory_fn()
    assert isinstance(prep, ModalityPreprocessor)


def test_create_preprocessor_for_known_modality() -> None:
    for name in [
        "demographics",
        "ses",
        "cognitive",
        "physical_activity",
        "medical_history",
        "lab_values",
        "lung_function",
    ]:
        prep = create_preprocessor_for_modality(name)
        assert isinstance(prep, ModalityPreprocessor), f"Failed for {name}"


def test_create_preprocessor_for_unknown_modality() -> None:
    assert create_preprocessor_for_modality("nonexistent") is None


def test_ses_preprocessor_drops_expected_columns() -> None:
    prep = create_ses_preprocessor()
    assert prep.drop_columns is not None
    assert "retired" in prep.drop_columns
    assert "income_weighted" in prep.drop_columns


def test_demographics_preprocessor_has_nominal_columns() -> None:
    prep = create_demographics_preprocessor()
    assert prep.nominal_columns == ["basis_sex", "basis_uort"]


# -- Nominal encoding --------------------------------------------------------


def test_nominal_encoding_consistent_columns() -> None:
    """OneHotEncoder should produce consistent columns regardless of test data categories."""
    train = pd.DataFrame(
        {
            "color": ["red", "green", "blue", "red", "blue"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    prep = ModalityPreprocessor(nominal_columns=["color"])
    prep.fit(train)
    train_out = prep.transform(train)

    # Subset of categories: should still get same columns
    test_subset = pd.DataFrame(
        {
            "color": ["red", "green"],
            "value": [10.0, 20.0],
        }
    )
    subset_out = prep.transform(test_subset)
    assert list(train_out.columns) == list(subset_out.columns)

    # Unknown category: should not error, and produce same columns
    test_unknown = pd.DataFrame(
        {
            "color": ["red", "purple"],
            "value": [10.0, 20.0],
        }
    )
    unknown_out = prep.transform(test_unknown)
    assert list(train_out.columns) == list(unknown_out.columns)


# -- LogTransformer ------------------------------------------------


def test_log_transformer_applies_log1p_to_listed_columns() -> None:
    df = pd.DataFrame(
        {
            "income_weighted": [0.0, 1.0, 9.0, 999.0],
            "education_years": [4, 8, 12, 20],
        }
    )
    out = LogTransformer(columns=["income_weighted"]).fit_transform(df)
    np.testing.assert_allclose(out["income_weighted"], np.log1p([0.0, 1.0, 9.0, 999.0]))
    # Untouched column passes through unchanged.
    assert (out["education_years"] == df["education_years"]).all()


def test_log_transformer_skips_missing_columns() -> None:
    """A configured column not present in X must be silently skipped."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    out = LogTransformer(columns=["missing_col", "a"]).fit_transform(df)
    np.testing.assert_allclose(out["a"], np.log1p([1.0, 2.0]))
    assert "missing_col" not in out.columns


def test_log_transformer_clips_negatives() -> None:
    """Negative values are clipped to 0 and a warning is logged."""
    df = pd.DataFrame({"crp": [-1.0, 0.0, 9.0]})
    out = LogTransformer(columns=["crp"]).fit_transform(df)
    np.testing.assert_allclose(out["crp"], np.log1p([0.0, 0.0, 9.0]))


def test_log_transformer_preserves_nan() -> None:
    df = pd.DataFrame({"crp": [1.0, np.nan, 9.0]})
    out = LogTransformer(columns=["crp"]).fit_transform(df)
    assert pd.isna(out["crp"].iloc[1])
    np.testing.assert_allclose(out["crp"].dropna().to_numpy(), np.log1p([1.0, 9.0]))


# -- Sentinel code consistency ------------------------------------------------


def test_kmatrix_sentinels_superset_of_psychometric() -> None:
    """Kmatrix sentinel codes must cover all psychometric missing codes."""
    from pcc_analysis.data_processing import (
        _KMATRIX_SENTINEL_CODES,
        _PSYCHOMETRIC_MISSING,
    )

    missing = set(_PSYCHOMETRIC_MISSING) - set(_KMATRIX_SENTINEL_CODES)
    assert not missing, f"Psychometric codes not in kmatrix sentinels: {missing}"
