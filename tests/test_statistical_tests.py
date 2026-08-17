"""Tests for Nadeau-Bengio corrected t-test and Holm-Bonferroni correction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pcc_analysis.statistical_tests import (
    compare_modalities_pairwise,
    holm_bonferroni_correction,
    nadeau_bengio_corrected_t_test,
)


class TestNadeauBengio:
    """Tests for the Nadeau-Bengio corrected t-test."""

    def test_identical_scores_give_p_one(self) -> None:
        """Identical scores should produce p=1."""
        scores = np.array([0.7, 0.72, 0.68, 0.71, 0.69])
        result = nadeau_bengio_corrected_t_test(scores, scores, n_train=800, n_test=200)
        assert result["p_value"] == pytest.approx(1.0)
        assert result["t_statistic"] == pytest.approx(0.0)
        assert result["mean_diff"] == pytest.approx(0.0)

    def test_different_scores_give_small_p(self) -> None:
        """Clearly different scores should give p < 0.05."""
        # Scores must have non-constant differences so variance > 0
        rng = np.random.default_rng(42)
        scores_a = 0.90 + rng.normal(0, 0.02, 10)
        scores_b = 0.50 + rng.normal(0, 0.02, 10)
        result = nadeau_bengio_corrected_t_test(
            scores_a, scores_b, n_train=900, n_test=100
        )
        assert result["p_value"] < 0.05
        assert result["mean_diff"] > 0

    def test_correction_more_conservative_than_naive(self) -> None:
        """NB correction should give larger p-values than uncorrected t-test."""
        from scipy.stats import ttest_rel

        rng = np.random.default_rng(42)
        a = rng.normal(0.75, 0.02, 5)
        b = rng.normal(0.70, 0.02, 5)

        nb_result = nadeau_bengio_corrected_t_test(a, b, n_train=800, n_test=200)
        _, naive_p = ttest_rel(a, b)
        # NB-corrected p should be >= naive p (more conservative)
        assert nb_result["p_value"] >= naive_p - 1e-10

    def test_output_keys(self) -> None:
        """Output should contain expected keys."""
        scores = np.array([0.7, 0.72, 0.68, 0.71, 0.69])
        result = nadeau_bengio_corrected_t_test(scores, scores, n_train=800, n_test=200)
        expected_keys = {
            "t_statistic",
            "p_value",
            "df",
            "mean_diff",
            "corrected_std",
        }
        assert set(result.keys()) == expected_keys

    def test_mismatched_lengths_raise(self) -> None:
        """Different-length score arrays should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            nadeau_bengio_corrected_t_test(
                np.array([0.7, 0.8]),
                np.array([0.7, 0.8, 0.9]),
                n_train=800,
                n_test=200,
            )


class TestHolmBonferroni:
    """Tests for the Holm-Bonferroni correction."""

    def test_single_p_value_unchanged(self) -> None:
        """A single p-value should be unchanged."""
        result = holm_bonferroni_correction([0.03])
        assert result == pytest.approx([0.03])

    def test_monotonicity(self) -> None:
        """Adjusted p-values (sorted) should be monotonically non-decreasing."""
        raw = [0.001, 0.04, 0.03, 0.5]
        adjusted = holm_bonferroni_correction(raw)
        sorted_adj = sorted(adjusted)
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1] - 1e-15

    def test_adjustment_never_below_raw(self) -> None:
        """Adjusted p-values should always be >= raw p-values."""
        raw = [0.01, 0.04, 0.03]
        adjusted = holm_bonferroni_correction(raw)
        for r, a in zip(raw, adjusted, strict=True):
            assert a >= r - 1e-15

    def test_clipped_to_one(self) -> None:
        """Adjusted p-values should never exceed 1.0."""
        raw = [0.5, 0.6, 0.7]
        adjusted = holm_bonferroni_correction(raw)
        for a in adjusted:
            assert a <= 1.0

    def test_dict_input(self) -> None:
        """Dict input should return dict output with same keys."""
        raw = {"mod_a": 0.01, "mod_b": 0.04, "mod_c": 0.03}
        adjusted = holm_bonferroni_correction(raw)
        assert isinstance(adjusted, dict)
        assert set(adjusted.keys()) == set(raw.keys())
        for k, raw_p in raw.items():
            assert adjusted[k] >= raw_p - 1e-15

    def test_empty_input(self) -> None:
        """Empty input should return empty output."""
        assert holm_bonferroni_correction([]) == []


class TestCompareModalitiesPairwise:
    """Tests for pairwise modality comparison."""

    def test_basic_comparison(self) -> None:
        """Should produce a DataFrame with expected columns."""
        df = pd.DataFrame(
            {
                "fold": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "modality": [
                    "demographics",
                    "demographics",
                    "demographics",
                    "lab_values",
                    "lab_values",
                    "lab_values",
                    "cognitive",
                    "cognitive",
                    "cognitive",
                ],
                "pr_auc": [0.5, 0.52, 0.48, 0.7, 0.72, 0.68, 0.6, 0.62, 0.58],
            }
        )
        result = compare_modalities_pairwise(
            df,
            reference_modality="demographics",
            metric="pr_auc",
            n_total=1000,
            n_folds=3,
        )
        assert "p_value_holm" in result.columns
        assert "p_value_raw" in result.columns
        assert len(result) == 2  # lab_values and cognitive
        # Holm should be >= raw
        for _, row in result.iterrows():
            assert row["p_value_holm"] >= row["p_value_raw"] - 1e-15


def test_an_absent_reference_modality_is_an_error_not_an_empty_table() -> None:
    """A missing reference must not pass for 'nothing was significant'.

    Split-modality runs carry mental health only as its five sub-scales. With
    no guard, the reference contributes zero folds, every modality is skipped
    for a fold-count mismatch, and the caller gets a valid, empty result.
    """
    df = pd.DataFrame(
        {
            "fold": [1, 2, 3, 1, 2, 3],
            "modality": ["mh_phq9"] * 3 + ["demographics"] * 3,
            "pr_auc": [0.5, 0.52, 0.48, 0.4, 0.42, 0.38],
        }
    )

    with pytest.raises(ValueError, match="mental_health"):
        compare_modalities_pairwise(
            df,
            reference_modality="mental_health",
            metric="pr_auc",
            n_total=1000,
            n_folds=3,
        )


def test_a_modality_scored_on_other_folds_is_skipped_not_mispaired(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Equal fold *counts* are not the same as equal folds.

    ``lab_values`` here is scored on folds 1, 2 and 4 while the reference has
    1, 2 and 3 — three each. Pairing two sorted arrays by position would put
    fold 4 opposite fold 3 and return a paired test of two different splits,
    with nothing in the output to show for it. Alignment is on the fold id, so
    the mismatch is reported and the row is dropped; ``cognitive``, which does
    share the folds, is unaffected.
    """
    df = pd.DataFrame(
        {
            "fold": [1, 2, 3, 1, 2, 4, 1, 2, 3],
            "modality": ["demographics"] * 3 + ["lab_values"] * 3 + ["cognitive"] * 3,
            "pr_auc": [0.5, 0.52, 0.48, 0.7, 0.72, 0.68, 0.6, 0.62, 0.58],
        }
    )

    with caplog.at_level(logging.WARNING):
        result = compare_modalities_pairwise(
            df,
            reference_modality="demographics",
            metric="pr_auc",
            n_total=1000,
            n_folds=3,
        )

    assert list(result["modality"]) == ["cognitive"]
    assert "lab_values" in caplog.text
    assert "same folds" in caplog.text
