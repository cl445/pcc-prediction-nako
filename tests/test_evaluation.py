"""Tests for evaluation metrics (balanced accuracy)."""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pcc_analysis.evaluation import (
    EvaluationMetrics,
    _seeded_global_rng,
    compute_decision_curve_analysis,
    expected_calibration_error,
    plot_modality_contribution_boxplot,
    quantile_bin_edges,
)


def test_balanced_accuracy_perfect_separation() -> None:
    """Perfect predictions should yield balanced_accuracy = 1.0."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
    y_pred_proba = np.array([0.0, 0.1, 0.1, 0.2, 0.2, 0.8, 0.9, 0.9, 1.0, 1.0])
    ev = EvaluationMetrics(y_true, y_pred_proba)
    metrics = ev.compute_all_metrics()

    assert metrics["balanced_accuracy"] == 1.0


def test_balanced_accuracy_random() -> None:
    """All-same predictions on balanced classes should give 0.5."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
    # Predict everything as class 1 -> sensitivity=1, specificity=0 -> BA=0.5
    y_pred_proba = np.ones(10)
    ev = EvaluationMetrics(y_true, y_pred_proba)
    metrics = ev.compute_all_metrics()

    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_balanced_accuracy_present_in_metrics() -> None:
    """balanced_accuracy should be included in compute_all_metrics output."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 50 + [1] * 50, dtype=np.float64)
    y_pred_proba = rng.uniform(0, 1, 100)
    ev = EvaluationMetrics(y_true, y_pred_proba)
    metrics = ev.compute_all_metrics()

    assert "balanced_accuracy" in metrics
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


def test_calibration_metrics_structure() -> None:
    """compute_all_metrics must include calibration keys."""
    y_true = np.array([0] * 50 + [1] * 50, dtype=np.float64)
    y_pred_proba = np.concatenate(
        [
            np.random.default_rng(0).uniform(0.1, 0.4, 50),
            np.random.default_rng(0).uniform(0.6, 0.9, 50),
        ]
    )
    ev = EvaluationMetrics(y_true, y_pred_proba)
    metrics = ev.compute_all_metrics()
    for key in ("calibration_slope", "calibration_intercept", "ece"):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_ece_range() -> None:
    """ECE must be between 0 and 1."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 50 + [1] * 50, dtype=np.float64)
    y_pred_proba = rng.uniform(0, 1, 100)
    ev = EvaluationMetrics(y_true, y_pred_proba)
    metrics = ev.compute_all_metrics()
    assert 0.0 <= metrics["ece"] <= 1.0


def test_roc_auc_perfect() -> None:
    """Perfect predictions should yield ROC-AUC = 1.0."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
    y_pred_proba = np.array([0.0, 0.1, 0.1, 0.2, 0.2, 0.8, 0.9, 0.9, 1.0, 1.0])
    ev = EvaluationMetrics(y_true, y_pred_proba)
    metrics = ev.compute_all_metrics()
    assert metrics["roc_auc"] == 1.0


def test_single_class_returns_nan() -> None:
    """When only one class is present, discriminative metrics should be NaN."""
    y_true = np.ones(20, dtype=np.float64)
    y_pred_proba = np.random.default_rng(0).uniform(0, 1, 20)
    ev = EvaluationMetrics(y_true, y_pred_proba)
    metrics = ev.compute_all_metrics()
    assert np.isnan(metrics["calibration_slope"])
    assert np.isnan(metrics["ece"])


# -- Decision Curve Analysis -------------------------------------------------


def test_dca_columns() -> None:
    y_true = np.array([0, 0, 1, 1, 1], dtype=np.float64)
    y_pred = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
    dca = compute_decision_curve_analysis(y_true, y_pred)
    assert list(dca.columns) == ["threshold", "net_benefit", "treat_all", "treat_none"]
    assert len(dca) == 99  # default 99 thresholds


def test_dca_custom_thresholds() -> None:
    y_true = np.array([0, 1, 1, 0, 1], dtype=np.float64)
    y_pred = np.array([0.2, 0.7, 0.8, 0.3, 0.9])
    thresholds = np.array([0.3, 0.5, 0.7])
    dca = compute_decision_curve_analysis(y_true, y_pred, thresholds=thresholds)
    assert len(dca) == 3
    assert dca["threshold"].tolist() == pytest.approx([0.3, 0.5, 0.7])


def test_dca_treat_none_is_zero() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.float64)
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    dca = compute_decision_curve_analysis(y_true, y_pred)
    assert (dca["treat_none"] == 0.0).all()


def test_dca_perfect_predictions() -> None:
    """Perfect separation should give positive net benefit."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    y_pred = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    dca = compute_decision_curve_analysis(y_true, y_pred, thresholds=np.array([0.5]))
    assert dca["net_benefit"].iloc[0] > 0


def test_modality_contribution_jitter_is_reproducible() -> None:
    """The same scores must place the strip-plot points identically."""
    scores = pd.DataFrame(
        {
            "modality": ["a"] * 10 + ["b"] * 10,
            "roc_auc": np.linspace(0.5, 0.9, 20),
        }
    )
    offsets = []
    for _ in range(2):
        _, ax = plt.subplots()
        plot_modality_contribution_boxplot(scores, ax=ax)
        offsets.append(
            np.concatenate([np.asarray(c.get_offsets()) for c in ax.collections])
        )
        plt.close("all")
    np.testing.assert_array_equal(offsets[0], offsets[1])


def test_seeded_global_rng_restores_the_stream() -> None:
    """Seeding for a figure must not shift anyone else's draws."""
    # Legacy API throughout, because the global stream is what is under test.
    np.random.seed(1234)  # noqa: NPY002
    before = np.random.get_state()  # noqa: NPY002
    with _seeded_global_rng(0):
        np.random.uniform(size=5)  # noqa: NPY002
    assert np.array_equal(before[1], np.random.get_state()[1])  # noqa: NPY002


def test_ece_bins_hold_equal_counts_not_equal_widths() -> None:
    """The partition must follow the data, not the unit interval.

    Predictions concentrated in a narrow band are the case the manuscript
    reports on. Under equal-width bins over [0, 1] such a sample lands in a
    handful of bins; under the equal-count binning used here every bin holds
    the same number of participants.
    """
    rng = np.random.default_rng(0)
    y_pred_proba = rng.uniform(0.10, 0.40, size=1000)

    edges = quantile_bin_edges(y_pred_proba, n_bins=10)
    counts = [
        int(((y_pred_proba > lo) & (y_pred_proba <= hi)).sum())
        for lo, hi in itertools.pairwise(edges)
    ]
    counts[0] += 1  # the lowest bin also owns its left edge

    assert len(edges) == 11
    assert counts == [100] * 10


def test_ece_sees_miscalibration_that_wide_bins_average_away() -> None:
    """Opposite-signed deviations inside one wide bin must not cancel.

    Two groups sit in the same equal-width bin and are wrong in opposite
    directions by the same amount. Equal-width binning reports them as
    calibrated; equal-count binning separates them and reports the error.
    """
    y_pred_proba = np.concatenate([np.full(500, 0.21), np.full(500, 0.29)])
    # Observed rates deviate by +0.10 and -0.10 from the predictions.
    y_true = np.concatenate(
        [
            np.repeat([1.0, 0.0], [155, 345]),
            np.repeat([1.0, 0.0], [95, 405]),
        ]
    )

    ece = expected_calibration_error(y_true, y_pred_proba, n_bins=10)

    assert ece == pytest.approx(0.10, abs=1e-9)


def test_ece_is_zero_for_a_perfectly_calibrated_sample() -> None:
    y_pred_proba = np.concatenate([np.full(1000, 0.25), np.full(1000, 0.75)])
    y_true = np.concatenate(
        [
            np.repeat([1.0, 0.0], [250, 750]),
            np.repeat([1.0, 0.0], [750, 250]),
        ]
    )

    assert expected_calibration_error(y_true, y_pred_proba) == pytest.approx(0.0)


def test_the_metrics_object_reports_the_same_ece_as_the_free_function() -> None:
    """Figures and metrics.csv must not drift apart again."""
    rng = np.random.default_rng(7)
    y_pred_proba = rng.uniform(0.05, 0.65, size=2000)
    y_true = rng.binomial(1, y_pred_proba).astype(np.float64)

    ev = EvaluationMetrics(y_true, y_pred_proba)

    assert ev.compute_ece() == expected_calibration_error(y_true, y_pred_proba)
