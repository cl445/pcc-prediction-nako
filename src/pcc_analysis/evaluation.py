"""Comprehensive evaluation metrics for PCC prediction.

Includes ROC-AUC, PR-AUC, calibration, DCA, bootstrap CIs,
and publication-ready plots.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ._tex_utils import latex_available

if TYPE_CHECKING:
    from ._types import EvaluationResult

logger = logging.getLogger(__name__)


def quantile_bin_edges(y_pred_proba: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Equal-count bin edges, matching sklearn's ``strategy="quantile"``.

    Degenerate edges (repeated quantiles, which happen when many predictions
    share a value) collapse into one, so the result can hold fewer than
    ``n_bins + 1`` edges.
    """
    edges = np.quantile(y_pred_proba, np.linspace(0, 1, n_bins + 1))
    return np.unique(edges)


def expected_calibration_error(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error over equal-count (quantile) bins.

    Equal-count rather than equal-width bins, because this model's predicted
    probabilities occupy a narrow band well inside [0, 1]. Equal-width bins
    over the full unit interval would put most participants into a handful of
    wide bins, and a wide bin averages away deviations that point in opposite
    directions: the same predictions score roughly three times better under
    equal-width binning than under equal-count binning, without being any
    better calibrated. Equal-count binning is also what the calibration curve
    in the manuscript is drawn with, so the reported number and the plotted
    curve describe the same partition.
    """
    edges = quantile_bin_edges(y_pred_proba, n_bins)
    total = 0.0
    for i in range(len(edges) - 1):
        # The lowest bin has to include its own left edge; every other bin
        # takes its left edge from the bin below it.
        if i == 0:
            mask = (y_pred_proba >= edges[i]) & (y_pred_proba <= edges[i + 1])
        else:
            mask = (y_pred_proba > edges[i]) & (y_pred_proba <= edges[i + 1])
        if mask.sum() == 0:
            continue
        total += mask.sum() * abs(y_true[mask].mean() - y_pred_proba[mask].mean())
    return float(total / len(y_true))


_MATPLOTLIB_CONFIGURED = False


def _configure_matplotlib() -> None:
    """Configure matplotlib for publication-quality plots (called once)."""
    global _MATPLOTLIB_CONFIGURED  # noqa: PLW0603
    if _MATPLOTLIB_CONFIGURED:
        return
    # text.usetex routes every string matplotlib draws through a latex
    # subprocess, so enabling it unconditionally makes a TeX distribution a
    # hard requirement for producing any figure at all — including in the
    # tests, which draw but never typeset for the paper. Without TeX,
    # matplotlib falls back to its own mathtext: the figures still render,
    # they just do not carry the paper's fonts. PGF output is dropped by
    # the same check at save time.
    has_latex = latex_available()
    if not has_latex:
        logger.warning(
            "LaTeX not found; rendering figures with matplotlib's own fonts "
            "instead of TeX. Paper-bound figures need a TeX distribution."
        )
    plt.rcParams.update(
        {
            "pgf.texsystem": "lualatex",
            "font.family": "serif",
            "text.usetex": has_latex,
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{fontspec}\setmainfont{TeX Gyre Termes}",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    _MATPLOTLIB_CONFIGURED = True


class EvaluationMetrics:
    """Comprehensive evaluation metrics for binary classification."""

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: np.ndarray | None = None,
    ) -> None:
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.y_pred = (y_pred_proba >= 0.5).astype(int) if y_pred is None else y_pred
        self.metrics_: dict[str, float] = {}

    def compute_all_metrics(self) -> dict[str, float]:
        n_classes = len(np.unique(self.y_true))
        if n_classes < 2:
            logger.warning("Only one class in y_true — metrics will be NaN")
            self.metrics_["roc_auc"] = float("nan")
            self.metrics_["pr_auc"] = float("nan")
        else:
            self.metrics_["roc_auc"] = float(
                roc_auc_score(self.y_true, self.y_pred_proba)
            )
            self.metrics_["pr_auc"] = float(
                average_precision_score(self.y_true, self.y_pred_proba)
            )
        self.metrics_["brier_score"] = float(
            brier_score_loss(self.y_true, self.y_pred_proba)
        )
        self.metrics_.update(self._calibration_metrics())
        self.metrics_.update(self._classification_metrics())
        return self.metrics_

    def _calibration_metrics(self) -> dict[str, float]:
        from sklearn.linear_model import LogisticRegression

        n_classes = len(np.unique(self.y_true))
        if n_classes < 2:
            return {
                "calibration_slope": float("nan"),
                "calibration_intercept": float("nan"),
                "ece": float("nan"),
            }

        proba_clip = np.clip(self.y_pred_proba, 1e-7, 1 - 1e-7)
        logits = np.log(proba_clip / (1 - proba_clip)).reshape(-1, 1)
        lr = LogisticRegression(C=1e10, max_iter=1000)
        lr.fit(logits, self.y_true)
        return {
            "calibration_slope": float(lr.coef_[0][0]),
            "calibration_intercept": float(lr.intercept_[0]),
            "ece": self.compute_ece(),
        }

    def compute_ece(self, n_bins: int = 10) -> float:
        return expected_calibration_error(self.y_true, self.y_pred_proba, n_bins=n_bins)

    def _classification_metrics(self) -> dict[str, float]:
        cm = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "balanced_accuracy": (sensitivity + specificity) / 2.0,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        }

    def plot_roc_curve(
        self,
        ax: mpl.axes.Axes | None = None,
        label: str = "Model",
    ) -> mpl.axes.Axes:
        _configure_matplotlib()
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        auc = roc_auc_score(self.y_true, self.y_pred_proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(
            xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve"
        )
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        return ax

    def plot_pr_curve(
        self,
        ax: mpl.axes.Axes | None = None,
        label: str = "Model",
    ) -> mpl.axes.Axes:
        _configure_matplotlib()
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        prec, rec, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        ap = average_precision_score(self.y_true, self.y_pred_proba)
        prevalence = self.y_true.mean()
        ax.plot(rec, prec, label=f"{label} (AP = {ap:.3f})", linewidth=2)
        ax.axhline(
            y=prevalence,
            color="k",
            linestyle="--",
            label=f"Baseline ({prevalence:.3f})",
        )
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        return ax

    def plot_calibration_curve(
        self,
        ax: mpl.axes.Axes | None = None,
        n_bins: int = 10,
    ) -> mpl.axes.Axes:
        _configure_matplotlib()
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        # Equal-count bins, matching :func:`expected_calibration_error`. The
        # ECE is stamped onto these very axes below, so drawing the curve on a
        # different partition would label one curve with another's number.
        prob_true, prob_pred = calibration_curve(
            self.y_true, self.y_pred_proba, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(prob_pred, prob_true, "s-", label="Model", linewidth=2, markersize=8)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(
            xlabel="Predicted Probability",
            ylabel="Observed Frequency",
            title="Calibration Curve",
        )
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        ece = self.compute_ece(n_bins)
        ax.text(
            0.95,
            0.05,
            f"ECE = {ece:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
        return ax

    def plot_all_curves(self, figsize: tuple[float, float] = (6.30, 2.8)) -> Figure:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        self.plot_roc_curve(ax=axes[0])
        self.plot_pr_curve(ax=axes[1])
        self.plot_calibration_curve(ax=axes[2])
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
    n_jobs: int = -1,
    stratify: bool = True,
) -> tuple[float, float, float]:
    from joblib import Parallel, delayed

    rng = np.random.default_rng(random_state)
    n = len(y_true)

    # Pre-generate all bootstrap indices for reproducibility.
    #
    # Stratified resampling draws the two classes separately, so every
    # bootstrap sample carries the original prevalence. That protects against
    # an all-one-class sample, on which the metric is undefined — a real
    # concern on a few dozen positives, and impossible on several thousand.
    #
    # It also removes prevalence as a source of variation, which narrows the
    # resulting interval. ROC-AUC is invariant to prevalence and unaffected;
    # PR-AUC is not, so its interval comes out materially too narrow. That is
    # why the callers below stratify the one and not the other.
    if stratify:
        idx_pos = np.where(y_true == 1)[0]
        idx_neg = np.where(y_true == 0)[0]
        indices = [
            np.concatenate(
                [
                    rng.choice(idx_neg, size=len(idx_neg), replace=True),
                    rng.choice(idx_pos, size=len(idx_pos), replace=True),
                ]
            )
            for _ in range(n_bootstrap)
        ]
    else:
        indices = [rng.choice(n, size=n, replace=True) for _ in range(n_bootstrap)]

    def _score_one(idx: np.ndarray) -> float | None:
        try:
            return metric_func(y_true[idx], y_pred_proba[idx])
        except Exception as e:
            logger.debug("Bootstrap iteration failed: %s", e)
            return None

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_score_one)(idx) for idx in indices
    )
    scores = [s for s in results if s is not None]

    n_valid = len(scores)
    if n_valid < n_bootstrap * 0.9:
        logger.warning(
            "Bootstrap CI: only %d/%d samples valid (%.0f%%) — "
            "CIs may be unreliable for this class distribution",
            n_valid,
            n_bootstrap,
            100 * n_valid / n_bootstrap,
        )

    alpha = 1 - confidence_level
    point = metric_func(y_true, y_pred_proba)
    arr = np.array(scores)
    return (
        point,
        float(np.percentile(arr, 100 * alpha / 2)),
        float(np.percentile(arr, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------------------
# Decision Curve Analysis
# ---------------------------------------------------------------------------


def compute_decision_curve_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    n = len(y_true)
    prevalence = y_true.mean()
    rows = []
    for t in thresholds:
        yp = (y_pred_proba >= t).astype(int)
        tp = int(np.sum((yp == 1) & (y_true == 1)))
        fp = int(np.sum((yp == 1) & (y_true == 0)))
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        ta = prevalence - (1 - prevalence) * (t / (1 - t))
        rows.append(
            {"threshold": t, "net_benefit": nb, "treat_all": ta, "treat_none": 0.0}
        )
    return pd.DataFrame(rows)


def plot_decision_curve(
    dca_df: pd.DataFrame,
    ax: mpl.axes.Axes | None = None,
    model_label: str = "Model",
) -> mpl.axes.Axes:
    _configure_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.30, 2.8))
    ax.plot(dca_df["threshold"], dca_df["net_benefit"], label=model_label, linewidth=2)
    ax.plot(
        dca_df["threshold"], dca_df["treat_all"], "--", label="Treat all", linewidth=1.5
    )
    ax.plot(
        dca_df["threshold"],
        dca_df["treat_none"],
        ":",
        label="Treat none",
        linewidth=1.5,
    )
    ax.set(
        xlabel="Threshold Probability",
        ylabel="Net Benefit",
        title="Decision Curve Analysis",
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    return ax


# ---------------------------------------------------------------------------
# Modality contribution plots
# ---------------------------------------------------------------------------

FIGURE_JITTER_SEED: int = 0
"""Seed for the point jitter in the modality-contribution strip plot.

seaborn draws the jitter from the legacy global ``np.random`` stream, so two
runs over identical data placed the points differently and the figure files
never compared equal — the one part of a run directory that could not be
diffed against another.
"""


@contextmanager
def _seeded_global_rng(seed: int) -> Iterator[None]:
    """Seed ``np.random`` for the block, then put the stream back as it was.

    Everything else in the pipeline draws from its own ``default_rng``, so no
    caller should be able to observe that this happened.
    """
    # The legacy global RNG is this function's subject, not an oversight:
    # seaborn draws the jitter from `np.random`, and only the legacy API can
    # seed and restore that stream (DECISIONS §2.19).
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(seed)  # noqa: NPY002
    try:
        yield
    finally:
        np.random.set_state(state)  # noqa: NPY002


def plot_modality_contribution_boxplot(
    modality_scores_df: pd.DataFrame,
    metric: str = "roc_auc",
    ax: mpl.axes.Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> mpl.axes.Axes:
    _configure_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    medians = (
        modality_scores_df.groupby("modality")[metric]
        .median()
        .sort_values(ascending=False)
    )
    order = medians.index.tolist()
    sns.boxplot(
        data=modality_scores_df,
        x="modality",
        y=metric,
        order=order,
        ax=ax,
        hue="modality",
        palette="Set2",
        width=0.6,
        linewidth=1.5,
        legend=False,
    )
    with _seeded_global_rng(FIGURE_JITTER_SEED):
        sns.stripplot(
            data=modality_scores_df,
            x="modality",
            y=metric,
            order=order,
            ax=ax,
            color="black",
            alpha=0.3,
            size=4,
        )
    ax.set_xlabel("Modality", fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").upper(), fontweight="bold")
    ax.set_title(
        f"Modality Contributions ({metric.replace('_', ' ').upper()})",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    for i, mod in enumerate(order):
        ax.text(
            i,
            ax.get_ylim()[1] * 0.95,
            f"{medians[mod]:.3f}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.3},
        )
    plt.tight_layout()
    return ax


def plot_modality_contribution_comparison(
    modality_scores_df: pd.DataFrame,
    metrics: list[str] | None = None,
    figsize: tuple[int, int] = (15, 6),
) -> Figure:
    if metrics is None:
        metrics = ["roc_auc", "pr_auc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    for i, m in enumerate(metrics):
        plot_modality_contribution_boxplot(modality_scores_df, metric=m, ax=axes[i])
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_figure_for_latex(
    fig: Figure, output_path: Path | str, formats: list[str] | None = None
) -> None:
    if formats is None:
        formats = ["pdf", "pgf", "png"]
    output_path = Path(output_path)

    if "pgf" in formats and not latex_available():
        logger.warning("LaTeX not found; skipping PGF format.")
        formats = [f for f in formats if f != "pgf"]

    for fmt in formats:
        try:
            fig.savefig(
                output_path.with_suffix(f".{fmt}"),
                format=fmt,
                bbox_inches="tight",
                dpi=300 if fmt in ("pdf", "png") else None,
                pad_inches=0.1 if fmt == "pgf" else None,
            )
            logger.info("Saved %s: %s", fmt.upper(), output_path.with_suffix(f".{fmt}"))
        except Exception as e:
            if fmt == "pgf" or "TeX" in str(e) or "LaTeX" in str(e):
                logger.warning("Could not save %s: %s", fmt.upper(), e)
            else:
                raise


# ---------------------------------------------------------------------------
# Threshold summary
# ---------------------------------------------------------------------------


def summarize_thresholds(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        yp = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        nb = (tp / len(y_true)) - (fp / len(y_true)) * (t / (1 - t)) if t < 1 else 0.0
        rows.append(
            {
                "threshold": t,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "specificity": spec,
                "npv": npv,
                "net_benefit": nb,
            }
        )
    df = pd.DataFrame(rows)

    def at(row: int, column: str) -> float:
        """Read one cell of *df* as a float, by row position."""
        return float(df[column].iloc[row])

    best_f1 = int(df["f1"].argmax())
    spec90 = int((df["specificity"] - 0.9).abs().argmin())
    best_nb = int(df["net_benefit"].argmax())
    summary = {
        "f1_optimal_threshold": at(best_f1, "threshold"),
        "f1_optimal_f1": at(best_f1, "f1"),
        "spec90_threshold": at(spec90, "threshold"),
        "spec90_recall": at(spec90, "recall"),
        "spec90_ppv": at(spec90, "precision"),
        "spec90_npv": at(spec90, "npv"),
        "net_benefit_threshold": at(best_nb, "threshold"),
        "net_benefit_value": at(best_nb, "net_benefit"),
    }
    return summary, df


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------


def evaluate_model_comprehensive(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    n_bootstrap: int = 1000,
    random_state: int | None = None,
    output_dir: Path | str | None = None,
    save_latex: bool = True,
    n_jobs: int = -1,
) -> EvaluationResult:
    evaluator = EvaluationMetrics(y_true, y_pred_proba)
    metrics = evaluator.compute_all_metrics()

    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        ci = {
            "roc_auc_ci": (float("nan"), float("nan")),
            "pr_auc_ci": (float("nan"), float("nan")),
        }
    else:
        _roc_auc, roc_lo, roc_hi = compute_bootstrap_ci(
            y_true,
            y_pred_proba,
            roc_auc_score,
            n_bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        # Unstratified, unlike the ROC interval above. Average precision is a
        # function of prevalence, so holding prevalence fixed across resamples
        # removes variation the estimate genuinely has and returns an interval
        # roughly a quarter too narrow.
        _, pr_lo, pr_hi = compute_bootstrap_ci(
            y_true,
            y_pred_proba,
            average_precision_score,
            n_bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
            stratify=False,
        )
        ci = {"roc_auc_ci": (roc_lo, roc_hi), "pr_auc_ci": (pr_lo, pr_hi)}

    if n_classes >= 2:
        dca_df = compute_decision_curve_analysis(y_true, y_pred_proba)
        fig_curves = evaluator.plot_all_curves()
        fig_dca, ax_dca = plt.subplots(figsize=(6.30, 2.8))
        plot_decision_curve(dca_df, ax=ax_dca, model_label=model_name)
    else:
        dca_df = pd.DataFrame()
        fig_curves = plt.figure()
        fig_dca = plt.figure()

    if save_latex and output_dir is not None and n_classes >= 2:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_figure_for_latex(fig_curves, output_dir / "evaluation_curves")
        save_figure_for_latex(fig_dca, output_dir / "decision_curve_analysis")

    plt.close(fig_curves)
    plt.close(fig_dca)

    threshold_summary, threshold_df = summarize_thresholds(y_true, y_pred_proba)

    roc_auc_val = metrics.get("roc_auc", float("nan"))
    roc_lo_val, roc_hi_val = ci.get("roc_auc_ci", (float("nan"), float("nan")))
    logger.info("ROC-AUC = %.3f [%.3f, %.3f]", roc_auc_val, roc_lo_val, roc_hi_val)
    return {
        "model_name": model_name,
        "metrics": metrics,
        "ci_results": ci,
        "dca": dca_df,
        "fig_curves": fig_curves,
        "fig_dca": fig_dca,
        "threshold_summary": threshold_summary,
        "threshold_detail": threshold_df,
    }
