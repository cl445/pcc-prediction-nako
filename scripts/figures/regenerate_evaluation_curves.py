"""Regenerate evaluation curves (ROC, PR, Calibration) with CIs on calibration bins."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _style import (  # configures matplotlib for PGF
    RUNS,
    STEEL_BLUE,
    figure_output_dir,
    save_pgf_pdf,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from pcc_analysis.evaluation import expected_calibration_error, quantile_bin_edges

# --- Config ---
PREDICTIONS_PATH = RUNS / "primary" / "predictions.csv"
N_BINS = 10
N_BOOTSTRAP = 1000
RNG_SEED = 42


def bootstrap_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals for calibration curve bins.

    Uses quantile-based binning so that each bin contains roughly the same
    number of observations. Avoids the misleading sparse-bin artefact at
    the top of the predicted-probability range that can arise with uniform
    binning when high probabilities are rare.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    prob_true, prob_pred = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy="quantile"
    )

    boot_rows: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            bt, _ = calibration_curve(
                y_true[idx], y_pred_proba[idx], n_bins=n_bins, strategy="quantile"
            )
            if len(bt) == len(prob_true):
                boot_rows.append(bt)
        except ValueError:
            continue

    boot_true = np.array(boot_rows)
    ci_low = np.percentile(boot_true, 2.5, axis=0)
    ci_high = np.percentile(boot_true, 97.5, axis=0)

    return prob_true, prob_pred, ci_low, ci_high


def count_per_bin(y_pred_proba: np.ndarray, n_bins: int = 10) -> list[int]:
    """Count observations per quantile-based calibration bin."""
    edges = quantile_bin_edges(y_pred_proba, n_bins)
    counts: list[int] = []
    for i in range(len(edges) - 1):
        if i == 0:
            mask = (y_pred_proba >= edges[i]) & (y_pred_proba <= edges[i + 1])
        else:
            mask = (y_pred_proba > edges[i]) & (y_pred_proba <= edges[i + 1])
        counts.append(int(mask.sum()))
    return counts


def main() -> None:
    df = pd.read_csv(PREDICTIONS_PATH)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred_proba"].to_numpy()
    prevalence = y_true.mean()

    print(f"Loaded {len(df)} predictions (prevalence: {prevalence:.3f})")

    # --- Compute metrics ---
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    # Calibration with bootstrap CIs
    print("Bootstrapping calibration CIs...")
    prob_true, prob_pred, ci_low, ci_high = bootstrap_calibration(
        y_true, y_pred, n_bins=N_BINS, n_bootstrap=N_BOOTSTRAP, seed=RNG_SEED
    )
    ece = expected_calibration_error(y_true, y_pred, n_bins=N_BINS)
    bin_counts = count_per_bin(y_pred, n_bins=N_BINS)

    print(f"ROC-AUC: {roc_auc:.3f}, AP: {ap:.3f}, ECE: {ece:.3f}")
    print(f"Bin counts: {bin_counts}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(6.30, 2.8))

    # (A) ROC Curve
    ax = axes[0]
    ax.plot(
        fpr, tpr, color=STEEL_BLUE, linewidth=1.5, label=f"Model (AUC = {roc_auc:.3f})"
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, linestyle="--")

    # (B) PR Curve
    ax = axes[1]
    ax.plot(
        recall,
        precision,
        color=STEEL_BLUE,
        linewidth=1.5,
        label=f"Model (AP = {ap:.3f})",
    )
    ax.axhline(
        y=prevalence,
        color="k",
        linestyle="--",
        linewidth=0.8,
        label=f"Baseline ({prevalence:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2, linestyle="--")

    # (C) Calibration Curve with CIs — zoomed to data range
    ax = axes[2]
    ax.plot(
        prob_pred,
        prob_true,
        "s-",
        color=STEEL_BLUE,
        linewidth=1.5,
        markersize=6,
        label="Model",
        zorder=3,
    )
    ax.fill_between(
        prob_pred,
        ci_low,
        ci_high,
        color=STEEL_BLUE,
        alpha=0.2,
        label=r"95\% CI",
        zorder=2,
    )
    # Perfect calibration line (only in visible range)
    xlim_max = max(prob_pred.max(), prob_true.max(), ci_high.max()) * 1.15
    xlim_max = min(xlim_max, 1.0)
    ax.plot(
        [0, xlim_max], [0, xlim_max], "k--", linewidth=0.8, label="Perfect calibration"
    )
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(0, min(ci_high.max() * 1.3, 1.0))
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curve", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2, linestyle="--")

    # Add ECE annotation
    ax.text(
        0.95,
        0.05,
        f"ECE = {ece:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
    )

    fig.tight_layout()

    # Save
    save_pgf_pdf(fig, figure_output_dir() / "evaluation_curves")
    print("Saved evaluation_curves.pgf/.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
