"""Generate all paper figures from pipeline results.

Produces four figures:
  1. Evaluation curves (ROC + PR + Calibration) -- main paper
  2. Decision curve analysis -- main paper
  3. Modality contributions ROC-AUC -- main paper
  4. Modality contributions PR-AUC -- supplementary

Usage:
    uv run python scripts/pipeline/04_generate_figures.py
    uv run python scripts/pipeline/04_generate_figures.py --results-dir results/pcc_pipeline_YYYYMMDD_HHMMSS
"""

import argparse
from pathlib import Path

import matplotlib as mpl

from pcc_analysis._tex_utils import latex_available

# The PGF backend and text.usetex both shell out to TeX. Without a TeX
# distribution the figures still render, in matplotlib's own fonts, and the
# .pgf the manuscript \inputs is skipped at save time.
HAS_LATEX = latex_available()

mpl.use("pgf" if HAS_LATEX else "agg")
mpl.rcParams.update(
    {
        "pgf.texsystem": "lualatex",
        "font.family": "serif",
        "text.usetex": HAS_LATEX,
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from pcc_analysis.config import get_results_dir
from pcc_analysis.evaluation import expected_calibration_error

# ── Display names for modality labels ──────────────────────────────────────

DISPLAY_NAMES = {
    "demographics": "Demographics",
    "ses": "SES",
    "cognitive": "Cognitive",
    "physical_activity": "Physical Activity",
    "medical_history": "Medical History",
    "lab_values": "Laboratory",
    "cardiovascular": "Cardiovascular",
    "lung_function": "Lung Function",
    "mental_health": "Mental Health",
    "mri_desikan": "MRI Desikan",
    "mri_destrieux": "MRI Destrieux",
    "mri_julich": r"MRI J\"ulich",
    "mri_yeo": "MRI Yeo",
    "mri_subcortical": "MRI Subcortical",
    "mri_cerebellar": "MRI Cerebellar",
}

MRI_MODALITIES = {
    "mri_desikan",
    "mri_destrieux",
    "mri_julich",
    "mri_yeo",
    "mri_subcortical",
    "mri_cerebellar",
}

COLOR_MRI = "#4682B4"
COLOR_OTHER = "#D2691E"

N_BOOTSTRAP = 1000
N_BINS = 10
RNG_SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────


def find_latest_results(results_root: Path) -> Path:
    """Return the most recent pcc_pipeline_* directory."""
    dirs = sorted(results_root.glob("pcc_pipeline_*"), reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No pipeline results found in {results_root}")
    return dirs[0]


def save_figure(fig: Figure, output_dir: Path, name: str) -> None:
    if HAS_LATEX:
        fig.savefig(output_dir / f"{name}.pgf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", dpi=300)
    print(f"  Saved {name}.pgf / .pdf" if HAS_LATEX else f"  Saved {name}.pdf")
    plt.close(fig)


# ── Figure 1: Evaluation Curves ───────────────────────────────────────────


def bootstrap_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals for calibration curve bins.

    Equal-count bins, matching
    :func:`pcc_analysis.evaluation.expected_calibration_error`, whose value is
    printed on the same axes.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy="quantile"
    )

    boot_true = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            bt, _ = calibration_curve(
                y_true[idx], y_pred[idx], n_bins=n_bins, strategy="quantile"
            )
            if len(bt) == len(prob_true):
                boot_true.append(bt)
        except ValueError:
            continue

    boot_true_arr = np.array(boot_true)
    ci_low = np.percentile(boot_true_arr, 2.5, axis=0)
    ci_high = np.percentile(boot_true_arr, 97.5, axis=0)
    return prob_true, prob_pred, ci_low, ci_high


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error, on the shared equal-count binning."""
    return expected_calibration_error(y_true, y_pred, n_bins=n_bins)


def figure_evaluation_curves(y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
    prevalence = y_true.mean()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    # Calibration
    prob_true, prob_pred, ci_low, ci_high = bootstrap_calibration(
        y_true, y_pred, n_bins=N_BINS, n_bootstrap=N_BOOTSTRAP, seed=RNG_SEED
    )
    ece = compute_ece(y_true, y_pred, n_bins=N_BINS)

    fig, axes = plt.subplots(1, 3, figsize=(6.30, 2.8))

    # (A) ROC
    ax = axes[0]
    ax.plot(
        fpr, tpr, color="#4682B4", linewidth=1.5, label=f"Model (AUC = {roc_auc:.3f})"
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, linestyle="--")

    # (B) PR
    ax = axes[1]
    ax.plot(
        recall,
        precision,
        color="#4682B4",
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

    # (C) Calibration
    ax = axes[2]
    ax.plot(
        prob_pred,
        prob_true,
        "s-",
        color="#4682B4",
        linewidth=1.5,
        markersize=6,
        label="Model",
        zorder=3,
    )
    ax.fill_between(
        prob_pred,
        ci_low,
        ci_high,
        color="#4682B4",
        alpha=0.2,
        label=r"95\% CI",
        zorder=2,
    )
    xlim_max = min(max(prob_pred.max(), prob_true.max(), ci_high.max()) * 1.15, 1.0)
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
    ax.text(
        0.95,
        0.05,
        f"ECE = {ece:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightsteelblue", "alpha": 0.5},
    )

    fig.tight_layout()
    return fig


# ── Figure 2: Decision Curve Analysis ─────────────────────────────────────


def compute_dca(
    y_true: np.ndarray, y_pred: np.ndarray, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 200)

    n = len(y_true)
    prevalence = y_true.mean()
    rows = []
    for t in thresholds:
        y_hat = (y_pred >= t).astype(int)
        tp = np.sum((y_hat == 1) & (y_true == 1))
        fp = np.sum((y_hat == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        ta = prevalence - (1 - prevalence) * (t / (1 - t))
        rows.append(
            {"threshold": t, "net_benefit": nb, "treat_all": ta, "treat_none": 0.0}
        )
    return pd.DataFrame(rows)


def figure_decision_curve(y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
    dca = compute_dca(y_true, y_pred)
    mask = dca["threshold"] <= 0.50
    df = dca[mask]

    fig, ax = plt.subplots(figsize=(6.30, 2.8))
    ax.plot(
        df["threshold"],
        df["net_benefit"],
        color="#4682B4",
        linewidth=1.8,
        label="PCC Multimodal Pipeline",
    )
    ax.plot(
        df["threshold"],
        df["treat_all"],
        color="#D2691E",
        linestyle="--",
        linewidth=1.2,
        label="Treat all",
    )
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1.2, label="Treat none")

    model_above = (df["net_benefit"] > 0) & (df["net_benefit"] > df["treat_all"])
    comparator = np.maximum(df["treat_all"].to_numpy(), 0)
    ax.fill_between(
        df["threshold"].to_numpy(),
        comparator,
        df["net_benefit"].to_numpy(),
        where=model_above.tolist(),
        color="#4682B4",
        alpha=0.10,
    )

    ax.set_xlim(0, 0.50)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


# ── Figure 3 / 4: Modality Contributions ──────────────────────────────────


def figure_modality_contributions(
    mod_df: pd.DataFrame, metric: str, ylabel: str, chance_line: float | None = None
) -> Figure:
    mod_df = mod_df.copy()
    mod_df["display_name"] = mod_df["modality"].map(DISPLAY_NAMES)
    mod_df["is_mri"] = mod_df["modality"].isin(MRI_MODALITIES)

    medians = mod_df.groupby("display_name")[metric].median().sort_values()
    order = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(6.30, 4.5))

    for i, name in enumerate(order):
        subset = mod_df[mod_df["display_name"] == name][metric].to_numpy()
        is_mri = mod_df[mod_df["display_name"] == name]["is_mri"].iloc[0]
        color = COLOR_MRI if is_mri else COLOR_OTHER

        ax.boxplot(
            subset,
            positions=[i],
            vert=False,
            widths=0.55,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.6, "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )

        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(subset))
        ax.scatter(
            subset,
            np.full_like(subset, i) + jitter,
            color="black",
            alpha=0.35,
            s=12,
            zorder=3,
        )

        med = np.median(subset)
        ax.text(med, i + 0.35, f"{med:.3f}", ha="center", va="bottom", fontsize=7.5)

    if chance_line is not None:
        ax.axvline(x=chance_line, color="gray", linestyle="--", linewidth=0.8, zorder=0)
        ax.text(
            chance_line - 0.002,
            len(order) - 1.5,
            "chance",
            ha="right",
            va="center",
            fontsize=7.5,
            color="gray",
            rotation=90,
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.set_xlabel(ylabel)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    legend_elements = [
        Patch(facecolor=COLOR_OTHER, alpha=0.6, label="Non-imaging"),
        Patch(facecolor=COLOR_MRI, alpha=0.6, label="Brain MRI"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument(
        "--results-dir", type=Path, default=None, help="Pipeline results directory."
    )
    args = parser.parse_args()

    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        results_dir = find_latest_results(get_results_dir())

    print(f"Using results from: {results_dir}")

    # Output directory for figures
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load predictions
    predictions = pd.read_csv(results_dir / "predictions.csv")
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred_proba"].to_numpy()
    print(f"Loaded {len(predictions):,} predictions (prevalence: {y_true.mean():.3f})")

    # Figure 1: Evaluation curves
    print("Generating evaluation curves...")
    fig = figure_evaluation_curves(y_true, y_pred)
    save_figure(fig, fig_dir, "evaluation_curves")

    # Figure 2: Decision curve analysis
    print("Generating decision curve analysis...")
    fig = figure_decision_curve(y_true, y_pred)
    save_figure(fig, fig_dir, "decision_curve_analysis")

    # Figure 3 & 4: Modality contributions
    mod_path = results_dir / "modality_scores_cv.csv"
    if mod_path.exists():
        mod_df = pd.read_csv(mod_path)
        print("Generating modality contributions (ROC-AUC)...")
        fig = figure_modality_contributions(
            mod_df, "roc_auc", "ROC-AUC", chance_line=0.5
        )
        save_figure(fig, fig_dir, "modality_contributions")

        print("Generating modality contributions (PR-AUC, supplementary)...")
        fig = figure_modality_contributions(mod_df, "pr_auc", "PR-AUC")
        save_figure(fig, fig_dir, "modality_contributions_prauc")
    else:
        print(f"  WARNING: {mod_path} not found, skipping modality figures.")

    print(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
