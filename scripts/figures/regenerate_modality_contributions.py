"""Regenerate modality contributions figure with professional labels."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _style import (  # configures matplotlib for PGF
    CHOCOLATE,
    RUNS,
    STEEL_BLUE,
    figure_output_dir,
    save_pgf_pdf,
)
from matplotlib.figure import Figure

# --- Config ---
DATA_PATH = RUNS / "primary" / "modality_scores_cv.csv"

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
    "mri_julich": "MRI Jülich",
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

COLOR_MRI = STEEL_BLUE
COLOR_OTHER = CHOCOLATE


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["display_name"] = df["modality"].map(DISPLAY_NAMES)
    df["is_mri"] = df["modality"].isin(MRI_MODALITIES)
    return df


def plot_single_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    chance_line: float | None = None,
) -> Figure:
    """Create a horizontal fold-level dot plot for one metric.

    With only k=5 cross-validation folds, quartile-based summaries
    (boxplots) are poorly defined. We therefore show the individual
    fold values, a min--max range bar, and the median explicitly.
    """
    # Sort by median (descending, so best at top)
    medians = df.groupby("display_name")[metric].median().sort_values()
    order = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(6.30, 4.5))

    positions = list(range(len(order)))
    for i, name in enumerate(order):
        subset = df[df["display_name"] == name][metric].to_numpy()
        is_mri = df[df["display_name"] == name]["is_mri"].iloc[0]
        color = COLOR_MRI if is_mri else COLOR_OTHER

        lo, hi = float(np.min(subset)), float(np.max(subset))
        med = float(np.median(subset))

        # Min--max range bar (thin, colored)
        ax.plot(
            [lo, hi],
            [i, i],
            color=color,
            alpha=0.55,
            linewidth=5.0,
            solid_capstyle="round",
            zorder=2,
        )

        # Individual fold points (no jitter: honest 1:1 mapping)
        ax.scatter(
            subset,
            np.full_like(subset, i, dtype=float),
            color="black",
            alpha=0.75,
            s=18,
            zorder=3,
        )

        # Median marker
        ax.plot(
            [med],
            [i],
            marker="|",
            color="black",
            markersize=14,
            markeredgewidth=1.6,
            zorder=4,
        )

        # Median annotation
        ax.text(
            med,
            i + 0.32,
            f"{med:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="black",
        )

    # Chance-level line
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

    ax.set_yticks(positions)
    ax.set_yticklabels(order)
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.set_xlabel(ylabel)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLOR_OTHER, alpha=0.55, label="Non-imaging (min--max)"),
        Patch(facecolor=COLOR_MRI, alpha=0.55, label="Brain MRI (min--max)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=5,
            label="Fold value",
        ),
        Line2D(
            [0],
            [0],
            marker="|",
            color="black",
            markersize=10,
            markeredgewidth=1.6,
            linestyle="none",
            label="Median",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    return fig


def main() -> None:
    df = load_data()
    output_dir = figure_output_dir()

    # Main paper figure: ROC-AUC only
    fig_roc = plot_single_metric(df, "roc_auc", "ROC-AUC", chance_line=0.5)
    save_pgf_pdf(fig_roc, output_dir / "modality_contributions")
    print("Saved main figure: modality_contributions.pgf/.pdf")
    plt.close(fig_roc)

    # Supplementary figure: PR-AUC
    fig_pr = plot_single_metric(df, "pr_auc", "PR-AUC", chance_line=None)
    save_pgf_pdf(fig_pr, output_dir / "modality_contributions_prauc")
    print("Saved supplementary figure: modality_contributions_prauc.pgf/.pdf")
    plt.close(fig_pr)


if __name__ == "__main__":
    main()
