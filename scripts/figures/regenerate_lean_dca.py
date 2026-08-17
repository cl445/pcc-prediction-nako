"""Regenerate combined Lean-Stack DCA (MRI vs. non-MRI) as PGF for LuaLaTeX.

Loads held-out predictions of the two Lean-Stack runs (separately fitted on
MRI and non-MRI cohorts) and overlays their decision curves on a single panel.
"""

import matplotlib.pyplot as plt
import pandas as pd
from _style import (  # configures matplotlib for PGF
    CHOCOLATE,
    RUNS,
    STEEL_BLUE,
    compute_dca,
    figure_output_dir,
    save_pgf_pdf,
)
from matplotlib.figure import Figure

# --- Config ---
MRI_PREDICTIONS = RUNS / "lean_mri" / "predictions.csv"
NON_MRI_PREDICTIONS = RUNS / "lean_non_mri" / "predictions.csv"


def plot_lean_dca(mri_dca: pd.DataFrame, non_mri_dca: pd.DataFrame) -> Figure:
    """Overlay MRI + non-MRI Lean decision curves on a single panel."""
    fig, ax = plt.subplots(figsize=(6.30, 3.2))

    # Clip to clinically relevant threshold range
    mask_mri = mri_dca["threshold"] <= 0.50
    mask_non = non_mri_dca["threshold"] <= 0.50
    mri = mri_dca[mask_mri]
    non = non_mri_dca[mask_non]

    # Model curves
    ax.plot(
        mri["threshold"],
        mri["net_benefit"],
        color=STEEL_BLUE,
        linewidth=1.8,
        label="Lean (MRI cohort)",
    )
    ax.plot(
        non["threshold"],
        non["net_benefit"],
        color=CHOCOLATE,
        linewidth=1.8,
        label="Lean (non-MRI cohort)",
    )

    # Reference: treat-all (use MRI prevalence for the baseline; both cohorts
    # are very close in prevalence, so a single reference curve is honest).
    ax.plot(
        mri["threshold"],
        mri["treat_all"],
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="Treat all",
    )
    ax.axhline(
        y=0,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        label="Treat none",
    )

    ax.set_xlim(0, 0.50)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    return fig


def main() -> None:
    mri_df = pd.read_csv(MRI_PREDICTIONS)
    non_df = pd.read_csv(NON_MRI_PREDICTIONS)

    print(
        f"MRI:     {len(mri_df)} predictions (prevalence: {mri_df['y_true'].mean():.3f})"
    )
    print(
        f"non-MRI: {len(non_df)} predictions (prevalence: {non_df['y_true'].mean():.3f})"
    )

    mri_dca = compute_dca(
        mri_df["y_true"].to_numpy(), mri_df["y_pred_proba"].to_numpy()
    )
    non_dca = compute_dca(
        non_df["y_true"].to_numpy(), non_df["y_pred_proba"].to_numpy()
    )

    fig = plot_lean_dca(mri_dca, non_dca)
    save_pgf_pdf(fig, figure_output_dir() / "lean_dca_mri_vs_non_mri")
    print("Saved lean_dca_mri_vs_non_mri.pgf/.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
