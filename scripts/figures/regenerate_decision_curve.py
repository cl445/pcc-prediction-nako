"""Regenerate Decision Curve Analysis figure with zoomed axes."""

import matplotlib.pyplot as plt
import numpy as np
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
PREDICTIONS_PATH = RUNS / "primary" / "predictions.csv"


def plot_dca(dca_df: pd.DataFrame) -> Figure:
    """Plot DCA with zoomed axes on the clinically relevant range."""
    fig, ax = plt.subplots(figsize=(6.30, 2.8))

    # Clip to relevant threshold range
    mask = dca_df["threshold"] <= 0.50
    df = dca_df[mask]

    # Model
    ax.plot(
        df["threshold"],
        df["net_benefit"],
        color=STEEL_BLUE,
        linewidth=1.8,
        label="PCC Multimodal Pipeline",
    )

    # Treat all
    ax.plot(
        df["threshold"],
        df["treat_all"],
        color=CHOCOLATE,
        linestyle="--",
        linewidth=1.2,
        label="Treat all",
    )

    # Treat none
    ax.axhline(
        y=0,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label="Treat none",
    )

    # Shade region where model has positive net benefit above both strategies
    model_above = (df["net_benefit"] > 0) & (df["net_benefit"] > df["treat_all"])
    if model_above.any():
        thresh_vals = df["threshold"].to_numpy()
        nb_vals = df["net_benefit"].to_numpy()
        ta_vals = df["treat_all"].to_numpy()

        # Find the better comparator at each point
        comparator = np.maximum(ta_vals, 0)
        ax.fill_between(
            thresh_vals,
            comparator,
            nb_vals,
            where=model_above.tolist(),
            color=STEEL_BLUE,
            alpha=0.10,
            label="_nolegend_",
        )

    ax.set_xlim(0, 0.50)
    ax.set_ylim(-0.15, 0.15)

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    return fig


def report_useful_range(y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
    """Print the threshold band in which the model beats both default strategies.

    The paper quotes this band and no artifact carries the curve itself, so it
    is printed here, on a grid fine enough to place the two crossings to the
    nearest percentage point (the plotting grid is coarser than that).
    """
    grid = np.round(np.arange(0.001, 0.999, 0.001), 3)
    dca_df = compute_dca(y_true, y_pred_proba, grid)
    above = dca_df["net_benefit"].to_numpy() > np.maximum(
        dca_df["treat_all"].to_numpy(), 0.0
    )
    if not above.any():
        print("Model never exceeds both default strategies")
        return
    inside = dca_df["threshold"].to_numpy()[above]
    print(
        f"Model above treat-all and treat-none for thresholds "
        f"{inside.min():.3f}-{inside.max():.3f} "
        f"({100 * inside.min():.0f}%-{100 * inside.max():.0f}%)"
    )


def main() -> None:
    df = pd.read_csv(PREDICTIONS_PATH)
    y_true = df["y_true"].to_numpy()
    y_pred_proba = df["y_pred_proba"].to_numpy()

    print(f"Loaded {len(df)} predictions (prevalence: {y_true.mean():.3f})")
    report_useful_range(y_true, y_pred_proba)

    dca_df = compute_dca(y_true, y_pred_proba)

    fig = plot_dca(dca_df)
    save_pgf_pdf(fig, figure_output_dir() / "decision_curve_analysis")
    print("Saved decision_curve_analysis.pgf/.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
