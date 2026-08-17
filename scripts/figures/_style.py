"""Shared style for paper figures: PGF + LuaLaTeX, TeX Gyre Termes, palette.

Importing this module configures matplotlib. Import it BEFORE
``matplotlib.pyplot`` so the backend switch lands.

The PGF backend and ``text.usetex`` both shell out to a TeX distribution, so
they are enabled only when one is installed. Without TeX the figures render in
matplotlib's own fonts through the Agg backend and no ``.pgf`` is written —
which is what the manuscript ``\\input``s, so a TeX-less machine can run
everything else but cannot produce paper-bound output.
"""

from pathlib import Path

import matplotlib as mpl

from pcc_analysis._tex_utils import latex_available

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

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Palette mirrors \definecolor entries in paper/main.tex.
STEEL_BLUE = "#4682B4"
CHOCOLATE = "#D2691E"

# Deterministic per-configuration run directories written by
# ``scripts/run_analysis.sh`` (see the configuration table in README.md).
RUNS = Path(__file__).resolve().parents[2] / "results" / "runs"


def figure_output_dir() -> Path:
    """Where to write paper figures.

    The sibling ``paper/`` repo when it is checked out next to ``code/``,
    otherwise a repo-local fallback so a standalone code checkout — which is
    all an external reader gets — still produces figures instead of crashing.
    """
    paper_figures = Path(__file__).resolve().parents[3] / "paper" / "figures"
    if paper_figures.is_dir():
        return paper_figures
    fallback = Path(__file__).resolve().parents[2] / "results" / "figures"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def save_pgf_pdf(fig: Figure, path: Path | str) -> None:
    """Save as ``.pgf`` (for ``\\input``) and ``.pdf`` (preview companion).

    The ``.pgf`` needs TeX and is skipped without it; the ``.pdf`` always lands.
    """
    path = Path(path)
    if HAS_LATEX:
        fig.savefig(path.with_suffix(".pgf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)


def compute_dca(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Decision Curve Analysis over a threshold grid."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 200)
    n = len(y_true)
    prevalence = float(np.asarray(y_true).mean())
    rows: list[dict[str, float]] = []
    for t in thresholds:
        yp = (y_pred_proba >= t).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        ta = prevalence - (1 - prevalence) * (t / (1 - t))
        rows.append(
            {
                "threshold": float(t),
                "net_benefit": nb,
                "treat_all": ta,
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)
