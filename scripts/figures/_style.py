"""Shared style for paper figures: PGF + LuaLaTeX, TeX Gyre Termes, palette.

Importing this module configures matplotlib. Import it BEFORE
``matplotlib.pyplot`` so the backend switch lands.

The PGF backend and ``text.usetex`` both shell out to a TeX distribution, so
they are enabled only when one is installed. Without TeX the figures render in
matplotlib's own fonts through the Agg backend and no ``.pgf`` is written —
which is what the manuscript ``\\input``s, so a TeX-less machine can run
everything else but cannot produce paper-bound output.
"""

import re
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


_PGF_COLOR = re.compile(
    r"^\\definecolor\{current(fill|stroke)\}\{rgb\}"
    r"\{([0-9.]+),([0-9.]+),([0-9.]+)\}(%?)$"
)
_PGF_OPACITY = re.compile(r"^\\pgfset(fill|stroke)opacity\{([0-9.]+)\}(%?)$")

# Graphics state carried through the file: the current fill and stroke colour,
# and the current fill and stroke opacity, each keyed by "fill"/"stroke".
type _Rgb = tuple[float, float, float]
type _Colours = dict[str, _Rgb]
type _Opacities = dict[str, float]


def _pgf_color(kind: str, rgb: tuple[float, float, float], suffix: str) -> str:
    r, g, b = (f"{c:.6f}" for c in rgb)
    return f"\\definecolor{{current{kind}}}{{rgb}}{{{r},{g},{b}}}{suffix}"


def _over_white(
    rgb: tuple[float, float, float], alpha: float
) -> tuple[float, float, float]:
    r, g, b = (1.0 - alpha * (1.0 - c) for c in rgb)
    return r, g, b


def flatten_pgf_opacity(path: Path) -> int:
    """Fold ``alpha`` into the colours of a ``.pgf`` file, against white.

    Where matplotlib writes a colour followed by ``\\pgfsetfillopacity{0.75}``,
    this writes the colour that opacity produces over white paper and sets the
    opacity to 1. The rendered result is identical wherever the element sits on
    the page background — which is every element that does not overlap another
    translucent one.

    Why bother: PDF/A-1 (ISO 19005-1, clause 6.4) forbids transparency
    outright, and the manuscript is bound into a dissertation that has to meet
    it for the university library. Transparency cannot be removed from a
    finished PDF without changing the picture, so it has to not be there in the
    first place.

    The one inexact case is overlap. A translucent marker drawn *on top of* a
    translucent bar blends with the bar, not with the paper, and comes out a
    shade darker here — measured on the modality-contribution figures at under
    0.6 % of the page area, invisible unless the two renderings are put side by
    side. Overlap is the reason this runs on the ``.pgf`` and is not simply left
    to matplotlib: only the caller knows which elements are meant to show
    through each other, and none of ours depends on it.

    Returns the number of places changed.
    """
    lines = path.read_text().split("\n")
    colour: _Colours = {"fill": (0.0, 0.0, 0.0), "stroke": (0.0, 0.0, 0.0)}
    opacity: _Opacities = {"fill": 1.0, "stroke": 1.0}
    scopes: list[tuple[_Colours, _Opacities]] = []
    out: list[str] = []
    changed = 0

    for line in lines:
        # PGF scopes restore the graphics state, so the state has to be stacked
        # with them — an opacity set inside a scope does not outlive it.
        if line.startswith(r"\begin{pgfscope}"):
            scopes.append((dict(colour), dict(opacity)))
            out.append(line)
            continue
        if line.startswith(r"\end{pgfscope}"):
            if scopes:
                saved_colour, saved_opacity = scopes.pop()
                colour, opacity = dict(saved_colour), dict(saved_opacity)
            out.append(line)
            continue

        match = _PGF_COLOR.match(line)
        if match:
            kind = match.group(1)
            rgb = (float(match.group(2)), float(match.group(3)), float(match.group(4)))
            colour[kind] = rgb
            if opacity[kind] < 1.0:
                out.append(
                    _pgf_color(kind, _over_white(rgb, opacity[kind]), match.group(5))
                )
                changed += 1
            else:
                out.append(line)
            continue

        match = _PGF_OPACITY.match(line)
        if match:
            kind, alpha, suffix = match.group(1), float(match.group(2)), match.group(3)
            opacity[kind] = alpha
            if alpha < 1.0:
                # The colour was set before the opacity, so restate it blended
                # rather than reaching back to rewrite the earlier line.
                out.append(_pgf_color(kind, _over_white(colour[kind], alpha), suffix))
                out.append(f"\\pgfset{kind}color{{current{kind}}}{suffix}")
                out.append(f"\\pgfset{kind}opacity{{1.000000}}{suffix}")
                changed += 1
            else:
                out.append(line)
            continue

        out.append(line)

    if changed:
        path.write_text("\n".join(out))
    return changed


def save_pgf_pdf(fig: Figure, path: Path | str) -> None:
    """Save as ``.pgf`` (for ``\\input``) and ``.pdf`` (preview companion).

    The ``.pgf`` needs TeX and is skipped without it; the ``.pdf`` always lands.

    The ``.pgf`` is written opacity-free (see ``flatten_pgf_opacity``) so that
    the manuscript stays convertible to PDF/A. Figure scripts can go on using
    ``alpha=`` as usual; it is folded into the colours on the way out. The
    ``.pdf`` companion is a preview and keeps its alpha channel — nothing binds
    it into the paper.
    """
    path = Path(path)
    if HAS_LATEX:
        pgf = path.with_suffix(".pgf")
        fig.savefig(pgf, bbox_inches="tight")
        flatten_pgf_opacity(pgf)
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
