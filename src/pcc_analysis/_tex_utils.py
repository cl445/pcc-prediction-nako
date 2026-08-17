"""Whether a TeX distribution is available to matplotlib.

Its own module, and deliberately free of matplotlib imports: the figure
scripts have to ask this question *before* importing ``matplotlib.pyplot``,
because the backend they select depends on the answer and a backend switch
only lands while pyplot is still unimported.
"""

from __future__ import annotations

import shutil


def latex_available() -> bool:
    """Whether a TeX distribution is installed.

    matplotlib shells out to `latex` for ``text.usetex`` and to the configured
    ``pgf.texsystem`` for PGF output. A TeX distribution ships all of them
    together, so the presence of one is a sound proxy for the rest.
    """
    return shutil.which("latex") is not None
