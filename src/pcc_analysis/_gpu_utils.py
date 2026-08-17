"""GPU detection utilities for XGBoost acceleration.

The question is whether XGBoost will actually compute on a GPU here, and it
cannot be answered by asking XGBoost to try. A fit with ``device="cuda"`` on
a machine that has no GPU does not raise: XGBoost moves the work to the CPU
and says so in a warning (``Device is changed from GPU to CPU as we couldn't
find any available GPU on the system``). A probe that only catches
exceptions reports a GPU on every machine, including a macOS wheel built
without CUDA at all.

Three things read the answer. The meta-learner passes ``n_jobs=1`` to its
randomised search when it is on a GPU, the per-fold permutation importance
does the same, and ``config.csv`` records the device in the config hash. A
wrong answer therefore costs parallelism, and makes the device stamp agree
between a GPU machine and a CPU one.
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

_CUDA_CHECKED = False
_CUDA_AVAILABLE = False

#: Substring of the XGBoost warning emitted when it falls back to the CPU.
_FALLBACK_WARNING = "changed from GPU to CPU"


def cuda_available() -> bool:
    """Check whether XGBoost will really use a GPU here.

    Two conditions, because either one alone is satisfied by a machine
    without a GPU: the installed XGBoost must be built with CUDA support,
    and a minimal fit must complete without the runtime announcing that it
    moved to the CPU. The result is cached after the first call.
    """
    global _CUDA_CHECKED, _CUDA_AVAILABLE  # noqa: PLW0603
    if _CUDA_CHECKED:
        return _CUDA_AVAILABLE

    _CUDA_CHECKED = True
    try:
        import numpy as np
        import xgboost as xgb

        if not bool(xgb.build_info().get("USE_CUDA", False)):
            # The macOS wheels, and any CPU-only build. Nothing to probe:
            # the fit below would succeed on the CPU and prove nothing.
            _CUDA_AVAILABLE = False
            logger.info("XGBoost built without CUDA support — using CPU")
            return _CUDA_AVAILABLE

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = xgb.XGBClassifier(
                device="cuda", n_estimators=1, max_depth=1, verbosity=1
            )
            model.fit(X, y)
        if any(_FALLBACK_WARNING in str(w.message) for w in caught):
            # CUDA-capable build, no usable GPU: a driver that is missing,
            # a card that is busy, a container without device access.
            _CUDA_AVAILABLE = False
            logger.info("XGBoost found no usable GPU — using CPU")
            return _CUDA_AVAILABLE

        _CUDA_AVAILABLE = True
        logger.info("CUDA detected — XGBoost will use GPU acceleration")
    except Exception:
        _CUDA_AVAILABLE = False
        logger.info("CUDA not available — XGBoost will use CPU")

    return _CUDA_AVAILABLE


def xgb_device() -> str:
    """Return the XGBoost device string ('cuda' or 'cpu')."""
    return "cuda" if cuda_available() else "cpu"
