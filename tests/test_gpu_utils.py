"""Tests for the XGBoost device detection.

Detection reads the build flag rather than probing with a fit, because
XGBoost 3.x does not raise when a GPU is missing — it moves the work to the
CPU and warns. A probe phrased as ``try: fit(device="cuda")`` therefore
answers "GPU" on every machine, including a macOS wheel built without CUDA.
These tests pin both halves: the build flag and the fallback warning.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from pcc_analysis import _gpu_utils

FALLBACK = (
    "[09:41:31] WARNING: src/context.cc:218: Device is changed from GPU to "
    "CPU as we couldn't find any available GPU on the system."
)


@pytest.fixture(autouse=True)
def _uncached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test asks the question fresh; the answer is cached per process."""
    monkeypatch.setattr(_gpu_utils, "_CUDA_CHECKED", False)
    monkeypatch.setattr(_gpu_utils, "_CUDA_AVAILABLE", False)


def _fake_classifier(warning: str | None) -> type:
    class FakeXGBClassifier:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def fit(self, _X: Any, _y: Any) -> None:
            if warning is not None:
                warnings.warn(warning, UserWarning, stacklevel=1)

    return FakeXGBClassifier


def test_a_build_without_cuda_needs_no_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU-only build cannot be probed by fitting: the fit would succeed."""

    def explode(**_kwargs: Any) -> None:
        raise AssertionError("the build flag already settled it")

    monkeypatch.setattr("xgboost.build_info", lambda: {"USE_CUDA": False})
    monkeypatch.setattr("xgboost.XGBClassifier", explode)

    assert _gpu_utils.cuda_available() is False
    assert _gpu_utils.xgb_device() == "cpu"


def test_a_fallback_warning_means_no_usable_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case that made the old probe wrong: a CUDA build, no card.

    The fit completes, so nothing is raised. Only the warning distinguishes
    it from a run that really used the GPU.
    """
    monkeypatch.setattr("xgboost.build_info", lambda: {"USE_CUDA": True})
    monkeypatch.setattr("xgboost.XGBClassifier", _fake_classifier(FALLBACK))

    assert _gpu_utils.cuda_available() is False
    assert _gpu_utils.xgb_device() == "cpu"


def test_a_quiet_fit_on_a_cuda_build_is_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must still let a real GPU through."""
    monkeypatch.setattr("xgboost.build_info", lambda: {"USE_CUDA": True})
    monkeypatch.setattr("xgboost.XGBClassifier", _fake_classifier(None))

    assert _gpu_utils.cuda_available() is True
    assert _gpu_utils.xgb_device() == "cuda"


def test_a_broken_probe_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection may not take a run down; the CPU path always works."""

    def explode() -> dict[str, bool]:
        raise RuntimeError("libxgboost is unhappy")

    monkeypatch.setattr("xgboost.build_info", explode)

    assert _gpu_utils.cuda_available() is False


def test_the_answer_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """It is asked once per fold and once per config record."""
    calls = 0

    def counting_build_info() -> dict[str, bool]:
        nonlocal calls
        calls += 1
        return {"USE_CUDA": False}

    monkeypatch.setattr("xgboost.build_info", counting_build_info)

    _gpu_utils.cuda_available()
    _gpu_utils.cuda_available()
    _gpu_utils.xgb_device()

    assert calls == 1
