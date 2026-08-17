"""Typed wrappers around sklearn helpers that discard type information."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sklearn.base import clone as _sklearn_clone

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def clone[EstimatorT](estimator: EstimatorT) -> EstimatorT:
    """``sklearn.base.clone`` with the estimator's own type carried through.

    sklearn annotates the return as ``BaseEstimator``, which drops ``predict``,
    ``predict_proba`` and every estimator-specific attribute at the call site.
    The runtime contract is stronger: the clone is a fresh unfitted instance of
    exactly the class that went in.

    The inbound cast bridges the other half of the same gap: sklearn accepts
    estimators nominally, while this codebase describes them structurally
    through :mod:`pcc_analysis.protocols`. Duck-typed estimators are exactly
    what ``clone`` supports at runtime.
    """
    return cast("EstimatorT", _sklearn_clone(cast("BaseEstimator", estimator)))
