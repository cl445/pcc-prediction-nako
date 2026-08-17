"""Custom Pipeline that supports passing confounders to orthogonalization step."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, override

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .orthogonalization import Orthogonalizer

if TYPE_CHECKING:
    from .protocols import Transformer

logger = logging.getLogger(__name__)


class ConfoundedPipeline(Pipeline):
    """Extended sklearn Pipeline that can pass confounders to specific steps.

    This allows orthogonalization steps to receive confounders during
    fit/transform without breaking sklearn's API.
    """

    # Narrows the inherited parameter types: sklearn's Pipeline accepts any
    # iterable, this one only what its transformers handle. The narrowing is
    # the point of the class, so the Liskov complaint is answered here.
    @override
    def fit(  # pyrefly: ignore[bad-override]
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        confounders: pd.DataFrame | None = None,
        **fit_params: Any,
    ) -> ConfoundedPipeline:
        """Fit all pipeline steps, routing confounders to Orthogonalizer steps.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.
        y : ndarray, optional
            Target vector.
        confounders : DataFrame, optional
            Confounder matrix (age, sex, center) passed to any
            :class:`Orthogonalizer` step for DML residualization.
        **fit_params
            Additional parameters forwarded to the final estimator
            (e.g. ``sample_weight``).
        """
        self._confounders = confounders

        # Extract sample_weight so transformer steps that support it
        # (e.g. StabilitySelector's internal estimator) can receive it.
        sample_weight = fit_params.get("sample_weight")

        # Collect extra confounders from pipeline steps (e.g. FeatureToConfounder)
        extra_confounders: list[pd.DataFrame] = []

        Xt = X
        for _step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            if confounders is not None and isinstance(transformer, Orthogonalizer):
                aug = self._augment_confounders(confounders, extra_confounders)
                logger.debug(
                    "Fitting %s with confounders (%d cols)", name, aug.shape[1]
                )
                Xt = transformer.fit_transform(Xt, y, confounders=aug)
            else:
                fitted = self._fit_transformer(transformer, Xt, y, sample_weight)
                Xt = fitted.transform(Xt)
                # Check if this step contributes extra confounders
                ec = getattr(fitted, "get_extra_confounders", None)
                if ec is not None:
                    extra = ec()
                    if extra is not None and not extra.empty:
                        extra_confounders.append(extra)

        self.steps[-1][1].fit(Xt, y, **fit_params)

        return self

    @staticmethod
    def _augment_confounders(
        confounders: pd.DataFrame, extras: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """Concatenate base confounders with any extra confounder columns."""
        if not extras:
            return confounders
        return pd.concat([confounders, *extras], axis=1)

    @staticmethod
    def _fit_transformer(
        transformer: Transformer,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None,
        sample_weight: np.ndarray | None,
    ) -> Transformer:
        """Fit a transformer, passing sample_weight and y if it accepts them.

        What the transformer accepts is read off its signature rather than
        discovered by catching ``TypeError``. A ``TypeError`` raised *inside*
        ``fit`` — a dtype it cannot handle, a bad comparison — is
        indistinguishable from one raised by the call itself, so catching it
        here would silently refit without the argument and hide a real defect
        behind a successful-looking fit.
        """
        params = inspect.signature(transformer.fit).parameters
        accepts = params.keys()
        takes_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        if sample_weight is not None and ("sample_weight" in accepts or takes_kwargs):
            transformer.fit(X, y, sample_weight=sample_weight)
        elif "y" in accepts or len(accepts) > 1:
            transformer.fit(X, y)
        else:
            transformer.fit(X)
        return transformer

    # Narrows the inherited parameter types: sklearn's Pipeline accepts any
    # iterable, this one only what its transformers handle. The narrowing is
    # the point of the class, so the Liskov complaint is answered here.
    @override
    def predict(  # pyrefly: ignore[bad-override]
        self, X: pd.DataFrame | np.ndarray, confounders: pd.DataFrame | None = None
    ) -> np.ndarray:
        """Predict class labels, applying orthogonalization with *confounders*."""
        Xt = self._transform(X, confounders)
        return np.asarray(self.steps[-1][1].predict(Xt))

    # Narrows the inherited parameter types: sklearn's Pipeline accepts any
    # iterable, this one only what its transformers handle. The narrowing is
    # the point of the class, so the Liskov complaint is answered here.
    @override
    def predict_proba(  # pyrefly: ignore[bad-override]
        self, X: pd.DataFrame | np.ndarray, confounders: pd.DataFrame | None = None
    ) -> np.ndarray:
        """Predict class probabilities, applying orthogonalization with *confounders*."""
        Xt = self._transform(X, confounders)
        return np.asarray(self.steps[-1][1].predict_proba(Xt))

    def _transform(
        self, X: pd.DataFrame | np.ndarray, confounders: pd.DataFrame | None = None
    ) -> pd.DataFrame | np.ndarray:
        extra_confounders: list[pd.DataFrame] = []
        Xt = X
        for _name, transformer in self.steps[:-1]:
            if confounders is not None and isinstance(transformer, Orthogonalizer):
                aug = self._augment_confounders(confounders, extra_confounders)
                Xt = transformer.transform(Xt, confounders=aug)
            else:
                Xt = transformer.transform(Xt)
                ec = getattr(transformer, "get_extra_confounders", None)
                if ec is not None:
                    extra = ec()
                    if extra is not None and not extra.empty:
                        extra_confounders.append(extra)
        return Xt

    # Narrows the inherited parameter types: sklearn's Pipeline accepts any
    # iterable, this one only what its transformers handle. The narrowing is
    # the point of the class, so the Liskov complaint is answered here.
    @override
    def fit_predict(  # pyrefly: ignore[bad-override]
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        confounders: pd.DataFrame | None = None,
        **fit_params: Any,
    ) -> np.ndarray:
        self.fit(X, y, confounders=confounders, **fit_params)
        return self.predict(X, confounders=confounders)
