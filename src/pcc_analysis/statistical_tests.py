"""Statistical tests for modality comparison.

Implements the Nadeau-Bengio corrected t-test for cross-validated
score differences and Holm-Bonferroni correction for multiple testing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, overload

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from ._types import NadeauBengioResult

logger = logging.getLogger(__name__)


def nadeau_bengio_corrected_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_train: int,
    n_test: int,
    *,
    alternative: str = "two-sided",
) -> NadeauBengioResult:
    """Nadeau-Bengio corrected paired t-test for CV scores.

    Corrects for the underestimation of variance caused by overlapping
    training sets in repeated/K-fold cross-validation.

    Parameters
    ----------
    scores_a, scores_b : ndarray, shape (K,)
        Per-fold metric scores for models A and B.
    n_train : int
        Number of training samples per fold.
    n_test : int
        Number of test samples per fold.
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    dict with keys: t_statistic, p_value, df, mean_diff, corrected_std
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    if len(scores_a) != len(scores_b):
        msg = "scores_a and scores_b must have the same length"
        raise ValueError(msg)

    k = len(scores_a)
    diffs = scores_a - scores_b
    mean_diff = float(np.mean(diffs))
    var_diffs = float(np.var(diffs, ddof=1))

    # Nadeau-Bengio correction: inflate variance estimate
    corrected_var = (1.0 / k + n_test / n_train) * var_diffs

    if corrected_var < 1e-15:
        # No variance — scores are identical
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "df": float(k - 1),
            "mean_diff": mean_diff,
            "corrected_std": 0.0,
        }

    corrected_std = float(np.sqrt(corrected_var))
    t_stat = mean_diff / corrected_std
    df = k - 1

    if alternative == "two-sided":
        p_value = float(2 * stats.t.sf(abs(t_stat), df))
    elif alternative == "greater":
        p_value = float(stats.t.sf(t_stat, df))
    elif alternative == "less":
        p_value = float(stats.t.cdf(t_stat, df))
    else:
        msg = f"alternative must be 'two-sided', 'greater', or 'less', got '{alternative}'"
        raise ValueError(msg)

    return {
        "t_statistic": float(t_stat),
        "p_value": p_value,
        "df": float(df),
        "mean_diff": mean_diff,
        "corrected_std": corrected_std,
    }


@overload
def holm_bonferroni_correction(p_values: dict[str, float]) -> dict[str, float]: ...


@overload
def holm_bonferroni_correction(p_values: list[float]) -> list[float]: ...


def holm_bonferroni_correction(
    p_values: dict[str, float] | list[float],
) -> dict[str, float] | list[float]:
    """Holm-Bonferroni step-down correction for multiple testing.

    Parameters
    ----------
    p_values : dict[str, float] or list[float]
        Raw p-values. If dict, keys are preserved in the output.

    Returns
    -------
    Same type as input with adjusted p-values.
    """
    if isinstance(p_values, dict):
        keys = list(p_values.keys())
        raw = np.array([p_values[k] for k in keys])
        adjusted = _holm_adjust(raw)
        return dict(zip(keys, adjusted, strict=True))

    raw = np.array(p_values, dtype=np.float64)
    adjusted = _holm_adjust(raw)
    return list(adjusted.tolist())


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Core Holm step-down adjustment."""
    m = len(p_values)
    if m == 0:
        return p_values.copy()

    # Sort indices by p-value (ascending)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    # Adjust: p_adj[i] = (m - i) * p_sorted[i]
    adjusted = np.empty(m)
    for i in range(m):
        adjusted[i] = (m - i) * sorted_p[i]

    # Enforce monotonicity (cumulative max)
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Unsort back to original order
    result = np.empty(m)
    result[order] = adjusted
    return result


def compare_modalities_pairwise(
    modality_scores_df: pd.DataFrame,
    reference_modality: str,
    metric: str = "pr_auc",
    n_total: int | None = None,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Compare each modality against a reference using NB-corrected t-test.

    The reference is required rather than defaulted, because the returned
    p-values mean nothing without it and the reference row is absent from the
    result: a caller that assumes the wrong reference reads every p-value as
    an answer to a question that was never asked.

    Parameters
    ----------
    modality_scores_df : DataFrame
        Must have columns 'fold', 'modality', and the ``metric`` column.
        Each row is one fold-modality score.
    reference_modality : str
        The baseline modality to compare against. It is excluded from the
        result, since it cannot be compared with itself.
    metric : str
        Which metric column to compare.
    n_total : int or None
        Total sample size. If None, estimated from fold count assuming
        equal-sized folds.
    n_folds : int
        Number of CV folds (used to estimate n_train, n_test if n_total
        is not given).

    Returns
    -------
    DataFrame with columns: modality, mean_diff, t_statistic, p_value_raw,
    p_value_holm.
    """
    present = list(modality_scores_df["modality"].unique())
    if reference_modality not in present:
        # Without this, an absent reference yields zero folds to compare
        # against, every modality is skipped for a fold-count mismatch, and
        # the caller receives a well-formed empty table. Split-modality runs
        # (where mental_health exists only as its five sub-scales) hit this.
        raise ValueError(
            f"reference modality {reference_modality!r} is not in the scores; "
            f"present: {sorted(present)}"
        )
    modalities = [m for m in present if m != reference_modality]

    # Per-fold scores for the reference, indexed by fold rather than by
    # position. The test is paired, so a modality's fold 7 has to meet the
    # reference's fold 7; lining two sorted arrays up by position gives the
    # right answer only while both cover exactly the same folds, and pairs
    # silently across a gap when they do not.
    ref_by_fold = (
        modality_scores_df[modality_scores_df["modality"] == reference_modality]
        .set_index("fold")[metric]
        .sort_index()
    )
    ref_scores = ref_by_fold.to_numpy()
    k = len(ref_scores)

    if n_total is None:
        # Rough estimate: total = k * n_test, n_test = n_total / k
        # Use k * 100 as default guess
        n_total = k * 100
        logger.info("n_total not provided; estimating as %d", n_total)

    n_test = n_total // n_folds
    n_train = n_total - n_test

    rows = []
    raw_p_values: dict[str, float] = {}

    for mod in modalities:
        mod_by_fold = (
            modality_scores_df[modality_scores_df["modality"] == mod]
            .set_index("fold")[metric]
            .sort_index()
        )
        if not mod_by_fold.index.equals(ref_by_fold.index):
            logger.warning(
                "Modality %s was scored on folds %s, the reference on %s; "
                "skipping, because a paired test needs the same folds on "
                "both sides",
                mod,
                sorted(mod_by_fold.index),
                sorted(ref_by_fold.index),
            )
            continue
        mod_scores = mod_by_fold.to_numpy()

        result = nadeau_bengio_corrected_t_test(mod_scores, ref_scores, n_train, n_test)
        rows.append(
            {
                "modality": mod,
                "mean_diff": result["mean_diff"],
                "t_statistic": result["t_statistic"],
                "p_value_raw": result["p_value"],
            }
        )
        raw_p_values[mod] = result["p_value"]

    if not rows:
        return pd.DataFrame(
            columns=[
                "modality",
                "mean_diff",
                "t_statistic",
                "p_value_raw",
                "p_value_holm",
            ]
        )

    # Apply Holm correction
    adjusted = holm_bonferroni_correction(raw_p_values)

    df = pd.DataFrame(rows)
    df["p_value_holm"] = df["modality"].map(adjusted)
    return df.sort_values("mean_diff", ascending=False).reset_index(drop=True)
