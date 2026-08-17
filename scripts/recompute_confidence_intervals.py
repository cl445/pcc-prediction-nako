"""Recompute `confidence_intervals.json` for every stored run.

The bootstrap consumes nothing but `y_true` and `y_pred_proba`, both of which
sit in each run's `predictions.csv`. Changing how it resamples therefore does
not need a retrained model — but nothing in the pipeline writes that file
outside a full run, so this script exists to do it.

It refuses to overwrite anything it cannot first reproduce. Every run is
recomputed under the *stored* settings and compared against the file on disk;
only if all of them match bit for bit does the new interval get written. That
is what makes the resulting diff attributable: the one thing that changed is
the one thing that was changed.

Usage:
    uv run python scripts/recompute_confidence_intervals.py            # check
    uv run python scripts/recompute_confidence_intervals.py --write    # apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from pcc_analysis.config import get_results_dir
from pcc_analysis.evaluation import compute_bootstrap_ci

# The settings a pipeline run uses. `n_bootstrap` is the config default; a run
# that used another value is detected by the reproduction check below rather
# than assumed away.
N_BOOTSTRAP = 1000
RANDOM_STATE = 42

# How the stored files were produced, and how they should be produced. ROC-AUC
# is invariant to prevalence, so stratifying costs it nothing; average
# precision is not, so stratifying narrows it.
STORED = {"roc_auc_ci": True, "pr_auc_ci": True}
TARGET = {"roc_auc_ci": True, "pr_auc_ci": False}

METRIC = {"roc_auc_ci": roc_auc_score, "pr_auc_ci": average_precision_score}


def _interval(preds: pd.DataFrame, key: str, *, stratify: bool) -> list[float]:
    _, lo, hi = compute_bootstrap_ci(
        preds["y_true"].to_numpy(),
        preds["y_pred_proba"].to_numpy(),
        METRIC[key],
        N_BOOTSTRAP,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return [lo, hi]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Directory of run directories (default: <results-dir>/runs).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the new intervals. Without it, only report.",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir or (get_results_dir() / "runs")
    runs = sorted(p.parent for p in runs_dir.glob("*/confidence_intervals.json"))
    if not runs:
        print(f"No run directories with confidence_intervals.json under {runs_dir}")
        return 1

    pending: list[tuple[Path, dict[str, list[float]]]] = []
    print(f"{'run':34s} {'metric':10s} {'stored':>22s} {'recomputed':>22s}")
    for run in runs:
        preds = pd.read_csv(run / "predictions.csv")
        stored = json.loads((run / "confidence_intervals.json").read_text())
        fresh: dict[str, list[float]] = {}

        for key, stored_stratify in STORED.items():
            if key not in stored:
                continue
            check = _interval(preds, key, stratify=stored_stratify)
            if max(abs(a - b) for a, b in zip(stored[key], check, strict=True)) > 1e-12:
                print(
                    f"\n{run.name}: {key} does not reproduce under the stored "
                    f"settings.\n  on disk    {stored[key]}\n  recomputed {check}\n"
                    "Refusing to write: a diff would not be attributable."
                )
                return 1
            fresh[key] = _interval(preds, key, stratify=TARGET[key])

        for key, value in fresh.items():
            moved = "" if value == stored[key] else "  <-"
            print(
                f"{run.name:34s} {key.replace('_ci', ''):10s} "
                f"[{stored[key][0]:.4f}, {stored[key][1]:.4f}] "
                f"[{value[0]:.4f}, {value[1]:.4f}]{moved}"
            )
        pending.append((run, {**stored, **fresh}))

    if not args.write:
        print("\nNothing written. Re-run with --write to apply.")
        return 0

    for run, payload in pending:
        (run / "confidence_intervals.json").write_text(json.dumps(payload) + "\n")
    print(f"\nWrote {len(pending)} confidence_intervals.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
