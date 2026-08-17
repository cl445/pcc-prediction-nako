"""Compute agreement statistics between PCC outcome definitions.

The primary outcome is the Bahmer weighted PCS (>10.75).  This script
quantifies agreement with two alternative operationalisations reported
in the Methods chapter as sensitivity checks:

  1. Bahmer weighted PCS (>10.75) — primary outcome
  2. Neurocognitive subtype (>=2 of four items: fatigue, reduced
     physical capacity, memory problems, concentration problems) —
     neurocognitive-subtype sensitivity
  3. Bahmer weighted PCS (>26.25) — severity sub-definition

Outputs:
  - Prevalence per definition
  - 2x2 cross-tabulations vs. the primary outcome
  - Cohen's kappa (agreement beyond chance)
  - LaTeX table fragment for supplementary materials

Usage:
    uv run python scripts/supplementary/outcome_agreement.py
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from pcc_analysis.config import (
    get_paper_constants_dir,
    get_processed_dir,
    get_results_dir,
)
from pcc_analysis.data_manager import (
    NEUROCOG_SYMPTOM_ITEMS,
    NakoDataManager,
    _compute_neurocog_label,
    load_pipeline_data,
    prepare_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
PRIMARY_LABEL = "Bahmer weighted PCS (>10.75)"
PRIMARY_COL = "bahmer_any_pcs"


def main() -> None:
    print("=" * 60)
    print("PCC Outcome Agreement Analysis")
    print("=" * 60)

    # ── 1. Load analytic sample (neuroimaging subsample) ──────────────
    print("\n[1/2] Loading pipeline data (neuroimaging subsample)...")
    y, _X_dict, _confounders = load_pipeline_data()
    print(f"  Analytic sample: N = {len(y):,}")

    # ── 2. Load alternative PCC definitions from corona2_pcc ──────────
    print("\n[2/2] Loading alternative PCC definitions...")
    processed_dir = get_processed_dir()
    mgr = NakoDataManager(processed_dir)
    pcc_all = prepare_index(mgr.load_corona2_pcc(), "PCC (all)")

    common_idx = y.index.intersection(pcc_all.index)
    print(f"  Aligned sample:  N = {len(common_idx):,}")

    outcomes = pd.DataFrame(index=common_idx)
    for col in [
        "any_pcc",
        "severe_pcc",
        "bahmer_any_pcs",
        "bahmer_severe_pcs",
        "had_covid",
        "n_symptoms",
        "bahmer_pcs_score",
        *NEUROCOG_SYMPTOM_ITEMS,
    ]:
        if col in pcc_all.columns:
            outcomes[col] = pcc_all.loc[common_idx, col]

    # Neurocognitive subtype (>=2 of 4 items) — constructed here from the
    # underlying symptom columns so the same clean-controls sample is
    # evaluated under an alternative operationalisation.
    outcomes["neurocog_ge2"] = _compute_neurocog_label(pcc_all.loc[common_idx])

    # ── Prevalence ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREVALENCE BY DEFINITION")
    print("=" * 60)

    definitions = {
        PRIMARY_LABEL: PRIMARY_COL,
        "Neurocognitive subtype ($\\geq 2$ of 4)": "neurocog_ge2",
        "Bahmer severity (>26.25)": "bahmer_severe_pcs",
    }

    prevalence: dict[str, dict[str, object]] = {}
    for label, col in definitions.items():
        if col in outcomes.columns:
            n_pos = int(outcomes[col].sum())
            pct = float(outcomes[col].mean()) * 100
            prevalence[label] = {"n": n_pos, "pct": pct, "col": col}
            print(f"  {label:35s}  n = {n_pos:6,}  ({pct:5.1f}%)")
        else:
            print(f"  {label:35s}  [column '{col}' not available]")

    # ── Cross-tabulations & Cohen's κ vs. primary ─────────────────────
    print("\n" + "=" * 60)
    print(f"AGREEMENT VS. PRIMARY OUTCOME ({PRIMARY_LABEL})")
    print("=" * 60)

    comparisons = [
        ("Primary vs. Neurocog (>=2 of 4)", PRIMARY_COL, "neurocog_ge2"),
        ("Primary vs. Bahmer severity", PRIMARY_COL, "bahmer_severe_pcs"),
    ]

    kappa_results: dict[str, float] = {}

    for label, col_a, col_b in comparisons:
        if col_a not in outcomes.columns or col_b not in outcomes.columns:
            print(f"\n  {label}: skipped (missing column)")
            continue

        mask = outcomes[[col_a, col_b]].notna().all(axis=1)
        a = outcomes.loc[mask, col_a].astype(int)
        b = outcomes.loc[mask, col_b].astype(int)

        kappa = float(cohen_kappa_score(a, b))
        kappa_results[label] = kappa

        cm = confusion_matrix(a, b, labels=[0, 1])
        concordant = int((a == b).sum())
        concordance_pct = concordant / len(a) * 100

        print(f"\n  {label}")
        print(f"  Cohen's κ = {kappa:.3f}")
        print(f"  Concordance: {concordant:,} / {len(a):,} ({concordance_pct:.1f}%)")
        print(f"  Confusion matrix (rows={col_a}, cols={col_b}):")
        print(f"    TN={cm[0, 0]:6,}  FP={cm[0, 1]:6,}")
        print(f"    FN={cm[1, 0]:6,}  TP={cm[1, 1]:6,}")

    # ── LaTeX table ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LATEX TABLE (for supplementary.tex)")
    print("=" * 60)

    latex_lines = []
    for label, info in prevalence.items():
        col = str(info["col"])
        if col == PRIMARY_COL:
            kappa_str = "---"
        elif col == "neurocog_ge2":
            kappa_str = f"\\num{{{kappa_results.get('Primary vs. Neurocog (>=2 of 4)', float('nan')):.3f}}}"
        elif col == "bahmer_severe_pcs":
            kappa_str = f"\\num{{{kappa_results.get('Primary vs. Bahmer severity', float('nan')):.3f}}}"
        else:
            kappa_str = "---"

        latex_lines.append(
            f"{label:40s} & \\num{{{info['n']}}} "
            f"& \\num{{{info['pct']:.1f}}} & {kappa_str} \\\\"
        )

    latex_table = "\n".join(latex_lines)
    print(f"\n{latex_table}")

    tex_path = get_paper_constants_dir() / "outcome_agreement.tex"
    tex_path.write_text(latex_table + "\n")
    print(f"\n  LaTeX table saved to: {tex_path}")

    # ── Save results ──────────────────────────────────────────────────
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "outcome_agreement.txt"

    with out_path.open("w") as f:
        f.write("PCC Outcome Agreement Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Primary outcome: {PRIMARY_LABEL} (column '{PRIMARY_COL}')\n\n")

        f.write("Prevalence:\n")
        for label, info in prevalence.items():
            f.write(f"  {label}: n={info['n']:,} ({info['pct']:.1f}%)\n")

        f.write("\nCohen's kappa vs. primary:\n")
        for label, kappa in kappa_results.items():
            f.write(f"  {label}: κ = {kappa:.3f}\n")

        f.write(f"\nLaTeX table rows:\n{latex_table}\n")

    print(f"\n  Results saved to: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
