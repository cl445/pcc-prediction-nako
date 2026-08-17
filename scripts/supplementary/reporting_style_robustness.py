"""Three robustness analyses addressing reporting-style confounding.

All three run on the existing processed parquets — no pipeline re-run required.

(1)  Chemosensory outcome. Does baseline mental health predict a
     style-robust outcome (loss of smell/taste) with a magnitude similar to
     the primary Bahmer OR? Smell/taste loss is the post-COVID symptom least
     driven by negative affectivity, so a comparable OR is hard to explain by
     reporting style alone.

(2)  Difference-in-association (a coarse but genuine second-hit test). Among
     ALL Corona-2 participants (infected and non-infected, who all answered
     the current PHQ-9/GAD-7), does baseline mental health predict CURRENT
     symptom load more strongly in the infected? A positive baseline-MH x
     infection interaction is what a second hit predicts. (An outcome-
     identical comparison is impossible because the 4-12 month PCC items were
     not administered to non-infected participants.)

(3)  Mixed-controls sensitivity. The baseline-MH -> PCC odds ratio under the
     alternative mixed-controls outcome (sub-threshold k0=1 symptomatics
     retained as PCC-negative), reported next to the clean-controls OR so the
     reader can judge whether the clean-controls design amplifies the effect.

Outputs:
    results/robustness_analyses.json
    results/robustness_analyses_tex.tex   (LaTeX constants)

Usage:
    uv run python scripts/supplementary/reporting_style_robustness.py
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pcc_analysis.config import get_paper_constants_dir, get_processed_dir
from pcc_analysis.sensitivity import composite_mh_positive

console = Console()


# --------------------------------------------------------------------------
# Result schemas (double as documentation of the emitted JSON structure)
# --------------------------------------------------------------------------
class ChemStat(TypedDict):
    n: int
    n_chemosensory_pos: int
    prevalence: float
    or_mh: float
    ci: list[float]
    p: float


# "or" is a Python keyword, so this one needs the functional TypedDict syntax.
Stratum = TypedDict("Stratum", {"n": int, "or": float, "ci": list[float], "p": float})


class DiaResult(TypedDict):
    n_total: int
    n_infected: int
    n_non_infected: int
    outcome: str
    interaction_or: float
    interaction_ci: list[float]
    interaction_p: float
    stratified: dict[str, Stratum]


class MixedStat(TypedDict):
    n: int
    n_pcc_pos: int
    prevalence: float
    or_mh: float
    ci: list[float]
    p: float


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _mri_ids() -> set[int]:
    processed = get_processed_dir()
    ids: set[int] = set()
    for f in [
        *processed.glob("mri_cortical_*.parquet"),
        processed / "mri_subcortical.parquet",
        processed / "mri_cerebellar.parquet",
    ]:
        ids |= set(pd.read_parquet(f, columns=["ID"])["ID"])
    return ids


def _symptom_flag(frame: pd.DataFrame, column: str) -> pd.Series:
    """Symptom column as a 0/1-coded series, absent column counting as 0."""
    if column not in frame.columns:
        return pd.Series(0, index=frame.index)
    return frame[column].fillna(0)


def _base_frame() -> pd.DataFrame:
    """Merge demographics, Corona-2 PCC items, baseline MH, current PHQ/GAD."""
    processed = get_processed_dir()
    demo = pd.read_parquet(
        processed / "demographics.parquet",
        columns=["ID", "basis_age", "basis_sex", "basis_uort"],
    )
    c2 = pd.read_parquet(processed / "corona2_pcc.parquet")
    mh = pd.read_parquet(processed / "mental_health.parquet")
    psy = pd.read_parquet(
        processed / "psychometric_scores.parquet"
    )  # current Corona-2 PHQ/GAD
    df = demo.merge(c2, on="ID", how="left")
    mh_cols = ["ID", "phq9_sum", "gad7_sum"]
    if "mini_major_depression" in mh.columns:
        mh_cols.append("mini_major_depression")
    df = df.merge(mh[mh_cols], on="ID", how="left")
    psy_cols = ["ID"] + [c for c in ("phq9_total", "gad7_total") if c in psy.columns]
    df = df.merge(psy[psy_cols], on="ID", how="left")
    df["has_mri"] = df["ID"].isin(_mri_ids())
    return df


def _current_mh_positive(df: pd.DataFrame) -> pd.Series:
    """Current (Corona-2) MH-positive: current PHQ-9>=10 OR GAD-7>=10."""
    phq = (df["phq9_total"] >= 10).fillna(False)
    gad = (df["gad7_total"] >= 10).fillna(False)
    return (phq | gad).astype(int)


def _logit_or(
    y: pd.Series,
    design: pd.DataFrame,
    term: str,
) -> tuple[float, float, float, float]:
    """Fit logistic regression; return OR, CI lo, CI hi, p for ``term``."""
    X = sm.add_constant(design, has_constant="add")
    res = sm.Logit(y.astype(float), X.astype(float), missing="drop").fit(disp=False)
    coef = res.params[term]
    lo, hi = res.conf_int().loc[term]
    return (
        float(np.exp(coef)),
        float(np.exp(lo)),
        float(np.exp(hi)),
        float(res.pvalues[term]),
    )


def _confounders(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["age"] = df["basis_age"].astype(float)
    out["sex"] = df["basis_sex"].astype("category").cat.codes.astype(float)
    centre = pd.get_dummies(
        df["basis_uort"].astype("category"), prefix="c", drop_first=True, dtype=float
    )
    return pd.concat([out, centre], axis=1)


def _clean_controls(df: pd.DataFrame, *, mri_only: bool) -> pd.DataFrame:
    m = (df["had_covid"] == 1) & (df["valid_symptoms"] == 1)
    if mri_only:
        m &= df["has_mri"]
    a = df.loc[m].copy()
    # clean controls: drop sub-threshold k0=1 with Bahmer-negative
    a = a.loc[~((a["d_co2_k0"] == 1) & (a["bahmer_any_pcs"] == 0))].copy()
    return a


# --------------------------------------------------------------------------
# (1) Chemosensory outcome
# --------------------------------------------------------------------------
def analysis_chemosensory(df: pd.DataFrame) -> dict[str, ChemStat]:
    results: dict[str, ChemStat] = {}
    for label, mri_only in (("mri", True), ("all_infected", False)):
        a = _clean_controls(df, mri_only=mri_only)
        chem = (
            (_symptom_flag(a, "symptom_loss_of_smell") == 1)
            | (_symptom_flag(a, "symptom_loss_of_taste") == 1)
        ).astype(int)
        exposure = composite_mh_positive(a)
        design = pd.concat([exposure.rename("mh_pos"), _confounders(a)], axis=1)
        orr, lo, hi, p = _logit_or(chem, design, "mh_pos")
        results[label] = {
            "n": len(a),
            "n_chemosensory_pos": int(chem.sum()),
            "prevalence": float(chem.mean()),
            "or_mh": orr,
            "ci": [lo, hi],
            "p": p,
        }
    return results


# --------------------------------------------------------------------------
# (2) Difference-in-association (baseline MH x infection on current load)
# --------------------------------------------------------------------------
def analysis_difference_in_association(df: pd.DataFrame) -> DiaResult:
    # Full Corona-2 sample with current PHQ/GAD and baseline MH observed.
    d = df.copy()
    d = d[d["had_covid"].isin([0, 1])]
    d = d.dropna(subset=["phq9_total", "gad7_total"])
    base = composite_mh_positive(d)
    cur = _current_mh_positive(d)
    infected = d["had_covid"].astype(int)
    inter = base * infected
    design = pd.concat(
        [
            base.rename("base_mh"),
            infected.rename("infected"),
            inter.rename("base_x_infected"),
            _confounders(d),
        ],
        axis=1,
    )
    orr, inter_lo, inter_hi, p = _logit_or(cur, design, "base_x_infected")

    # Stratified ORs (baseline MH -> current MH-positive) within each arm.
    strata: dict[str, Stratum] = {}
    for name, val in (("infected", 1), ("non_infected", 0)):
        sub = d[infected == val]
        b = composite_mh_positive(sub)
        c = _current_mh_positive(sub)
        des = pd.concat([b.rename("base_mh"), _confounders(sub)], axis=1)
        o, lo, hi, pp = _logit_or(c, des, "base_mh")
        strata[name] = {"n": len(sub), "or": o, "ci": [lo, hi], "p": pp}

    return {
        "n_total": len(d),
        "n_infected": int((infected == 1).sum()),
        "n_non_infected": int((infected == 0).sum()),
        "outcome": "current Corona-2 MH-positive (PHQ-9>=10 OR GAD-7>=10)",
        "interaction_or": orr,
        "interaction_ci": [inter_lo, inter_hi],
        "interaction_p": p,
        "stratified": strata,
    }


# --------------------------------------------------------------------------
# (3) Mixed-controls sensitivity of the baseline-MH OR
# --------------------------------------------------------------------------
def analysis_mixed_controls(df: pd.DataFrame) -> dict[str, MixedStat]:
    out: dict[str, MixedStat] = {}
    for name, clean in (("clean_controls", True), ("mixed_controls", False)):
        m = (df["had_covid"] == 1) & (df["valid_symptoms"] == 1) & df["has_mri"]
        a = df.loc[m].copy()
        if clean:
            a = a.loc[~((a["d_co2_k0"] == 1) & (a["bahmer_any_pcs"] == 0))].copy()
        # else: keep sub-threshold k0=1 as PCC-negative (mixed controls)
        y = a["bahmer_any_pcs"].astype(int)
        exposure = composite_mh_positive(a)
        design = pd.concat([exposure.rename("mh_pos"), _confounders(a)], axis=1)
        orr, lo, hi, p = _logit_or(y, design, "mh_pos")
        out[name] = {
            "n": len(a),
            "n_pcc_pos": int(y.sum()),
            "prevalence": float(y.mean()),
            "or_mh": orr,
            "ci": [lo, hi],
            "p": p,
        }
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]Robustness analyses — reporting-style confounding")
    df = _base_frame()

    chem = analysis_chemosensory(df)
    dia = analysis_difference_in_association(df)
    mixed = analysis_mixed_controls(df)

    summary = {
        "chemosensory_outcome": chem,
        "difference_in_association": dia,
        "mixed_controls": mixed,
    }
    results_dir = get_paper_constants_dir()
    (results_dir / "robustness_analyses.json").write_text(json.dumps(summary, indent=2))

    # --- LaTeX constants ---
    cm = chem["mri"]
    ca = chem["all_infected"]
    mc_clean = mixed["clean_controls"]
    mc_mixed = mixed["mixed_controls"]
    tex = [
        "% Generated by scripts/supplementary/reporting_style_robustness.py — do not edit by hand.",
        f"\\newcommand{{\\resChemOrMRI}}{{{cm['or_mh']:.2f}}}",
        f"\\newcommand{{\\resChemOrMRIlo}}{{{cm['ci'][0]:.2f}}}",
        f"\\newcommand{{\\resChemOrMRIhi}}{{{cm['ci'][1]:.2f}}}",
        f"\\newcommand{{\\resChemNposMRI}}{{{cm['n_chemosensory_pos']}}}",
        f"\\newcommand{{\\resChemOrAll}}{{{ca['or_mh']:.2f}}}",
        f"\\newcommand{{\\resChemOrAlllo}}{{{ca['ci'][0]:.2f}}}",
        f"\\newcommand{{\\resChemOrAllhi}}{{{ca['ci'][1]:.2f}}}",
        f"\\newcommand{{\\resChemNposAll}}{{{ca['n_chemosensory_pos']}}}",
        f"\\newcommand{{\\resDiaInteractionOr}}{{{dia['interaction_or']:.2f}}}",
        f"\\newcommand{{\\resDiaInteractionOrlo}}{{{dia['interaction_ci'][0]:.2f}}}",
        f"\\newcommand{{\\resDiaInteractionOrhi}}{{{dia['interaction_ci'][1]:.2f}}}",
        f"\\newcommand{{\\resDiaOrInfected}}{{{dia['stratified']['infected']['or']:.2f}}}",
        f"\\newcommand{{\\resDiaOrNonInfected}}{{{dia['stratified']['non_infected']['or']:.2f}}}",
        f"\\newcommand{{\\resMixedOr}}{{{mc_mixed['or_mh']:.2f}}}",
        f"\\newcommand{{\\resMixedOrlo}}{{{mc_mixed['ci'][0]:.2f}}}",
        f"\\newcommand{{\\resMixedOrhi}}{{{mc_mixed['ci'][1]:.2f}}}",
        f"\\newcommand{{\\resMixedN}}{{{mc_mixed['n']}}}",
        f"\\newcommand{{\\resMixedPrev}}{{{100 * mc_mixed['prevalence']:.1f}}}",
        f"\\newcommand{{\\resCleanOrCheck}}{{{mc_clean['or_mh']:.2f}}}",
    ]
    (results_dir / "robustness_analyses_tex.tex").write_text("\n".join(tex) + "\n")

    # --- Pretty-print ---
    t = Table(title="Robustness analyses", show_header=True)
    t.add_column("Analysis", style="bold")
    t.add_column("Result", justify="right")
    t.add_row(
        "Chemosensory OR (MRI sample)",
        f"{cm['or_mh']:.2f} [{cm['ci'][0]:.2f}, {cm['ci'][1]:.2f}], {cm['n_chemosensory_pos']} events",
    )
    t.add_row(
        "Chemosensory OR (all infected)",
        f"{ca['or_mh']:.2f} [{ca['ci'][0]:.2f}, {ca['ci'][1]:.2f}], {ca['n_chemosensory_pos']} events",
    )
    t.add_row(
        "DiA interaction OR",
        f"{dia['interaction_or']:.2f} [{dia['interaction_ci'][0]:.2f}, {dia['interaction_ci'][1]:.2f}], p={dia['interaction_p']:.2g}",
    )
    t.add_row("  base-MH OR | infected", f"{dia['stratified']['infected']['or']:.2f}")
    t.add_row(
        "  base-MH OR | non-infected", f"{dia['stratified']['non_infected']['or']:.2f}"
    )
    t.add_row("Clean-controls OR (check)", f"{mc_clean['or_mh']:.2f}")
    t.add_row(
        "Mixed-controls OR",
        f"{mc_mixed['or_mh']:.2f} [{mc_mixed['ci'][0]:.2f}, {mc_mixed['ci'][1]:.2f}]",
    )
    console.print(t)
    console.print("[green]Wrote results/robustness_analyses.json + _tex.tex")


if __name__ == "__main__":
    main()
