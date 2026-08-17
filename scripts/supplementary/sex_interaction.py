"""Sex modification of the Baseline-Mental-Health -> PCC association.

Whether the baseline mental-health effect is the same in both sexes, or
whether sex modifies it.

The primary multi-modal pipeline already orthogonalises mental health
against demographics (age, sex, centre) via DML, so the reported effect
is "over and above" sex. What that does *not* show is whether the effect
*differs* by sex — an association that is absent in one sex and twice as
large in the other averages to the same adjusted estimate as one that is
uniform. This script adds two complementary, pre-specified
sensitivity analyses on the same composite baseline-MH exposure and the
same clean-controls MRI analytic sample used for the E-value
(``evalue.py``), without any pipeline re-fit:

1. Multiplicative interaction model
       logit P(PCC) ~ MH + female + MH:female + age + centre
   The interaction odds ratio exp(beta_{MH:female}) is the ratio of the
   MH odds ratio in females to the MH odds ratio in males; it is tested
   both with a Wald test and with a likelihood-ratio test against the
   no-interaction model fitted on the identical rows.

2. Sex-stratified models
       logit P(PCC) ~ MH + age + centre        (separately by sex)
   reporting the adjusted baseline-MH odds ratio within each sex.

Usage:
    uv run python scripts/supplementary/sex_interaction.py
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from scipy import stats
from sklearn.metrics import roc_auc_score

from pcc_analysis.config import get_paper_constants_dir
from pcc_analysis.data_manager import load_pipeline_data
from pcc_analysis.sensitivity import composite_mh_positive

console = Console()


def _base_confounders(conf: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Age (continuous) and centre (drop-first dummies), as in ``evalue.py``."""
    age = conf["basis_age"].astype(float)
    center = pd.get_dummies(
        conf["basis_uort"].astype("category"),
        prefix="center",
        drop_first=True,
        dtype=float,
    )
    return age, center


def _or_ci_p(result: Any, name: str) -> tuple[float, float, float, float]:
    """Odds ratio, 95% CI lo/hi, p-value for a single coefficient."""
    coef = result.params[name]
    se = result.bse[name]
    p = result.pvalues[name]
    return (
        float(np.exp(coef)),
        float(np.exp(coef - 1.96 * se)),
        float(np.exp(coef + 1.96 * se)),
        float(p),
    )


def fit_interaction(
    y: pd.Series, exposure: pd.Series, conf: pd.DataFrame
) -> dict[str, Any]:
    """MH x sex interaction model with Wald and likelihood-ratio tests.

    ``female`` is coded 1 = female, 0 = male, so the MH main effect is the
    MH odds ratio in males and exp(MH:female) is the female/male ratio of
    MH odds ratios.
    """
    # NAKO basis_sex codes: 1 = male, 2 = female.
    female = (conf["basis_sex"].astype(float) == 2.0).astype(float)
    age, center = _base_confounders(conf)

    X = pd.DataFrame(index=exposure.index)
    X["mh_positive"] = exposure.astype(float).to_numpy()
    X["female"] = female.to_numpy()
    X["mh_x_female"] = X["mh_positive"] * X["female"]
    X["age"] = age.to_numpy()
    X = pd.concat([X, center], axis=1)
    X = sm.add_constant(X, has_constant="add")

    # Fit full and reduced (no interaction) on the *identical* row set so the
    # likelihood-ratio test is valid.
    data = pd.concat([y.astype(float).rename("y"), X], axis=1).dropna()
    y_clean = data["y"]
    X_full = data.drop(columns=["y"])
    X_red = X_full.drop(columns=["mh_x_female"])

    full = sm.Logit(y_clean, X_full).fit(disp=False)
    reduced = sm.Logit(y_clean, X_red).fit(disp=False)

    lr_stat = float(2.0 * (full.llf - reduced.llf))
    lr_p = float(stats.chi2.sf(lr_stat, df=1))

    or_int, lo_int, hi_int, p_int = _or_ci_p(full, "mh_x_female")
    or_mh_male, lo_mh_male, hi_mh_male, _ = _or_ci_p(full, "mh_positive")

    return {
        "n": len(y_clean),
        "mh_main_effect_or_in_males": {
            "point": or_mh_male,
            "ci_95": [lo_mh_male, hi_mh_male],
        },
        "interaction_or_female_vs_male": {
            "point": or_int,
            "ci_95": [lo_int, hi_int],
            "wald_p": p_int,
        },
        "likelihood_ratio_test": {"chi2_df1": lr_stat, "p_value": lr_p},
    }


def fit_stratum(
    y: pd.Series, exposure: pd.Series, conf: pd.DataFrame, mask: pd.Series
) -> dict[str, Any]:
    """Adjusted baseline-MH odds ratio (age + centre) within one sex stratum."""
    yy = y[mask].astype(float)
    ee = exposure[mask].astype(float)
    age, center = _base_confounders(conf[mask])

    X = pd.DataFrame(index=ee.index)
    X["mh_positive"] = ee.to_numpy()
    X["age"] = age.to_numpy()
    X = pd.concat([X, center], axis=1)
    X = sm.add_constant(X, has_constant="add")
    # Drop centre dummies that are all-zero within this stratum (avoid singular).
    X = X.loc[:, (X != 0).any(axis=0)]

    data = pd.concat([yy.rename("y"), X], axis=1).dropna()
    fit = sm.Logit(data["y"], data.drop(columns=["y"])).fit(disp=False)
    or_, lo, hi, p = _or_ci_p(fit, "mh_positive")

    n = len(data)
    n_exposed = int(data["mh_positive"].sum())
    n_outcome = int(data["y"].sum())
    return {
        "n": n,
        "n_mh_positive": n_exposed,
        "n_pcc": n_outcome,
        "prevalence_pcc": float(data["y"].mean()),
        "odds_ratio": {"point": or_, "ci_95": [lo, hi], "p_value": p},
    }


def demographics_decomposition(y: pd.Series, conf: pd.DataFrame) -> dict[str, Any]:
    """Is sex the dominant feature within the demographics modality?

    The demographics modality (age, sex, centre) ranks second overall; this
    asks which feature carries that signal. Reports univariable standalone
    discrimination (\\acs{ROC}-\\acs{AUC} of the raw feature as a score, which
    has no fitted parameters and hence no overfitting) and the univariable
    odds ratio, for sex and for age.
    """
    yv = y.astype(float)
    female = (conf["basis_sex"].astype(float) == 2.0).astype(float)
    age = conf["basis_age"].astype(float)

    def _univariable_or(predictor: pd.Series, scale: float = 1.0) -> dict[str, float]:
        X = sm.add_constant(
            pd.DataFrame({"x": predictor.astype(float) / scale}), has_constant="add"
        )
        data = pd.concat([yv.rename("y"), X], axis=1).dropna()
        fit = sm.Logit(data["y"], data.drop(columns=["y"])).fit(disp=False)
        return dict(
            zip(
                ("point", "ci_lo", "ci_hi", "p_value"),
                _or_ci_p(fit, "x"),
                strict=True,
            )
        )

    return {
        "sex_female": {
            "standalone_auc": float(roc_auc_score(yv, female)),
            "univariable_or_female_vs_male": _univariable_or(female),
        },
        "age": {
            "standalone_auc": float(roc_auc_score(yv, age)),
            "univariable_or_per_10y": _univariable_or(age, scale=10.0),
        },
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]Sex modification of baseline MH -> PCC")

    y, X, conf = load_pipeline_data(
        target="bahmer", clean_controls=True, cohort="mri", stack_variant="full"
    )
    exposure = composite_mh_positive(X["mental_health"])

    interaction = fit_interaction(y, exposure, conf)

    # NAKO basis_sex codes: 1 = male, 2 = female.
    is_female = conf["basis_sex"].astype(float) == 2.0
    is_male = conf["basis_sex"].astype(float) == 1.0
    female = fit_stratum(y, exposure, conf, is_female)
    male = fit_stratum(y, exposure, conf, is_male)

    demo = demographics_decomposition(y, conf)

    summary = {
        "sample": {
            "cohort": "mri",
            "clean_controls": True,
            "target": "bahmer",
            "n_total": len(y),
            "n_female": int(is_female.sum()),
            "n_male": int(is_male.sum()),
        },
        "exposure_definition": (
            "Composite baseline mental-health positivity: PHQ-9 sum >= 10 "
            "OR GAD-7 sum >= 10 OR MINI major depression == 1 "
            "(missing components treated as non-positive)."
        ),
        "interaction_model": {
            "specification": "logit P(PCC) ~ MH + female + MH:female + age + centre",
            **interaction,
        },
        "stratified_models": {
            "specification": "logit P(PCC) ~ MH + age + centre, fitted within each sex",
            "female": female,
            "male": male,
        },
        "demographics_decomposition": {
            "specification": (
                "Univariable standalone discrimination (ROC-AUC of the raw "
                "feature) and univariable OR for each demographic feature, to "
                "show whether the second-ranked demographics modality is carried "
                "by sex or by age."
            ),
            **demo,
        },
    }

    out_path = get_paper_constants_dir() / "sex_interaction.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Pretty-print
    tbl = Table(
        title="Baseline-MH -> PCC: sex interaction and stratified ORs",
        show_header=True,
    )
    tbl.add_column("Quantity", style="bold")
    tbl.add_column("Value", justify="right")
    im = interaction["interaction_or_female_vs_male"]
    lrt = interaction["likelihood_ratio_test"]
    mm = interaction["mh_main_effect_or_in_males"]
    tbl.add_row("N (interaction model)", f"{interaction['n']}")
    tbl.add_row(
        "MH OR in males (main effect)",
        f"{mm['point']:.2f} [{mm['ci_95'][0]:.2f}, {mm['ci_95'][1]:.2f}]",
    )
    tbl.add_row(
        "Interaction OR (female/male)",
        f"{im['point']:.2f} [{im['ci_95'][0]:.2f}, {im['ci_95'][1]:.2f}] (Wald p={im['wald_p']:.2g})",
    )
    tbl.add_row("LR test (df=1)", f"chi2={lrt['chi2_df1']:.2f}, p={lrt['p_value']:.2g}")
    tbl.add_section()
    fo = female["odds_ratio"]
    mo = male["odds_ratio"]
    tbl.add_row(
        f"MH OR | female (n={female['n']})",
        f"{fo['point']:.2f} [{fo['ci_95'][0]:.2f}, {fo['ci_95'][1]:.2f}] (p={fo['p_value']:.2g})",
    )
    tbl.add_row(
        f"MH OR | male (n={male['n']})",
        f"{mo['point']:.2f} [{mo['ci_95'][0]:.2f}, {mo['ci_95'][1]:.2f}] (p={mo['p_value']:.2g})",
    )
    tbl.add_section()
    sx = demo["sex_female"]
    ag = demo["age"]
    sxor = sx["univariable_or_female_vs_male"]
    agor = ag["univariable_or_per_10y"]
    tbl.add_row(
        "Sex (female): standalone AUC / OR",
        f"{sx['standalone_auc']:.3f} / OR {sxor['point']:.2f} [{sxor['ci_lo']:.2f}, {sxor['ci_hi']:.2f}]",
    )
    tbl.add_row(
        "Age: standalone AUC / OR per 10y",
        f"{ag['standalone_auc']:.3f} / OR {agor['point']:.2f} [{agor['ci_lo']:.2f}, {agor['ci_hi']:.2f}]",
    )
    console.print(tbl)
    console.print(f"[green]Wrote {out_path}")


if __name__ == "__main__":
    main()
