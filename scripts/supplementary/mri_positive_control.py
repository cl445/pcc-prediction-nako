"""MRI feature-extraction positive control.

The primary analysis reports that all six structural-MRI parcellations
perform at or near chance for predicting PCC. The embedded mental-health
positive control validates only the *meta-learner* sensitivity, not that
the MRI feature extraction and pre-processing deliver usable information.
The decisive control for the latter is to show that the very same MRI
features — fed through the same kind of regularised base learner, *before*
the DML orthogonalisation against age/sex/centre — predict age and sex
almost perfectly.

A high age-R^2 and sex-ROC-AUC establish that the MRI pipeline is intact
and information-rich, so "pipeline insensitivity" cannot explain the PCC
null. It also makes explicit *why* the orthogonalised PCC signal is near
chance: the DML step removes exactly the age/sex/ICV axes along which the
volumetric features vary most, i.e. the axes that "brain reserve" would
live on.

Runs on the identical clean-controls MRI analytic sample (N = 8461) and
the identical processed volumetric features used by the main pipeline.

Outputs:
    results/mri_positive_control.json
    results/mri_positive_control_tex.tex   (LaTeX constants)

Usage:
    uv run python scripts/supplementary/mri_positive_control.py
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pcc_analysis.config import get_paper_constants_dir
from pcc_analysis.data_manager import load_pipeline_data

console = Console()

RANDOM_STATE = 42
N_FOLDS = 10

MRI_MODALITIES = [
    "mri_desikan",
    "mri_destrieux",
    "mri_julich",
    "mri_yeo",
    "mri_subcortical",
    "mri_cerebellar",
]

# Display labels matching the paper's atlas naming.
MRI_LABELS = {
    "mri_desikan": "Desikan-Killiany",
    "mri_destrieux": "Destrieux",
    "mri_julich": "Julich",
    "mri_yeo": "Yeo Networks",
    "mri_subcortical": "Subcortical",
    "mri_cerebellar": "Cerebellar",
}

# TeX-macro-safe suffixes (no digits / underscores).
TEX_SUFFIX = {
    "mri_desikan": "Desikan",
    "mri_destrieux": "Destrieux",
    "mri_julich": "Julich",
    "mri_yeo": "Yeo",
    "mri_subcortical": "Subcortical",
    "mri_cerebellar": "Cerebellar",
    "combined": "Combined",
}


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the atlas volume features (``ma_*``), coerced to float.

    Drops the identifier column and the ``etiv`` (intracranial-volume)
    covariate that ``load_pipeline_data`` attaches to each MRI modality for
    ICV adjustment. Excluding ICV keeps the feature counts identical to the
    atlas counts reported in Table 1 / Figure 2, and makes the positive
    control a test of the regional volumes themselves rather than of raw
    head size (which is a trivial, sex-dimorphic shortcut).
    """
    cols = [c for c in df.columns if c.startswith("ma_")]
    return pd.DataFrame(df[cols]).astype(float)


class PosControlResult(TypedDict):
    """Age- and sex-prediction metrics for one MRI feature set."""

    n_features: int
    age: dict[str, float]
    sex: dict[str, float]


def predict_age(features: pd.DataFrame, age: np.ndarray) -> dict[str, float]:
    """Out-of-fold age prediction (RidgeCV regression)."""
    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )
    oof = np.asarray(cross_val_predict(model, features, age, cv=N_FOLDS), dtype=float)
    return {
        "r2": float(r2_score(age, oof)),
        "mae_years": float(mean_absolute_error(age, oof)),
        "pearson_r": float(np.corrcoef(np.asarray(age, dtype=float), oof)[0, 1]),
    }


def predict_sex(features: pd.DataFrame, sex: np.ndarray) -> dict[str, float]:
    """Out-of-fold sex prediction (L2 logistic regression), ROC-AUC."""
    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    proba_oof = np.asarray(
        cross_val_predict(model, features, sex, cv=cv, method="predict_proba"),
        dtype=float,
    )
    return {"roc_auc": float(roc_auc_score(sex, proba_oof[:, 1]))}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]MRI feature-extraction positive control (age / sex)")

    y, X, conf = load_pipeline_data(
        target="bahmer", clean_controls=True, cohort="mri", stack_variant="full"
    )
    n = len(y)

    age = conf["basis_age"].astype(float).to_numpy()
    # basis_sex: 1 = male, 2 = female -> female indicator (positive class).
    sex = (conf["basis_sex"].astype(float).to_numpy() == 2).astype(int)

    console.log(f"Analytic sample N = {n}, female share = {sex.mean():.3f}")

    feature_frames = {m: _feature_frame(X[m]) for m in MRI_MODALITIES}
    feature_frames["combined"] = pd.concat(
        [feature_frames[m] for m in MRI_MODALITIES], axis=1
    )

    results: dict[str, PosControlResult] = {}
    for key, feats in feature_frames.items():
        console.log(f"[{key}] {feats.shape[1]} features ...")
        age_res = predict_age(feats, age)
        sex_res = predict_sex(feats, sex)
        results[key] = {
            "n_features": int(feats.shape[1]),
            "age": age_res,
            "sex": sex_res,
        }

    summary = {
        "sample": {
            "cohort": "mri",
            "clean_controls": True,
            "n": int(n),
            "female_share": float(sex.mean()),
            "age_mean": float(age.mean()),
            "age_sd": float(age.std(ddof=1)),
        },
        "method": (
            f"{N_FOLDS}-fold out-of-fold prediction of baseline age "
            "(RidgeCV regression; R^2, MAE) and sex (L2 logistic regression, "
            "balanced; ROC-AUC) from the same processed volumetric MRI "
            "features used by the main pipeline, before DML orthogonalisation."
        ),
        "results": results,
    }

    constants_dir = get_paper_constants_dir()
    json_path = constants_dir / "mri_positive_control.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # --- LaTeX constants -------------------------------------------------
    tex_lines = [
        "% Generated by scripts/supplementary/mri_positive_control.py — do not edit by hand.",
        "% MRI feature-extraction positive control: age R^2 and sex ROC-AUC",
        "% from the same volumetric features used by the main pipeline,",
        f"% on the clean-controls MRI analytic sample (N = {n}).",
    ]
    for key, res in results.items():
        suf = TEX_SUFFIX[key]
        tex_lines.append(
            f"\\newcommand{{\\resMriPosAgeRtwo{suf}}}{{{res['age']['r2']:.2f}}}"
        )
        tex_lines.append(
            f"\\newcommand{{\\resMriPosAgeMae{suf}}}{{{res['age']['mae_years']:.1f}}}"
        )
        tex_lines.append(
            f"\\newcommand{{\\resMriPosSexRoc{suf}}}{{{res['sex']['roc_auc']:.3f}}}"
        )
    tex_path = constants_dir / "mri_positive_control_tex.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")

    # --- Pretty-print ----------------------------------------------------
    tbl = Table(title=f"MRI positive control (N = {n})", show_header=True)
    tbl.add_column("Feature set", style="bold")
    tbl.add_column("p", justify="right")
    tbl.add_column("Age R²", justify="right")
    tbl.add_column("Age MAE (y)", justify="right")
    tbl.add_column("Sex ROC-AUC", justify="right")
    for key, res in results.items():
        tbl.add_row(
            MRI_LABELS.get(key, "All atlases combined"),
            f"{res['n_features']}",
            f"{res['age']['r2']:.2f}",
            f"{res['age']['mae_years']:.1f}",
            f"{res['sex']['roc_auc']:.3f}",
        )
    console.print(tbl)
    console.print(f"[green]Wrote {json_path}")
    console.print(f"[green]Wrote {tex_path}")


if __name__ == "__main__":
    main()
