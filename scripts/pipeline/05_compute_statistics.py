"""Compute descriptive statistics and generate LaTeX tables.

Generates:
  - Table 1: Baseline characteristics by PCC status
  - Table 2: Data availability by modality
  - Table 3: MRI selection bias comparison
  - Outcome distribution summary (JSON)

Usage:
    uv run python scripts/pipeline/05_compute_statistics.py
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from pcc_analysis.config import (
    get_paper_constants_dir,
    get_paper_tables_dir,
    get_processed_dir,
)
from pcc_analysis.data_manager import CONFOUNDER_COLUMNS, FULL_STACK_JOINS

# ── Configuration ──────────────────────────────────────────────────────────

PCC_COLUMN = "bahmer_any_pcs"
RANDOM_STATE = 42
ID_COLUMN = "ID"

# Table-2 row label -> the modality key ``load_pipeline_data`` uses. Needed
# because the joins that widen a modality beyond its own parquet are declared
# against the pipeline keys (``FULL_STACK_JOINS``), not against these labels.
MODALITY_PIPELINE_KEYS: dict[str, str] = {
    "Demographics": "demographics",
    "Socioeconomic": "ses",
    "Cognitive": "cognitive",
    "Physical Activity": "physical_activity",
    "Medical History": "medical_history",
    "Laboratory": "lab_values",
    "Cardiovascular": "cardiovascular",
    "Lung Function": "lung_function",
    "Mental Health": "mental_health",
}

# Feature counts are not listed here. ``generate_table2`` counts the real
# columns; the fallbacks this dict used to carry were stale (MRI 314 against
# an actual 599, lung function 7 against 9) and surfaced only when a parquet
# was missing, i.e. exactly when the table had nothing to describe. A count
# nobody recomputes is a table cell that quietly stops describing the model.
MODALITIES: dict[str, dict[str, str | list[str]]] = {
    "Demographics": {
        "file": "demographics.parquet",
        # demographics.parquet also carries basis_lvl and basis_status_mrt,
        # which describe the sample rather than predict with it. This is the
        # whitelist load_pipeline_data selects, imported rather than repeated.
        "features": list(CONFOUNDER_COLUMNS),
    },
    "Socioeconomic": {
        "file": "socioeconomic_status.parquet",
        "exclude": ["ID"],
    },
    "Cognitive": {
        "file": "cognitive_tests.parquet",
        "exclude": ["ID"],
    },
    "Physical Activity": {
        "file": "physical_activity.parquet",
        "exclude": ["ID"],
    },
    "Medical History": {
        "file": "medical_history.parquet",
        "exclude": ["ID"],
    },
    "Laboratory": {"file": "lab_values.parquet", "exclude": ["ID"]},
    "Cardiovascular": {
        "file": "cardiovascular.parquet",
        "exclude": ["ID"],
    },
    "Lung Function": {
        "file": "lung_function.parquet",
        "exclude": ["ID"],
    },
    "Mental Health": {
        "file": "mental_health.parquet",
        "exclude": ["ID"],
    },
    "MRI": {
        "files": [
            "mri_subcortical.parquet",
            "mri_cortical_desikan_killiany.parquet",
            "mri_cortical_destrieux.parquet",
            "mri_cortical_julich.parquet",
            "mri_cortical_yeo_networks.parquet",
            "mri_cerebellar.parquet",
        ],
    },
}

MRI_SUBMODALITIES: dict[str, str] = {
    "Desikan-Killiany": "mri_cortical_desikan_killiany.parquet",
    "Destrieux": "mri_cortical_destrieux.parquet",
    "Julich": "mri_cortical_julich.parquet",
    "Yeo Networks": "mri_cortical_yeo_networks.parquet",
    "Subcortical": "mri_subcortical.parquet",
    "Cerebellar": "mri_cerebellar.parquet",
}

TABLE1_VARIABLES: dict[str, dict[str, str | int | dict[str, str]]] = {
    "Age, years": {"col": "basis_age", "type": "continuous"},
    "Female sex": {"col": "basis_sex", "type": "binary", "positive_value": 2},
    "BMI, kg/m²": {"col": "bmi_self_reported", "type": "continuous"},
    "BMI category": {
        "col": "bmi_category",
        "type": "categorical",
        "categories": {
            "underweight": "Underweight (<18.5)",
            "normal": "Normal (18.5--24.9)",
            "overweight": "Overweight (25--29.9)",
            "obese": r"Obese ($\geq$30)",
        },
    },
    "Education (ISCED level)": {"col": "education_isced_level", "type": "continuous"},
    "Currently employed": {
        "col": "employment_status",
        "type": "binary",
        "positive_value": 1,
    },
    "Hypertension": {"col": "has_hypertension", "type": "binary"},
    "Cancer history": {"col": "has_cancer_history", "type": "binary"},
    r"Polypharmacy ($\geq$5 medications)": {"col": "polypharmacy", "type": "binary"},
    "Had COVID-19": {"col": "had_covid", "type": "binary"},
    "Number of symptoms": {"col": "n_symptoms", "type": "continuous"},
    "PHQ-9 score (0-27)": {"col": "phq9_total", "type": "continuous"},
    "GAD-7 score (0-21)": {"col": "gad7_total", "type": "continuous"},
    "PHQ-9 baseline (0-27)": {"col": "phq9_sum", "type": "continuous"},
    "GAD-7 baseline (0-21)": {"col": "gad7_sum", "type": "continuous"},
    "MINI major depression (baseline)": {
        "col": "mini_major_depression",
        "type": "binary",
    },
}


# ── Data loading ───────────────────────────────────────────────────────────


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    if "bmi_category" in df.columns:
        df["bmi_category"] = (
            df["bmi_category"]
            .astype(str)
            .replace({"obese_1": "obese", "obese_2": "obese", "obese_3": "obese"})
        )
    return df


def load_all_data(data_dir: Path) -> pd.DataFrame:
    print("Loading data...")

    df = pd.read_parquet(data_dir / "demographics.parquet")
    print(f"  Demographics: {len(df):,} rows")

    sex_age_path = data_dir / "baseline_sex_age.parquet"
    if sex_age_path.exists():
        sex_age = pd.read_parquet(sex_age_path)
        df = df.merge(sex_age, on="ID", how="left")

    files_to_merge = [
        "corona2_pcc.parquet",
        "socioeconomic_status.parquet",
        "medical_history.parquet",
        "cognitive_tests.parquet",
        "physical_activity.parquet",
        "lab_values.parquet",
        "cardiovascular.parquet",
        "lung_function.parquet",
        "mental_health.parquet",
        "psychometric_scores.parquet",
    ]

    for fname in files_to_merge:
        fpath = data_dir / fname
        if fpath.exists():
            temp = pd.read_parquet(fpath)
            cols_to_add = [c for c in temp.columns if c not in df.columns or c == "ID"]
            df = df.merge(temp[cols_to_add], on="ID", how="left")
            print(f"  {fname}: merged")
        else:
            print(f"  {fname}: NOT FOUND")

    # MRI indicator: present in any atlas parquet, which is the definition
    # data_manager._mri_participant_ids uses to select the cohort. An earlier
    # version additionally required a non-null feature value, which excluded
    # two participants who carry an all-null row in the Desikan-Killiany and
    # Yeo files. Those two are in the analytic sample the model was fitted on
    # (they drop out later, at the per-modality readiness intersection), so
    # counting them out here made the descriptive tables describe a cohort of
    # 19240 while the pipeline trained on one of 19242 — and the manuscript's
    # participant flow, which subtracts one from the other, stopped adding up.
    mri_files = MODALITIES["MRI"]["files"]
    mri_ids: set[object] = set()
    if isinstance(mri_files, list):
        for mri_file in mri_files:
            fpath = data_dir / str(mri_file)
            if fpath.exists():
                mdf = pd.read_parquet(fpath, columns=["ID"])
                mri_ids.update(mdf["ID"].values)
    df["has_mri"] = df["ID"].isin(mri_ids)
    print(f"  MRI data: {len(mri_ids):,} participants in an atlas parquet")

    print(f"\nTotal merged: {len(df):,} rows, {len(df.columns)} columns")
    return preprocess_data(df)


# ── Statistical functions ──────────────────────────────────────────────────


def compute_continuous_stats(series: pd.Series) -> dict[str, Any]:
    valid = series.dropna()
    if len(valid) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "missing": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100)
            if len(series) > 0
            else 0.0,
        }
    return {
        "n": len(valid),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "median": float(valid.median()),
        "q25": float(valid.quantile(0.25)),
        "q75": float(valid.quantile(0.75)),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "missing": int(series.isna().sum()),
        "missing_pct": float(series.isna().mean() * 100),
    }


def compute_binary_stats(series: pd.Series, positive_value: Any = 1) -> dict[str, Any]:
    valid = series.dropna()
    n_positive = int((valid == positive_value).sum())
    return {
        "n": len(valid),
        "n_positive": n_positive,
        "pct_positive": n_positive / len(valid) * 100 if len(valid) > 0 else 0,
        "missing": int(series.isna().sum()),
        "missing_pct": float(series.isna().mean() * 100),
    }


def compute_categorical_stats(
    series: pd.Series, categories: dict[str, str]
) -> dict[str, Any]:
    valid = series.dropna()
    counts = valid.value_counts()
    result: dict[str, Any] = {
        "n": len(valid),
        "missing": int(series.isna().sum()),
        "categories": {},
    }
    for val, label in categories.items():
        n = int(counts.get(val, 0))
        result["categories"][label] = {
            "n": n,
            "pct": n / len(valid) * 100 if len(valid) > 0 else 0,
        }
    return result


def compare_groups_continuous(g1: pd.Series, g2: pd.Series) -> dict[str, Any]:
    v1, v2 = g1.dropna(), g2.dropna()
    t_stat, t_pval = stats.ttest_ind(v1, v2, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(v1, v2, alternative="two-sided")
    pooled_std = np.sqrt(
        ((len(v1) - 1) * v1.std() ** 2 + (len(v2) - 1) * v2.std() ** 2)
        / (len(v1) + len(v2) - 2)
    )
    cohens_d = float((v1.mean() - v2.mean()) / pooled_std) if pooled_std > 0 else 0.0
    return {
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pval),
        "u_statistic": float(u_stat),
        "u_pvalue": float(u_pval),
        "cohens_d": cohens_d,
    }


def compare_groups_categorical(g1: pd.Series, g2: pd.Series) -> dict[str, Any]:
    v1, v2 = g1.dropna(), g2.dropna()
    all_cats = set(v1.unique()) | set(v2.unique())
    observed = np.array([[sum(v1 == c), sum(v2 == c)] for c in all_cats])
    if observed.min() >= 5:
        chi2, pval, dof, _ = stats.chi2_contingency(observed)
    elif observed.shape == (2, 2):
        _, pval = stats.fisher_exact(observed)
        chi2, dof = float("nan"), float("nan")
    else:
        chi2, pval, dof, _ = stats.chi2_contingency(observed)
    return {
        "chi2_statistic": float(chi2),
        "chi2_pvalue": float(pval),
        "dof": float(dof),
    }


def format_pvalue(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


# ── Table generators ───────────────────────────────────────────────────────


def generate_table1(df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    print("\nGenerating Table 1: Baseline Characteristics...")
    pcc_neg = df[df[PCC_COLUMN] == 0]
    pcc_pos = df[df[PCC_COLUMN] == 1]
    print(f"  PCC- (n={len(pcc_neg):,}), PCC+ (n={len(pcc_pos):,})")

    results: dict[str, Any] = {}
    latex_rows: list[str] = []

    for var_name, var_config in TABLE1_VARIABLES.items():
        col = str(var_config["col"])
        var_type = str(var_config["type"])
        if col not in df.columns:
            print(f"  WARNING: Column '{col}' not found, skipping {var_name}")
            continue

        if var_type == "continuous":
            sn = compute_continuous_stats(pcc_neg[col])
            sp = compute_continuous_stats(pcc_pos[col])
            cmp = compare_groups_continuous(pcc_neg[col], pcc_pos[col])
            neg_str = f"{sn['mean']:.1f} ({sn['std']:.1f})"
            pos_str = f"{sp['mean']:.1f} ({sp['std']:.1f})"
            latex_rows.append(
                f"  {var_name} & {neg_str} & {pos_str} & {format_pvalue(cmp['t_pvalue'])} \\\\"
            )
            results[var_name] = {
                "type": "continuous",
                "pcc_neg": sn,
                "pcc_pos": sp,
                "comparison": cmp,
            }

        elif var_type == "binary":
            pv = var_config.get("positive_value", 1)
            sn = compute_binary_stats(pcc_neg[col], pv)
            sp = compute_binary_stats(pcc_pos[col], pv)
            cmp = compare_groups_categorical(pcc_neg[col], pcc_pos[col])
            neg_str = f"{sn['n_positive']:,} ({sn['pct_positive']:.1f}\\%)"
            pos_str = f"{sp['n_positive']:,} ({sp['pct_positive']:.1f}\\%)"
            latex_rows.append(
                f"  {var_name}, n (\\%) & {neg_str} & {pos_str} & {format_pvalue(cmp['chi2_pvalue'])} \\\\"
            )
            results[var_name] = {
                "type": "binary",
                "pcc_neg": sn,
                "pcc_pos": sp,
                "comparison": cmp,
            }

        elif var_type == "categorical":
            categories_raw = var_config["categories"]
            if not isinstance(categories_raw, dict):
                continue
            categories: dict[str, str] = {
                str(k): str(v) for k, v in categories_raw.items()
            }
            sn = compute_categorical_stats(pcc_neg[col], categories)
            sp = compute_categorical_stats(pcc_pos[col], categories)
            cmp = compare_groups_categorical(pcc_neg[col], pcc_pos[col])
            latex_rows.append(
                f"  {var_name}, n (\\%) & & & {format_pvalue(cmp['chi2_pvalue'])} \\\\"
            )
            for cat_label in categories.values():
                nc = sn["categories"].get(cat_label, {"n": 0, "pct": 0})
                pc = sp["categories"].get(cat_label, {"n": 0, "pct": 0})
                latex_rows.append(
                    f"  \\quad {cat_label} & {nc['n']:,} ({nc['pct']:.1f}\\%) & {pc['n']:,} ({pc['pct']:.1f}\\%) & \\\\"
                )
            results[var_name] = {
                "type": "categorical",
                "pcc_neg": sn,
                "pcc_pos": sp,
                "comparison": cmp,
            }

    latex = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{Baseline characteristics by PCC status}\n"
        "\\label{tab:baseline_characteristics}\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        f"\\textbf{{Variable}} & \\textbf{{PCC$-$ (n={len(pcc_neg):,})}} & \\textbf{{PCC$+$ (n={len(pcc_pos):,})}} & \\textbf{{p-value}} \\\\\n"
        "\\midrule\n" + "\n".join(latex_rows) + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n\\small\n"
        "\\item Continuous variables presented as mean (SD); categorical variables as n (\\%).\n"
        "\\item P-values from Welch's t-test (continuous) or Chi-square test (categorical).\n"
        "\\item PCC defined as Bahmer weighted post-COVID syndrome score $>10.75$ \\citep{bahmer_2022} among infected participants.\n"
        "\\end{tablenotes}\n\\end{table}\n"
    )
    return latex, results


def _read_feature_frame(
    path: Path, *, features: list[str] | None, exclude: list[str]
) -> pd.DataFrame:
    """Feature columns of a processed parquet, indexed by participant ID.

    ``features`` is a whitelist where the loader selects one (demographics);
    otherwise every column outside ``exclude`` counts as a feature.
    """
    frame = pd.read_parquet(path)
    if ID_COLUMN in frame.columns:
        frame = frame.set_index(ID_COLUMN)
    if features is not None:
        keep = [c for c in features if c in frame.columns]
    else:
        keep = [c for c in frame.columns if c not in exclude]
    return frame[keep]


def _model_input_frame(
    data_dir: Path, modality_name: str, modality_config: dict[str, Any]
) -> pd.DataFrame | None:
    """The modality frame the Full Stack reads, or *None* if unavailable.

    This is the frame ``load_pipeline_data`` assembles, minus the cohort and
    outcome filters: the modality's own parquet, restricted to the columns the
    loader selects, plus whatever ``FULL_STACK_JOINS`` left-joins onto it.
    Counting the parquet alone would describe an artifact the model never
    sees — ``mental_health`` reaches the base learners eleven columns wider
    than its own file, ``medical_history`` six wider.

    Two differences from the loader are deliberate, because this table
    describes the delivered data rather than one analytic sample: it counts
    over all participants, not the cohort, so a column that happens to be
    empty inside the MRI arm still counts here; and it counts raw input
    variables, before the per-modality preprocessor drops and one-hot
    expansions, which is the convention the manuscript states.
    """
    fname = modality_config.get("file")
    if not isinstance(fname, str):
        return None
    fpath = data_dir / fname
    if not fpath.exists():
        return None

    features_raw = modality_config.get("features")
    features = features_raw if isinstance(features_raw, list) else None
    exclude_raw = modality_config.get("exclude", [ID_COLUMN])
    exclude = exclude_raw if isinstance(exclude_raw, list) else [ID_COLUMN]
    frame = _read_feature_frame(fpath, features=features, exclude=exclude)

    join_stem = FULL_STACK_JOINS.get(MODALITY_PIPELINE_KEYS.get(modality_name, ""))
    if join_stem is not None:
        join_path = data_dir / f"{join_stem}.parquet"
        if join_path.exists():
            extra = _read_feature_frame(join_path, features=None, exclude=[ID_COLUMN])
            frame = frame.join(extra, how="left")
        else:
            print(
                f"  WARNING: {join_path.name} not found; {modality_name} "
                "counted without the block joined into it at load time"
            )
    return frame


def generate_table2(df: pd.DataFrame, data_dir: Path) -> tuple[str, dict[str, Any]]:
    print("\nGenerating Table 2: Data Availability by Modality...")
    results: dict[str, Any] = {}
    latex_rows: list[str] = []

    for modality_name, modality_config in MODALITIES.items():
        # Always counted from the data. ``None`` means the parquet is not
        # there, and the row says so instead of printing a plausible number.
        n_features: int | None = None

        if modality_name == "MRI":
            mri_files = modality_config.get("files")
            if isinstance(mri_files, list) and all(
                (data_dir / fname).exists() for fname in mri_files
            ):
                # Footer schema only: the six atlases hold ~30,000 rows each,
                # and nothing here needs a single value out of them. All six
                # or none, because a partial sum would look like a full one.
                n_features = sum(
                    len(
                        [
                            c
                            for c in pq.read_schema(data_dir / fname).names
                            if c != ID_COLUMN
                        ]
                    )
                    for fname in mri_files
                )
            n_complete = int(df["has_mri"].sum()) if "has_mri" in df.columns else 0
            missing_pct = (
                float((1 - df["has_mri"].mean()) * 100)
                if "has_mri" in df.columns
                else 100.0
            )
            pattern = "Systematic (Level 2)"
        else:
            frame = _model_input_frame(data_dir, modality_name, modality_config)
            if frame is not None:
                n_features = frame.shape[1]
                has_any = frame.notna().any(axis=1)
                n_complete = int(has_any.sum())
                missing_pct = (1 - has_any.mean()) * 100
                pattern = (
                    "Complete"
                    if missing_pct < 5
                    else ("Sporadic" if missing_pct < 20 else "Systematic")
                )
            elif isinstance(modality_config.get("file"), str):
                n_complete, missing_pct, pattern = 0, 100.0, "Not available"
            else:
                n_complete, missing_pct, pattern = len(df), 0.0, "Complete"

        features_cell = "n/a" if n_features is None else str(n_features)
        latex_rows.append(
            f"  {modality_name} & {features_cell} & {n_complete:,} & {missing_pct:.1f}\\% & {pattern} \\\\"
        )
        results[modality_name] = {
            "n_features": n_features,
            "n_complete": n_complete,
            "missing_pct": missing_pct,
            "pattern": pattern,
        }

    latex = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{Data availability by modality}\n"
        "\\label{tab:modality_availability}\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "\\textbf{Modality} & \\textbf{Features} & \\textbf{Complete Cases} & \\textbf{Missing} & \\textbf{Pattern} \\\\\n"
        "\\midrule\n" + "\n".join(latex_rows) + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n\\small\n"
        "\\item Complete cases defined as having at least one non-missing feature value.\n"
        "\\item Features counts raw input variables prior to preprocessing, including the "
        "blocks that arrive in a separate NAKO delivery and are joined onto a host modality "
        "at load time: the item-level baseline PHQ-9 responses into Mental Health, the "
        "baseline tobacco block into Medical History.\n"
        "\\item Systematic (Level 2): MRI available only in 20\\% subsample by design.\n"
        "\\end{tablenotes}\n\\end{table}\n"
    )
    return latex, results


def generate_table3_mri_selection(df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    print("\nGenerating Table 3: MRI Selection Bias...")
    if "has_mri" not in df.columns:
        print("  WARNING: has_mri column not found")
        return "", {}

    mri_yes = df[df["has_mri"]]
    mri_no = df[~df["has_mri"]]
    print(f"  MRI+ (n={len(mri_yes):,}), MRI- (n={len(mri_no):,})")

    if len(mri_no) == 0 or len(mri_yes) == 0:
        print("  WARNING: one group is empty, skipping comparison")
        return "", {}

    compare_vars: dict[str, dict[str, str | int]] = {
        "Age, years": {"col": "basis_age", "type": "continuous"},
        "Female sex, n (\\%)": {
            "col": "basis_sex",
            "type": "binary",
            "positive_value": 2,
        },
        "BMI, kg/m²": {"col": "bmi_self_reported", "type": "continuous"},
        "Education (ISCED)": {"col": "education_isced_level", "type": "continuous"},
        "PHQ-9 score (0-27)": {"col": "phq9_total", "type": "continuous"},
        "PCC prevalence, n (\\%)": {"col": PCC_COLUMN, "type": "binary"},
    }

    results: dict[str, Any] = {}
    latex_rows: list[str] = []

    for var_name, vc in compare_vars.items():
        col = str(vc["col"])
        if col not in df.columns:
            continue
        if vc["type"] == "continuous":
            sn = compute_continuous_stats(mri_no[col])
            sy = compute_continuous_stats(mri_yes[col])
            cmp = compare_groups_continuous(mri_no[col], mri_yes[col])
            latex_rows.append(
                f"  {var_name} & {sn['mean']:.1f} ({sn['std']:.1f}) & {sy['mean']:.1f} ({sy['std']:.1f}) & {format_pvalue(cmp['t_pvalue'])} \\\\"
            )
        else:
            pv = vc.get("positive_value", 1)
            sn = compute_binary_stats(mri_no[col], pv)
            sy = compute_binary_stats(mri_yes[col], pv)
            cmp = compare_groups_categorical(mri_no[col], mri_yes[col])
            latex_rows.append(
                f"  {var_name} & {sn['n_positive']:,} ({sn['pct_positive']:.1f}\\%) & {sy['n_positive']:,} ({sy['pct_positive']:.1f}\\%) & {format_pvalue(cmp['chi2_pvalue'])} \\\\"
            )
        results[var_name] = {"mri_no": sn, "mri_yes": sy, "comparison": cmp}

    latex = (
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption{Comparison of participants with and without MRI data}\n"
        "\\label{tab:mri_selection}\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        f"\\textbf{{Variable}} & \\textbf{{No MRI (n={len(mri_no):,})}} & \\textbf{{MRI (n={len(mri_yes):,})}} & \\textbf{{p-value}} \\\\\n"
        "\\midrule\n" + "\n".join(latex_rows) + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n\\small\n"
        "\\item MRI available in Level 2 subsample (~20\\% of participants).\n"
        "\\item P-values from Welch's t-test (continuous) or Chi-square test (categorical).\n"
        "\\end{tablenotes}\n\\end{table}\n"
    )
    return latex, results


def generate_table_s1_missingness(df: pd.DataFrame, data_dir: Path) -> dict[str, Any]:
    """Compute feature-level missingness per modality for the MRI subsample.

    Returns dict with per-modality n_features, cell_level_pct, complete_pct,
    all_missing_pct for Table S1 (supplement).

    Counts the same frames as :func:`generate_table2` — the modality parquet
    restricted to the columns the loader selects, plus the blocks
    ``FULL_STACK_JOINS`` attaches to it. This is the table the manuscript's
    supplement transcribes, so it is the one that must not describe an
    artifact the model never reads.
    """
    print("\nGenerating Table S1: Feature-Level Missingness (MRI subsample)...")
    mri_ids = pd.Index(df.loc[df["has_mri"], "ID"])
    n_mri = len(mri_ids)
    print(f"  MRI subsample: N={n_mri:,}")

    results: dict[str, Any] = {"n_mri_subsample": n_mri}

    def _missingness_stats(frame: pd.DataFrame, ids: pd.Index) -> dict[str, Any]:
        mod_df = frame.loc[frame.index.intersection(ids)]
        n_features = mod_df.shape[1]
        n_rows = len(mod_df)
        if n_rows == 0 or n_features == 0:
            return {
                "n_features": n_features,
                "cell_level_pct": 0.0,
                "complete_pct": 0.0,
                "all_missing_pct": 0.0,
            }
        na_matrix = mod_df.isna()
        na_count = int(na_matrix.sum().sum())
        cell_level_pct = na_count / (n_rows * n_features) * 100
        complete_pct = int((~na_matrix).all(axis=1).sum()) / n_rows * 100
        all_missing_pct = int(na_matrix.all(axis=1).sum()) / n_rows * 100
        return {
            "n_features": n_features,
            "cell_level_pct": round(cell_level_pct, 2),
            "complete_pct": round(complete_pct, 2),
            "all_missing_pct": round(all_missing_pct, 2),
        }

    # Non-MRI modalities
    for mod_name, mod_config in MODALITIES.items():
        if mod_name == "MRI":
            continue
        frame = _model_input_frame(data_dir, mod_name, mod_config)
        if frame is None:
            print(f"  WARNING: {mod_config.get('file')} not found, skipping {mod_name}")
            continue
        stats = _missingness_stats(frame, mri_ids)
        results[mod_name] = stats
        print(
            f"  {mod_name:20s}  features={stats['n_features']:3d}  "
            f"cell={stats['cell_level_pct']:5.2f}%  "
            f"complete={stats['complete_pct']:6.2f}%  "
            f"all_miss={stats['all_missing_pct']:5.2f}%"
        )

    # MRI sub-modalities. No entry in FULL_STACK_JOINS: eTIV is joined onto
    # each atlas at load time but ``FeatureToConfounder`` removes it from X
    # again and routes it to the orthogonaliser, so it is an adjustment
    # covariate rather than a feature and does not belong in a feature count.
    for sub_name, sub_file in MRI_SUBMODALITIES.items():
        fpath = data_dir / sub_file
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping MRI/{sub_name}")
            continue
        stats = _missingness_stats(
            _read_feature_frame(fpath, features=None, exclude=[ID_COLUMN]), mri_ids
        )
        key = f"MRI: {sub_name}"
        results[key] = stats
        print(
            f"  {key:20s}  features={stats['n_features']:3d}  "
            f"cell={stats['cell_level_pct']:5.2f}%  "
            f"complete={stats['complete_pct']:6.2f}%  "
            f"all_miss={stats['all_missing_pct']:5.2f}%"
        )

    return results


def compute_outcome_distribution(df: pd.DataFrame) -> dict[str, Any]:
    print("\nComputing outcome distribution...")
    result: dict[str, Any] = {"total_n": len(df)}

    if "had_covid" in df.columns:
        result["had_covid"] = {
            "n": int(df["had_covid"].sum()),
            "pct": float(df["had_covid"].mean() * 100),
        }
    for col_name, key in [
        ("bahmer_any_pcs", "bahmer_any_pcs"),
        ("bahmer_severe_pcs", "bahmer_severe_pcs"),
        ("any_pcc", "diexer_any_pcc"),
        ("severe_pcc", "diexer_severe_pcc"),
    ]:
        if col_name in df.columns:
            result[key] = {
                "n": int(df[col_name].sum()),
                "pct": float(df[col_name].mean() * 100),
            }
    if "n_symptoms" in df.columns:
        result["n_symptoms"] = compute_continuous_stats(df["n_symptoms"])
    if "bahmer_pcs_score" in df.columns:
        result["bahmer_pcs_score"] = compute_continuous_stats(df["bahmer_pcs_score"])

    return result


# ── Main ───────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate descriptive statistics tables for the PCC paper.",
    )
    parser.add_argument(
        "--cohort",
        choices=["mri", "non_mri", "all"],
        default="mri",
        help=(
            "Analytic cohort for Table 1 and Table S1.  'mri' (default) "
            "reproduces the primary analytic sample.  'non_mri' produces "
            "the non-MRI within-study replication tables.  'all' "
            "skips the MRI/non-MRI filter entirely."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("Descriptive Statistics for PCC Prediction Paper")
    print(f"Cohort: {args.cohort}")
    print("=" * 60)

    data_dir = get_processed_dir()
    tables_dir = get_paper_tables_dir()

    # Output-filename suffix so MRI and non-MRI tables can coexist in
    # ``results/tables/`` without overwriting each other.
    tag = "" if args.cohort == "mri" else f"_{args.cohort}"

    df = load_all_data(data_dir)
    all_results: dict[str, Any] = {}

    all_results["outcome_distribution"] = compute_outcome_distribution(df)

    # Table 1: clean-controls analytic sample for the selected cohort.
    # The primary analysis uses the MRI arm; the non-MRI within-study
    # replication requires the non-MRI arm for its parallel table.
    mri_mask = df["has_mri"]
    if args.cohort == "mri":
        cohort_mask = mri_mask
        cohort_label = "MRI + infected + valid symptoms - sub-threshold k0=1"
    elif args.cohort == "non_mri":
        cohort_mask = ~mri_mask
        cohort_label = "non-MRI + infected + valid symptoms - sub-threshold k0=1"
    else:  # all
        cohort_mask = pd.Series(True, index=df.index)
        cohort_label = "all + infected + valid symptoms - sub-threshold k0=1"
    valid_mask = df.get("valid_symptoms", pd.Series(True, index=df.index)) == 1
    infected_mask = df.get("had_covid", pd.Series(0, index=df.index)) == 1
    y_label = df.get(PCC_COLUMN)
    k0 = df.get("d_co2_k0")
    analytic_mask = cohort_mask & valid_mask & infected_mask
    if y_label is not None and k0 is not None:
        subthreshold_mask = (k0 == 1) & (y_label == 0)
        analytic_mask &= ~subthreshold_mask
    df_clean = df[analytic_mask].copy()
    print(
        f"\nClean-controls analytic sample for Table 1 "
        f"(cohort={args.cohort}): N={len(df_clean):,} ({cohort_label})"
    )
    t1_latex, t1_res = generate_table1(df_clean)
    all_results["table1_baseline"] = t1_res
    t1_path = tables_dir / f"table1_baseline_characteristics{tag}.tex"
    t1_path.write_text(t1_latex)
    print(f"  Saved: {t1_path}")

    t2_latex, t2_res = generate_table2(df, data_dir)
    all_results["table2_modality"] = t2_res
    (tables_dir / "table2_modality_availability.tex").write_text(t2_latex)
    print(f"  Saved: {tables_dir / 'table2_modality_availability.tex'}")

    # Table 3 (MRI selection bias) compares MRI vs. non-MRI — always
    # reports the same comparison regardless of which cohort Table 1
    # is conditioned on.  Only run once, when cohort='mri' (default) or
    # when explicitly asked via cohort='all'; skip in the non_mri run
    # so we don't produce redundant files.
    if args.cohort != "non_mri":
        t3_latex, t3_res = generate_table3_mri_selection(df)
        all_results["table3_mri_selection"] = t3_res
        if t3_latex:
            (tables_dir / "table3_mri_selection_bias.tex").write_text(t3_latex)
            print(f"  Saved: {tables_dir / 'table3_mri_selection_bias.tex'}")

    # Table S1 missingness: computed on the clean-controls analytic sample
    # (same restriction as Table 1) so the footer percentages match the
    # sample reported in the main paper.
    s1_res = generate_table_s1_missingness(df_clean, data_dir)
    all_results["table_s1_missingness"] = s1_res

    stats_path = get_paper_constants_dir() / f"descriptive_stats{tag}.json"
    with stats_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"  Saved: {stats_path}")

    print("\n" + "=" * 60)
    print("Done!")
    outcome = all_results["outcome_distribution"]
    print(f"  Total participants: {outcome['total_n']:,}")
    if "had_covid" in outcome:
        print(
            f"  Had COVID-19: {outcome['had_covid']['n']:,} ({outcome['had_covid']['pct']:.1f}%)"
        )
    if "bahmer_any_pcs" in outcome:
        print(
            f"  PCC (Bahmer PCS>10.75): {outcome['bahmer_any_pcs']['n']:,} ({outcome['bahmer_any_pcs']['pct']:.1f}%)"
        )


if __name__ == "__main__":
    main()
