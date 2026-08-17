"""Side-by-side Table 1: MRI vs. non-MRI analytic sample.

Imports the data-loading and per-variable stats helpers from
``pipeline/05_compute_statistics.py`` (same module-global schemas so Table 1
rows stay aligned) and produces a three-column LaTeX table plus a
JSON payload with the per-row statistics, standardized mean
differences (SMDs) and two-sample tests.

Output
------
- ``results/tables/table1_baseline_comparison.tex`` — LaTeX table
- ``results/tables/table1_baseline_comparison.json`` — row-by-row
  statistics for reproducibility + direct access from paper tooling.

Usage
-----
    uv run python scripts/supplementary/cohort_comparison.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

from pcc_analysis.config import get_paper_tables_dir, get_processed_dir

# ``05_compute_statistics.py`` starts with a digit, so it cannot be
# ``import``-ed the normal way.  Load it explicitly so we can reuse the
# variable schema and the per-variable stat helpers — keeping schemas
# in one place avoids silent drift between the single-cohort Table 1
# and this comparison table.
_STATS_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "pipeline" / "05_compute_statistics.py"
)


def _load_stats_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "pcc_statistics_module", _STATS_MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PCC_COLUMN = "bahmer_any_pcs"


def _build_analytic_mask(df: pd.DataFrame, cohort: str) -> pd.Series:
    """Reproduce the clean-controls analytic sample filter per cohort."""
    mri_mask = df["has_mri"]
    if cohort == "mri":
        cohort_mask = mri_mask
    elif cohort == "non_mri":
        cohort_mask = ~mri_mask
    else:
        raise ValueError(f"Unsupported cohort: {cohort!r}")
    valid_mask = df.get("valid_symptoms", pd.Series(True, index=df.index)) == 1
    infected_mask = df.get("had_covid", pd.Series(0, index=df.index)) == 1
    y_label = df.get(_PCC_COLUMN)
    k0 = df.get("d_co2_k0")
    analytic = cohort_mask & valid_mask & infected_mask
    if y_label is not None and k0 is not None:
        subthreshold = (k0 == 1) & (y_label == 0)
        analytic &= ~subthreshold
    return cast("pd.Series", analytic)


def _standardized_mean_difference_continuous(
    x_mri: pd.Series, x_non: pd.Series
) -> float:
    """Cohen's d with pooled SD.  Returns NaN if either side empty."""
    a = x_mri.dropna().to_numpy(dtype=float)
    b = x_non.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def _standardized_mean_difference_binary(p_mri: float, p_non: float) -> float:
    """Standardised proportion difference (Cohen's h-style pooled SD)."""
    if not (np.isfinite(p_mri) and np.isfinite(p_non)):
        return float("nan")
    pooled = np.sqrt((p_mri * (1 - p_mri) + p_non * (1 - p_non)) / 2)
    if pooled == 0:
        return float("nan")
    return float((p_mri - p_non) / pooled)


def _welch_p(x_mri: pd.Series, x_non: pd.Series) -> float:
    a = x_mri.dropna().to_numpy(dtype=float)
    b = x_non.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    _t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(p)


def _chi2_p(s_mri: pd.Series, s_non: pd.Series, positive_value: Any = 1) -> float:
    a = s_mri.dropna()
    b = s_non.dropna()
    n1, n0 = int((a == positive_value).sum()), int(len(a) - (a == positive_value).sum())
    m1, m0 = int((b == positive_value).sum()), int(len(b) - (b == positive_value).sum())
    table = np.array([[n1, n0], [m1, m0]])
    if (
        table.sum() == 0
        or (table.sum(axis=1) == 0).any()
        or (table.sum(axis=0) == 0).any()
    ):
        return float("nan")
    try:
        _chi2, p, _dof, _expected = stats.chi2_contingency(table)
    except ValueError:
        return float("nan")
    return float(p)


def _chi2_categorical_p(
    s_mri: pd.Series, s_non: pd.Series, categories: dict[str, str]
) -> float:
    cats = list(categories.keys())
    counts_mri = [int((s_mri.dropna() == c).sum()) for c in cats]
    counts_non = [int((s_non.dropna() == c).sum()) for c in cats]
    table = np.array([counts_mri, counts_non])
    if (
        table.sum() == 0
        or (table.sum(axis=1) == 0).any()
        or (table.sum(axis=0) == 0).any()
    ):
        return float("nan")
    try:
        _chi2, p, _dof, _expected = stats.chi2_contingency(table)
    except ValueError:
        return float("nan")
    return float(p)


def _format_continuous(stats_dict: dict[str, Any]) -> str:
    if stats_dict["n"] == 0:
        return "--"
    return f"{stats_dict['mean']:.1f} ({stats_dict['std']:.1f})"


def _format_binary(stats_dict: dict[str, Any]) -> str:
    if stats_dict["n"] == 0:
        return "--"
    return f"{stats_dict['n_positive']:,} ({stats_dict['pct_positive']:.1f}\\%)"


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return r"$<0.001$"
    return f"{p:.3f}"


def _format_smd(smd: float) -> str:
    if not np.isfinite(smd):
        return "--"
    return f"{smd:+.2f}"


def compare_cohorts(
    df: pd.DataFrame, variables: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    stats_mod = _load_stats_module()

    mri_mask = _build_analytic_mask(df, "mri")
    non_mask = _build_analytic_mask(df, "non_mri")
    df_mri = df.loc[mri_mask].copy()
    df_non = df.loc[non_mask].copy()

    rows: list[dict[str, Any]] = []
    for label, spec in variables.items():
        col = cast("str", spec["col"])
        vtype = cast("str", spec["type"])
        if col not in df.columns:
            rows.append({"label": label, "type": vtype, "warning": "column not found"})
            continue
        s_mri = df_mri[col]
        s_non = df_non[col]

        entry: dict[str, Any] = {"label": label, "type": vtype, "column": col}
        if vtype == "continuous":
            entry["mri"] = stats_mod.compute_continuous_stats(s_mri)
            entry["non_mri"] = stats_mod.compute_continuous_stats(s_non)
            entry["smd"] = _standardized_mean_difference_continuous(s_mri, s_non)
            entry["p_value"] = _welch_p(s_mri, s_non)
        elif vtype == "binary":
            positive_value = spec.get("positive_value", 1)
            entry["mri"] = stats_mod.compute_binary_stats(s_mri, positive_value)
            entry["non_mri"] = stats_mod.compute_binary_stats(s_non, positive_value)
            p_mri = entry["mri"]["pct_positive"] / 100.0
            p_non = entry["non_mri"]["pct_positive"] / 100.0
            entry["smd"] = _standardized_mean_difference_binary(p_mri, p_non)
            entry["p_value"] = _chi2_p(s_mri, s_non, positive_value)
        elif vtype == "categorical":
            categories = cast("dict[str, str]", spec["categories"])
            entry["mri"] = stats_mod.compute_categorical_stats(s_mri, categories)
            entry["non_mri"] = stats_mod.compute_categorical_stats(s_non, categories)
            entry["smd"] = float(
                "nan"
            )  # no single SMD for multi-category; tableize per level
            entry["p_value"] = _chi2_categorical_p(s_mri, s_non, categories)
        else:
            raise ValueError(f"Unknown variable type: {vtype!r}")
        rows.append(entry)

    return {
        "n_mri": int(mri_mask.sum()),
        "n_non_mri": int(non_mask.sum()),
        "rows": rows,
    }


def render_latex(comparison: dict[str, Any]) -> str:
    n_mri = comparison["n_mri"]
    n_non = comparison["n_non_mri"]
    header = (
        r"\begin{table}[H]"
        "\n"
        r"\centering"
        "\n"
        r"\caption{Baseline characteristics of the MRI and non-MRI clean-controls analytic samples.  Continuous variables reported as mean (SD); binary variables as n (\%).  SMD = standardised mean difference (Cohen's $d$ for continuous, standardised proportion difference for binary).  $p$-values from Welch $t$-test (continuous) or $\chi^{2}$-test (binary / categorical); at the sample sizes shown, SMD is the primary effect-size summary.}"
        "\n"
        r"\label{tab:baseline_comparison}"
        "\n"
        r"\begin{tabular}{lrrrr}"
        "\n"
        r"\toprule"
        "\n"
        f"Variable & MRI (n={n_mri:,}) & non-MRI (n={n_non:,}) & SMD & $p$ \\\\"
        "\n"
        r"\midrule"
        "\n"
    )
    body_lines: list[str] = []
    for r in comparison["rows"]:
        label = r["label"]
        if r.get("warning"):
            body_lines.append(f"{label} & -- & -- & -- & -- \\\\")
            continue
        if r["type"] in ("continuous", "binary"):
            mri_s = (
                _format_continuous(r["mri"])
                if r["type"] == "continuous"
                else _format_binary(r["mri"])
            )
            non_s = (
                _format_continuous(r["non_mri"])
                if r["type"] == "continuous"
                else _format_binary(r["non_mri"])
            )
            body_lines.append(
                f"{label} & {mri_s} & {non_s} & {_format_smd(r['smd'])} & {_format_p(r['p_value'])} \\\\"
            )
        elif r["type"] == "categorical":
            body_lines.append(f"{label} & & & & {_format_p(r['p_value'])} \\\\")
            for cat_label, mri_cat in r["mri"]["categories"].items():
                non_cat = r["non_mri"]["categories"].get(
                    cat_label, {"n": 0, "pct": 0.0}
                )
                mri_cell = f"{mri_cat['n']:,} ({mri_cat['pct']:.1f}\\%)"
                non_cell = f"{non_cat['n']:,} ({non_cat['pct']:.1f}\\%)"
                body_lines.append(
                    f"\\quad {cat_label} & {mri_cell} & {non_cell} & & \\\\"
                )
    footer = (
        r"\bottomrule"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\end{table}"
        "\n"
    )
    return header + "\n".join(body_lines) + "\n" + footer


def main() -> None:
    stats_mod = _load_stats_module()
    data_dir = get_processed_dir()
    tables_dir = get_paper_tables_dir()

    df = stats_mod.load_all_data(data_dir)
    comparison = compare_cohorts(df, stats_mod.TABLE1_VARIABLES)

    json_path = tables_dir / "table1_baseline_comparison.json"
    with json_path.open("w") as f:
        json.dump(comparison, f, indent=2, default=float)
    print(f"Saved: {json_path}")

    tex_path = tables_dir / "table1_baseline_comparison.tex"
    tex_path.write_text(render_latex(comparison))
    print(f"Saved: {tex_path}")

    print(
        f"\nCohorts: MRI n={comparison['n_mri']:,}, "
        f"non-MRI n={comparison['n_non_mri']:,}"
    )


if __name__ == "__main__":
    main()
