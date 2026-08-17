"""Generate synthetic smoke-test parquet files from a JSON profile."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.typing import NAType

from smoke_test._profile_types import (
    COLUMN_DTYPES,
    BooleanColumnProfile,
    ColumnProfile,
    DerivationGroupProfile,
    IntegerColumnProfile,
    ModalityProfile,
    Profile,
)

# Bahmer (2022) symptom-complex → (symptom list, weight).  Kept local to
# the smoke-test module (mirror of ``data_processing._BAHMER_MAPPING``)
# so the smoke test produces Bahmer scores that match the real pipeline
# rather than an ad-hoc uniform placeholder.
_BAHMER_MAPPING: dict[str, tuple[list[str], float]] = {
    "fatigue": (["symptom_fatigue"], 7.0),
    "cough_wheeze": (["symptom_cough"], 7.0),
    "neurological": (
        [
            "symptom_memory_problems",
            "symptom_concentration_problems",
            "symptom_headache",
            "symptom_nerve_problems",
        ],
        6.5,
    ),
    "joint_muscle_pain": (["symptom_joint_muscle_pain"], 6.5),
    "ent_ailments": (["symptom_runny_nose"], 5.5),
    "gastrointestinal": (
        ["symptom_gastrointestinal_problems", "symptom_loss_of_appetite"],
        5.0,
    ),
    "sleep_disturbance": (["symptom_sleep_problems"], 5.0),
    "exercise_intolerance": (
        ["symptom_breathing_problems", "symptom_reduced_physical_capacity"],
        4.0,
    ),
    "infection_signs": (["symptom_fever", "symptom_sweating"], 3.5),
    "chemosensory_deficits": (
        ["symptom_loss_of_smell", "symptom_loss_of_taste"],
        3.5,
    ),
    "chest_pain": (["symptom_chest_tightness"], 3.5),
    "dermatological": (["symptom_hair_loss"], 2.0),
}

# Logistic clustering strength.  symptom_i ~ Bernoulli(sigmoid(logit(p_i) +
# LIABILITY_ALPHA * (liability − mean_liability))).  Increase for stronger
# co-endorsement, decrease for closer-to-independent sampling.  The value
# is tuned so the downstream Bahmer-score prevalence lands close to the
# real-data target (~26 %) without running away from the marginal
# symptom prevalences we profile in.
LIABILITY_ALPHA: float = 3.0


def _to_int8(arr: np.ndarray) -> pd.arrays.IntegerArray:
    """Wrap an ndarray as nullable Int8 (works around pandas-stubs limitation)."""
    return pd.array(arr.astype(np.int8), dtype="Int8")


def _ids(n: int) -> np.ndarray:
    """Sequential participant IDs starting at 100_000."""
    return np.arange(100_000, 100_000 + n, dtype=np.uint32)


def _generate_column(
    col_name: str,
    col_profile: ColumnProfile,
    n: int,
    rng: np.random.Generator,
) -> pd.Series:
    """Dispatch to the right generator based on ``dtype`` in the profile.

    Each branch tests ``col_profile["dtype"]`` directly rather than a local
    copy of it: that is what narrows :data:`ColumnProfile` to the one member
    carrying the fields the branch reads.
    """
    if col_profile["dtype"] == "numeric":
        std = max(col_profile["std"], 1e-9)
        values = rng.normal(col_profile["mean"], std, size=n).clip(
            col_profile["q01"], col_profile["q99"]
        )
        return pd.Series(values, name=col_name)

    if col_profile["dtype"] == "integer":
        std = max(col_profile["std"], 1e-9)
        drawn = rng.normal(col_profile["mean"], std, size=n).clip(
            col_profile["q01"], col_profile["q99"]
        )
        values = np.rint(drawn).astype(np.int64)
        return pd.Series(values, name=col_name, dtype=col_profile["pandas_dtype"])

    if col_profile["dtype"] == "boolean":
        values = (rng.random(n) < col_profile["true_frac"]).astype(np.int8)
        return pd.Series(pd.array(values, dtype="Int8"), name=col_name)

    if col_profile["dtype"] == "categorical":
        categories = col_profile["categories"]
        labels = list(categories.keys())
        probs = np.array(list(categories.values()), dtype=np.float64)
        probs = probs / probs.sum()  # re-normalise
        values = rng.choice(labels, size=n, p=probs)
        return pd.Series(pd.Categorical(values, categories=labels), name=col_name)

    if col_profile["dtype"] == "datetime":
        # Generate NaT — the analysis pipeline doesn't use date values
        return pd.Series(pd.array([pd.NaT] * n), name=col_name, dtype="string")

    if col_profile["dtype"] == "string":
        return pd.Series([pd.NA] * n, name=col_name, dtype="string")

    # Fallback: NaN
    return pd.Series(np.full(n, np.nan), name=col_name)


def _inject_missingness(
    df: pd.DataFrame,
    rng: np.random.Generator,
    col_profiles: dict[str, ColumnProfile],
    systematic_missing_frac: float = 0.0,
) -> None:
    """Set random cells to NaN/NA according to per-column ``missing_frac``.

    When *systematic_missing_frac* > 0, the profiled ``missing_frac`` values
    already include the systematic component (entire rows missing).  To avoid
    double-counting, we deconvolve the sporadic fraction:

        sporadic = (total - systematic) / (1 - systematic)

    This ensures that subjects who *do* have data see only the residual
    sporadic missingness, not the full rate.
    """
    for col, profile in col_profiles.items():
        if col not in df.columns or col == "ID":
            continue
        frac = profile.get("missing_frac", 0.0)
        if frac <= 0:
            continue
        if systematic_missing_frac > 0:
            frac = max(
                0.0, (frac - systematic_missing_frac) / (1 - systematic_missing_frac)
            )
        if frac <= 0:
            continue
        mask = rng.random(len(df)) < frac
        if mask.any():
            df.loc[mask, col] = _na_for_dtype(df[col])


def _na_for_dtype(series: pd.Series) -> float | NAType:
    """Return the appropriate NA sentinel for a series dtype."""
    dtype_str = str(series.dtype)
    if dtype_str in ("Int8", "Int64", "UInt8", "UInt16", "Float64"):
        return pd.NA
    if dtype_str == "category":
        return np.nan
    if dtype_str in ("string", "string[python]"):
        return pd.NA
    return np.nan


def _inject_systematic_missingness(
    df: pd.DataFrame,
    rng: np.random.Generator,
    frac: float,
) -> None:
    """Set entire rows to NaN (all features except ID)."""
    if frac <= 0:
        return
    n = len(df)
    n_missing = int(n * frac)
    if n_missing == 0:
        return
    indices = rng.choice(n, size=n_missing, replace=False)
    feat_cols = [c for c in df.columns if c != "ID"]
    df.loc[df.index[indices], feat_cols] = np.nan


def _inject_signal(
    df: pd.DataFrame,
    pcc: np.ndarray,
    rng: np.random.Generator,
    n_signal_features: int = 3,
    effect_size: float = 0.3,
) -> None:
    """Shift a few continuous features by PCC status (in-place).

    Adds ``effect_size * std * pcc`` to randomly chosen float columns,
    creating weak but learnable associations (Cohen's d ~ effect_size).
    This ensures SAGA converges quickly instead of hitting max_iter.

    Integer-typed columns are skipped so the quality harness' dtype
    invariants continue to hold. Skipping them is not enough for the
    plausibility ranges, though: a bounded float — a percentage, a
    proportion — leaves its range under any positive shift, and
    ``sleep_efficiency_pct`` really did come out above 100. The shifted
    column is therefore clipped back to the values it held before, which
    the generator had already bounded by the profiled 1st and 99th
    percentiles. The association survives; it is a shift *within* the
    range rather than out of it.
    """
    numeric_cols = [
        c
        for c in df.select_dtypes(include=np.number).columns
        if c != "ID" and pd.api.types.is_float_dtype(df[c])
    ]
    if not numeric_cols:
        return
    chosen = rng.choice(
        numeric_cols, size=min(n_signal_features, len(numeric_cols)), replace=False
    )
    for col in chosen:
        std = df[col].std()
        if std > 0 and not np.isnan(std):
            lo, hi = df[col].min(), df[col].max()
            df[col] = (df[col] + effect_size * std * pcc).clip(lo, hi)


def _generate_derivation_group(
    group_def: DerivationGroupProfile,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample mutually exclusive boolean dummies from a latent categorical."""
    columns = group_def["columns"]
    proportions = group_def["proportions"]
    probs = np.array([proportions[c] for c in columns], dtype=np.float64)
    probs = probs / probs.sum()

    latent = rng.choice(len(columns), size=n, p=probs)
    df = pd.DataFrame()
    for i, col in enumerate(columns):
        df[col] = pd.array((latent == i).astype(np.int8), dtype="Int8")
    return df


# ---------------------------------------------------------------------------
# Modality-specific generators
# ---------------------------------------------------------------------------


def _generate_generic(
    mod_profile: ModalityProfile,
    ids: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generic generator: one ``_generate_column`` call per profiled column."""
    n = len(ids)
    col_profiles: dict[str, ColumnProfile] = mod_profile["columns"]
    sys_frac = mod_profile.get("systematic_missing_frac", 0.0)

    parts: list[pd.Series] = [pd.Series(ids, name="ID")]
    for col_name, col_prof in col_profiles.items():
        if col_prof.get("derived") or col_prof.get("in_derivation_group"):
            continue
        parts.append(_generate_column(col_name, col_prof, n, rng))

    df = pd.concat(parts, axis=1)
    _inject_missingness(df, rng, col_profiles, systematic_missing_frac=sys_frac)
    return df


def _sample_symptom_with_liability(
    marginal_p: float,
    liability_shift: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a binary symptom with a per-subject logit shift.

    Preserves the marginal prevalence approximately while inducing
    positive correlation between symptoms that share the same liability
    draw — matches the real-data pattern where PCC symptoms co-occur
    rather than being endorsed independently.
    """
    # Clip to avoid log(0).  Very rare symptoms (p < 1e-6) are treated
    # as absent to keep the logit finite.
    p = float(np.clip(marginal_p, 1e-6, 1 - 1e-6))
    logit_p = float(np.log(p / (1.0 - p)))
    p_i = 1.0 / (1.0 + np.exp(-(logit_p + liability_shift)))
    return (rng.random(len(liability_shift)) < p_i).astype(np.int8)


def _compute_bahmer_score(df: pd.DataFrame) -> pd.Series:
    """Weighted PCS score per ``_BAHMER_MAPPING`` (same logic as the
    real pipeline; see ``data_processing._calculate_bahmer_pcs_score``)."""
    scores = pd.Series(0.0, index=df.index)
    for symptoms, weight in _BAHMER_MAPPING.values():
        avail = [s for s in symptoms if s in df.columns]
        if not avail:
            continue
        present = (df[avail].fillna(0).sum(axis=1) > 0).astype(float)
        scores = scores + present * weight
    return scores


def _generate_corona2_pcc(
    mod_profile: ModalityProfile,
    ids: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Corona-2 PCC: generate symptoms with a latent PCC liability, then
    derive all downstream scores deterministically.

    The liability is a per-subject Beta-distributed variable (zero for
    non-infected participants) that modulates every symptom's
    endorsement probability.  This reproduces the empirical symptom
    clustering — participants who endorse one neurocognitive item are
    substantially more likely to endorse others — which an independent
    per-item Bernoulli sampler would miss.
    """
    n = len(ids)
    col_profiles: dict[str, ColumnProfile] = mod_profile["columns"]
    df = pd.DataFrame({"ID": ids})

    # had_covid first — liability is only defined for infected subjects.
    if "had_covid" in col_profiles:
        df["had_covid"] = _generate_column(
            "had_covid", col_profiles["had_covid"], n, rng
        )
    else:
        df["had_covid"] = pd.array(
            rng.choice([0, 1], size=n, p=[0.41, 0.59]).astype(np.int8),
            dtype="Int8",
        )
    had_covid = df["had_covid"].fillna(0).to_numpy().astype(int)

    # Latent PCC liability: Beta(0.6, 1.4) for infected (right-skewed,
    # mean ~0.30), zero for non-infected.  Then center around the
    # infected mean so the logit shift is signed (some boost, some
    # penalty) and the marginal symptom prevalences are not inflated.
    raw_liab = rng.beta(0.6, 1.4, size=n).astype(np.float64)
    liability = np.where(had_covid == 1, raw_liab, 0.0)
    infected_mask = had_covid == 1
    mean_liab = float(liability[infected_mask].mean()) if infected_mask.any() else 0.0
    liability_shift_inf = LIABILITY_ALPHA * (liability - mean_liab)
    liability_shift = np.where(infected_mask, liability_shift_inf, 0.0)

    # Rescale each symptom's marginal from the population-wide profile
    # value to an infected-only target:  in the real Corona-2 data,
    # non-infected respondents skip the symptom block and are coded as
    # 0, so the population marginal understates the infected-subject
    # base rate by a factor of ``had_covid.mean()``.  Sampling on
    # infected with the population marginal reproduces that dilution
    # (infected prevalences drop by ≈ 2×) and prevents the liability
    # model from generating realistic symptom clusters.
    infected_rate = float(infected_mask.mean())
    rescale = 1.0 / max(infected_rate, 0.05)

    # Sample symptoms with the liability shift; all other non-derived
    # columns use the standard generator.
    symptom_cols: list[str] = []
    for col_name, col_prof in col_profiles.items():
        if col_prof.get("derived") or col_name in ("had_covid", "ID"):
            continue
        if col_name.startswith("symptom_") and col_prof["dtype"] == "boolean":
            p_population = col_prof["true_frac"]
            p_infected = min(p_population * rescale, 0.95)
            sampled = _sample_symptom_with_liability(
                marginal_p=p_infected,
                liability_shift=liability_shift,
                rng=rng,
            )
            # Structural zero for non-infected (they skip the block).
            sampled = np.where(infected_mask, sampled, 0).astype(np.int8)
            df[col_name] = pd.array(sampled, dtype="Int8")
            symptom_cols.append(col_name)
        else:
            df[col_name] = _generate_column(col_name, col_prof, n, rng)

    # Derive counts and the Diexer-style flags (≥1, ≥9 symptoms AND infected).
    symptom_count = (
        df[symptom_cols].fillna(0).sum(axis=1)
        if symptom_cols
        else pd.Series(0, index=df.index)
    )
    df["n_symptoms"] = _to_int8(np.asarray(symptom_count, dtype=np.int8))
    df["any_pcc"] = _to_int8(
        ((symptom_count >= 1) & (had_covid == 1)).to_numpy().astype(np.int8)
    )
    df["severe_pcc"] = _to_int8(
        ((symptom_count >= 9) & (had_covid == 1)).to_numpy().astype(np.int8)
    )

    # Auxiliary severity columns: the real ``corona2_pcc`` parquet carries
    # them (see ``data_processing/pcc_outcome.py``), so the fixture has to
    # as well; no analysis step reads them, so noise is enough.
    df["pcc_severity_score"] = rng.uniform(0, 20, size=n)
    df["pcc_severity_weighted"] = rng.uniform(0, 40, size=n)
    df["n_domains_affected"] = _to_int8(
        np.asarray(rng.integers(0, 8, size=n), dtype=np.int8)
    )

    # Bahmer weighted PCS — recomputed from the sampled symptoms so the
    # score actually reflects the clustered symptom pattern.  Apply the
    # infected restriction (non-infected skip the symptom block, so
    # their Bahmer score is zero by convention).
    bahmer = _compute_bahmer_score(df)
    bahmer = bahmer.where(df["had_covid"].fillna(0) == 1, 0.0)
    df["bahmer_pcs_score"] = bahmer

    bahmer_cats = ["none", "mild", "moderate", "severe"]
    df["bahmer_severity"] = pd.cut(
        df["bahmer_pcs_score"],
        bins=[-1, 10.75, 26.25, 41.5, 60],
        labels=bahmer_cats,
    )
    df["bahmer_any_pcs"] = _to_int8(
        (df["bahmer_pcs_score"] > 10.75).to_numpy().astype(np.int8)
    )
    df["bahmer_severe_pcs"] = _to_int8(
        (df["bahmer_pcs_score"] > 26.25).to_numpy().astype(np.int8)
    )

    # Synthetic `valid_symptoms` flag — the real loader filters on this,
    # so emit a realistic proportion of ones.  The real data shows ≈95 %
    # valid among infected respondents; keep the same marginal here.
    df["valid_symptoms"] = _to_int8(
        np.where(had_covid == 0, 1, (rng.random(n) < 0.95).astype(np.int8)).astype(
            np.int8
        )
    )

    # Synthetic ``d_co2_k0`` routing-gate — needed by the real loader when
    # ``clean_controls=True`` (see the routing logic in data_manager.py).
    # In the production NAKO:
    #   k0=1  → kmatrix answered (symptomatic)
    #   k0=2  → kmatrix skipped (symptom-free)
    #   7775  → routing question not shown (questionnaire break-off)
    #   8888  → refused
    # Non-infected participants never reach this question, so we leave
    # ``k0`` as NA for them.
    k0 = np.zeros(n, dtype=np.int16)
    n_sym = (
        np.asarray(symptom_count, dtype=np.int64)
        if symptom_cols
        else np.zeros(n, dtype=np.int64)
    )
    for i in range(n):
        if not infected_mask[i]:
            k0[i] = 0  # placeholder; converted to NA below
            continue
        draw = rng.random()
        if n_sym[i] >= 1:
            # Symptomatic: overwhelmingly k0=1; small chance of k0=2 or
            # sentinel (e.g. participant routed past question for other
            # reasons).
            if draw < 0.94:
                k0[i] = 1
            elif draw < 0.98:
                k0[i] = 2
            elif draw < 0.995:
                k0[i] = 7775
            else:
                k0[i] = 8888
        # Asymptomatic infected: overwhelmingly k0=2.  A minority
        # endorsed the gate but had all-zero items, or fell into the
        # 7775/8888 sentinel buckets.
        elif draw < 0.94:
            k0[i] = 2
        elif draw < 0.98:
            k0[i] = 1
        elif draw < 0.995:
            k0[i] = 7775
        else:
            k0[i] = 8888
    k0_arr = pd.array(k0.astype(np.int16), dtype="Int16")
    # Mark non-infected as NA so the clean-controls filter in the loader
    # (which only cares about infected participants anyway) sees a
    # well-formed column.
    k0_arr[~infected_mask] = pd.NA
    df["d_co2_k0"] = k0_arr

    sys_frac = mod_profile.get("systematic_missing_frac", 0.0)
    _inject_missingness(df, rng, col_profiles, systematic_missing_frac=sys_frac)
    return df


def _generate_ses(
    mod_profile: ModalityProfile,
    ids: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """SES: derivation groups + deterministic derived columns."""
    n = len(ids)
    col_profiles: dict[str, ColumnProfile] = mod_profile["columns"]
    df = pd.DataFrame({"ID": ids})

    # Generate non-derived, non-group columns
    for col_name, col_prof in col_profiles.items():
        if col_prof.get("derived") or col_prof.get("in_derivation_group"):
            continue
        df[col_name] = _generate_column(col_name, col_prof, n, rng)

    # Generate derivation groups
    groups = mod_profile.get("derivation_groups", {})
    for group_def in groups.values():
        group_df = _generate_derivation_group(group_def, n, rng)
        for col in group_df.columns:
            df[col] = group_df[col]

    # Derive deterministic columns
    if "household_size" in df.columns:
        df["living_alone"] = _to_int8((df["household_size"] == 1).to_numpy())
    if "number_children" in df.columns:
        df["has_children"] = _to_int8((df["number_children"] > 0).to_numpy())
    if "income_weighted" in df.columns and "needs_weighted" in df.columns:
        df["income_adequacy"] = df["income_weighted"] / df["needs_weighted"]

    sys_frac = mod_profile.get("systematic_missing_frac", 0.0)
    _inject_missingness(df, rng, col_profiles, systematic_missing_frac=sys_frac)
    return df


def _generate_medical_history(
    mod_profile: ModalityProfile,
    ids: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Medical history: generate source columns, derive flags/counts."""
    n = len(ids)
    col_profiles: dict[str, ColumnProfile] = mod_profile["columns"]
    df = pd.DataFrame({"ID": ids})

    # Generate non-derived columns
    for col_name, col_prof in col_profiles.items():
        if col_prof.get("derived"):
            continue
        df[col_name] = _generate_column(col_name, col_prof, n, rng)

    # Derive bmi_category from bmi_self_reported
    if "bmi_self_reported" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi_self_reported"],
            bins=[0, 18.5, 25, 30, 35, 40, 100],
            labels=[
                "underweight",
                "normal",
                "overweight",
                "obese_1",
                "obese_2",
                "obese_3",
            ],
            include_lowest=True,
        )
        df["overweight"] = _to_int8((df["bmi_self_reported"] >= 25).to_numpy())
        df["obese"] = _to_int8((df["bmi_self_reported"] >= 30).to_numpy())

    # Derive hypertension flag
    if "hypertension_current" in df.columns:
        df["has_hypertension"] = _to_int8((df["hypertension_current"] == 1).to_numpy())

    # Cancer history
    cancer_cols = [
        c for c in df.columns if c.startswith("cancer_") and c.endswith("_age")
    ]
    if cancer_cols:
        df["has_cancer_history"] = _to_int8(
            df[cancer_cols].notna().any(axis=1).to_numpy()
        )
        df["number_cancers"] = _to_int8(df[cancer_cols].notna().sum(axis=1).to_numpy())

    # Infection history
    infection_cols = [c for c in df.columns if c.startswith("infection_")]
    if infection_cols:
        df["has_infection_history"] = _to_int8(
            df[infection_cols].notna().any(axis=1).to_numpy()
        )

    # Surgery history
    surgery_cols = [
        c
        for c in df.columns
        if c.startswith("surgery_general_anesthesia_") and c.endswith("_age")
    ]
    if surgery_cols:
        n_surg = df[surgery_cols].notna().sum(axis=1).to_numpy()
        df["number_surgeries"] = _to_int8(n_surg)
        df["has_surgery_history"] = _to_int8(n_surg > 0)

    # Medication count
    med_cols = [c for c in df.columns if c.startswith("medication_")]
    if med_cols:
        med_sum = sum((df[c] == 1).astype("Int8") for c in med_cols)
        n_meds: np.ndarray = np.asarray(med_sum, dtype=np.int8)
        df["number_medications"] = _to_int8(n_meds)
        df["polypharmacy"] = _to_int8(n_meds >= 5)

    # Neurological disease
    neuro_cols = [
        c for c in ["stroke_age", "epilepsy_age", "parkinsons_age"] if c in df.columns
    ]
    if neuro_cols:
        df["has_neurological_disease"] = _to_int8(
            df[neuro_cols].notna().any(axis=1).to_numpy()
        )

    # CV risk
    cv_flags: list[np.ndarray] = []
    if "has_hypertension" in df.columns:
        cv_flags.append(df["has_hypertension"].fillna(0).to_numpy())
    if "obese" in df.columns:
        cv_flags.append(df["obese"].fillna(0).to_numpy())
    if "polypharmacy" in df.columns:
        cv_flags.append(df["polypharmacy"].fillna(0).to_numpy())
    if cv_flags:
        cv_sum = np.sum(cv_flags, axis=0)
        df["cv_risk_factor_count"] = _to_int8(cv_sum)
        df["high_cv_risk"] = _to_int8(cv_sum >= 2)

    sys_frac = mod_profile.get("systematic_missing_frac", 0.0)
    _inject_missingness(df, rng, col_profiles, systematic_missing_frac=sys_frac)
    return df


def _generate_physical_activity(
    mod_profile: ModalityProfile,
    ids: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Physical activity: generic columns + systematic missingness."""
    df = _generate_generic(mod_profile, ids, rng)
    sys_frac = mod_profile.get("systematic_missing_frac", 0.0)
    if sys_frac > 0:
        _inject_systematic_missingness(df, rng, sys_frac)
    return df


def _generate_mental_health(
    mod_profile: ModalityProfile,
    ids: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Baseline mental health: sample sum scores + derive threshold flags.

    The PHQ-9 moderate-depression flag, the GAD-7 moderate-anxiety flag,
    and the stress moderate flag are deterministic from the sum scores
    (PHQ-9 ≥ 10, GAD-7 ≥ 10, PHQ-stress ≥ 10), so they must track the
    sampled sums to stay internally consistent — otherwise the
    mental-health modality carries contradictory signal and the base
    learner either overfits or no-ops.
    """
    df = _generate_generic(mod_profile, ids, rng)

    # Re-derive threshold flags that the profiler marked as derived.
    def _ge(col: str, threshold: float) -> pd.arrays.IntegerArray:
        return _to_int8(
            (df[col].fillna(0).to_numpy().astype(float) >= threshold).astype(np.int8)
        )

    if "phq9_sum" in df.columns:
        df["phq9_moderate_depression"] = _ge("phq9_sum", 10)
        # Severity buckets 0-4: 0=<5, 1=5-9, 2=10-14, 3=15-19, 4=≥20
        df["phq9_severity_category"] = _to_int8(
            np.digitize(df["phq9_sum"].fillna(0).to_numpy(), [5, 10, 15, 20]).astype(
                np.int8
            )
        )
        # DSM-5 criteria count (stubbed 0/1/2 from PHQ-9 severity)
        df["phq9_dsm_criteria"] = _to_int8(
            np.clip(
                df["phq9_severity_category"].fillna(0).to_numpy().astype(np.int8) // 2,
                0,
                2,
            ).astype(np.int8)
        )

    if "gad7_sum" in df.columns:
        df["gad7_moderate_anxiety"] = _ge("gad7_sum", 10)
        df["gad7_diagnosis"] = _to_int8(
            np.digitize(df["gad7_sum"].fillna(0).to_numpy(), [5, 10, 15]).astype(
                np.int8
            )
        )

    if "phq_stress" in df.columns:
        df["phq_stress_moderate"] = _ge("phq_stress", 10)

    return df


# Acute-care levels the severity flags threshold on (mirror of
# ``data_processing.acute_infection.HOSPITALISED_LEVEL`` / ``ICU_LEVEL``).
# Kept local for the same reason as ``_BAHMER_MAPPING``: this module builds
# fixtures out of a JSON profile and imports nothing from the analysis
# package.
_HOSPITALISED_LEVEL: int = 2
_ICU_LEVEL: int = 3


_CARE_LEVEL_RATE_TOLERANCE: float = 1e-9
"""Slack allowed when the reconstructed level rates are checked for unity."""


def _acute_care_level_rates(
    level: IntegerColumnProfile,
    hospitalised: BooleanColumnProfile,
    icu: BooleanColumnProfile,
) -> np.ndarray:
    """Rates of acute-care levels 0..3, read off the profile's summaries.

    ``acute_care_level``'s own summary cannot supply them. Hospital and
    intensive care are rare enough to sit above its 99th percentile, and
    :func:`_generate_column` clips every draw to ``[q01, q99]``, so a level
    drawn from that summary never exceeds 1. The generator's contract is
    that the synthetic column has the shape the profile records, and a level
    that cannot reach 2 breaks it outright: the profile states a hospital
    rate the column then contradicts in every row.

    The flags pin the tail the clip drops, because they are exactly its
    thresholds: the ``acute_icu`` rate is P(level = 3), and what is left of
    the ``acute_hospitalised`` rate is P(level = 2). The profiled level mean,
    being a mean over all four levels, splits the remainder between "no care"
    and "seen by a physician". Every probability therefore comes out of the
    profile rather than out of a chosen constant.

    Three recorded summaries of one variable can also disagree — no level
    mean small enough to fund the hospital and intensive-care rates, or an
    intensive-care rate above the hospital rate that contains it. That is a
    defect in the profile, and clamping or renormalising it would restore
    the silent inconsistency this reconstruction exists to remove, so it
    raises instead.
    """
    p_icu = icu["true_frac"]
    p_ward = hospitalised["true_frac"] - p_icu
    # mean = 0*p0 + 1*p1 + 2*p2 + 3*p3, solved for p1.
    p_physician = level["mean"] - _HOSPITALISED_LEVEL * p_ward - _ICU_LEVEL * p_icu
    p_none = 1.0 - p_physician - p_ward - p_icu
    rates = np.array([p_none, p_physician, p_ward, p_icu], dtype=np.float64)
    if rates.min() < 0.0 or abs(rates.sum() - 1.0) > _CARE_LEVEL_RATE_TOLERANCE:
        raise ValueError(
            "acute_infection profile: acute_hospitalised true_frac="
            f"{hospitalised['true_frac']}, acute_icu true_frac={icu['true_frac']} "
            f"and acute_care_level mean={level['mean']} imply level rates "
            f"{rates.tolist()} for levels 0..3, which are not a distribution. "
            "One of the three summaries is wrong; the flags are thresholds of "
            "the level, so all three describe the same variable."
        )
    return rates


def _generate_acute_infection(
    mod_profile: ModalityProfile,
    ids: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Acute course: draw the care level, derive what is keyed to it.

    ``acute_hospitalised``, ``acute_icu``, ``hospital_days`` and ``icu_days``
    are exact functions of ``acute_care_level`` in the extractor (see
    ``data_processing/acute_infection.py``): the flags are its thresholds,
    and a level below a threshold makes the matching day count a structural
    zero rather than a measurement. Drawn independently they contradict each
    other within the row — level 0 beside a hospital stay, intensive-care
    days beside a cleared ICU flag — and
    ``scripts/supplementary/severity_interaction.py`` then runs under
    ``--smoke`` against data on which no severity recode can be wrong, which
    is the one regression that check exists to catch.

    The level keeps the missing values the profile asks for and the derived
    columns follow it into them, so the fixture also exercises the
    unknown-level path rather than handing every participant a level.
    """
    n = len(ids)
    col_profiles: dict[str, ColumnProfile] = mod_profile["columns"]
    df = _generate_generic(mod_profile, ids, rng)

    level_profile = col_profiles.get("acute_care_level")
    hospitalised_profile = col_profiles.get("acute_hospitalised")
    icu_profile = col_profiles.get("acute_icu")
    if level_profile is None or hospitalised_profile is None or icu_profile is None:
        return df
    if (
        level_profile["dtype"] != "integer"
        or hospitalised_profile["dtype"] != "boolean"
        or icu_profile["dtype"] != "boolean"
    ):
        return df

    rates = _acute_care_level_rates(level_profile, hospitalised_profile, icu_profile)
    drawn = rng.choice(len(rates), size=n, p=rates).astype(np.int8)
    # Redraw the values, keep the missingness ``_generate_generic`` injected.
    known = df["acute_care_level"].notna()
    sampled = pd.Series(drawn, index=df.index, dtype=level_profile["pandas_dtype"])
    level = sampled.where(known)
    df["acute_care_level"] = level

    df["acute_hospitalised"] = (level >= _HOSPITALISED_LEVEL).astype("Int8")
    df["acute_icu"] = (level == _ICU_LEVEL).astype("Int8")

    for days, admitted_from in (
        ("hospital_days", _HOSPITALISED_LEVEL),
        ("icu_days", _ICU_LEVEL),
    ):
        days_profile = col_profiles.get(days)
        if days_profile is None:
            continue
        reported = (
            df[days]
            if days in df.columns
            else _generate_column(days, days_profile, n, rng)
        )
        below = (level < admitted_from).fillna(False).to_numpy(dtype=bool)
        df[days] = reported.mask(below, 0).where(known)

    return df


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_ModalityGenerator = Callable[
    [ModalityProfile, np.ndarray, np.random.Generator], pd.DataFrame
]

_SPECIAL_GENERATORS: dict[str, _ModalityGenerator] = {
    "corona2_pcc": _generate_corona2_pcc,
    "socioeconomic_status": _generate_ses,
    "medical_history": _generate_medical_history,
    "physical_activity": _generate_physical_activity,
    "mental_health": _generate_mental_health,
    "acute_infection": _generate_acute_infection,
}


_MRI_FRACTION: float = 0.4
"""Fraction of synthetic participants assigned to the MRI sub-cohort.

Production NAKO has ~17 % MRI coverage (8.5 k of 50 k Corona-2
participants).  Smoke-test uses 40 % so both the MRI and non-MRI
arms retain enough participants for end-to-end CV validation after
the clean-controls filter.  Keep this in sync with the non_mri
smoke-test assertions if you lower it.
"""

_MRI_PARQUET_NAMES: tuple[str, ...] = (
    "mri_cortical_desikan_killiany",
    "mri_cortical_destrieux",
    "mri_cortical_julich",
    "mri_cortical_yeo_networks",
    "mri_subcortical",
    "mri_cerebellar",
    "mri_etiv",
)
"""Modality names thinned to the MRI sub-cohort in ``generate_all``."""


def generate_all(
    profile: Profile,
    n_subjects: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate all modality DataFrames from a profile.

    Returns a dict mapping modality name to DataFrame.
    """
    rng = np.random.default_rng(seed)
    ids = _ids(n_subjects)
    modalities = profile["modalities"]
    result: dict[str, pd.DataFrame] = {}

    # corona2_pcc first (provides PCC target for signal injection)
    if "corona2_pcc" in modalities:
        result["corona2_pcc"] = _generate_corona2_pcc(
            modalities["corona2_pcc"], ids, rng
        )

    # All other modalities
    for name, mod_profile in modalities.items():
        if name == "corona2_pcc":
            continue
        generator = _SPECIAL_GENERATORS.get(name, _generate_generic)
        result[name] = generator(mod_profile, ids, rng)

    # Inject weak PCC signal into all modalities except corona2_pcc.
    # Runs BEFORE MRI thinning so every modality's row count still
    # matches the full ``pcc_target`` vector.
    pcc_target: np.ndarray = (
        result["corona2_pcc"]["bahmer_any_pcs"].to_numpy(dtype=float)
        if "corona2_pcc" in result
        else np.zeros(n_subjects)
    )
    for name, df in result.items():
        if name != "corona2_pcc":
            _inject_signal(df, pcc_target, rng)

    # Thin MRI modalities to the MRI sub-cohort so the smoke test can
    # exercise both ``cohort='mri'`` and ``cohort='non_mri'`` pipeline
    # paths.  Without this, every synthetic subject has MRI data and the
    # non-MRI arm is empty.  The split mirrors the cohort architecture,
    # where only a subset of participants carry MRI modalities.
    mri_n = max(1, round(n_subjects * _MRI_FRACTION))
    mri_ids = list(rng.choice(ids, size=mri_n, replace=False).tolist())
    for mri_name in _MRI_PARQUET_NAMES:
        if mri_name not in result:
            continue
        df = result[mri_name]
        result[mri_name] = df[df["ID"].isin(mri_ids)].reset_index(drop=True)

    return result


def load_profile(path: Path) -> Profile:
    """Load a JSON profile from disk and check it has the expected shape.

    Profiles are generated from real data on a different machine and checked
    in, so the file can predate the schema the generators expect. Validating
    the parts the generators index into turns that into one clear error here
    instead of a ``KeyError`` deep inside generation.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    _check_profile_shape(raw, path)
    return raw


def _check_profile_shape(raw: object, path: Path) -> None:
    """Raise ``ValueError`` unless *raw* matches :class:`Profile`.

    Every defect in a profile file is reported as ``ValueError``, including the
    ones a caller reaches by handing over JSON of the wrong shape. Splitting the
    shape checks off into ``TypeError`` would leave callers catching two
    exception types to report one malformed file, so ``TRY004`` is waived here.
    """
    if not isinstance(raw, dict) or not {"meta", "modalities"} <= raw.keys():
        raise ValueError(f"{path}: not a profile — expected 'meta' and 'modalities'")
    modalities = raw["modalities"]
    if not isinstance(modalities, dict):
        raise ValueError(f"{path}: 'modalities' must be an object")  # noqa: TRY004
    for name, modality in modalities.items():
        columns = modality.get("columns") if isinstance(modality, dict) else None
        if not isinstance(columns, dict):
            raise ValueError(  # noqa: TRY004
                f"{path}: modality {name!r} has no 'columns' object"
            )
        for column, profile in columns.items():
            dtype = profile.get("dtype") if isinstance(profile, dict) else None
            if dtype not in COLUMN_DTYPES:
                raise ValueError(
                    f"{path}: column {name}.{column} has unknown dtype {dtype!r}"
                )
