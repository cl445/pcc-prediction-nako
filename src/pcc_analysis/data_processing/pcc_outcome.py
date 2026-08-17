"""PCC outcome derivation: Diexer symptom-count and Bahmer-weighted PCS score.

Operates on the Corona-2 ``d_co2_kmatrix_*`` symptom items. Two
operationalisations are produced:

* **Diexer et al. (2025)** -- ``any_pcc`` (>=1 symptom) and ``severe_pcc``
  (>=9 symptoms) within 4-12 months post-infection.
* **Bahmer et al. (2022)** -- weighted PCS score (0-59) plus the four-level
  severity category (none/mild/moderate/severe).

Constants kept private to this module, with the exception of
``_KMATRIX_SENTINEL_CODES`` which is reached from
``tests/test_preprocessing.py`` via re-export from the package root.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


_KMATRIX_SYMPTOM_NAMES: dict[str, str] = {
    "k1": "fever",
    "k2": "loss_of_smell",
    "k3": "loss_of_taste",
    "k4": "joint_muscle_pain",
    "k5": "fatigue",
    "k6": "sleep_problems",
    "k7": "reduced_physical_capacity",
    "k8": "sweating",
    "k9": "memory_problems",
    "k10": "concentration_problems",
    "k11": "headache",
    "k12": "runny_nose",
    "k13": "cough",
    "k14": "breathing_problems",
    "k15": "chest_tightness",
    "k16": "heart_problems",
    "k17": "gastrointestinal_problems",
    "k18": "loss_of_appetite",
    "k19": "hair_loss",
    "k19a": "circulation_problems",
    "k19b": "nerve_problems",
}

# Sentinel codes that indicate missing / invalid responses in kmatrix symptom
# items.  This is the union of the general and survey-specific sentinel codes
# that can appear in the Corona-2 binary symptom questions.
_KMATRIX_SENTINEL_CODES: list[int] = [
    -88,  # refused to answer
    -9,  # general missing
    -8,  # not determinable
    6666,  # Corona-2: not applicable / filter skip
    7775,  # question not shown
    7776,  # question skipped
    7777,  # not assessed
    8886,  # not evaluable
    8888,  # refused
    8889,  # don't know
    9999,  # general missing
]

_SEVERITY_WEIGHTS: dict[str, float] = {
    "symptom_fatigue": 3.0,
    "symptom_reduced_physical_capacity": 3.0,
    "symptom_breathing_problems": 2.5,
    "symptom_memory_problems": 2.5,
    "symptom_concentration_problems": 2.5,
    "symptom_heart_problems": 2.5,
    "symptom_sleep_problems": 2.0,
    "symptom_chest_tightness": 2.0,
    "symptom_joint_muscle_pain": 2.0,
    "symptom_nerve_problems": 2.0,
    "symptom_circulation_problems": 2.0,
    "symptom_headache": 1.5,
    "symptom_gastrointestinal_problems": 1.5,
    "symptom_loss_of_smell": 1.5,
    "symptom_loss_of_taste": 1.5,
    "symptom_loss_of_appetite": 1.5,
    "symptom_sweating": 1.0,
    "symptom_hair_loss": 1.0,
    "symptom_cough": 1.0,
    "symptom_runny_nose": 0.5,
    "symptom_fever": 0.5,
}

_SYMPTOM_DOMAINS: dict[str, list[str]] = {
    "functional": [
        "symptom_fatigue",
        "symptom_reduced_physical_capacity",
        "symptom_sleep_problems",
        "symptom_joint_muscle_pain",
    ],
    "cognitive": [
        "symptom_memory_problems",
        "symptom_concentration_problems",
    ],
    "cardiopulmonary": [
        "symptom_breathing_problems",
        "symptom_chest_tightness",
        "symptom_heart_problems",
        "symptom_circulation_problems",
    ],
    "sensory": [
        "symptom_loss_of_smell",
        "symptom_loss_of_taste",
    ],
    "neurological": [
        "symptom_headache",
        "symptom_nerve_problems",
    ],
    "respiratory": [
        "symptom_cough",
        "symptom_runny_nose",
    ],
    "gastrointestinal": [
        "symptom_gastrointestinal_problems",
        "symptom_loss_of_appetite",
    ],
    "other": [
        "symptom_sweating",
        "symptom_hair_loss",
        "symptom_fever",
    ],
}

_FUNCTIONAL_SYMPTOMS: list[str] = [
    "symptom_fatigue",
    "symptom_reduced_physical_capacity",
    "symptom_breathing_problems",
    "symptom_memory_problems",
    "symptom_concentration_problems",
    "symptom_sleep_problems",
]

# Bahmer et al. (2022) symptom-complex -> (symptom list, weight)
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


def _calculate_bahmer_pcs_score(symptom_df: DataFrame) -> pd.Series:
    """Weighted PCS score (0-59) per Bahmer et al. (2022)."""
    scores = pd.Series(0.0, index=symptom_df.index)
    for symptoms, weight in _BAHMER_MAPPING.values():
        avail = [s for s in symptoms if s in symptom_df.columns]
        if not avail:
            continue
        present = (symptom_df[avail].sum(axis=1) > 0).astype("Int8")
        scores += present * weight
    return scores


def _determine_pcc_status(df: DataFrame) -> DataFrame:
    """Build comprehensive PCC DataFrame from Corona-2 survey.

    Implements two operationalisations:

    * **Diexer et al. (2025)** -- symptom-count based
      (``any_pcc``: >= 1 symptom; ``severe_pcc``: >= 9 symptoms
      in the 4-12 month window after SARS-CoV-2 infection)

    * **Bahmer et al. (2022)** -- weighted PCS score (0-59)
      with severity categories none/mild/moderate/severe.
    """
    had_covid = df["d_co2_h1"] == 1
    logger.info(
        "Participants with COVID-19: %s (%.1f%%)",
        f"{had_covid.sum():,}",
        had_covid.mean() * 100,
    )

    # kmatrix symptom columns (4-12 months after first infection)
    kmatrix_cols = sorted(
        c
        for c in df.columns
        if c.startswith("d_co2_kmatrix_k")
        and not any(x in c.lower() for x in ["datum", "vorfilter", "vorhanden"])
    )
    logger.info("Found %d kmatrix symptom columns", len(kmatrix_cols))

    # Binary symptom matrix
    symptom_data = df[kmatrix_cols].copy()
    symptom_data = symptom_data.replace(_KMATRIX_SENTINEL_CODES, pd.NA)

    # ── Symptom routing gate (d_co2_k0) ──
    # The Corona-2 questionnaire uses a routing question before the kmatrix
    # section asking whether the participant experienced symptoms 4-12 months
    # post-infection.  Values:
    #   1    = yes, symptom detail section administered
    #   2    = no symptoms reported, kmatrix skipped (valid symptom-free)
    #   7775 = routing question not shown (questionnaire dropout before section)
    #   8888 = refused / not evaluable
    # Only k0=1 (assessed) and k0=2 (explicitly symptom-free) provide
    # observable symptom status.  k0 ∈ {7775, 8888} means the outcome is
    # not observed and these participants must be excluded.
    k0 = df.get("d_co2_k0")
    if k0 is not None:
        routing_not_observed = had_covid & k0.isin([7775, 8888])
        n_not_shown = int((had_covid & (k0 == 7775)).sum())
        n_refused = int((had_covid & (k0 == 8888)).sum())
        logger.info(
            "Symptom routing (d_co2_k0): %d infected not shown (7775), "
            "%d refused (8888) -> excluded",
            n_not_shown,
            n_refused,
        )
    else:
        routing_not_observed = pd.Series(False, index=df.index)
        logger.warning("d_co2_k0 not found; cannot apply routing filter")

    # ── Validity filter: ≥80% non-missing for assessed participants ──
    # Among infected participants who WERE routed to the kmatrix (k0=1),
    # require ≥80% of items answered.  Infected with k0=2 (explicitly
    # symptom-free) have all-NaN kmatrix by design; fillna(False) below
    # correctly codes them as symptom-free.
    n_items = len(kmatrix_cols)
    valid_frac = symptom_data.notna().sum(axis=1) / n_items
    was_assessed = symptom_data.notna().any(axis=1)
    insufficient = had_covid & was_assessed & (valid_frac < 0.80)
    n_insufficient = int(insufficient.sum())
    if n_insufficient > 0:
        logger.info(
            "Symptom validity filter: %d infected participants with <80%% valid "
            "responses (of %d assessed) -> symptom items set to NA",
            n_insufficient,
            int((had_covid & was_assessed).sum()),
        )
        symptom_data.loc[insufficient] = pd.NA

    symptom_binary = (symptom_data == 1).fillna(False).astype("Int8")

    result = pd.DataFrame()
    result["ID"] = df["ID"]
    result["had_covid"] = had_covid.astype("Int8")
    # Store the raw routing code so downstream consumers can distinguish
    # "declared symptom-free" (k0=2) from "assessed with sub-threshold
    # symptoms" (k0=1 ∩ Bahmer ≤ 10.75) — see load_pipeline_data's
    # ``clean_controls`` option for why this matters for the PCC-
    # control-group definition.
    if k0 is not None:
        result["d_co2_k0"] = pd.to_numeric(k0, errors="coerce").astype("Int16")
    # valid_symptoms = 0 for two disjoint groups of infected participants:
    #   (a) routing not observed: k0 ∈ {7775, 8888} — outcome not observable
    #   (b) insufficient responses: assessed (k0=1) but <80% items valid
    # Non-infected and k0=2 (explicitly symptom-free) are always valid.
    # Downstream consumers (load_pipeline_data) MUST filter on this column.
    invalid = insufficient | routing_not_observed
    result["valid_symptoms"] = (~invalid).astype("Int8")

    # One-hot encoded symptoms with readable names
    for col in kmatrix_cols:
        suffix = col.replace("d_co2_kmatrix_", "")
        readable = _KMATRIX_SYMPTOM_NAMES.get(suffix, suffix)
        result[f"symptom_{readable}"] = symptom_binary[col]

    symptom_count = symptom_binary.sum(axis=1)
    result["n_symptoms"] = symptom_count

    # -- Diexer operationalisation --
    result["any_pcc"] = ((symptom_count >= 1) & had_covid).astype("Int8")
    result["severe_pcc"] = ((symptom_count >= 9) & had_covid).astype("Int8")

    # -- Severity scores --
    symptom_cols = [c for c in result.columns if c.startswith("symptom_")]

    functional_count = result[
        [s for s in _FUNCTIONAL_SYMPTOMS if s in symptom_cols]
    ].sum(axis=1)
    other_cols = [s for s in symptom_cols if s not in _FUNCTIONAL_SYMPTOMS]
    other_count = result[other_cols].sum(axis=1)
    result["pcc_severity_score"] = (functional_count * 2.0) + (other_count * 0.5)

    weighted_sum = sum(result[c] * _SEVERITY_WEIGHTS.get(c, 1.0) for c in symptom_cols)
    result["pcc_severity_weighted"] = weighted_sum

    domain_flags = []
    for dom_symptoms in _SYMPTOM_DOMAINS.values():
        avail = [s for s in dom_symptoms if s in symptom_cols]
        if avail:
            domain_flags.append((result[avail].sum(axis=1) > 0).astype("Int8"))
    result["n_domains_affected"] = sum(domain_flags) if domain_flags else 0

    # -- Bahmer PCS score --
    result["bahmer_pcs_score"] = _calculate_bahmer_pcs_score(result)
    conditions = [
        result["bahmer_pcs_score"] == 0,
        (result["bahmer_pcs_score"] > 0) & (result["bahmer_pcs_score"] <= 10.75),
        (result["bahmer_pcs_score"] > 10.75) & (result["bahmer_pcs_score"] <= 26.25),
        result["bahmer_pcs_score"] > 26.25,
    ]
    categories = ["none", "mild", "moderate", "severe"]
    result["bahmer_severity"] = pd.Categorical(
        np.select(conditions, categories, default="none"),
        categories=categories,
        ordered=True,
    )
    result["bahmer_any_pcs"] = (result["bahmer_pcs_score"] > 10.75).astype("Int8")
    result["bahmer_severe_pcs"] = (result["bahmer_pcs_score"] > 26.25).astype("Int8")

    return result
