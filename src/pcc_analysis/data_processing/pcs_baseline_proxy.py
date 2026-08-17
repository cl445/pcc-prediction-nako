"""Pre-pandemic PCS-equivalent baseline proxy from KVAD ambulatory ICD-10 claims.

Provides a third PCS measurement at T0 for the longitudinal RI-CLPM model
(pre-pandemic, 2013-2019) to complement the Corona-1 acute self-report at
T1 and the Bahmer persistent self-report at T2. Without this, the cross-
lag identification collapses to a 2-wave reduction; with it, the
random-intercept (trait) absorption survives.

Method (informed by GCAT Catalonia 2025, BMC Medicine
``10.1186/s12916-025-04427-x``)
-----------------------------------------------------------------------
1. Restrict the ambulatory-claims stream to the pre-pandemic window
   (default: ``kvad_year < 2020``). Diagnoses certainty filter: drop
   "ausgeschlossen" (sure==3); keep "gesichert" (1), "Verdacht" (2),
   "Zustand nach" (4) -- the latter two carry symptom-load signal even
   if not formally confirmed.
2. Map each Bahmer post-COVID symptom item (the 21 columns from
   ``corona2_pcc.parquet``) to one or more 3-character ICD-10 codes
   (``kvad_a_amb_diag_3``); see ``BAHMER_TO_ICD3`` below for the curated
   table and the per-item rationale.
3. Per participant, mark a Bahmer item as "PCS-T0 positive" if at least
   one of its mapped ICD-3 codes appears in the pre-pandemic window.
4. Sum the per-item indicators: ``pcs_t0_proxy_count`` (0..21), directly
   comparable to a Bahmer item-count score at T2.

Sensitivity variants (GCAT-aligned)
-----------------------------------
The default specification differs from GCAT 2025 in two parameters that
the GCAT pipeline pinned down. To probe whether those choices materially
shift the T0 score, ``process_kvad_pcs_proxy`` accepts a ``variant``
argument that emits parquets for direct comparison:

* ``"default"`` -- as above (baseline used by the main analyses).
* ``"chronic"`` -- restrict the ICD-3 mapping to codes classified as
  chronic by HCUP's Chronic Conditions Indicator (CCI 2023). All R-chapter
  symptom codes (R00, R05, R06, R07, R10, R11, R19, R20, R41, R43, R50,
  R51, R53, R61, R63) drop out as acute; F/G/I/J/K/L/M chapter codes in
  the mapping survive. This is what GCAT does upstream of trajectory
  construction. Eight Bahmer items become trivially zero (no chronic ICD
  mapping); 13 retain at least one chronic code.
* ``"strict"`` -- restrict diagnose-certainty to ``gesichert`` (1) only,
  dropping ``Verdacht`` (2) and ``Zustand nach`` (4). This is the
  conservative reading typical of GCAT-style chronic-disease cohort
  reconstructions, where uncertain or resolved diagnoses are treated as
  null evidence.

Output filenames key off the variant: ``kvad_pcs_proxy.parquet``,
``kvad_pcs_proxy_chronic.parquet``, ``kvad_pcs_proxy_strict.parquet``.
The column schema is identical across variants so the downstream
convergence validator can ingest all three with one loader.

Caveats:

- **Measurement-mode heterogeneity across waves.** T0 is claims-derived
  (clinician-recorded ICD codes), T1 is self-reported acute symptoms
  (``cov87s``), T2 is self-reported persistent symptoms (Bahmer). A
  random-intercept model accommodates this via free per-wave loadings
  but the strong-invariance assumption is violated; we accept configural
  invariance and report it as a limitation.
- **Subcohort restriction.** Only NAKO participants with linked KVAD
  ambulatory claims (~36k of ~200k) have a meaningful T0 PCS measurement.
  The RI-CLPM runs on this subset. Other methods that don't need a
  T0 PCS proxy run on the full analytic sample.
- **Underdiagnosis.** Claims data captures only those symptoms that
  prompted a clinician contact AND a coded diagnosis. Mild or
  unattended symptoms vanish. The score is therefore a *floor* on
  somatic-symptom load, not a complete inventory.
- **Severity not captured.** Each Bahmer item collapses to a binary
  (any pre-pandemic claim vs. none). Severity discrimination requires
  weighting by claim count or quarter coverage; left as a sensitivity
  variant for a follow-up iteration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pandas import DataFrame

from pcc_analysis.data_processing._common import _load_nako_csv, _save_parquet

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)

ProxyVariant = Literal["default", "chronic", "strict"]


# Per-symptom ICD-3 mapping. Each entry maps a Bahmer ``symptom_*`` column
# (the canonical post-COVID item set in ``corona2_pcc.parquet``) to one or
# more 3-character ICD-10 codes that operationalise the same symptom domain
# in claims data. Choices favour the ICD chapter R (Symptoms, signs and
# abnormal findings not classified elsewhere) and selected functional
# F-/G-/J-/K-/M-codes that the literature treats as direct symptom proxies.
#
# References for individual mappings:
#   - R-chapter codes follow the WHO ICD-10 2019 listings.
#   - F45/F48 (somatoform / neurasthenia) per Kroenke 2007 PHQ-15
#     interpretation of "medically unexplained symptoms".
#   - J44/J45 (COPD/Asthma) per Stewart 2023 long-COVID respiratory
#     codings.
#   - GCAT Catalonia 2025 used 3-digit grouping with a chronic-conditions
#     filter; we keep the 3-digit grain but accept all certainty flags
#     except "ausgeschlossen".
#
# Intentional cross-mappings (one ICD code feeds two Bahmer items):
#   - ``R20`` (cutaneous-sensation disorders): both
#     ``symptom_circulation_problems`` and ``symptom_nerve_problems``.
#     A single R20 record therefore contributes ``+2`` to
#     ``pcs_t0_proxy_count``. This mirrors how a participant at T2
#     could endorse both Bahmer items independently from the same
#     underlying complaint, so the count remains comparable to
#     a Bahmer-item-count score rather than under-counting at T0.
#   - ``R43`` (disturbances of smell and taste): both
#     ``symptom_loss_of_smell`` and ``symptom_loss_of_taste``. Same
#     rationale — ICD-10 collapses what the Bahmer inventory lists as
#     two distinct items.
BAHMER_TO_ICD3: dict[str, tuple[str, ...]] = {
    "symptom_fever": ("R50",),
    "symptom_concentration_problems": ("R41",),
    "symptom_headache": ("R51", "G43", "G44"),
    "symptom_runny_nose": ("J30", "J31"),
    "symptom_cough": ("R05",),
    "symptom_breathing_problems": ("R06", "J45", "J44"),
    "symptom_chest_tightness": ("R07",),
    "symptom_heart_problems": ("R00", "I49"),
    "symptom_gastrointestinal_problems": ("R10", "R11", "R19", "K58", "K59"),
    "symptom_loss_of_appetite": ("R63",),
    "symptom_hair_loss": ("L65",),
    "symptom_circulation_problems": ("I73", "R20"),
    "symptom_nerve_problems": ("R20", "G50", "G56", "G57", "G58"),
    "symptom_loss_of_smell": ("R43",),
    "symptom_loss_of_taste": ("R43",),
    "symptom_joint_muscle_pain": ("M25", "M54", "M79", "M62"),
    "symptom_fatigue": ("R53", "F48"),
    "symptom_sleep_problems": ("G47", "F51"),
    "symptom_reduced_physical_capacity": ("R53", "F48"),
    "symptom_sweating": ("R61",),
    "symptom_memory_problems": ("R41", "F06"),
}


# Diagnose-certainty values to keep when aggregating. NAKO codes:
#   1=G (gesichert), 2=V (Verdacht), 3=A (ausgeschlossen), 4=Z (Zustand nach).
# Default drops only "ausgeschlossen" and the explicit-missing -9 sentinel;
# the remaining flags all carry signal that the patient consulted a clinician
# for that symptom complex. The strict variant additionally drops V (2) and
# Z (4) so only clinically confirmed diagnoses contribute to the score.
_CERTAINTY_DROP_DEFAULT: frozenset[int] = frozenset({3, -9})
_CERTAINTY_DROP_STRICT: frozenset[int] = frozenset({2, 3, 4, -9})

# CCI-2023-aligned chronicity classification, restricted to the codes that
# appear in BAHMER_TO_ICD3. R-chapter codes ("Symptoms, signs and abnormal
# findings, NEC") index transient clinician contacts and are classified
# acute by CCI 2023; F/G/I/J/K/L/M chapter codes in our mapping are chronic
# (mental, neurological, circulatory, respiratory, digestive, integumentary,
# musculoskeletal). The chronic-only filter is GCAT-aligned: GCAT 2025 runs
# the same CCI 2023 indicator before any aggregation. Note R53 ("malaise
# and fatigue") sits at the boundary -- CCI 2023 classifies the 4-digit
# R53.82 (chronic fatigue) as chronic, but at the 3-digit level R53 covers
# both acute and chronic forms, so we conservatively keep it acute.
_ICD3_CHRONIC: frozenset[str] = frozenset(
    {
        # F-chapter (mental, behavioural, somatoform)
        "F06",
        "F45",
        "F48",
        "F51",
        # G-chapter (nervous system: migraine, sleep, mononeuropathies)
        "G43",
        "G44",
        "G47",
        "G50",
        "G56",
        "G57",
        "G58",
        # I-chapter (circulatory: arrhythmia, peripheral vascular)
        "I49",
        "I73",
        # J-chapter (respiratory: rhinitis, COPD, asthma)
        "J30",
        "J31",
        "J44",
        "J45",
        # K-chapter (digestive: IBS, other functional)
        "K58",
        "K59",
        # L-chapter (skin appendage: hair loss)
        "L65",
        # M-chapter (musculoskeletal)
        "M25",
        "M54",
        "M62",
        "M79",
    }
)

_PRE_PANDEMIC_YEAR_MAX: int = 2019  # inclusive; calendar pre-pandemic cutoff
_KVAD_YEAR_MIN: int = 2010
"""Earliest KVAD ambulatory-claims coverage year.

Statutory ambulatory-claims coverage in the linked NAKO/KVAD subset
starts in 2010; rows before that surface only as data-entry errors and
would otherwise inflate the chronic-symptom load with synthetic
out-of-window observations.
"""


def _filter_pre_pandemic(
    diag: DataFrame, *, year_max: int = _PRE_PANDEMIC_YEAR_MAX
) -> DataFrame:
    """Restrict KVAD ambulatory diagnoses to the pre-pandemic window.

    The window is calendar-based (``_KVAD_YEAR_MIN`` ≤ year ≤ ``year_max``).
    A per-participant "pre-baseline-interview" window would be more precise
    but requires the interview-date column from baseline_sex_age, which is
    not always linked 1:1 with KVAD; the calendar cutoff keeps the join
    graph minimal. Rows with a missing ``kvad_year`` are dropped (logged
    separately) because ``Series.between`` evaluates to ``False`` on NaN
    and silent attrition would mask data-quality regressions in KVAD
    deliveries.
    """
    n0 = len(diag)
    n_year_missing = int(diag["kvad_year"].isna().sum())
    if n_year_missing:
        logger.info(
            "KVAD pre-pandemic filter: dropping %d row(s) with missing kvad_year",
            n_year_missing,
        )
    in_window = diag["kvad_year"].between(_KVAD_YEAR_MIN, year_max, inclusive="both")
    out = diag[in_window].copy()
    logger.info(
        "KVAD pre-pandemic filter (year in [%d, %d]): %d -> %d rows",
        _KVAD_YEAR_MIN,
        year_max,
        n0,
        len(out),
    )
    return out


def _filter_certainty(
    diag: DataFrame, *, drop_codes: frozenset[int] = _CERTAINTY_DROP_DEFAULT
) -> DataFrame:
    """Drop diagnoses with disallowed certainty codes.

    The default *drop_codes* removes only ``ausgeschlossen`` (3) and the
    explicit-missing sentinel (-9). Pass ``_CERTAINTY_DROP_STRICT`` to
    additionally drop ``Verdacht`` (2) and ``Zustand nach`` (4) for the
    GCAT-aligned strict variant.
    """
    n0 = len(diag)
    out = diag[~diag["kvad_amb_diag_sure"].isin(list(drop_codes))].copy()
    logger.info(
        "KVAD certainty filter (drop %s): %d -> %d rows",
        sorted(drop_codes),
        n0,
        len(out),
    )
    return out


def _restrict_mapping_to_chronic(
    mapping: dict[str, tuple[str, ...]],
    *,
    chronic_codes: frozenset[str] = _ICD3_CHRONIC,
) -> dict[str, tuple[str, ...]]:
    """Drop non-chronic ICD-3 codes from each Bahmer item's mapping.

    Items whose mapping becomes empty (all codes acute) are retained with
    an empty tuple so they still appear as zero-valued indicators in the
    output -- this preserves a stable column schema across variants and
    makes "no chronic ICD signal at T0" explicit rather than a missing
    column.
    """
    return {
        symptom: tuple(c for c in codes if c in chronic_codes)
        for symptom, codes in mapping.items()
    }


def _aggregate_to_bahmer_indicators(
    diag: DataFrame, *, mapping: dict[str, tuple[str, ...]]
) -> DataFrame:
    """Per-participant binary indicators per Bahmer symptom + the sum.

    Returns one row per participant with columns:

    - ``ID``                            uint32
    - ``pcs_t0_proxy_<symptom>``        Int8 (0/1) per Bahmer item
    - ``pcs_t0_proxy_count``            UInt8 (0..21) sum of indicators
    - ``pcs_t0_proxy_n_diagnoses``      UInt32 raw row count (audit trail)

    The output frame contains every participant in ``diag`` — including
    those with KVAD claims but **no** Bahmer-mapped ICD code (their
    indicators are all 0, count = 0). The proxy is a *floor* on somatic-
    symptom load (see module docstring), so an explicit zero is the
    correct signal — silently dropping zero-score participants would
    restrict M1 to the symptomatic sub-cohort and leave the healthy
    controls indistinguishable from "no KVAD link at all".
    """
    # Index over every participant present in the (already filtered) diag
    # frame — this ensures zero-score participants get an explicit row
    # rather than being silently dropped. Drop NA IDs defensively.
    all_ids = pd.Index(
        sorted(diag["ID"].dropna().unique()),
        name="ID",
    )
    out = pd.DataFrame(index=all_ids)

    # Build per-symptom binary hit columns over the full participant index.
    # Each column is 1 iff the participant has ≥1 diagnosis in the
    # symptom's ICD set, 0 otherwise (NaN never occurs here because
    # reindex(fill_value=0) over the full participant index).
    for symptom, icd3_codes in mapping.items():
        mask = diag["kvad_a_amb_diag_3"].isin(icd3_codes)
        hit_ids = diag.loc[mask, "ID"].drop_duplicates()
        col = f"pcs_t0_proxy_{symptom.removeprefix('symptom_')}"
        out[col] = (
            pd.Series(1, index=pd.Index(hit_ids, name="ID"))
            .reindex(out.index, fill_value=0)
            .astype("Int8")
        )

    out["pcs_t0_proxy_count"] = out.sum(axis=1).astype("UInt8")
    out["pcs_t0_proxy_n_diagnoses"] = (
        diag.groupby("ID").size().reindex(out.index, fill_value=0).astype("UInt32")
    )
    return out.reset_index()


_VARIANT_PARQUET_NAME: dict[ProxyVariant, str] = {
    "default": "kvad_pcs_proxy",
    "chronic": "kvad_pcs_proxy_chronic",
    "strict": "kvad_pcs_proxy_strict",
}


def process_kvad_pcs_proxy(
    paths: NAKOPaths,
    output_dir: Path,
    *,
    diag_df: DataFrame | None = None,
    year_max: int = _PRE_PANDEMIC_YEAR_MAX,
    variant: ProxyVariant = "default",
) -> Path:
    """Build the pre-pandemic PCS-equivalent baseline proxy parquet.

    Reads ``exportfile_kv_amb_diag_mapped.csv``, restricts to the pre-
    pandemic window, drops "ausgeschlossen" diagnoses (and -- in the
    ``"strict"`` variant -- ``Verdacht`` / ``Zustand nach`` as well),
    aggregates ICD-3 codes into Bahmer-symptom indicators, and writes
    ``kvad_pcs_proxy[_<variant>].parquet``.

    Parameters
    ----------
    variant
        ``"default"`` (used by the main analyses),
        ``"chronic"`` (CCI-2023-aligned chronic-only ICD set, GCAT-style),
        or ``"strict"`` (only ``gesichert`` certainty).

    See module docstring for the methodological rationale and caveats.
    """
    if variant not in _VARIANT_PARQUET_NAME:
        raise ValueError(
            f"variant must be one of {sorted(_VARIANT_PARQUET_NAME)}; got {variant!r}"
        )
    logger.info("Processing KVAD pre-pandemic PCS proxy (variant=%s) ...", variant)
    diag = diag_df if diag_df is not None else _load_nako_csv(paths.kvad_csv)
    diag = _filter_pre_pandemic(diag, year_max=year_max)
    drop_codes = (
        _CERTAINTY_DROP_STRICT if variant == "strict" else _CERTAINTY_DROP_DEFAULT
    )
    diag = _filter_certainty(diag, drop_codes=drop_codes)
    mapping = (
        _restrict_mapping_to_chronic(BAHMER_TO_ICD3)
        if variant == "chronic"
        else BAHMER_TO_ICD3
    )
    out = _aggregate_to_bahmer_indicators(diag, mapping=mapping)
    return _save_parquet(out, _VARIANT_PARQUET_NAME[variant], output_dir)


def process_kvad_pcs_proxy_all_variants(
    paths: NAKOPaths,
    output_dir: Path,
    *,
    diag_df: DataFrame | None = None,
    year_max: int = _PRE_PANDEMIC_YEAR_MAX,
) -> dict[ProxyVariant, Path]:
    """Emit all three sensitivity variants from the same source frame.

    Loads the KVAD CSV at most once (when *diag_df* is None) and reuses
    the in-memory frame across variants. Returns a mapping from variant
    name to the written parquet path so the downstream convergence
    validator can pick them up.
    """
    diag = diag_df if diag_df is not None else _load_nako_csv(paths.kvad_csv)
    variants: tuple[ProxyVariant, ...] = ("default", "chronic", "strict")
    return {
        variant: process_kvad_pcs_proxy(
            paths,
            output_dir,
            diag_df=diag,
            year_max=year_max,
            variant=variant,
        )
        for variant in variants
    }
