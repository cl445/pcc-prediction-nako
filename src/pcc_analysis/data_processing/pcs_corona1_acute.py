"""Corona-1 acute self-reported symptom inventory (cov87s) -> PCS-T1 proxy.

In the Corona-1 questionnaire (May 2020) NAKO administered an
identification-only checklist of 13 COVID-suspect symptoms (question
COV87s, items ``d_co1_cov87s_1..13``):

    1  Abgeschlagenheit       fatigue
    2  Atemprobleme           breathing problems
    3  Kopfschmerzen          headache
    4  Übelkeit               nausea
    5  Fieber                 fever
    6  Schüttelfrost          chills
    7  Gliederschmerzen       joint/muscle pain
    8  Husten                 cough
    9  Schnupfen              runny nose
    10 Durchfall              diarrhoea
    11 Riechstörung           loss of smell
    12 Geschmacksstörung      loss of taste
    13 Andere                 other

Each item is a binary 0/1 indicator (``AusNichtAus``: 0=not selected,
1=selected). Coverage is broad: ~92.5 % of Corona-1 participants
answered the block. Item endorsement is sparse (0.2-1.2 % per item),
consistent with the pandemic phase (May 2020) and the population-based
sample.

Items 1-12 map cleanly to Bahmer T2 symptom items (~10 of 12 have a
direct Bahmer pendant; see the rename map below). Item 13 ("Andere")
is kept as a flag but excluded from the sum.

The output parquet ``pcs_corona1_acute.parquet`` lives alongside the
mh_longitudinal panel and provides the T1 PCS wave
in the 3-wave RI-CLPM with the KVAD-derived T0 PCS proxy
[``kvad_pcs_proxy.parquet``] and the Bahmer T2 PCS items
[``corona2_pcc.parquet``]).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

from pcc_analysis.data_processing._common import _load_nako_csv, _save_parquet
from pcc_analysis.data_processing._likert import _PSYCHOMETRIC_MISSING
from pcc_analysis.data_processing.corona import nako_corona1_csv

if TYPE_CHECKING:
    from pcc_analysis.config import NAKOPaths

logger = logging.getLogger(__name__)


# Map cov87s_<n> -> canonical Bahmer-aligned symptom name. The names mirror
# the Bahmer ``symptom_*`` columns in ``corona2_pcc.parquet`` so M1's
# wave-pivot can use a single rename map across all three PCS waves.
COV87S_TO_BAHMER_NAME: dict[int, str] = {
    1: "fatigue",  # Abgeschlagenheit
    2: "breathing_problems",  # Atemprobleme
    3: "headache",  # Kopfschmerzen
    4: "loss_of_appetite",  # Übelkeit -> appetite proxy
    5: "fever",  # Fieber
    6: "sweating",  # Schüttelfrost -> sweating/chills proxy
    7: "joint_muscle_pain",  # Gliederschmerzen
    8: "cough",  # Husten
    9: "runny_nose",  # Schnupfen
    10: "gastrointestinal_problems",  # Durchfall -> GI proxy
    11: "loss_of_smell",  # Riechstörung
    12: "loss_of_taste",  # Geschmacksstörung
    # 13 = "Andere" -> kept as a flag but excluded from the sum.
}


def _read_corona1_block(corona1_df: DataFrame) -> DataFrame:
    """Pull cov87s_1..13 from the Corona-1 raw CSV and coerce to nullable Int8."""
    cols = [f"d_co1_cov87s_{i}" for i in range(1, 14)]
    present = [c for c in cols if c in corona1_df.columns]
    if not present:
        msg = (
            "Corona-1 raw data does not contain any d_co1_cov87s_* columns; "
            "make sure the file passed in is export_corona.csv (Q10a symptom block)."
        )
        raise ValueError(msg)
    out = corona1_df[["ID", *present]].copy()
    # Sentinel codes (7775..9999) become NA.
    out[present] = out[present].replace(_PSYCHOMETRIC_MISSING, pd.NA).astype("Int8")
    return out


_COV87S_MAX_MISSING: int = 2
"""Tolerance for missing items when aggregating ``pcs_t1_acute_count``.

Up to two NA items are permitted; the sum is then computed from the
observed items and treated as a count. Beyond two missings the row's
sum is set to NA. With ~92.5 % full-block coverage on Corona-1 this
keeps the vast majority of partial responders in the analytic frame
instead of nulling them out at the first sentinel.

Deliberately *not* the complete-case rule the psychometric sums follow
(:func:`pcc_analysis.data_processing._likert._score_likert_questionnaire`,
``max_missing=0``, per Streit et al. 2023). That rule exists because a PHQ-9
or GAD-7 sum is a validated instrument score whose interpretation depends on
all nine or seven items being answered. This is a symptom count over an
unweighted checklist: a participant who ticked four of twelve and skipped two
has at least four acute symptoms whether or not the last two were answered,
so the tolerance costs no construct validity and keeps a stratum that
complete-case scoring would drop.
"""


def _aggregate_cov87s(raw: DataFrame) -> DataFrame:
    """Rename items, derive sum + answered-count, finalise dtypes."""
    rename = {
        f"d_co1_cov87s_{idx}": f"pcs_t1_acute_{name}"
        for idx, name in COV87S_TO_BAHMER_NAME.items()
    }
    rename["d_co1_cov87s_13"] = "pcs_t1_acute_other_flag"
    out = raw.rename(columns=rename)
    bahmer_cols = [f"pcs_t1_acute_{n}" for n in COV87S_TO_BAHMER_NAME.values()]
    present_bahmer = [c for c in bahmer_cols if c in out.columns]
    # Sum across the 12 Bahmer-mapped items (Andere excluded). Up to
    # ``_COV87S_MAX_MISSING`` NA items are tolerated (see that constant for
    # why a checklist count may, where a psychometric sum may not); beyond
    # that the sum is set to NA so partial responders don't bias the count
    # downward. ``min_count`` is therefore
    # ``len(present_bahmer) - _COV87S_MAX_MISSING`` so pandas only emits a
    # numeric sum when at least that many items are observed.
    min_observed = max(1, len(present_bahmer) - _COV87S_MAX_MISSING)
    out["pcs_t1_acute_count"] = (
        out[present_bahmer].sum(axis=1, min_count=min_observed).astype("Int8")
    )
    out["pcs_t1_acute_n_answered"] = (
        out[present_bahmer].notna().sum(axis=1).astype("UInt8")
    )
    return out


def process_pcs_corona1_acute(
    paths: NAKOPaths,
    output_dir: Path,
    *,
    corona1_df: DataFrame | None = None,
) -> Path:
    """Build the Corona-1 acute symptom inventory parquet.

    See module docstring for the rationale. Output:
    ``pcs_corona1_acute.parquet``.
    """
    logger.info("Processing Corona-1 acute symptom inventory (cov87s) ...")
    raw = (
        corona1_df
        if corona1_df is not None
        else _load_nako_csv(nako_corona1_csv(paths))
    )
    block = _read_corona1_block(raw)
    out = _aggregate_cov87s(block)
    return _save_parquet(out, "pcs_corona1_acute", output_dir)
