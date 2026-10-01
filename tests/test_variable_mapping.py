"""Pin the raw NAKO variable behind each extracted column that was once wrong.

Until 2026-09-28 several extractors gave a NAKO variable a name it does not
have, and two derived features rested on that name (DECISIONS §2.34). Each
test feeds a one-row raw frame in which only the variable under test is set
and checks that the value lands in the column that means the same thing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcc_analysis.data_processing.cognitive import _extract_cognitive_tests
from pcc_analysis.data_processing.medical_history import (
    _derive_medical_metrics,
    _extract_medical_history,
)
from pcc_analysis.data_processing.ses import _derive_ses_metrics, _extract_ses
from pcc_analysis.preprocessing import (
    create_cognitive_preprocessor,
    create_ses_preprocessor,
)

_MEDICAL_RAW = [
    "a_an_cvht_age", "a_anthro_bmi_eig", "a_anthro_gew_eig", "a_anthro_groe_eig",
    "a_antidiabetika", "a_antiepilept", "a_antihypertens", "a_antiparkinson",
    "a_betablock", "a_gerinnung", "a_hte_ad_cur", "a_hte_ad_onset",
    "a_hte_ad_onset_cat", "a_hte_psa_cur", "a_hte_pso_cur", "a_hte_pso_onset",
    "a_inf_age_hepb", "a_inf_age_hepc", "a_inf_age_hiv", "a_inf_age_seps",
    "a_inf_age_tub", "a_inf_age_zos", "a_kre_1_alter", "a_kre_1_nc",
    "a_kre_2_alter", "a_kre_3_alter", "a_kre_4_alter", "a_lipidsenkend",
    "a_op_alg1a_age", "a_op_alg1b_age", "a_op_alg1c_age", "a_op_alg1d_age",
    "a_op_alg1e_age", "a_op_alg1f_age", "a_op_alg1g_age", "a_op_entf1g_age",
    "d_an_cv_6",
]  # fmt: skip

_COGNITIVE_RAW = [
    "a_npsy_difts2_21", "a_npsy_difts2_23", "a_npsy_difts3_32", "a_npsy_sumts1",
    "a_npsy_sumts2_1", "a_npsy_sumts2_2", "a_npsy_sumts2_3", "a_npsy_sumts3_2",
    "a_npsy_sumts3_3", "a_npsy_sumts4", "a_npsy_sumts5", "a_npsy_sumts6",
]  # fmt: skip

_SES_RAW = [
    "a_isco_code", "a_isco_major", "a_isco_minor", "a_isco_skill",
    "a_isco_submajor", "a_isei", "a_kldb_anf", "a_kldb_code", "a_kldb_fuehr",
    "a_kldb_major", "a_kldb_seg", "a_kldb_sek", "a_nempl", "a_selfemp",
    "a_ses_bedarfgw", "a_ses_beruf", "a_ses_child", "a_ses_deutsch",
    "a_ses_el_datum", "a_ses_el_gesamt", "a_ses_el_seit_j", "a_ses_ewstat",
    "a_ses_famst", "a_ses_househ", "a_ses_inc", "a_ses_incgw", "a_ses_incpos",
    "a_ses_isced97_level", "a_ses_isced97_years", "a_ses_partner",
    "a_ses_rentenalter", "a_ses_workh", "a_siops",
]  # fmt: skip


def _raw(columns: list[str], **values: object) -> pd.DataFrame:
    row: dict[str, object] = {"ID": 1, **dict.fromkeys(columns, np.nan)}
    row.update(values)
    return pd.DataFrame([row])


def test_hypertension_comes_from_the_doctor_diagnosis_item() -> None:
    raw = _raw(_MEDICAL_RAW, d_an_cv_6=1, a_an_cvht_age=52, a_hte_ad_cur=2)
    out = _extract_medical_history(raw)
    assert bool(out.loc[0, "has_hypertension"]) is True
    assert out.loc[0, "hypertension_age_onset"] == 52
    assert bool(out.loc[0, "has_atopic_eczema"]) is False


def test_atopic_eczema_is_not_hypertension() -> None:
    raw = _raw(_MEDICAL_RAW, d_an_cv_6=2, a_hte_ad_cur=1, a_hte_ad_onset=12)
    out = _extract_medical_history(raw)
    assert bool(out.loc[0, "has_atopic_eczema"]) is True
    assert out.loc[0, "atopic_eczema_age_onset"] == 12
    assert bool(out.loc[0, "has_hypertension"]) is False


def test_cardiovascular_procedures_are_counted_as_such() -> None:
    raw = _raw(_MEDICAL_RAW, a_op_alg1b_age=60, a_op_alg1d_age=65)
    out = _derive_medical_metrics(_extract_medical_history(raw))
    assert out.loc[0, "cv_procedure_ptca_age"] == 60
    assert out.loc[0, "cv_procedure_pacemaker_age"] == 65
    assert out.loc[0, "number_cv_procedures"] == 2
    assert not any(c.startswith("surgery_") for c in out.columns)


def test_pegboard_and_number_series_are_told_apart() -> None:
    raw = _raw(_COGNITIVE_RAW, a_npsy_sumts5=14, a_npsy_sumts6=502.9)
    out = _extract_cognitive_tests(raw)
    assert out.loc[0, "number_series_ability"] == 502.9
    assert out.loc[0, "purdue_pegboard_pairs"] == 14
    drops = create_cognitive_preprocessor().drop_columns or []
    assert "number_series_ability" in drops
    assert "purdue_pegboard_pairs" not in drops


def test_equivalised_income_is_not_divided_twice() -> None:
    raw = _raw(_SES_RAW, a_ses_inc=3050, a_ses_bedarfgw=1.5, a_ses_incgw=2033.3)
    out = _derive_ses_metrics(_extract_ses(raw))
    assert out.loc[0, "equivalised_income"] == 2033.3
    assert "income_adequacy" not in out.columns
    drops = create_ses_preprocessor().drop_columns or []
    assert "equivalised_income" not in drops


def test_unemployment_columns_carry_their_meaning() -> None:
    raw = _raw(_SES_RAW, a_ses_el_seit_j=2, a_ses_el_gesamt=5, a_ses_deutsch=1)
    out = _extract_ses(raw)
    assert out.loc[0, "unemployment_duration_current_years"] == 2
    assert out.loc[0, "unemployment_duration_total_years"] == 5
    assert out.loc[0, "german_language_proficiency"] == 1
