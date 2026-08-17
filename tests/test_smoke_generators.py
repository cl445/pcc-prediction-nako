"""The synthetic acute-course fixture must obey the extractor's recodes.

``acute_infection.parquet`` has a single consumer,
``scripts/supplementary/severity_interaction.py``, and under ``--smoke`` that
script reads what :mod:`smoke_test.generators` writes. The run is worth doing
only if a broken severity recode would surface in it, which it cannot while
the fixture itself contains rows whose care level and severity flags
contradict each other.

The rules under test are the extractor's, from
``pcc_analysis.data_processing.acute_infection``: hospitalisation is
``acute_care_level >= 2``, intensive care is ``acute_care_level == 3``, and a
day count below the level that would make it a measurement is a structural
zero. An unknown level propagates: it leaves every column keyed to it
unknown rather than guessing at a reference category.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
from smoke_test._profile_types import Profile
from smoke_test.generators import generate_all, load_profile

PROFILE_PATH = Path(__file__).resolve().parents[1] / "smoke_test/profiles/default.json"

HOSPITALISED_LEVEL = 2
ICU_LEVEL = 3

# Enough participants for the rare upper levels to appear at their profiled
# rates: hospitalisation runs at 0.7 % of the ~56 % whose level is known.
N_SUBJECTS = 4000
SEED = 7


def _acute_only(profile: Profile) -> Profile:
    """The profile reduced to the acute-course modality.

    ``generate_all`` walks every modality it is given, and the MRI stack
    alone is several hundred columns; the modalities are independent, so
    dropping them costs the test nothing and keeps the draw sequence for
    this one fixed regardless of what the rest of the profile does.
    """
    return Profile(
        meta=profile["meta"],
        modalities={"acute_infection": profile["modalities"]["acute_infection"]},
    )


@pytest.fixture(scope="module")
def acute() -> pd.DataFrame:
    profile = _acute_only(load_profile(PROFILE_PATH))
    return generate_all(profile, n_subjects=N_SUBJECTS, seed=SEED)["acute_infection"]


def test_severity_flags_are_thresholds_of_the_care_level(acute: pd.DataFrame) -> None:
    known = acute[acute["acute_care_level"].notna()]
    assert not known.empty
    level = known["acute_care_level"]
    assert (known["acute_hospitalised"] == (level >= HOSPITALISED_LEVEL)).all()
    assert (known["acute_icu"] == (level == ICU_LEVEL)).all()


def test_an_unknown_care_level_leaves_everything_keyed_to_it_unknown(
    acute: pd.DataFrame,
) -> None:
    unknown = acute["acute_care_level"].isna()
    assert unknown.any()
    for column in ("acute_hospitalised", "acute_icu", "hospital_days", "icu_days"):
        assert acute.loc[unknown, column].isna().all()


def test_day_counts_are_zero_below_the_level_that_measures_them(
    acute: pd.DataFrame,
) -> None:
    level = acute["acute_care_level"]
    for column, admitted_from in (
        ("hospital_days", HOSPITALISED_LEVEL),
        ("icu_days", ICU_LEVEL),
    ):
        never_admitted = (level < admitted_from).fillna(False)
        assert never_admitted.any()
        assert (acute.loc[never_admitted, column] == 0).all()


def test_the_care_level_reaches_the_hospital_rate_the_profile_records(
    acute: pd.DataFrame,
) -> None:
    """The upper levels sit above the 99th percentile the draws clip to.

    A level sampled from its own summary alone stops at 1, so every column
    thresholded on it is constant and the fixture contradicts the hospital
    rate the profile states in every row.
    """
    profiled = load_profile(PROFILE_PATH)["modalities"]["acute_infection"]["columns"]
    hospitalised = profiled["acute_hospitalised"]
    assert hospitalised["dtype"] == "boolean"
    rate = float(acute["acute_hospitalised"].dropna().mean())
    assert 0.0 < rate < 3 * hospitalised["true_frac"]
    assert (acute["acute_care_level"] == ICU_LEVEL).any()


def test_summaries_that_cannot_describe_one_variable_are_rejected() -> None:
    """The three summaries are one variable seen three ways, so they can
    disagree — and a profile that says more participants reached intensive
    care than reached a hospital ward describes no distribution over levels.
    Clamping it back into one would reinstate exactly the silent
    inconsistency the derivation removes."""
    profile = copy.deepcopy(_acute_only(load_profile(PROFILE_PATH)))
    columns = profile["modalities"]["acute_infection"]["columns"]
    icu = columns["acute_icu"]
    hospitalised = columns["acute_hospitalised"]
    assert icu["dtype"] == "boolean"
    assert hospitalised["dtype"] == "boolean"
    icu["true_frac"] = hospitalised["true_frac"] * 2

    with pytest.raises(ValueError, match="acute_icu true_frac"):
        generate_all(profile, n_subjects=100, seed=SEED)
