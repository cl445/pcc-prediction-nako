"""Smoke tests for ModalityPipelineFactory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pcc_analysis.custom_pipeline import ConfoundedPipeline
from pcc_analysis.modality_pipelines import ModalityPipelineFactory
from pcc_analysis.orthogonalization import Orthogonalizer
from pcc_analysis.stability_selection import StabilitySelector


@pytest.fixture
def factory() -> ModalityPipelineFactory:
    return ModalityPipelineFactory(
        random_state=42, orthogonalize=True, cv=3, n_jobs=1, n_subsamples=10
    )


@pytest.fixture
def factory_no_ortho() -> ModalityPipelineFactory:
    return ModalityPipelineFactory(
        random_state=42, orthogonalize=False, cv=3, n_jobs=1, n_subsamples=10
    )


# -- Helper ------------------------------------------------------------------


def _step_names(pipeline: Pipeline) -> list[str]:
    return [name for name, _ in pipeline.steps]


def _has_step_type(pipeline: Pipeline, cls: type) -> bool:
    return any(isinstance(step, cls) for _, step in pipeline.steps)


# -- Demographics (no orthogonalization, no stability selection) --------------


def test_demographics_pipeline_structure(factory: ModalityPipelineFactory) -> None:
    pipe = factory.create_demographics_pipeline()
    assert isinstance(pipe, ConfoundedPipeline)
    names = _step_names(pipe)
    assert "preprocessor" in names
    assert "scaler" in names
    assert "classifier" in names
    # Demographics should never have orthogonalization
    assert not _has_step_type(pipe, Orthogonalizer)


# -- SES ---------------------------------------------------------------------


def test_ses_pipeline_structure(factory: ModalityPipelineFactory) -> None:
    pipe = factory.create_ses_pipeline()
    assert isinstance(pipe, ConfoundedPipeline)
    names = _step_names(pipe)
    assert "preprocessor" in names
    assert "missingness" in names
    assert "imputer" in names
    assert "orthogonalizer" in names
    assert "classifier" in names


def test_ses_pipeline_no_ortho(factory_no_ortho: ModalityPipelineFactory) -> None:
    pipe = factory_no_ortho.create_ses_pipeline()
    assert not _has_step_type(pipe, Orthogonalizer)


# -- Cognitive ---------------------------------------------------------------


def test_cognitive_pipeline_has_stability(factory: ModalityPipelineFactory) -> None:
    pipe = factory.create_cognitive_pipeline()
    assert _has_step_type(pipe, StabilitySelector)
    assert _has_step_type(pipe, Orthogonalizer)


# -- Physical Activity -------------------------------------------------------


def test_physical_activity_pipeline_has_stability(
    factory: ModalityPipelineFactory,
) -> None:
    pipe = factory.create_physical_activity_pipeline()
    assert _has_step_type(pipe, StabilitySelector)


# -- Medical History ---------------------------------------------------------


def test_medical_history_pipeline_has_stability(
    factory: ModalityPipelineFactory,
) -> None:
    pipe = factory.create_medical_history_pipeline()
    assert _has_step_type(pipe, StabilitySelector)


# -- Lab Values --------------------------------------------------------------


def test_lab_values_pipeline_has_stability(factory: ModalityPipelineFactory) -> None:
    pipe = factory.create_lab_values_pipeline()
    assert _has_step_type(pipe, StabilitySelector)


# -- Cardiovascular (no preprocessor, no stability selection) ----------------


def test_cardiovascular_pipeline_structure(factory: ModalityPipelineFactory) -> None:
    pipe = factory.create_cardiovascular_pipeline()
    names = _step_names(pipe)
    assert "preprocessor" not in names
    assert "missingness" in names
    assert not _has_step_type(pipe, StabilitySelector)


# -- Lung Function -----------------------------------------------------------


def test_lung_function_pipeline_no_stability(
    factory: ModalityPipelineFactory,
) -> None:
    pipe = factory.create_lung_function_pipeline()
    assert not _has_step_type(pipe, StabilitySelector)
    assert "preprocessor" in _step_names(pipe)


# -- MRI pipelines -----------------------------------------------------------


@pytest.mark.parametrize(
    "atlas",
    ["desikan", "destrieux", "julich", "yeo", "subcortical", "cerebellar"],
)
def test_mri_pipeline_valid_atlases(
    factory: ModalityPipelineFactory, atlas: str
) -> None:
    pipe = factory.create_mri_pipeline(atlas)
    assert isinstance(pipe, ConfoundedPipeline)
    names = _step_names(pipe)
    assert "etiv_confounder" in names
    assert "classifier" in names


@pytest.mark.parametrize(
    "atlas", ["desikan", "destrieux", "julich", "yeo", "subcortical"]
)
def test_mri_pipeline_pca_atlases(factory: ModalityPipelineFactory, atlas: str) -> None:
    pipe = factory.create_mri_pipeline(atlas)
    assert "pca" in _step_names(pipe)


def test_mri_cerebellar_no_pca(factory: ModalityPipelineFactory) -> None:
    pipe = factory.create_mri_pipeline("cerebellar")
    assert "pca" not in _step_names(pipe)


def test_mri_unknown_atlas_raises(factory: ModalityPipelineFactory) -> None:
    with pytest.raises(ValueError, match="Unknown atlas"):
        factory.create_mri_pipeline("nonexistent")


# -- create_all_pipelines ----------------------------------------------------


def test_create_all_pipelines_count(factory: ModalityPipelineFactory) -> None:
    all_pipes = factory.create_all_pipelines()
    # 9 non-MRI + 6 MRI atlases = 15
    assert len(all_pipes) == 15
    assert "demographics" in all_pipes
    assert "mental_health" in all_pipes
    assert "mri_desikan" in all_pipes
    assert "mri_cerebellar" in all_pipes


# -- Fit smoke test (demographics, simplest pipeline) ------------------------


def test_demographics_pipeline_fit_predict(
    factory: ModalityPipelineFactory,
) -> None:
    """Demographics pipeline should fit and predict on dummy data."""
    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame(
        {
            "basis_sex": rng.choice(["male", "female"], n),
            "basis_uort": rng.choice(["urban", "rural"], n),
            "age": rng.uniform(20, 80, n),
        }
    )
    y = np.zeros(n, dtype=np.float64)
    y[rng.choice(n, size=25, replace=False)] = 1.0

    pipe = factory.create_demographics_pipeline()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.all((proba >= 0) & (proba <= 1))


# -- Fit smoke test (cardiovascular, no preprocessor) ------------------------


def test_cardiovascular_pipeline_fit_predict(
    factory: ModalityPipelineFactory,
) -> None:
    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame(rng.standard_normal((n, 7)), columns=[f"cv_{i}" for i in range(7)])
    y = np.zeros(n, dtype=np.float64)
    y[rng.choice(n, size=25, replace=False)] = 1.0

    confounders = pd.DataFrame(
        {"age": rng.uniform(20, 80, n), "sex": rng.choice([0.0, 1.0], n)}
    )

    pipe = factory.create_cardiovascular_pipeline()
    pipe.fit(X, y, confounders=confounders)
    proba = pipe.predict_proba(X, confounders=confounders)
    assert proba.shape == (n, 2)
