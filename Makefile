SOURCES := src scripts tests smoke_test

# Supplementary analyses that read only the processed parquets. The
# transport/clinical-utility/sub-modality panels are not here because they
# consume specific pipeline run directories — `make analysis` runs those.
# This list is the same set, in the same order, as the `supp` block in
# scripts/run_analysis.sh.
PARQUET_SUPPLEMENTARY := \
	outcome_agreement \
	cohort_comparison \
	evalue \
	sex_interaction \
	severity_interaction \
	smoking_adjustment \
	persistence_adjustment \
	symptom_decomposition \
	mh_trajectory_adjustment \
	mri_positive_control \
	inclusion_comparison \
	reporting_style_robustness

# The two of those the synthetic fixtures cannot exercise:
#   persistence_adjustment      reads the raw NAKO export directly rather than
#                               the processed parquets, so no fixture reaches it
#   reporting_style_robustness  fits a design that is singular at the fixture's
#                               size (~180 rows, ~15 positives); that is the
#                               fixture's fault, not the script's
#
# Membership here depends on the fixtures, not only on the code: a script can
# leave this list because the generated data changed under it. Derive the set
# from an actual `--smoke` run rather than editing it from memory -- the run
# prints what it skipped and what failed, and those two lists are the answer.
FIXTURE_CANNOT_RUN := persistence_adjustment reporting_style_robustness
SMOKE_SUPPLEMENTARY := $(filter-out $(FIXTURE_CANNOT_RUN),$(PARQUET_SUPPLEMENTARY))

.PHONY: all analysis analysis-smoke check-runs data pipeline figures statistics supplementary \
        lint format format-check fix test typecheck check check-decision-refs \
        smoke-data smoke-test smoke-test-mri-lean smoke-test-nonmri-lean \
        smoke-test-transfer-nonmri-lean smoke-test-all smoke-supplementary clean

# -- Full analysis -----------------------------------------------------------

# Run the complete analysis: data -> 10 pipeline runs -> transfer -> figures
# + statistics -> supplementary analyses. Writes every paper constant and
# table under results/paper/. Takes days, and splits across two machines with
# --only/--except (see scripts/run_analysis.sh). Re-running it on the same
# data reproduces its results.
analysis:
	./scripts/run_analysis.sh

# Are the run directories under RUNS_DIR comparable to each other? Run after
# collecting the halves of a two-machine rerun, before any number moves into
# the manuscript. Non-zero exit means a delta across them would measure the
# environment as well as the analysis. RUNS_DIR is overridable so the same
# target reaches a smoke run: `make check-runs RUNS_DIR=smoke_results/runs`.
RUNS_DIR ?= results/runs
check-runs:
	uv run python scripts/check_run_stacks.py --runs-dir $(RUNS_DIR)

# Same graph as `analysis`, but on generated synthetic data (config_smoke.toml,
# output to smoke_data/ and smoke_results/). Fast end-to-end wiring check; the
# numbers are not scientifically meaningful.
analysis-smoke:
	./scripts/run_analysis.sh --smoke

# Core pipeline only (primary run + figures + statistics).
all: data pipeline figures statistics

data:
	uv run python scripts/pipeline/01_process_data.py

pipeline:
	uv run python scripts/pipeline/02_run_pipeline.py

# --results-dir is explicit because the script's own default globs for
# timestamped pcc_pipeline_* directories, while `make analysis` writes the
# deterministic run names under results/runs/.
figures:
	uv run python scripts/pipeline/04_generate_figures.py --results-dir results/runs/primary

statistics:
	uv run python scripts/pipeline/05_compute_statistics.py

# Every supplementary analysis that needs no pipeline run directory; see
# PARQUET_SUPPLEMENTARY at the top of this file.
supplementary:
	@for s in $(PARQUET_SUPPLEMENTARY); do \
	    echo "==> $$s"; \
	    uv run python scripts/supplementary/$$s.py || exit 1; \
	done

# -- Code quality ------------------------------------------------------------

lint:
	uv run ruff check $(SOURCES)

format:
	uv run ruff format $(SOURCES)

format-check:
	uv run ruff format --check $(SOURCES)

fix:
	uv run ruff check --fix $(SOURCES)
	uv run ruff format $(SOURCES)

test:
	uv run python -m pytest tests

typecheck:
	uv run pyrefly check

# What CI runs, in CI's order. format-check rather than format: `make check`
# reports, `make fix` and the pre-commit hook repair.
# Every `§N.M` cited anywhere in the tree names a section DECISIONS.md has.
# The citations are plain strings, so nothing else connects them to the
# document: renumbering it silently repoints all of them, and a citation that
# was wrong from the start looks exactly like one that is right.
check-decision-refs:
	uv run python scripts/check_decision_refs.py

check: lint format-check typecheck check-decision-refs test

# -- Smoke test --------------------------------------------------------------
# End-to-end checks on synthetic data (see smoke_test/). Fast, not
# scientifically meaningful; output goes to smoke_data/ and smoke_results/.

smoke-data:
	uv run python -m smoke_test generate --output-dir smoke_data

# Default smoke pipeline: implicit cohort=all + stack=full (coincides with
# cohort=mri + stack=full on the thinned smoke fixtures).
smoke-test: smoke-data
	uv run python scripts/pipeline/02_run_pipeline.py --config config_smoke.toml

# Lean Universal Classifier in each cohort arm.
smoke-test-mri-lean: smoke-data
	uv run python scripts/pipeline/02_run_pipeline.py --config config_smoke.toml \
	    --cohort mri --stack lean

smoke-test-nonmri-lean: smoke-data
	uv run python scripts/pipeline/02_run_pipeline.py --config config_smoke.toml \
	    --cohort non_mri --stack lean

# Transfer-validation smoke: train Lean-MRI, then apply the pickled model to
# the non-MRI sample without refit.
smoke-test-transfer-nonmri-lean: smoke-data
	uv run python scripts/pipeline/02_run_pipeline.py --config config_smoke.toml \
	    --cohort mri --stack lean
	uv run python scripts/pipeline/03_apply_transfer.py --config config_smoke.toml \
	    --from-run "$$(ls -d smoke_results/pcc_pipeline_mri_lean_* | tail -1)" \
	    --target-cohort non_mri --n-bootstrap 50

# Full smoke matrix: primary (full stack) + both Lean cohorts + transfer
# validation. Reuses a single generated fixture.
smoke-test-all: smoke-data
	uv run python scripts/pipeline/02_run_pipeline.py --config config_smoke.toml
	uv run python scripts/pipeline/02_run_pipeline.py --config config_smoke.toml \
	    --cohort mri --stack lean
	uv run python scripts/pipeline/02_run_pipeline.py --config config_smoke.toml \
	    --cohort non_mri --stack lean
	uv run python scripts/pipeline/03_apply_transfer.py --config config_smoke.toml \
	    --from-run "$$(ls -d smoke_results/pcc_pipeline_mri_lean_* | tail -1)" \
	    --target-cohort non_mri --n-bootstrap 50

# The supplementary analyses against the synthetic fixtures. `make
# supplementary` needs NAKO parquets and therefore runs nowhere but a
# workstation, which leaves these scripts unexercised between full analyses
# unless something runs them here. PCC_CONFIG points them at smoke_data/ and
# smoke_results/, so nothing here reads or writes real data.
#
# The set is PARQUET_SUPPLEMENTARY minus FIXTURE_CANNOT_RUN; both are declared
# at the top of this file, with the reason each exclusion is out.
smoke-supplementary: smoke-data
	@for s in $(SMOKE_SUPPLEMENTARY); do \
	    echo "==> $$s"; \
	    PCC_CONFIG=config_smoke.toml uv run python scripts/supplementary/$$s.py || exit 1; \
	done

# -- Cleanup -----------------------------------------------------------------

clean:
	rm -rf results/ data/processed/ smoke_results/ smoke_data/
