#!/usr/bin/env bash
#
# run_analysis.sh — run the complete PCC analysis, end to end.
#
# This is the analysis as performed for the paper: it runs the full graph
# in dependency order and writes every paper-bound artifact (LaTeX
# constants and tables) under results/paper/. Pipeline runs go to
# deterministic directories under results/runs/ so the downstream
# figure/panel scripts find them without hand-pasted timestamps.
#
# Re-running this script on the same data therefore *reproduces* its
# results; reproduction needs no extra steps beyond running it again.
#
# Requirements:
#   - NAKO data access, configured in config.toml (see config.toml.example)
#   - `uv sync` has been run
#
# All steps read config.toml for paths and pipeline settings.
#
# Runtime: DAYS. Ten nested 10x5 cross-validation runs dominate the
# wall-clock; each is independent and restartable. The two Lean fits on the
# large cohorts are most of it — measured at roughly 25 h (lean_non_mri) and
# 47 h (lean_pooled) against 1.5-2 h for each MRI-cohort run, because both
# are fitted on tens of thousands of participants rather than the imaged
# subsample.
#
# Usage:
#   ./scripts/run_analysis.sh                      # on real NAKO data
#   ./scripts/run_analysis.sh --only lean_pooled   # just that step
#   ./scripts/run_analysis.sh --except lean_pooled # everything else
#   ./scripts/run_analysis.sh --list               # the step names
#   ./scripts/run_analysis.sh --smoke              # wiring check on synthetic data
#
# --only/--except exist for one purpose: splitting a rerun across two
# machines without running the pipeline commands by hand. Doing that by hand
# is what loses the guards this script carries — the split-MH validation, the
# control deltas, the constants snapshot and its diff, and the comparability
# check at the end. A step whose input another machine produced is not
# silently skipped: it names the directory it needs and stops.
#
# The split that balances is by configuration, not by fold: `--only
# lean_pooled` on one machine against `--except lean_pooled` on the other
# puts ~46 h opposite ~38 h. Whichever way it is cut, both machines must
# install from the same lockfile — `check_run_stacks.py` at the end says
# whether they did, but only for the runs it can see, so run it once more
# over the collected directories after copying them together.
#
# --smoke runs the identical graph on generated synthetic fixtures using
# config_smoke.toml (PCC_CONFIG), writing to smoke_data/ and smoke_results/.
# The numbers are meaningless; what it checks is that the steps wire
# together. Two caveats on its coverage, both reported explicitly at the
# end of the run rather than folded into one warning:
#   - statsmodels-based supplementary fits may not converge on random data
#     and are reported as warnings instead of aborting the check;
#   - steps that read the raw NAKO CSV export are skipped outright, because
#     the generator only synthesises the processed parquets.
set -euo pipefail

cd "$(dirname "$0")/.."

PY="uv run python"
SMOKE=0
SUPP_WARN=()
SUPP_SKIPPED=()

# Every step this script can run, in dependency order. The pipeline
# configurations are named as their run directories, so `--only split_mh`
# reads the same as the directory it writes.
STEPS="data primary control neurocog mixed_controls no_dml split_mh \
lean_mri control_lean lean_non_mri lean_pooled transfer figures statistics \
supplementary panels"

ONLY=""
EXCEPT=""
ONLY_GIVEN=0
EXCEPT_GIVEN=0
RESET_CONSTANTS_BASELINE=0

usage() {
    cat <<'USAGE'
Usage: run_analysis.sh [--smoke] [--only STEPS] [--except STEPS] [--list]

  --only STEPS     run only these steps (comma- or space-separated)
  --except STEPS   run everything but these
  --list           print the step names, one per line
  --smoke          run the same graph on synthetic fixtures
  --reset-constants-baseline
                   start a fresh "what this rerun moved" baseline instead of
                   keeping the one an earlier invocation left

--only/--except split a rerun across two machines while keeping this
script's guards. A selected step whose input is missing stops and names
the directory it needs, so the halves can be produced in either order and
assembled afterwards.
USAGE
}

# Reject a misspelt step name instead of silently running nothing: `--only
# lean-pooled` would otherwise look like a no-op run that succeeded.
validate_steps() {
    local requested
    for requested in $1; do
        case " $STEPS " in
            *" $requested "*) ;;
            *)
                echo "Unknown step: $requested" >&2
                echo "Valid steps: $STEPS" >&2
                exit 2
                ;;
        esac
    done
}

# A flag whose value was forgotten is a typo, not a request to run
# everything: without this, `--smoke --only` would shift past the end and
# die on `set -e` with no message at all.
needs_value() {
    [ "$1" -gt 0 ] || {
        echo "$2 needs a step list, e.g. $2 lean_pooled" >&2
        exit 2
    }
}

while [ $# -gt 0 ]; do
    case "$1" in
        --smoke) SMOKE=1 ;;
        --only)
            shift
            needs_value $# --only
            ONLY="$ONLY $(echo "$1" | tr ',' ' ')"
            ONLY_GIVEN=1
            ;;
        --only=*) ONLY="$ONLY $(echo "${1#*=}" | tr ',' ' ')"; ONLY_GIVEN=1 ;;
        --except)
            shift
            needs_value $# --except
            EXCEPT="$EXCEPT $(echo "$1" | tr ',' ' ')"
            EXCEPT_GIVEN=1
            ;;
        --except=*) EXCEPT="$EXCEPT $(echo "${1#*=}" | tr ',' ' ')"; EXCEPT_GIVEN=1 ;;
        --reset-constants-baseline) RESET_CONSTANTS_BASELINE=1 ;;
        --list) echo "$STEPS" | tr ' ' '\n' | grep -v '^$'; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Try --help, or --list for the step names." >&2
            exit 2
            ;;
    esac
    shift
done
validate_steps "$ONLY"
validate_steps "$EXCEPT"

# Collapse the accumulated whitespace, then catch the two selections that
# parse but mean "run nothing": an empty value (`--only=`, `--only ""`,
# `--only ,`) and a step named on both sides. Both would otherwise exit 0
# having run nothing at all.
ONLY="$(echo "$ONLY" | xargs || true)"
EXCEPT="$(echo "$EXCEPT" | xargs || true)"
if [ "$ONLY_GIVEN" = 1 ] && [ -z "$ONLY" ]; then
    echo "--only was given an empty step list." >&2
    echo "Valid steps: $STEPS" >&2
    exit 2
fi
if [ "$EXCEPT_GIVEN" = 1 ] && [ -z "$EXCEPT" ]; then
    echo "--except was given an empty step list." >&2
    echo "Valid steps: $STEPS" >&2
    exit 2
fi
for selected in $ONLY; do
    case " $EXCEPT " in
        *" $selected "*)
            echo "Step '${selected}' is in both --only and --except." >&2
            echo "Nothing would run. Drop it from one of them." >&2
            exit 2
            ;;
    esac
done

should_run() {
    case " $EXCEPT " in *" $1 "*) return 1 ;; esac
    [ -z "$ONLY" ] && return 0
    case " $ONLY " in *" $1 "*) return 0 ;; esac
    return 1
}

# A step whose input is missing stops the run and says who produces it. The
# directory may equally have come from the other machine, so the message
# names copying as the second way out rather than assuming a single host.
#
# Gated on config.csv rather than on the directory: config.csv is the last
# file a run writes, so its presence is the one cheap marker that a run
# finished. A directory alone proves nothing — a run killed at hour 30 and an
# rsync still in flight both leave one, complete with the provenance.json
# that was written before the first fold.
finished_run() {
    [ -f "$1/config.csv" ]
}

require_run_dir() {
    local step="$1" dir="$2" producer="$3"
    if finished_run "$dir"; then
        return 0
    fi
    if [ -d "$dir" ]; then
        echo "ERROR: step '${step}' needs ${dir}, which exists but did not" >&2
        echo "       finish: it has no config.csv, which a run writes last." >&2
        echo "       An interrupted run or a copy still in flight looks like" >&2
        echo "       this. Re-run step '${producer}', or finish the copy." >&2
    else
        echo "ERROR: step '${step}' needs ${dir}, which does not exist." >&2
        echo "       Run step '${producer}' here first, or copy its run" >&2
        echo "       directory over from the machine that produced it." >&2
    fi
    exit 1
}

if [ "$SMOKE" = 1 ]; then
    export PCC_CONFIG="config_smoke.toml"
    RUNS="smoke_results/runs"
    echo "==> SMOKE mode: synthetic data, config_smoke.toml (PCC_CONFIG)"
    if should_run data; then
        echo "==> [1/6] Generating synthetic data"
        $PY -m smoke_test generate --output-dir smoke_data
    fi
else
    RUNS="results/runs"
    if should_run data; then
        echo "==> [1/6] Processing data"
        $PY scripts/pipeline/01_process_data.py
    fi
fi

if [ -n "$ONLY" ] || [ -n "$EXCEPT" ]; then
    echo "==> Partial run.${ONLY:+ only:$ONLY}${EXCEPT:+ except:$EXCEPT}"
    echo "    The steps left out are not skipped silently: anything selected"
    echo "    that needs their output will stop and name what is missing."
fi

# Run a supplementary analysis. In --smoke mode a failure is recorded and
# the run continues (random data routinely makes a logistic fit singular);
# on real data the step is strict and a failure aborts via `set -e`.
supp() {
    if [ "$SMOKE" = 1 ]; then
        $PY "$@" || SUPP_WARN+=("$1")
    else
        $PY "$@"
    fi
}

# Same, but for steps that read the raw NAKO CSV export instead of the
# processed parquets. The synthetic generator produces only parquets, so
# under --smoke there is nothing for these to read: skip them outright
# rather than let them fail and be misread as a model-fit problem.
supp_raw_nako() {
    if [ "$SMOKE" = 1 ]; then
        SUPP_SKIPPED+=("$1")
    else
        $PY "$@"
    fi
}

# --- 2. Pipeline runs (one deterministic dir per configuration) -----------
# Each is a full nested 10x5 CV. Names map to the paper as follows:
#   primary        -> main result (MRI, full stack, DML)
#   control        -> pre-amendment specification, for attributing the shift
#   neurocog       -> neurocognitive-subtype outcome sensitivity
#   mixed_controls -> mixed-controls outcome sensitivity
#   no_dml         -> orthogonalization-disabled sensitivity
#   split_mh       -> mental-health sub-modality decomposition
#   lean_mri       -> Lean Universal Classifier, MRI cohort (transfer source)
#   control_lean   -> pre-rerun search width on the Lean Stack
#   lean_non_mri   -> Lean Universal Classifier, non-MRI cohort
#   lean_pooled    -> Lean Universal Classifier, pooled cohorts
echo "==> [2/6] Pipeline runs (10 configurations; this is the long part)"
if should_run primary; then
    $PY scripts/pipeline/02_run_pipeline.py --output-dir "${RUNS}/primary"
fi

# The control run: the primary configuration with the amendment features
# withheld and the meta-learner's hyperparameter search back at the width the
# earlier run used. It answers the one question the primary run cannot.
#
# Three things separate this stack from an earlier one, and a headline that
# moves has to be attributable to one of them rather than to all three at
# once: the amendment features, a hyperparameter search of 50 draws against
# the earlier run's 10, and the dependency bump.
#
# The third is already accounted for. DECISIONS §2.9 measured it on the smoke
# graph: xgboost 3.2.0 to 3.4.0 changes 13 of 16 output files, every other
# bumped package leaves them byte-identical, and the full bump reproduces the
# xgboost-only bump exactly. So the library effect is isolated by measurement,
# and this run isolates the other two by construction — together they
# decompose the delta.
#
# What §2.9 also establishes is that an earlier figure cannot be reproduced
# across an xgboost change at all, so checking the chain end to end needs a
# run pinned to the version that produced it. The check below therefore says
# which of the two situations it is in rather than reporting a deviation it
# already expects.
#
# Costs an eighth of the wall-clock of this step. Discovering afterwards that
# the shift is unattributable costs the whole rerun again.
CONTROL_HYPERPARAM_ITERATIONS=10
if should_run control; then
    $PY scripts/pipeline/02_run_pipeline.py \
        --without-amendment-features \
        --hyperparam-iterations "$CONTROL_HYPERPARAM_ITERATIONS" \
        --output-dir "${RUNS}/control"
fi

# Print the comparison rather than leaving it to be done by hand later. The
# control figure is only useful next to the reference it is meant to
# reproduce, and a run that has to be compared manually across two CSVs three
# weeks after the fact usually is not.
report_control_delta() {
    $PY - "$1" "$2" "$3" <<'PYTHON'
import sys
from pathlib import Path

import pandas as pd

what, control_dir, reference_dir = sys.argv[1:4]
control = float(pd.read_csv(Path(control_dir) / "metrics.csv")["roc_auc"].iloc[0])
reference = float(pd.read_csv(Path(reference_dir) / "metrics.csv")["roc_auc"].iloc[0])
print(f"    {what}")
print(f"      control (pre-rerun spec): ROC-AUC {control:.4f}")
print(f"      current spec:             ROC-AUC {reference:.4f}")
print(f"      delta attributable to the change: {reference - control:+.4f}")
PYTHON
}

# Both halves of a delta have to exist, and — once a rerun is split across
# two machines — they have to have been produced under the same stack.
# Neither is a reason to abort here: by this point the expensive runs are
# behind us, and killing the remaining steps over a comparison would cost
# more than the comparison is worth. The check at the end of the script is
# the strict one; this says what the delta is worth as it prints it.
delta_across_runs() {
    local what="$1" control_dir="$2" reference_dir="$3"
    # Both sides must have *finished*, not merely exist. A directory left
    # behind by an interrupted run has no metrics.csv, and reading it here
    # would kill the whole invocation under `set -e` — including a
    # `--only lean_pooled` that never asked about either of these runs.
    if ! finished_run "$control_dir" || ! finished_run "$reference_dir"; then
        echo "    ${what}: no delta — it needs a finished run in both"
        echo "      ${control_dir} and ${reference_dir}. Run or copy the"
        echo "      missing one; the delta prints on the next invocation"
        echo "      that sees both."
        return 0
    fi
    if ! $PY scripts/check_run_stacks.py "$control_dir" "$reference_dir" >/dev/null
    then
        echo "    WARNING: these two runs record different environments, so the"
        echo "    delta below is not attributable to the specification alone."
        echo "    Run scripts/check_run_stacks.py on them for the detail."
    fi
    report_control_delta "$what" "$control_dir" "$reference_dir"
}

delta_across_runs "Full Stack, MRI cohort" "${RUNS}/control" "${RUNS}/primary"

# An earlier Full-Stack figure to compare the control against, and the xgboost
# it was produced under. Both come from the environment and both are optional:
# a first run has nothing to compare to, and hard-coding one run's number here
# would make every later reader inherit it as though it meant something.
#
#     CONTROL_REFERENCE_ROC=<roc> CONTROL_REFERENCE_XGBOOST=<version> ./scripts/run_analysis.sh
#
# Checked rather than printed: the split-MH guard above exists because a number
# nobody compares is a number nobody notices, and leaving this one to be
# matched from memory is the same bet. A warning, not an abort — a deviation
# is a finding to investigate, and the remaining runs are worth having either
# way.
CONTROL_REFERENCE_ROC="${CONTROL_REFERENCE_ROC:-}"
CONTROL_REFERENCE_XGBOOST="${CONTROL_REFERENCE_XGBOOST:-}"
CONTROL_TOLERANCE=0.001

check_control_against_reference() {
    $PY - "$1" "$CONTROL_REFERENCE_ROC" "$CONTROL_TOLERANCE" "$CONTROL_REFERENCE_XGBOOST" <<'PYTHON'
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

run_dir, reference, tolerance, reference_xgboost = (
    sys.argv[1],
    float(sys.argv[2]),
    float(sys.argv[3]),
    sys.argv[4],
)
control = float(pd.read_csv(Path(run_dir) / "metrics.csv")["roc_auc"].iloc[0])
installed_xgboost = version("xgboost")

if reference_xgboost and installed_xgboost != reference_xgboost:
    # Not a deviation to investigate: DECISIONS §2.9 measured that xgboost
    # moves these numbers, so the comparison is answering a question nobody
    # asked. Saying what would make it answerable beats printing a warning
    # whose cause is already documented.
    print(
        f"      control ROC-AUC {control:.4f}; the reference {reference:.3f} was\n"
        f"      produced under xgboost {reference_xgboost} and this run uses\n"
        f"      {installed_xgboost}, which DECISIONS §2.9 measured as moving\n"
        "      13 of 16 output files. Reproducing it end to end would need a\n"
        f"      run pinned to xgboost {reference_xgboost}; the delta above is\n"
        "      still the feature set and search width, which this stack holds\n"
        "      fixed."
    )
elif abs(control - reference) <= tolerance:
    print(f"      control reproduces the reference ROC-AUC of {reference:.3f}")
else:
    print(
        f"      WARNING: control ROC-AUC {control:.4f} against the reference "
        f"{reference:.3f} ({control - reference:+.4f}),\n"
        f"      on the same xgboost {installed_xgboost}. The earlier\n"
        "      specification no longer reproduces its own figure, so something\n"
        "      else moved: another library, the processed parquets, or the\n"
        "      pipeline. Explain it before reading the delta above as the\n"
        "      feature set."
    )
PYTHON
}

if [ "$SMOKE" = 1 ]; then
    echo "      (reference check skipped: synthetic data)"
elif [ -z "$CONTROL_REFERENCE_ROC" ]; then
    echo "      (no CONTROL_REFERENCE_ROC set; reporting the delta only)"
elif finished_run "${RUNS}/control"; then
    check_control_against_reference "${RUNS}/control"
fi
echo "    The Lean control has no pinned reference; read its delta"
echo "    against the lean_mri run below."

if should_run neurocog; then
    $PY scripts/pipeline/02_run_pipeline.py --target neurocog --output-dir "${RUNS}/neurocog"
fi
if should_run mixed_controls; then
    $PY scripts/pipeline/02_run_pipeline.py --no-clean-controls --output-dir "${RUNS}/mixed_controls"
fi
if should_run no_dml; then
    $PY scripts/pipeline/02_run_pipeline.py --no-orthogonalize --output-dir "${RUNS}/no_dml"
fi
if should_run split_mh; then
    $PY scripts/pipeline/02_run_pipeline.py --split-mh-submodalities --output-dir "${RUNS}/split_mh"
fi

# The split-MH run is only identifiable by its modality list: the flag
# contributes nothing to config.csv, so a run that silently lost the
# sub-modalities looks exactly like a primary run. That failure is not
# hypothetical — a run has trained with no mental-health input at all, and
# the only symptom was a ROC-AUC roughly six points lower. Check here rather
# than discovering it in the reported numbers.
check_split_mh() {
    local shap="${RUNS}/split_mh/shap_modality_importance.csv"
    if [ ! -f "$shap" ]; then
        echo "ERROR: $shap missing — SHAP analysis failed silently (it is" >&2
        echo "       caught and logged as a warning, not raised)." >&2
        return 1
    fi
    local n_mh
    n_mh=$(grep -c '^mh_' "$shap" || true)
    if [ "$n_mh" != "5" ]; then
        echo "ERROR: expected 5 mh_* rows in $shap, found ${n_mh}." >&2
        echo "       The sub-modality split was dropped; this run is unusable." >&2
        return 1
    fi
    echo "    split_mh validated: ${n_mh} mh_* sub-modalities present"
}

# Fatal on real data, a warning under --smoke, like every other step that
# depends on a model fit succeeding. The failure this guards against is
# _compute_shap swallowing its own exception and returning None, which is
# precisely what tiny synthetic fixtures provoke in the TreeExplainer; on
# synthetic data an abort here would take out the three Lean runs, the
# transfer step, the figures and all eleven supplementary scripts over a
# number that was never meaningful.
#
# Gated on the step or on a *finished* directory, and fatal only when this
# invocation depends on the answer. A split-MH run copied in from the other
# machine is as worth validating as one produced here, but an invocation
# that neither produced it nor consumes it (`--only lean_pooled`, say) must
# not abort over it — least of all over a directory still being written.
split_mh_is_consumed_here() {
    should_run split_mh || should_run panels
}

if should_run split_mh || finished_run "${RUNS}/split_mh"; then
    if ! check_split_mh; then
        if [ "$SMOKE" = 1 ]; then
            SUPP_WARN+=("split_mh validation")
        elif split_mh_is_consumed_here; then
            exit 1
        else
            echo "    (not fatal here: this invocation neither produced the" >&2
            echo "     split_mh run nor runs the panel that reads it — but it" >&2
            echo "     must be fixed before the panel step is run)" >&2
        fi
    fi
fi

if should_run lean_mri; then
    $PY scripts/pipeline/02_run_pipeline.py --cohort mri --stack lean --output-dir "${RUNS}/lean_mri"
fi

# The Lean Stack needs its own control, for a reason that is easy to miss:
# PLAN_FULL_RERUN §2a says the Lean Stack and the §A2a transfer are unchanged,
# which holds for the features — the amendment joins are gated on the Full
# Stack — but not for the meta-learner, whose search width is global. All
# three Lean runs and the transfer built on them therefore move too, and the
# TOST verdict they feed is already marginal on two of its three criteria
# (calibration intercept CI upper +0.065 against a band top of +0.05, slope
# 1.043 [1.005, 1.085]). If the abstract's "2 of 3 equivalence criteria" is
# going to change, the reason has to be attributable to the search width
# rather than merely coincident with it.
#
# No --without-amendment-features here: the Lean Stack never had them, so the
# search width is the only thing this control has to hold fixed.
if should_run control_lean; then
    $PY scripts/pipeline/02_run_pipeline.py --cohort mri --stack lean \
        --hyperparam-iterations "$CONTROL_HYPERPARAM_ITERATIONS" \
        --output-dir "${RUNS}/control_lean"
fi
delta_across_runs "Lean Stack, MRI cohort" "${RUNS}/control_lean" "${RUNS}/lean_mri"

if should_run lean_non_mri; then
    $PY scripts/pipeline/02_run_pipeline.py --cohort non_mri --stack lean --output-dir "${RUNS}/lean_non_mri"
fi
# The longest single step of the whole graph: 46.5 h at N=49865 on the
# 2026-05 run, against ~38 h for everything else put together. If the rerun
# is split across two machines, this is the half.
if should_run lean_pooled; then
    $PY scripts/pipeline/02_run_pipeline.py --cohort mri_plus_non_mri --stack lean --output-dir "${RUNS}/lean_pooled"
fi

# --- 3. Transfer validation (lean MRI model -> non-MRI cohort, no refit) ---
if should_run transfer; then
    echo "==> [3/6] Transfer validation"
    require_run_dir transfer "${RUNS}/lean_mri" lean_mri
    $PY scripts/pipeline/03_apply_transfer.py \
        --from-run "${RUNS}/lean_mri" --target-cohort non_mri \
        --output-dir "${RUNS}/transfer_lean_mri_to_non_mri"
fi

# --- 4. Figures (from the primary run) and descriptive statistics ----------
if should_run figures || should_run statistics; then
    echo "==> [4/6] Figures and statistics"
fi

# Which of the remaining steps write into results/paper/constants. The
# snapshot below is taken only if at least one of them runs: a machine that
# was asked for `--only lean_pooled` writes no constant, and resetting the
# baseline there would leave the next real run diffing against itself.
# `figures` is not among them — 04_generate_figures.py writes figures, and
# not one line of results/paper/constants.
writes_constants() {
    should_run statistics || should_run supplementary || should_run panels
}

# Snapshot the constants before anything overwrites them, so the run can say
# at the end which numbers it moved. The expensive half of a rerun is not the
# cross-validation but carrying the changed constants into the manuscript by
# hand, and that is done from whatever list exists — so the run produces one
# instead of leaving it to be reconstructed from two directories afterwards.
#
# An existing baseline is kept rather than replaced, because a rerun is no
# longer one invocation. Overwriting it on the second constants-writing
# invocation — the other half of the split, or a repeat of one panel —
# replaces the pre-rerun state with what the first invocation had just
# written, and the closing diff then compares the new constants against
# themselves, losing the list the paper cascade is worked from.
#
# The cost of keeping it is a baseline from an *older* rerun going
# unnoticed, so its date is printed and --reset-constants-baseline replaces
# it deliberately.
PAPER_DIR="$(dirname "$RUNS")/paper"
CONSTANTS_DIR="${PAPER_DIR}/constants"
CONSTANTS_SNAPSHOT="${PAPER_DIR}/constants_previous"
CONSTANTS_SNAPSHOT_TAKEN=0
if [ -d "$CONSTANTS_DIR" ] && writes_constants; then
    if [ -d "$CONSTANTS_SNAPSHOT" ] && [ "$RESET_CONSTANTS_BASELINE" != 1 ]; then
        echo "    Constants baseline kept from $(ls -ld "$CONSTANTS_SNAPSHOT" |
            awk '{print $6, $7, $8}'), so the diff at the end covers every"
        echo "    invocation of this rerun. --reset-constants-baseline takes a"
        echo "    fresh one."
    else
        rm -rf "$CONSTANTS_SNAPSHOT"
        cp -R "$CONSTANTS_DIR" "$CONSTANTS_SNAPSHOT"
    fi
    CONSTANTS_SNAPSHOT_TAKEN=1
fi

if should_run figures; then
    require_run_dir figures "${RUNS}/primary" primary
    $PY scripts/pipeline/04_generate_figures.py --results-dir "${RUNS}/primary"
fi
if should_run statistics; then
    $PY scripts/pipeline/05_compute_statistics.py
fi

# --- 5. Supplementary analyses that read only the processed parquets -------
if should_run supplementary; then
    echo "==> [5/6] Supplementary analyses"
    supp scripts/supplementary/outcome_agreement.py
    supp scripts/supplementary/cohort_comparison.py
    supp scripts/supplementary/evalue.py
    supp scripts/supplementary/sex_interaction.py
    supp scripts/supplementary/severity_interaction.py
    supp scripts/supplementary/smoking_adjustment.py
    supp_raw_nako scripts/supplementary/persistence_adjustment.py
    supp scripts/supplementary/symptom_decomposition.py
    supp scripts/supplementary/mh_trajectory_adjustment.py
    supp scripts/supplementary/mri_positive_control.py
    supp scripts/supplementary/inclusion_comparison.py
    supp scripts/supplementary/reporting_style_robustness.py
fi

# --- 6. Supplementary panels that consume the pipeline run directories -----
# Each of these reads two run directories and reports a difference between
# them, so each is a place where two machines' output meets. They refuse the
# comparison themselves if the two runs disagree on their stack; the
# directory checks here are only about the input being present at all.
if should_run panels; then
    echo "==> [6/6] Transfer/clinical-utility panels"
    require_run_dir panels "${RUNS}/lean_mri" lean_mri
    require_run_dir panels "${RUNS}/transfer_lean_mri_to_non_mri" transfer
    require_run_dir panels "${RUNS}/lean_non_mri" lean_non_mri
    require_run_dir panels "${RUNS}/split_mh" split_mh
    require_run_dir panels "${RUNS}/primary" primary
    supp scripts/supplementary/transport_panel.py \
        --source-run "${RUNS}/lean_mri" \
        --transfer-run "${RUNS}/transfer_lean_mri_to_non_mri"
    supp scripts/supplementary/lean_clinical_utility.py \
        --mri-run "${RUNS}/lean_mri" --non-mri-run "${RUNS}/lean_non_mri"
    supp scripts/supplementary/mh_submodality_panel.py \
        --run-dir "${RUNS}/split_mh" --primary-run "${RUNS}/primary"
fi

echo ""
# Every run directory present, checked against every other. Under --only
# that is a partial picture — the point of the check is the assembled set,
# so run it again over the collected directories once both machines are
# done. Its exit status is carried to the end of this script rather than
# aborting here: the constants diff below is the thing a rerun is read for,
# and a stack mismatch is a reason to distrust it, not to withhold it.
echo "==> Run comparability"
STACKS_OK=1
$PY scripts/check_run_stacks.py --runs-dir "$RUNS" || STACKS_OK=0
echo ""
if [ "$CONSTANTS_SNAPSHOT_TAKEN" = 1 ]; then
    echo "==> Constants changed by this run"
    $PY scripts/compare_constants.py \
        --before "$CONSTANTS_SNAPSHOT" --after "$CONSTANTS_DIR" || true
    echo ""
else
    echo "==> No constants diff: this run found no ${CONSTANTS_DIR} to snapshot,"
    echo "    so every constant it wrote is new. That is the normal first run,"
    echo "    not an error."
    if [ -d "$CONSTANTS_SNAPSHOT" ]; then
        echo "    ${CONSTANTS_SNAPSHOT} is an older run's baseline, kept as it"
        echo "    stands: diffing against it would credit this run with the"
        echo "    movement of the run that wrote it."
    fi
    echo ""
fi
if [ "${#SUPP_SKIPPED[@]}" -gt 0 ]; then
    echo "==> ${#SUPP_SKIPPED[@]} supplementary step(s) SKIPPED, not exercised by this check:"
    echo "    they read the raw NAKO export, which --smoke does not synthesise."
    for s in "${SUPP_SKIPPED[@]}"; do echo "      - $s"; done
    echo ""
fi
if [ "${#SUPP_WARN[@]}" -gt 0 ]; then
    echo "==> WARNING: ${#SUPP_WARN[@]} supplementary step(s) did not complete."
    echo "    A singular or non-converging fit is expected on random data;"
    echo "    any other traceback is a real failure worth reading."
    for s in "${SUPP_WARN[@]}"; do echo "      - $s"; done
    echo ""
fi
if [ "$STACKS_OK" = 0 ]; then
    echo "==> WARNING: the run directories under ${RUNS} were not all produced"
    echo "    in the same environment (see the report above). Deltas across"
    echo "    them — the controls, the transport panel, the sub-modality"
    echo "    panel — measure that difference as well as the analysis."
    echo ""
fi
echo "==> Done. All paper constants and tables are under results/paper/"
echo "    constants/  LaTeX \\newcommand files + JSON payloads"
echo "    tables/     generated LaTeX tables"

# A non-zero exit for a run that computed everything it was asked to: the
# artifacts are on disk, but they are not ready to be read against each
# other, and the exit code is the signal that survives `nohup`.
if [ "$STACKS_OK" = 0 ]; then
    exit 1
fi
