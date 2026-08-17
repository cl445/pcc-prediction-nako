"""Run the full nested cross-validation PCC prediction pipeline.

Loads processed Parquet files, binarises the Bahmer weighted post-COVID
syndrome score as the PCC target, runs 10x5 nested CV with per-modality
base learners and an XGBoost meta-learner, then saves predictions,
metrics, and feature importances.

Usage:
    # Full run:
    uv run python scripts/pipeline/02_run_pipeline.py

    # Distributed: compute specific folds
    uv run python scripts/pipeline/02_run_pipeline.py --folds 0,1,2 --output-dir results/distributed/
    uv run python scripts/pipeline/02_run_pipeline.py --folds 3,4 --output-dir results/distributed/

    # Merge fold artifacts into final results
    uv run python scripts/pipeline/02_run_pipeline.py --merge-folds results/distributed/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from rich.console import Console
from rich.logging import RichHandler

from pcc_analysis._gpu_utils import cuda_available
from pcc_analysis.config import (
    get_pipeline_config,
    get_processed_dir,
    get_results_dir,
    load_config,
)
from pcc_analysis.data_manager import load_pipeline_data
from pcc_analysis.meta_learner import META_HYPERPARAM_ITERATIONS
from pcc_analysis.orchestration import merge_fold_results, run_pcc_pipeline
from pcc_analysis.run_comparability import (
    fingerprint_analysis_data,
    fingerprint_modality,
)

if TYPE_CHECKING:
    import pandas as pd

console = Console()


class _RunKwargs(TypedDict):
    """Everything ``run_pcc_pipeline`` needs except ``fold_subset``.

    Kept separate so the full run and the fold-subset run can share one set
    of arguments while still passing ``fold_subset`` explicitly — that
    argument selects which of the two overloads applies, and with it whether
    the call returns a complete or a partial result.
    """

    y: pd.Series
    X_dict: dict[str, pd.DataFrame]
    output_dir: Path
    confounders: pd.DataFrame | None
    n_outer_folds: int
    n_inner_folds: int
    orthogonalize: bool
    random_state: int
    n_jobs: int
    meta_permutation_iterations: int
    n_bootstrap_eval: int
    cohort: str
    stack_variant: str
    target: str
    clean_controls: bool
    split_mh_submodalities: bool
    amendment_features: bool
    hyperparam_iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCC Multimodal Prediction Pipeline",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated 0-based fold indices to compute (e.g. '0,1,2')",
    )
    group.add_argument(
        "--merge-folds",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory with fold_*.pkl files to merge into final results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override automatic timestamp output directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: auto-detected)",
    )
    parser.add_argument(
        "--target",
        choices=["bahmer", "neurocog"],
        default="bahmer",
        help=(
            "PCC target to model.  'bahmer' = weighted PCS > 10.75 (primary); "
            "'neurocog' = >=2 of fatigue/reduced-physical-capacity/memory/"
            "concentration (neurocognitive-subtype sensitivity analysis)."
        ),
    )
    parser.add_argument(
        "--no-clean-controls",
        action="store_true",
        help=(
            "Disable the clean-controls filter (keep k0=1 sub-threshold "
            "symptomatic participants as PCC-negatives).  Default design "
            "excludes them so the control arm contains only k0=2 "
            "declared-symptom-free participants."
        ),
    )
    parser.add_argument(
        "--no-orthogonalize",
        action="store_true",
        help=(
            "Disable DML-based orthogonalization against demographics "
            "(age, sex, centre).  Intended for the no-orthogonalization "
            "sensitivity analysis — checks whether DML masks "
            "age-/sex-dependent second-hit effects."
        ),
    )
    parser.add_argument(
        "--cohort",
        choices=["mri", "non_mri", "mri_plus_non_mri", "all"],
        default="all",
        help=(
            "Analytic cohort filter (relative to MRI availability). "
            "'mri' = MRI sub-study participants (primary analysis); "
            "'non_mri' = infected participants WITHOUT MRI data "
            "(within-study replication — requires "
            "--stack=lean); 'mri_plus_non_mri' = pooled MRI + "
            "non-MRI for Lean-Stack pooled training (also requires "
            "--stack=lean); 'all' = no filter (legacy default)."
        ),
    )
    parser.add_argument(
        "--stack",
        choices=["full", "lean"],
        default="full",
        help=(
            "Model specification.  'full' = ten modalities incl. six MRI "
            "atlases (primary analysis); 'lean' = four self-report "
            "modalities (demographics, ses, medical_history, "
            "mental_health) = Lean Universal Classifier for cross-cohort "
            "replication."
        ),
    )
    parser.add_argument(
        "--split-mh-submodalities",
        action="store_true",
        help=(
            "Split the monolithic mental_health modality into five "
            "instrument-specific sub-modalities (PHQ-9, GAD-7, MINI, "
            "PHQ-Panic, PHQ-Stress) at load time, for assessing "
            "mental-health instrument specificity."
        ),
    )
    parser.add_argument(
        "--without-amendment-features",
        action="store_true",
        help=(
            "Withhold every feature the NAKO amendment deliveries added "
            "(baseline PHQ-9 items, tobacco block, GPAQ), reproducing the "
            "pre-amendment feature set.  For the control run: combine with "
            "--hyperparam-iterations set to the value the earlier run "
            "used, which holds the feature set and the search width at "
            "the values behind the draft's figures, and so makes the delta against the "
            "primary run attributable to those two.  Reproducing the "
            "that earlier figure end to end is a separate check that needs a "
            "run pinned to xgboost 3.2.0, since 3.4.0 moves the numbers on "
            "its own (DECISIONS §2.20)."
        ),
    )
    parser.add_argument(
        "--fingerprint-only",
        action="store_true",
        help=(
            "Load the data this configuration would be fitted on, print its "
            "digest per modality, and exit without computing anything. Run "
            "it on both machines of a distributed rerun and compare: the "
            "digest is what the config hash carries, so a difference here is "
            "a difference that would only surface at the end of the run, or "
            "not at all."
        ),
    )
    parser.add_argument(
        "--hyperparam-iterations",
        type=int,
        default=META_HYPERPARAM_ITERATIONS,
        metavar="N",
        help=(
            "Draws in the meta-learner's randomised hyperparameter search "
            f"(default {META_HYPERPARAM_ITERATIONS}, capped at 50 by the "
            "search itself; 0 uses XGBoost defaults without searching)."
        ),
    )
    return parser.parse_args()


def _parse_fold_indices(fold_str: str, n_outer_folds: int) -> list[int]:
    """Parse and validate comma-separated fold indices."""
    try:
        folds = [int(f.strip()) for f in fold_str.split(",")]
    except ValueError:
        console.print(f"[red]Invalid fold specification: '{fold_str}'")
        console.print("Expected comma-separated integers, e.g. '0,1,2'")
        sys.exit(1)

    invalid = [f for f in folds if f < 0 or f >= n_outer_folds]
    if invalid:
        console.print(f"[red]Fold indices {invalid} out of range [0, {n_outer_folds})")
        sys.exit(1)

    return folds


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    console.rule("[bold]PCC Multimodal Prediction Pipeline")

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        console.print(f"[red]{e}")
        sys.exit(1)

    # Load data (analytic sample is restricted to SARS-CoV-2 infected
    # participants).  The processed-data path is
    # derived from the *loaded* config so an explicit ``--config`` (e.g.
    # the smoke-test config) actually takes effect — without threading
    # it through here, ``load_pipeline_data`` silently re-reads the
    # default ``config.toml`` and points at production data instead.
    processed_dir = get_processed_dir(config)
    clean_controls = not args.no_clean_controls
    amendment_features = not args.without_amendment_features
    console.print(
        f"[bold]Loading processed data (target={args.target}, "
        f"cohort={args.cohort}, stack={args.stack}, "
        f"clean_controls={clean_controls}, "
        f"amendment_features={amendment_features}) from {processed_dir}..."
    )
    try:
        y, X_dict, confounders = load_pipeline_data(
            processed_dir=processed_dir,
            target=args.target,
            clean_controls=clean_controls,
            cohort=args.cohort,
            stack_variant=args.stack,
            split_mh_submodalities=args.split_mh_submodalities,
            amendment_features=amendment_features,
        )
    except FileNotFoundError as e:
        console.print(f"[red]{e}")
        console.print("Run [bold]scripts/pipeline/01_process_data.py[/bold] first.")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}")
        sys.exit(1)

    console.print(f"  Target:          {args.target}")
    console.print(f"  Cohort:          {args.cohort}")
    console.print(f"  Stack variant:   {args.stack}")
    console.print(f"  Clean controls:  {clean_controls}")
    console.print(f"  Analytic sample: {len(y):,}")
    console.print(f"  PCC prevalence:  {float(y.mean()) * 100:.1f}%")
    console.print(f"  Modalities:      {len(X_dict)} ({', '.join(X_dict.keys())})")

    # Returns before the pipeline configuration, the GPU probe and anything
    # that writes: the digest is a property of the loaded data alone, and
    # this mode is run on two machines to compare it.
    if args.fingerprint_only:
        console.rule("[bold]Data fingerprint")
        console.print(f"  [bold]{fingerprint_analysis_data(y, X_dict)}[/bold]")
        console.print(f"  y ({y.name}): N={len(y):,}, positives={int(y.sum()):,}")
        for name in sorted(X_dict):
            frame = X_dict[name]
            console.print(
                f"  {name:<24} {frame.shape[0]:>6} x {frame.shape[1]:<4} "
                f"{fingerprint_modality(name, frame)}"
            )
        console.print(
            "\n  Same digest on both machines means the same analysis data. "
            "A difference\n  in one modality line says which parquet moved; a "
            "difference in the top\n  line alone means the labels or the "
            "sample did."
        )
        return

    # Pipeline config
    pipeline_cfg = get_pipeline_config(config)
    results_dir = get_results_dir(config)
    n_outer_folds = int(pipeline_cfg["n_outer_folds"])

    # GPU detection
    gpu = cuda_available()
    console.print(
        f"\n[bold]Hardware:[/bold] XGBoost using "
        f"{'[green]CUDA GPU[/green]' if gpu else '[yellow]CPU[/yellow]'}"
    )

    # --- Merge mode ---
    if args.merge_folds is not None:
        merge_dir = args.merge_folds
        if not merge_dir.is_dir():
            console.print(f"[red]Directory not found: {merge_dir}")
            sys.exit(1)

        out = args.output_dir or merge_dir
        console.print(f"\n[bold]Merging fold artifacts from {merge_dir}...")
        console.print(f"  Output: {out}")

        results = merge_fold_results(
            fold_dir=merge_dir,
            y=y,
            X_dict=X_dict,
            confounders=confounders,
            n_outer_folds=n_outer_folds,
            n_inner_folds=int(pipeline_cfg["n_inner_folds"]),
            orthogonalize=(
                bool(pipeline_cfg["orthogonalize"]) and not args.no_orthogonalize
            ),
            random_state=int(pipeline_cfg["random_state"]),
            n_jobs=int(pipeline_cfg["n_jobs"]),
            meta_permutation_iterations=int(
                pipeline_cfg["meta_permutation_iterations"]
            ),
            n_bootstrap_eval=int(pipeline_cfg["n_bootstrap"]),
            output_dir=out,
            cohort=args.cohort,
            stack_variant=args.stack,
            target=args.target,
            clean_controls=clean_controls,
            split_mh_submodalities=args.split_mh_submodalities,
            amendment_features=amendment_features,
            hyperparam_iterations=args.hyperparam_iterations,
        )

        console.rule("[bold green]Merge Complete")
        metrics = results["evaluation"]["metrics"]
        console.print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
        console.print(f"  PR-AUC:  {metrics['pr_auc']:.3f}")
        console.print(f"  Output:  {out}")
        return

    # --- Fold subset or full run ---
    fold_subset = None
    if args.folds is not None:
        fold_subset = _parse_fold_indices(args.folds, n_outer_folds)

    from datetime import datetime

    dml_tag = "noDML_" if args.no_orthogonalize else ""
    cohort_tag = "" if args.cohort == "all" else f"{args.cohort}_"
    stack_tag = "" if args.stack == "full" else f"{args.stack}_"
    control_tag = "" if amendment_features else "preAmendment_"
    output_dir = args.output_dir or results_dir / (
        f"pcc_pipeline_{cohort_tag}{stack_tag}{args.target}_"
        f"{control_tag}{dml_tag}{datetime.now().astimezone():%Y%m%d_%H%M%S}"
    )

    console.print("\n[bold]Pipeline configuration:")
    for k, v in pipeline_cfg.items():
        console.print(f"  {k}: {v}")
    console.print(f"  hyperparam_iterations: {args.hyperparam_iterations}")
    console.print(f"  output_dir: {output_dir}")
    if fold_subset is not None:
        console.print(f"  fold_subset: {fold_subset}")

    # Run pipeline
    mode = f"folds {fold_subset}" if fold_subset else "full"
    console.print(f"\n[bold]Running nested CV pipeline ({mode})...")
    run_kwargs: _RunKwargs = {
        "y": y,
        "X_dict": X_dict,
        "output_dir": output_dir,
        "confounders": confounders,
        "n_outer_folds": n_outer_folds,
        "n_inner_folds": int(pipeline_cfg["n_inner_folds"]),
        "orthogonalize": (
            bool(pipeline_cfg["orthogonalize"]) and not args.no_orthogonalize
        ),
        "random_state": int(pipeline_cfg["random_state"]),
        "n_jobs": int(pipeline_cfg["n_jobs"]),
        "meta_permutation_iterations": int(pipeline_cfg["meta_permutation_iterations"]),
        "n_bootstrap_eval": int(pipeline_cfg["n_bootstrap"]),
        "cohort": args.cohort,
        "stack_variant": args.stack,
        "target": args.target,
        "clean_controls": clean_controls,
        "split_mh_submodalities": args.split_mh_submodalities,
        "amendment_features": amendment_features,
        "hyperparam_iterations": args.hyperparam_iterations,
    }

    # Summary
    if fold_subset is not None:
        run_pcc_pipeline(**run_kwargs, fold_subset=fold_subset)
        console.rule("[bold green]Fold Subset Complete")
        console.print(f"  Computed folds: {fold_subset}")
        console.print(f"  Artifacts saved to: {output_dir}")
        console.print(
            "  Next: copy fold_*.pkl files together and run with --merge-folds"
        )
    else:
        full_results = run_pcc_pipeline(**run_kwargs, fold_subset=None)
        console.rule("[bold green]Pipeline Complete")
        metrics = full_results["evaluation"]["metrics"]
        console.print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
        console.print(f"  PR-AUC:  {metrics['pr_auc']:.3f}")
        console.print(f"  Output:  {output_dir}")


if __name__ == "__main__":
    main()
