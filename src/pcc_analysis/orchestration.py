"""Nested CV pipeline orchestration for PCC multimodal prediction."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, overload

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ._gpu_utils import cuda_available, xgb_device
from ._provenance import package_versions, write_provenance
from .evaluation import (
    evaluate_model_comprehensive,
    plot_modality_contribution_comparison,
    save_figure_for_latex,
)
from .meta_learner import (
    META_HYPERPARAM_ITERATIONS,
    MetaLearner,
    create_oof_predictions,
    train_meta_learner,
)
from .modality_pipelines import ModalityPipelineFactory
from .run_comparability import fingerprint_analysis_data

if TYPE_CHECKING:
    from ._types import (
        CVResults,
        EvaluationResult,
        FinalModelResult,
        FoldArtifact,
        FoldPermImportance,
        FoldResult,
        ModalityScore,
        PartialCVResults,
        PerFoldOoS,
        PermutationTestResult,
        PipelineConfig,
        PipelineResult,
        ShapResults,
    )
    from .protocols import ConfoundedEstimator

logger = logging.getLogger(__name__)


class PCCMultimodalPipeline:
    """Full nested-CV pipeline orchestrator.

    Parameters
    ----------
    output_dir : Path
        Directory for outputs.
    n_outer_folds, n_inner_folds : int
        CV fold counts.
    orthogonalize : bool
        Whether to orthogonalize features (DML).
    random_state : int
        Seed.
    n_jobs : int
        Parallelism.
    meta_permutation_iterations : int
        Permutation test iterations for the meta-learner.
    hyperparam_iterations : int
        Draws in the meta-learner's randomised hyperparameter search.
        Clamped to :data:`META_HYPERPARAM_ITERATIONS`; must not be negative.
    provenance_filename : str
        Name of the provenance record written into *output_dir*.  A merge
        writes a second one under its own name so it does not overwrite the
        fold runs' record.
    data_fingerprint : str, optional
        Digest of the analysis data, from
        :func:`~pcc_analysis.run_comparability.fingerprint_analysis_data`.
        :meth:`run` computes it from the data it is handed, so only a
        caller that skips ``run`` — the fold merge — has to pass it.
    """

    def __init__(
        self,
        output_dir: Path | str,
        n_outer_folds: int = 10,
        n_inner_folds: int = 5,
        orthogonalize: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        meta_permutation_iterations: int = 1000,
        n_subsamples: int = 100,
        n_bootstrap_eval: int = 1000,
        cohort: str = "all",
        stack_variant: str = "full",
        target: str = "bahmer",
        clean_controls: bool = True,
        split_mh_submodalities: bool = False,
        amendment_features: bool = True,
        hyperparam_iterations: int = META_HYPERPARAM_ITERATIONS,
        *,
        provenance_filename: str = "provenance.json",
        data_fingerprint: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.orthogonalize = orthogonalize
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.meta_permutation_iterations = meta_permutation_iterations
        self.n_subsamples = n_subsamples
        self.n_bootstrap_eval = n_bootstrap_eval
        # These six are recorded for provenance only: the pipeline itself
        # runs on whatever (y, X_dict) it is given; the filters and the
        # outcome definition were applied upstream in
        # ``load_pipeline_data``.  Storing them here lets ``config.csv``
        # round-trip the analysis configuration so distributed reruns stay
        # identifiable.  Without ``target`` and ``clean_controls`` the
        # primary and the two outcome-sensitivity runs write identical
        # configuration records and are told apart only by their directory
        # name — the same class of silent mix-up the split-MH guard in
        # ``scripts/run_analysis.sh`` exists to catch.
        self.cohort = cohort
        self.stack_variant = stack_variant
        self.target = target
        self.clean_controls = clean_controls
        self.split_mh_submodalities = split_mh_submodalities
        self.amendment_features = amendment_features
        # Unlike the tags above, this one is not provenance-only: it is
        # passed to the meta-learner below. It is recorded for the same
        # reason all the same — the search width differs across runs of
        # this analysis (10 draws for the earlier run, 50 by default
        # here), and it changes the fitted model without otherwise
        # leaving a trace in the run directory.
        #
        # Stored clamped to the ceiling the randomised search enforces on
        # itself, so the number in ``config.csv`` is the number of draws the
        # meta-learner actually took: an unclamped 200 would describe a
        # search that never ran, and folds computed at 60 and at 200 would
        # carry different config hashes while holding bit-identical models,
        # refusing a merge that is in fact sound. Zero stays legal — the
        # ablation and incremental-performance contrasts below pass it to
        # hold every pairwise comparison at untuned XGBoost defaults —
        # but a negative count is a typo, not a quieter zero.
        if hyperparam_iterations < 0:
            raise ValueError(
                "hyperparam_iterations must be >= 0 (0 selects default "
                f"XGBoost parameters), got {hyperparam_iterations}"
            )
        self.hyperparam_iterations = min(
            hyperparam_iterations, META_HYPERPARAM_ITERATIONS
        )
        self.data_fingerprint = data_fingerprint

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        # Written up front rather than next to ``config.csv`` at the end:
        # the runs this describes take hours, and one that dies in hour
        # three still has to be identifiable afterwards.
        #
        # The name is a parameter because a merge is a second run over the
        # directory the folds were computed into, and the two answer
        # different questions: the fold run's ``provenance.json`` says what
        # produced the cross-validation, the merge's record says what
        # produced the final model. Sharing one filename would let a merge
        # started from another checkout, or on another machine, erase the
        # only record of what computed the folds.
        write_provenance(self.output_dir, filename=provenance_filename)

    def _setup_logging(self) -> None:
        # Local wall-clock, so the log file sorts next to the run directory a
        # human is looking at. The provenance record below uses UTC instead.
        log_file = (
            self.output_dir
            / f"pipeline_{datetime.now().astimezone():%Y%m%d_%H%M%S}.log"
        )
        # Attach to the package logger so all pcc_analysis.* modules
        # (meta_learner, orthogonalization, etc.) write to the log file.
        pkg_logger = logging.getLogger("pcc_analysis")
        for h in pkg_logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                pkg_logger.removeHandler(h)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        pkg_logger.addHandler(fh)
        self._file_handler = fh

    def _config_hash(self) -> str:
        """Deterministic hash of pipeline configuration for consistency checks."""
        cfg = self._get_config()
        config_str = json.dumps(
            {k: v for k, v in cfg.items() if k != "timestamp"},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _save_fold_artifact(self, fold_idx: int, artifact: FoldArtifact) -> None:
        """Save a single fold result as a pickle file."""
        path = self.output_dir / f"fold_{fold_idx}.pkl"
        with path.open("wb") as f:
            pickle.dump(artifact, f)
        logger.info("Saved fold %d artifact to %s", fold_idx, path)

    # ------------------------------------------------------------------

    def run(
        self,
        y: pd.Series,
        X_dict: dict[str, pd.DataFrame],
        confounders: pd.DataFrame | None = None,
        fold_subset: list[int] | None = None,
    ) -> CVResults | PipelineResult:
        """Execute the full nested-CV pipeline.

        The analytic sample is assumed to be restricted to SARS-CoV-2
        infected participants upstream (see ``load_pipeline_data``),
        since the PCC outcome is only defined for the infected cohort.
        The PCC target is the Bahmer weighted post-COVID syndrome
        score binarised at ``> 10.75``.

        Parameters
        ----------
        y : Series
            Binary PCC labels (Bahmer ``any_pcs``), indexed by participant.
        X_dict : dict[str, DataFrame]
            Per-modality feature DataFrames, keyed by modality name.
        confounders : DataFrame, optional
            Demographics (age, sex, center) for DML orthogonalization.
        fold_subset : list[int], optional
            If set, only compute these outer fold indices (0-based)
            and save per-fold artifacts for later merging.

        Returns
        -------
        CVResults | PipelineResult
            Partial fold results when *fold_subset* is given, otherwise
            the full pipeline result including evaluation and SHAP.
        """
        logger.info("=" * 80)
        logger.info("STARTING PCC MULTIMODAL PIPELINE")
        if fold_subset is not None:
            logger.info("FOLD SUBSET MODE: computing folds %s", fold_subset)
        logger.info("=" * 80)

        # Validate fold indices
        if fold_subset is not None:
            invalid = [f for f in fold_subset if f < 0 or f >= self.n_outer_folds]
            if invalid:
                raise ValueError(
                    f"Fold indices {invalid} out of range [0, {self.n_outer_folds})"
                )

        # Before anything is fitted, so the digest describes what this run
        # was actually given and every fold artifact carries the same one.
        if self.data_fingerprint is None:
            self.data_fingerprint = fingerprint_analysis_data(y, X_dict)
        logger.info("Data fingerprint: %s", self.data_fingerprint)

        y_arr = np.asarray(y, dtype=np.int8)
        target_label = str(y.name) if y.name else "y"
        logger.info(
            "Target: %s, N=%d, prevalence=%.1f%%",
            target_label,
            len(y_arr),
            float(y_arr.mean()) * 100,
        )

        # 2. Pipelines
        factory = ModalityPipelineFactory(
            random_state=self.random_state,
            orthogonalize=self.orthogonalize,
            cv=self.n_inner_folds,
            n_jobs=self.n_jobs,
            n_subsamples=self.n_subsamples,
        )
        pipelines = self._create_pipelines(factory, list(X_dict.keys()))

        # 3. Nested CV
        cv_results = self._nested_cv(
            X_dict,
            y_arr,
            pipelines,
            confounders,
            fold_subset=fold_subset,
        )

        # When running a fold subset, skip post-CV steps
        if fold_subset is not None:
            logger.info(
                "Fold subset run complete. Artifacts saved to %s", self.output_dir
            )
            # Clean up file handler
            if hasattr(self, "_file_handler"):
                self._file_handler.close()
                logging.getLogger("pcc_analysis").removeHandler(self._file_handler)
            return cv_results

        # Post-CV steps
        results = self._run_post_cv_steps(
            cv_results,
            X_dict,
            y_arr,
            pipelines,
            confounders,
        )
        return results

    def _run_post_cv_steps(
        self,
        cv_results: CVResults,
        X_dict: dict[str, pd.DataFrame],
        y: np.ndarray,
        pipelines: dict[str, ConfoundedEstimator],
        confounders: pd.DataFrame | None,
    ) -> PipelineResult:
        """Run all steps after nested CV: permutation test, final model, evaluation."""
        # 3b. Permutation test on CV predictions
        perm_test = None
        if self.meta_permutation_iterations > 0:
            perm_test = self._permutation_test_cv(
                cv_results["y_true"], cv_results["y_pred_proba"]
            )

        # 4. Statistical modality comparisons (Nadeau-Bengio + Holm)
        modality_comparisons = None
        mod_df = pd.DataFrame(cv_results["modality_scores"])
        if not mod_df.empty and len(mod_df["modality"].unique()) > 1:
            modality_comparisons = self._compare_modalities(mod_df, n_total=len(y))

        # 5. Final model (returns OoF predictions for downstream analyses)
        final_model = self._train_final_model(
            X_dict,
            y,
            pipelines,
            confounders,
        )

        # 6. SHAP analysis on final model
        shap_results = self._compute_shap(final_model)

        # 7. Modality ablation on final model (in-sample at meta-learner level)
        dropout_results = self._modality_ablation(
            final_model["meta_learner"],
            final_model["oof_predictions"],
            y,
            final_model["modality_names"],
        )

        # 7b. Out-of-sample modality ablation (uses per-fold meta-learners
        # on held-out test predictions; skipped in distributed fold_subset runs)
        per_fold_oos = getattr(self, "_per_fold_oos", [])
        dropout_oos_results = self._modality_ablation_oos(per_fold_oos)

        # 8. Incremental performance (in-sample at meta-learner level)
        incremental_results = self._incremental_performance(
            final_model["oof_predictions"],
            y,
            final_model["modality_names"],
        )

        # 8b. Out-of-sample incremental performance
        incremental_oos_results = self._incremental_performance_oos(per_fold_oos)

        # 9. Evaluation
        evaluation = self._evaluate(cv_results["y_true"], cv_results["y_pred_proba"])

        # 10. Save
        results: PipelineResult = {
            "cv_results": cv_results,
            "permutation_test": perm_test,
            "modality_comparisons": modality_comparisons,
            "final_model": final_model,
            "shap_results": shap_results,
            "modality_ablation": dropout_results,
            "modality_ablation_oos": dropout_oos_results,
            "incremental_performance": incremental_results,
            "incremental_performance_oos": incremental_oos_results,
            "evaluation": evaluation,
            "config": self._get_config(),
        }
        self._save_results(results)

        logger.info(
            "PIPELINE COMPLETED — ROC-AUC: %.3f, PR-AUC: %.3f",
            evaluation["metrics"]["roc_auc"],
            evaluation["metrics"]["pr_auc"],
        )
        self.results_ = results

        # Clean up file handler to prevent handler leak
        if hasattr(self, "_file_handler"):
            self._file_handler.close()
            logging.getLogger("pcc_analysis").removeHandler(self._file_handler)

        return results

    # ------------------------------------------------------------------

    def _create_pipelines(
        self, factory: ModalityPipelineFactory, available: list[str]
    ) -> dict[str, ConfoundedEstimator]:
        """Build per-modality pipelines for every key the loader produced.

        Prefix conventions:
        - ``mri_<atlas>`` — six MRI parcellations.
        - ``mh_<instrument>`` — five MH sub-modalities produced by
          ``load_pipeline_data(split_mh_submodalities=True)``
          (``mh_phq9``, ``mh_gad7``, ``mh_mini``, ``mh_panic``,
          ``mh_stress``); each shares the mental-health preprocessing
          pipeline because the sub-modality DataFrame already carries
          only the instrument-specific columns.
        - everything else must be in ``mapping``.

        Unmatched modality names raise ``ValueError`` rather than being
        silently dropped: a dispatch that falls through to a
        ``logger.warning`` lets the split-MH sub-modalities disappear
        without stopping the run, producing a model trained without
        mental-health input that still looks like a complete one.
        """
        mapping = {
            "demographics": factory.create_demographics_pipeline,
            "ses": factory.create_ses_pipeline,
            "cognitive": factory.create_cognitive_pipeline,
            "physical_activity": factory.create_physical_activity_pipeline,
            "medical_history": factory.create_medical_history_pipeline,
            "lab_values": factory.create_lab_values_pipeline,
            "cardiovascular": factory.create_cardiovascular_pipeline,
            "lung_function": factory.create_lung_function_pipeline,
            "mental_health": factory.create_mental_health_pipeline,
        }
        pipes: dict[str, ConfoundedEstimator] = {}
        unmatched: list[str] = []
        for mod in available:
            try:
                if mod in mapping:
                    pipes[mod] = mapping[mod]()
                elif mod.startswith("mri_"):
                    pipes[mod] = factory.create_mri_pipeline(mod.replace("mri_", ""))
                elif mod.startswith("mh_"):
                    pipes[mod] = factory.create_mental_health_pipeline()
                else:
                    unmatched.append(mod)
            except Exception:
                logger.exception("Pipeline creation failed for %s", mod)
                raise
        if unmatched:
            raise ValueError(
                "_create_pipelines could not match modality name(s): "
                f"{unmatched}. Either extend the dispatch table, register a "
                "new prefix branch (mri_/mh_), or remove these from the "
                "X_dict before pipeline construction. Silent drops are not "
                "tolerated."
            )
        return pipes

    def _nested_cv(
        self,
        X_dict: dict[str, pd.DataFrame],
        y: np.ndarray,
        pipelines: dict[str, ConfoundedEstimator],
        confounders: pd.DataFrame | None,
        *,
        fold_subset: list[int] | None = None,
    ) -> CVResults | PartialCVResults:
        n = len(y)
        y_oof = np.zeros(n)
        y_true = np.zeros(n)
        fold_results: list[FoldResult] = []
        modality_scores: list[ModalityScore] = []
        fold_perm_importances: list[FoldPermImportance] = []
        # Per-fold OoS artifacts for out-of-sample meta-learner analyses
        # (ablation + incremental). Only collected for full runs — distributed
        # fold_subset runs produce per-fold .pkl artifacts but skip OoS.
        per_fold_oos: list[PerFoldOoS] = []

        cv_outer = StratifiedKFold(
            n_splits=self.n_outer_folds, shuffle=True, random_state=self.random_state
        )
        first_key = next(iter(X_dict))

        for fold_idx, (train_idx, test_idx) in enumerate(
            cv_outer.split(X_dict[first_key], y)
        ):
            if fold_subset is not None and fold_idx not in fold_subset:
                continue
            logger.info("Outer fold %d/%d", fold_idx + 1, self.n_outer_folds)
            X_tr = {k: v.iloc[train_idx] for k, v in X_dict.items()}
            X_te = {k: v.iloc[test_idx] for k, v in X_dict.items()}
            y_tr, y_te = y[train_idx], y[test_idx]
            c_tr = confounders.iloc[train_idx] if confounders is not None else None
            c_te = confounders.iloc[test_idx] if confounders is not None else None

            # Inner OoF + fit on full training data in one pass
            _t0 = time.monotonic()
            oof_train, mod_names, fitted_pipes = create_oof_predictions(
                pipelines,
                X_tr,
                y_tr,
                self.n_inner_folds,
                self.random_state,
                c_tr,
                return_fitted=True,
            )
            logger.info("  OoF + refit: %.1fs", time.monotonic() - _t0)
            _t1 = time.monotonic()
            meta = train_meta_learner(
                oof_train,
                y_tr,
                mod_names,
                add_interactions=False,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                hyperparam_iterations=self.hyperparam_iterations,
            )
            logger.info("  Meta-learner: %.1fs", time.monotonic() - _t1)

            # Test predictions using already-fitted pipelines
            _t2 = time.monotonic()
            assert fitted_pipes is not None
            test_preds = np.full((len(test_idx), len(pipelines)), np.nan)
            for i, mod_name in enumerate(mod_names):
                p = fitted_pipes.get(mod_name)
                if p is None:
                    continue
                X_tem = X_te[mod_name]
                te_ok = ~pd.isna(X_tem).all(axis=1).to_numpy()
                if te_ok.sum() == 0:
                    continue
                cf_te = c_te[te_ok] if c_te is not None else None
                preds = p.predict_proba(X_tem[te_ok], confounders=cf_te)[:, 1]
                test_preds[te_ok, i] = preds

                try:
                    valid_y = y_te[te_ok]
                    if len(np.unique(valid_y)) >= 2:
                        score: ModalityScore = {
                            "fold": fold_idx + 1,
                            "modality": mod_name,
                            "roc_auc": float(roc_auc_score(valid_y, preds)),
                            "pr_auc": float(average_precision_score(valid_y, preds)),
                        }
                        modality_scores.append(score)
                except Exception as e:
                    logger.debug(
                        "Modality score failed for %s fold %d: %s",
                        mod_name,
                        fold_idx + 1,
                        e,
                    )

            logger.info("  Test predictions: %.1fs", time.monotonic() - _t2)
            y_pred_fold = meta.predict_proba(test_preds)[:, 1]
            y_oof[test_idx] = y_pred_fold
            y_true[test_idx] = y_te

            if fold_subset is None:
                per_fold_oos.append(
                    {
                        "fold_idx": fold_idx,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "y_tr": y_tr,
                        "y_te": y_te,
                        "oof_train": oof_train,
                        "test_preds": test_preds,
                        "modality_names": list(mod_names),
                        "meta": meta,
                    }
                )

            # Permutation importance on held-out test data
            try:
                X_meta_test = meta.transform_meta_features(test_preds)
                perm = permutation_importance(
                    meta.calibrated_model_,
                    X_meta_test,
                    y_te,
                    n_repeats=20,
                    random_state=self.random_state,
                    scoring="roc_auc",
                    n_jobs=1 if cuda_available() else self.n_jobs,
                )
                fold_perm_importances.append(
                    {
                        "fold": fold_idx + 1,
                        "feature_names": meta.feature_names_,
                        # ``Bunch`` is a dict subclass, so the arrays are
                        # reachable by key; ``asarray`` pins the element type.
                        "importances_mean": np.asarray(perm["importances_mean"]),
                        "importances_std": np.asarray(perm["importances_std"]),
                    }
                )
            except Exception as e:
                logger.warning(
                    "Fold %d permutation importance failed: %s", fold_idx + 1, e
                )

            try:
                fold_roc = float(roc_auc_score(y_te, y_pred_fold))
                fold_pr = float(average_precision_score(y_te, y_pred_fold))
            except ValueError:
                fold_roc = fold_pr = float("nan")
            fold_result: FoldResult = {
                "fold": fold_idx + 1,
                "roc_auc": fold_roc,
                "pr_auc": fold_pr,
            }
            fold_results.append(fold_result)
            logger.info(
                "  Fold %d: ROC-AUC=%.3f, PR-AUC=%.3f", fold_idx + 1, fold_roc, fold_pr
            )

            # Save fold artifact for distributed computation
            if fold_subset is not None:
                # Collect modality scores for this fold only
                fold_modality_scores = [
                    s for s in modality_scores if s["fold"] == fold_idx + 1
                ]
                # Collect perm importance for this fold only
                fold_perm = next(
                    (p for p in fold_perm_importances if p["fold"] == fold_idx + 1),
                    None,
                )
                self._save_fold_artifact(
                    fold_idx,
                    {
                        "fold_idx": fold_idx,
                        "test_idx": test_idx,
                        "y_pred_fold": y_pred_fold,
                        "y_te": y_te,
                        "fold_result": fold_result,
                        "modality_scores_fold": fold_modality_scores,
                        "fold_perm_importance": fold_perm,
                        "config_hash": self._config_hash(),
                    },
                )

        # When running a fold subset, skip CSV/plot generation
        if fold_subset is not None:
            computed: list[int] = fold_subset if fold_results else []
            logger.info("Fold subset complete: computed folds %s", computed)
            return {
                "y_true": y_true,
                "y_pred_proba": y_oof,
                "fold_results": fold_results,
                "modality_scores": modality_scores,
                "feature_importance_permutation": None,
                "partial": True,
                "computed_folds": computed,
            }

        pd.DataFrame(fold_results).to_csv(
            self.output_dir / "cv_fold_results.csv", index=False
        )
        mod_df = pd.DataFrame(modality_scores)
        mod_df.to_csv(self.output_dir / "modality_scores_cv.csv", index=False)

        if not mod_df.empty:
            fig = plot_modality_contribution_comparison(mod_df)
            save_figure_for_latex(fig, self.output_dir / "modality_contributions")

        # Aggregate permutation importances across folds
        perm_importance_df = None
        if fold_perm_importances:
            feature_names = fold_perm_importances[0]["feature_names"]
            all_means = np.array([f["importances_mean"] for f in fold_perm_importances])
            perm_importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance_mean": all_means.mean(axis=0),
                    "importance_std": all_means.std(axis=0),
                }
            ).sort_values("importance_mean", ascending=False)
            perm_importance_df.to_csv(
                self.output_dir / "feature_importance_permutation.csv", index=False
            )

        # Stash OoS artifacts on the instance so _run_post_cv_steps can use
        # them without threading a large list through the return type.
        self._per_fold_oos = per_fold_oos

        return {
            "y_true": y_true,
            "y_pred_proba": y_oof,
            "fold_results": fold_results,
            "modality_scores": modality_scores,
            "feature_importance_permutation": perm_importance_df,
        }

    def _permutation_test_cv(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> PermutationTestResult:
        """Permutation test on nested CV OOF predictions.

        Permutes labels against fixed held-out predictions to build a null
        distribution. Valid because OOF predictions are truly held-out;
        under H0 any label assignment is equally likely.

        Uses vectorised batch computation for performance.
        """
        rng = np.random.default_rng(self.random_state)
        real_score = float(average_precision_score(y_true, y_pred_proba))

        # Vectorised: generate all permutation indices at once
        n = len(y_true)
        perm_indices = np.column_stack(
            [rng.permutation(n) for _ in range(self.meta_permutation_iterations)]
        )
        perm_scores = np.array(
            [
                average_precision_score(y_true[perm_indices[:, i]], y_pred_proba)
                for i in range(self.meta_permutation_iterations)
            ]
        )

        p_value = float(
            (np.sum(perm_scores >= real_score) + 1)
            / (self.meta_permutation_iterations + 1)
        )
        summary: PermutationTestResult = {
            "score": real_score,
            "p_value": p_value,
            "perm_mean": float(np.mean(perm_scores)),
            "n_permutations": self.meta_permutation_iterations,
        }
        with (self.output_dir / "meta_permutation_test.json").open("w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Permutation test: score=%.3f, p=%.4f", real_score, p_value)
        return summary

    def _compare_modalities(
        self,
        modality_scores_df: pd.DataFrame,
        n_total: int,
    ) -> pd.DataFrame | None:
        """Run Nadeau-Bengio corrected pairwise modality comparisons."""
        from .statistical_tests import compare_modalities_pairwise

        try:
            comparisons = compare_modalities_pairwise(
                modality_scores_df,
                # Mental health is the modality the manuscript's claim is
                # about, so it is the one every other modality is tested
                # against.
                reference_modality="mental_health",
                metric="pr_auc",
                n_total=n_total,
                n_folds=self.n_outer_folds,
            )
            comparisons.to_csv(
                self.output_dir / "modality_comparisons_nb.csv", index=False
            )
            logger.info(
                "Modality comparisons (NB + Holm): %d comparisons", len(comparisons)
            )
        except Exception as e:
            logger.warning("Modality comparison failed: %s", e)
            return None
        else:
            return comparisons

    def _train_final_model(
        self,
        X_dict: dict[str, pd.DataFrame],
        y: np.ndarray,
        pipelines: dict[str, ConfoundedEstimator],
        confounders: pd.DataFrame | None,
    ) -> FinalModelResult:
        # ``return_fitted=True`` additionally fits each per-modality
        # pipeline on the FULL source-cohort sample (not the CV folds).
        # Those fitted pipelines are needed for transfer-validation runs
        # (scripts/pipeline/03_apply_transfer.py) so the target
        # cohort can be scored without re-training.
        oof, mod_names, fitted_pipelines = create_oof_predictions(
            pipelines,
            X_dict,
            y,
            self.n_inner_folds,
            self.random_state,
            confounders,
            return_fitted=True,
        )
        if fitted_pipelines is None:  # pragma: no cover — return_fitted is True
            raise RuntimeError(
                "create_oof_predictions did not return fitted pipelines even "
                "though return_fitted=True; cannot persist a transferable "
                "final model."
            )
        meta = train_meta_learner(
            oof,
            y,
            mod_names,
            add_interactions=False,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            hyperparam_iterations=self.hyperparam_iterations,
        )
        importance = meta.get_feature_importance()
        importance.to_csv(self.output_dir / "feature_importance.csv", index=False)

        # The source-cohort confounder frame is persisted alongside the
        # pipelines so transfer runs can compute KS-distance
        # between source and target confounder distributions without
        # re-reading the source parquets.  Empty frame when the pipeline
        # was run without confounders.
        #
        # Privacy: store a column-wise shuffled snapshot with a fresh
        # RangeIndex.  KS distance is a per-column statistic, so this
        # preserves the transfer-report's drift signal exactly while
        # making it impossible to reconstruct per-participant
        # (age, sex, centre) tuples from the pickle if it is shared
        # outside the source-cohort access perimeter.
        if confounders is not None:
            rng_priv = np.random.default_rng(self.random_state)
            shuffled: dict[str, np.ndarray] = {}
            for col in confounders.columns:
                vals = confounders[col].to_numpy(copy=True)
                rng_priv.shuffle(vals)
                shuffled[col] = vals
            confounders_train = pd.DataFrame(shuffled)
        else:
            confounders_train = pd.DataFrame(index=range(len(y)))

        # Capture the exact column layout that the per-modality pipelines
        # saw at fit time.  A transfer run must reindex its target-cohort
        # DataFrames to these columns (filling absent columns with NaN)
        # before calling predict_proba, otherwise the KNN imputer — which
        # stores feature_names_in_ from fit — rejects the input.  Storing
        # the layout at the pipeline level is simpler than walking the
        # fitted sklearn steps.
        modality_column_layout: dict[str, list[str]] = {
            name: list(X_dict[name].columns) for name in mod_names
        }

        final: FinalModelResult = {
            "meta_learner": meta,
            "modality_names": mod_names,
            "feature_importance": importance,
            "oof_predictions": oof,
            "modality_pipelines": fitted_pipelines,
            "modality_column_layout": modality_column_layout,
            "confounders_train": confounders_train,
        }
        with (self.output_dir / "final_model.pkl").open("wb") as f:
            pickle.dump(final, f)
        return final

    def _compute_shap(self, final_model: FinalModelResult) -> ShapResults | None:
        """Compute SHAP values on the final meta-learner."""
        try:
            from .shap_analysis import compute_meta_learner_shap

            shap_results = compute_meta_learner_shap(
                final_model["meta_learner"],
                final_model["oof_predictions"],
                final_model["modality_names"],
            )
            shap_results["feature_importance"].to_csv(
                self.output_dir / "shap_feature_importance.csv", index=False
            )
            shap_results["modality_importance"].to_csv(
                self.output_dir / "shap_modality_importance.csv", index=False
            )
            logger.info("SHAP analysis completed")
        except Exception as e:
            logger.warning("SHAP analysis failed: %s", e)
            return None
        else:
            return shap_results

    def _modality_ablation(
        self,
        meta_learner: MetaLearner,
        oof_predictions: np.ndarray,
        y: np.ndarray,
        modality_names: list[str],
    ) -> pd.DataFrame | None:
        """Modality ablation: replace each modality's OoF column with NaN.

        Measures performance drop when a modality is removed.
        XGBoost handles NaN natively, so no retraining is needed.
        """
        try:
            # Full-model performance
            y_pred_full = meta_learner.predict_proba(oof_predictions)[:, 1]
            roc_full = float(roc_auc_score(y, y_pred_full))
            pr_full = float(average_precision_score(y, y_pred_full))

            rows = []
            for i, mod_name in enumerate(modality_names):
                oof_dropped = oof_predictions.copy()
                oof_dropped[:, i] = np.nan

                y_pred_dropped = meta_learner.predict_proba(oof_dropped)[:, 1]
                roc_dropped = float(roc_auc_score(y, y_pred_dropped))
                pr_dropped = float(average_precision_score(y, y_pred_dropped))

                rows.append(
                    {
                        "modality": mod_name,
                        "roc_auc_full": roc_full,
                        "roc_auc_dropped": roc_dropped,
                        "delta_roc_auc": roc_full - roc_dropped,
                        "pr_auc_full": pr_full,
                        "pr_auc_dropped": pr_dropped,
                        "delta_pr_auc": pr_full - pr_dropped,
                    }
                )

            df = pd.DataFrame(rows).sort_values("delta_pr_auc", ascending=False)
            df.to_csv(self.output_dir / "modality_ablation.csv", index=False)
            logger.info("Modality ablation completed for %d modalities", len(rows))
        except Exception as e:
            logger.warning("Modality ablation failed: %s", e)
            return None
        else:
            return df

    def _incremental_performance(
        self,
        oof_predictions: np.ndarray,
        y: np.ndarray,
        modality_names: list[str],
        baseline_modality: str = "demographics",
    ) -> pd.DataFrame | None:
        """Compute incremental performance over baseline for each modality.

        Trains a meta-learner on [baseline + modality] OoF predictions
        and compares against a baseline-only meta-learner.
        """
        try:
            if baseline_modality not in modality_names:
                logger.warning(
                    "Baseline modality '%s' not found; skipping incremental",
                    baseline_modality,
                )
                return None

            baseline_idx = modality_names.index(baseline_modality)
            baseline_oof = oof_predictions[:, [baseline_idx]]

            # Baseline-only meta-learner.  hyperparam_iterations=0 uses
            # default XGBoost parameters (no randomized search) so that all
            # pairwise [baseline + modality] comparisons are evaluated under
            # identical meta-learner settings.
            meta_baseline = train_meta_learner(
                baseline_oof,
                y,
                [baseline_modality],
                add_interactions=False,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                hyperparam_iterations=0,
            )
            y_pred_base = meta_baseline.predict_proba(baseline_oof)[:, 1]
            pr_baseline = float(average_precision_score(y, y_pred_base))

            rows = []
            for i, mod_name in enumerate(modality_names):
                if mod_name == baseline_modality:
                    continue

                # Combine baseline + this modality
                augmented_oof = np.column_stack([baseline_oof, oof_predictions[:, [i]]])
                augmented_names = [baseline_modality, mod_name]

                meta_aug = train_meta_learner(
                    augmented_oof,
                    y,
                    augmented_names,
                    add_interactions=False,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    hyperparam_iterations=0,
                )
                y_pred_aug = meta_aug.predict_proba(augmented_oof)[:, 1]
                pr_augmented = float(average_precision_score(y, y_pred_aug))

                rows.append(
                    {
                        "modality": mod_name,
                        "pr_auc_baseline": pr_baseline,
                        "pr_auc_augmented": pr_augmented,
                        "delta_pr_auc": pr_augmented - pr_baseline,
                    }
                )

            df = pd.DataFrame(rows).sort_values("delta_pr_auc", ascending=False)
            df.to_csv(self.output_dir / "incremental_performance.csv", index=False)
            logger.info("Incremental performance computed for %d modalities", len(rows))
        except Exception as e:
            logger.warning("Incremental performance failed: %s", e)
            return None
        else:
            return df

    def _modality_ablation_oos(
        self,
        per_fold_oos: list[PerFoldOoS],
    ) -> pd.DataFrame | None:
        """Out-of-sample modality ablation across outer folds.

        For each fold the test-fold base-learner predictions are masked with
        ``NaN`` in one modality column at a time and re-scored by the
        fold-specific meta-learner.  Per-fold scores are concatenated across
        folds so each participant's ablated prediction comes from a
        meta-learner that never saw them during training.

        This complements :meth:`_modality_ablation`, which operates on the
        single final-model OoF matrix (in-sample at the meta-learner level).
        """
        if not per_fold_oos:
            logger.info("Skipping OoS modality ablation: no per-fold artifacts")
            return None
        try:
            modality_names = per_fold_oos[0]["modality_names"]
            n_total = sum(len(f["y_te"]) for f in per_fold_oos)
            y_true_cat = np.empty(n_total, dtype=np.int8)
            y_full_cat = np.empty(n_total, dtype=np.float64)
            y_dropped_cat = {
                mod: np.empty(n_total, dtype=np.float64) for mod in modality_names
            }

            offset = 0
            for fold in per_fold_oos:
                meta = fold["meta"]
                test_preds = fold["test_preds"]
                n_te = len(fold["y_te"])
                y_true_cat[offset : offset + n_te] = fold["y_te"]
                y_full_cat[offset : offset + n_te] = meta.predict_proba(test_preds)[
                    :, 1
                ]
                for i, mod in enumerate(modality_names):
                    masked = test_preds.copy()
                    masked[:, i] = np.nan
                    y_dropped_cat[mod][offset : offset + n_te] = meta.predict_proba(
                        masked
                    )[:, 1]
                offset += n_te

            roc_full = float(roc_auc_score(y_true_cat, y_full_cat))
            pr_full = float(average_precision_score(y_true_cat, y_full_cat))

            rows = []
            for mod in modality_names:
                y_drop = y_dropped_cat[mod]
                roc_dropped = float(roc_auc_score(y_true_cat, y_drop))
                pr_dropped = float(average_precision_score(y_true_cat, y_drop))
                rows.append(
                    {
                        "modality": mod,
                        "roc_auc_full": roc_full,
                        "roc_auc_dropped": roc_dropped,
                        "delta_roc_auc": roc_full - roc_dropped,
                        "pr_auc_full": pr_full,
                        "pr_auc_dropped": pr_dropped,
                        "delta_pr_auc": pr_full - pr_dropped,
                    }
                )

            df = pd.DataFrame(rows).sort_values("delta_pr_auc", ascending=False)
            df.to_csv(self.output_dir / "modality_ablation_oos.csv", index=False)
            logger.info(
                "OoS modality ablation: full PR-AUC=%.3f, %d modalities, "
                "max delta=%.4f",
                pr_full,
                len(rows),
                float(df["delta_pr_auc"].iloc[0]) if len(df) else 0.0,
            )
        except Exception as e:
            logger.warning("OoS modality ablation failed: %s", e)
            return None
        else:
            return df

    def _incremental_performance_oos(
        self,
        per_fold_oos: list[PerFoldOoS],
        baseline_modality: str = "demographics",
    ) -> pd.DataFrame | None:
        """Out-of-sample incremental performance across outer folds.

        For each fold we train a baseline-only meta-learner and one
        [baseline + modality] meta-learner per non-baseline modality on the
        training-fold OoF stacking matrix, then evaluate on the held-out
        test-fold base-learner predictions.  Test-fold scores are
        concatenated across folds before computing PR-AUC, so the
        comparison is genuinely out-of-sample.

        Uses ``hyperparam_iterations=0`` (default XGBoost params) for all
        pairwise comparisons, matching :meth:`_incremental_performance`.
        """
        if not per_fold_oos:
            logger.info("Skipping OoS incremental performance: no per-fold artifacts")
            return None
        try:
            modality_names = per_fold_oos[0]["modality_names"]
            if baseline_modality not in modality_names:
                logger.warning(
                    "Baseline modality '%s' not found in OoS folds; skipping",
                    baseline_modality,
                )
                return None
            baseline_idx = modality_names.index(baseline_modality)
            other_mods = [m for m in modality_names if m != baseline_modality]

            n_total = sum(len(f["y_te"]) for f in per_fold_oos)
            y_true_cat = np.empty(n_total, dtype=np.int8)
            y_base_cat = np.empty(n_total, dtype=np.float64)
            y_aug_cat = {mod: np.empty(n_total, dtype=np.float64) for mod in other_mods}

            offset = 0
            for fold in per_fold_oos:
                oof_tr = fold["oof_train"]
                test_preds = fold["test_preds"]
                n_te = len(fold["y_te"])
                y_true_cat[offset : offset + n_te] = fold["y_te"]

                meta_base = train_meta_learner(
                    oof_tr[:, [baseline_idx]],
                    fold["y_tr"],
                    [baseline_modality],
                    add_interactions=False,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    hyperparam_iterations=0,
                )
                y_base_cat[offset : offset + n_te] = meta_base.predict_proba(
                    test_preds[:, [baseline_idx]]
                )[:, 1]

                for mod in other_mods:
                    i = modality_names.index(mod)
                    train_cols = oof_tr[:, [baseline_idx, i]]
                    test_cols = test_preds[:, [baseline_idx, i]]
                    meta_aug = train_meta_learner(
                        train_cols,
                        fold["y_tr"],
                        [baseline_modality, mod],
                        add_interactions=False,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                        hyperparam_iterations=0,
                    )
                    y_aug_cat[mod][offset : offset + n_te] = meta_aug.predict_proba(
                        test_cols
                    )[:, 1]

                offset += n_te

            pr_baseline = float(average_precision_score(y_true_cat, y_base_cat))
            rows = []
            for mod in other_mods:
                pr_augmented = float(
                    average_precision_score(y_true_cat, y_aug_cat[mod])
                )
                rows.append(
                    {
                        "modality": mod,
                        "pr_auc_baseline": pr_baseline,
                        "pr_auc_augmented": pr_augmented,
                        "delta_pr_auc": pr_augmented - pr_baseline,
                    }
                )

            df = pd.DataFrame(rows).sort_values("delta_pr_auc", ascending=False)
            df.to_csv(self.output_dir / "incremental_performance_oos.csv", index=False)
            logger.info(
                "OoS incremental performance: baseline PR-AUC=%.3f, top gain %s=%.4f",
                pr_baseline,
                df["modality"].iloc[0] if len(df) else "—",
                float(df["delta_pr_auc"].iloc[0]) if len(df) else 0.0,
            )
        except Exception as e:
            logger.warning("OoS incremental performance failed: %s", e)
            return None
        else:
            return df

    def _evaluate(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> EvaluationResult:
        evaluation = evaluate_model_comprehensive(
            y_true,
            y_pred_proba,
            model_name="PCC Multimodal Pipeline",
            n_bootstrap=self.n_bootstrap_eval,
            random_state=self.random_state,
            output_dir=self.output_dir,
            save_latex=True,
            n_jobs=self.n_jobs,
        )
        pd.DataFrame([evaluation["metrics"]]).to_csv(
            self.output_dir / "metrics.csv", index=False
        )
        ci = evaluation.get("ci_results", {})
        if ci:
            ci_out = {k: [float(v[0]), float(v[1])] for k, v in ci.items()}
            with (self.output_dir / "confidence_intervals.json").open("w") as f:
                json.dump(ci_out, f, indent=2)
        ts = evaluation.get("threshold_summary", {})
        if ts:
            pd.DataFrame([ts]).to_json(
                self.output_dir / "threshold_summary.json", orient="records", indent=2
            )
        return evaluation

    def _save_results(self, results: PipelineResult) -> None:
        pd.DataFrame(
            {
                "y_true": results["cv_results"]["y_true"],
                "y_pred_proba": results["cv_results"]["y_pred_proba"],
            }
        ).to_csv(self.output_dir / "predictions.csv", index=False)
        pd.DataFrame([results["config"]]).to_csv(
            self.output_dir / "config.csv", index=False
        )
        logger.info("Results saved to %s", self.output_dir)

    def _get_config(self) -> PipelineConfig:
        return {
            "n_outer_folds": self.n_outer_folds,
            "n_inner_folds": self.n_inner_folds,
            "meta_model": "gbdt",
            "orthogonalize": self.orthogonalize,
            "random_state": self.random_state,
            # UTC with an explicit offset: this record is read by people in
            # other timezones, and a bare local timestamp cannot be resolved.
            "timestamp": datetime.now(UTC).isoformat(),
            "cohort": self.cohort,
            "stack_variant": self.stack_variant,
            "package_versions": package_versions(),
            # Which XGBoost backend fitted the meta-learner. In the hash for
            # the same reason as the stack stamp: `cuda` and `cpu` build
            # their histograms differently, so folds computed on a machine
            # with a GPU and folds computed on one without are not the same
            # analysis — and nothing else in this record would say so, since
            # the device is detected rather than configured.
            "device": xgb_device(),
            # Which data went in.  `run` fills this from the frames it is
            # given; a merge is handed it.  The fold merge validates `y_te`
            # per fold, so a changed outcome is caught there — this is what
            # catches a changed *feature* matrix, which that check cannot
            # see.
            "data_fingerprint": self.data_fingerprint or "unrecorded",
            "target": self.target,
            "clean_controls": self.clean_controls,
            "split_mh_submodalities": self.split_mh_submodalities,
            "amendment_features": self.amendment_features,
            "hyperparam_iterations": self.hyperparam_iterations,
            # Stability selection resamples this many times per modality, so
            # it decides which features survive into every base learner.
            # ``n_jobs``, ``meta_permutation_iterations`` and
            # ``n_bootstrap_eval`` are deliberately absent: they change how
            # long a run takes, how tight its p-value is and how wide its
            # confidence interval is, but not the model that was fitted.
            "n_subsamples": self.n_subsamples,
        }


@overload
def run_pcc_pipeline(
    y: pd.Series,
    X_dict: dict[str, pd.DataFrame],
    output_dir: Path | str,
    confounders: pd.DataFrame | None = ...,
    n_outer_folds: int = ...,
    n_inner_folds: int = ...,
    orthogonalize: bool = ...,
    random_state: int = ...,
    n_jobs: int = ...,
    meta_permutation_iterations: int = ...,
    n_subsamples: int = ...,
    n_bootstrap_eval: int = ...,
    fold_subset: None = ...,
    cohort: str = ...,
    stack_variant: str = ...,
    target: str = ...,
    clean_controls: bool = ...,
    split_mh_submodalities: bool = ...,
    amendment_features: bool = ...,
    hyperparam_iterations: int = ...,
) -> PipelineResult: ...


@overload
def run_pcc_pipeline(
    y: pd.Series,
    X_dict: dict[str, pd.DataFrame],
    output_dir: Path | str,
    confounders: pd.DataFrame | None = ...,
    n_outer_folds: int = ...,
    n_inner_folds: int = ...,
    orthogonalize: bool = ...,
    random_state: int = ...,
    n_jobs: int = ...,
    meta_permutation_iterations: int = ...,
    n_subsamples: int = ...,
    n_bootstrap_eval: int = ...,
    *,
    fold_subset: list[int],
    cohort: str = ...,
    stack_variant: str = ...,
    target: str = ...,
    clean_controls: bool = ...,
    split_mh_submodalities: bool = ...,
    amendment_features: bool = ...,
    hyperparam_iterations: int = ...,
) -> CVResults: ...


def run_pcc_pipeline(
    y: pd.Series,
    X_dict: dict[str, pd.DataFrame],
    output_dir: Path | str,
    confounders: pd.DataFrame | None = None,
    n_outer_folds: int = 10,
    n_inner_folds: int = 5,
    orthogonalize: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
    meta_permutation_iterations: int = 1000,
    n_subsamples: int = 100,
    n_bootstrap_eval: int = 1000,
    fold_subset: list[int] | None = None,
    cohort: str = "all",
    stack_variant: str = "full",
    target: str = "bahmer",
    clean_controls: bool = True,
    split_mh_submodalities: bool = False,
    amendment_features: bool = True,
    hyperparam_iterations: int = META_HYPERPARAM_ITERATIONS,
) -> CVResults | PipelineResult:
    """Convenience wrapper to run the full pipeline.

    ``fold_subset`` decides the return type: a subset run computes only those
    outer folds and returns the partial :class:`CVResults`, a full run returns
    the complete :class:`PipelineResult`. The overloads above let callers see
    which one they get instead of narrowing the union themselves.

    Parameters
    ----------
    fold_subset : list[int] | None
        If set, only compute these outer fold indices (0-based).
        Fold artifacts are saved to output_dir for later merging.
    cohort, stack_variant, target, clean_controls, split_mh_submodalities,
    amendment_features
        Provenance tags mirrored into ``config.csv`` for reproducibility.
        Must match the values passed to ``load_pipeline_data`` upstream —
        they describe the data this run was given, and nothing here
        re-derives or validates them.
    hyperparam_iterations : int
        Draws in the meta-learner's randomised hyperparameter search.
        Recorded like a provenance tag but actually applied; pass the
        earlier run's width of 10 to fit a control run against it.  Values
        above :data:`META_HYPERPARAM_ITERATIONS` are clamped to it, since
        that is the ceiling the search enforces on itself.
    """
    pipeline = PCCMultimodalPipeline(
        output_dir=output_dir,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        orthogonalize=orthogonalize,
        random_state=random_state,
        n_jobs=n_jobs,
        meta_permutation_iterations=meta_permutation_iterations,
        n_subsamples=n_subsamples,
        n_bootstrap_eval=n_bootstrap_eval,
        cohort=cohort,
        stack_variant=stack_variant,
        target=target,
        clean_controls=clean_controls,
        split_mh_submodalities=split_mh_submodalities,
        amendment_features=amendment_features,
        hyperparam_iterations=hyperparam_iterations,
    )
    return pipeline.run(
        y,
        X_dict,
        confounders,
        fold_subset=fold_subset,
    )


def merge_fold_results(
    fold_dir: Path | str,
    y: pd.Series,
    X_dict: dict[str, pd.DataFrame],
    confounders: pd.DataFrame | None = None,
    n_outer_folds: int = 10,
    n_inner_folds: int = 5,
    orthogonalize: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
    meta_permutation_iterations: int = 1000,
    n_subsamples: int = 100,
    n_bootstrap_eval: int = 1000,
    output_dir: Path | str | None = None,
    cohort: str = "all",
    stack_variant: str = "full",
    target: str = "bahmer",
    clean_controls: bool = True,
    split_mh_submodalities: bool = False,
    amendment_features: bool = True,
    hyperparam_iterations: int = META_HYPERPARAM_ITERATIONS,
) -> PipelineResult:
    """Merge fold artifacts from distributed runs and complete the pipeline.

    Provenance goes to ``provenance_merge.json`` rather than over the
    ``provenance.json`` the fold runs left in the same directory: the fold
    run's record answers what produced the cross-validation, this one
    answers what produced the final model, and a merge on a second machine
    must not overwrite the first answer with its own.

    Parameters
    ----------
    fold_dir : Path
        Directory containing fold_*.pkl files.
    output_dir : Path | None
        Output directory for merged results. Defaults to fold_dir.
    cohort, stack_variant, target, clean_controls, split_mh_submodalities,
    amendment_features, hyperparam_iterations
        The configuration the merged run is finished under; pass the same
        values the fold-subset runs were given.  A merge is not a passive
        concatenation — it fits the final meta-learner and writes
        ``config.csv`` — so these are checked against the folds' config hash
        and a mismatch aborts the merge.  Omitting a flag here that the fold
        runs were given would otherwise fit and publish a model under
        settings no fold ever saw.

    The same check covers the environment and the data: the hash carries
    the library stack, the XGBoost device and a digest of ``y``/``X_dict``,
    so folds computed on a second machine against a different stack or a
    different vintage of the processed parquets fail here rather than
    merging into a model that no fold was scored under.
    """
    fold_dir = Path(fold_dir)
    if output_dir is None:
        output_dir = fold_dir
    output_dir = Path(output_dir)

    # 1. Load all fold artifacts
    fold_files = sorted(fold_dir.glob("fold_*.pkl"))
    if not fold_files:
        raise FileNotFoundError(f"No fold_*.pkl files found in {fold_dir}")

    artifacts: dict[int, FoldArtifact] = {}
    for fp in fold_files:
        try:
            with fp.open("rb") as f:
                art = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load {fp}: {e}") from e
        artifacts[art["fold_idx"]] = art

    # 2. Check completeness
    expected = set(range(n_outer_folds))
    found = set(artifacts.keys())
    missing = expected - found
    if missing:
        raise ValueError(
            f"Missing folds: {sorted(missing)}. "
            f"Found folds: {sorted(found)}. "
            f"Expected {n_outer_folds} folds."
        )

    # 3. Config hash consistency between the folds
    hashes = {idx: art["config_hash"] for idx, art in artifacts.items()}
    unique_hashes = set(hashes.values())
    if len(unique_hashes) > 1:
        raise ValueError(
            f"Config mismatch between folds. "
            f"Found {len(unique_hashes)} different configs: {hashes}"
        )
    (fold_hash,) = unique_hashes

    # 4. Use provided labels directly (Bahmer any_pcs)
    y_arr = np.asarray(y, dtype=np.int8)

    # 5. Validate y values and reassemble arrays
    n = len(y_arr)
    y_oof = np.zeros(n)
    y_true = np.zeros(n)
    fold_results: list[FoldResult] = []
    modality_scores: list[ModalityScore] = []
    fold_perm_importances: list[FoldPermImportance] = []

    for fold_idx in range(n_outer_folds):
        art = artifacts[fold_idx]
        test_idx = art["test_idx"]

        # Validate that targets match
        expected_y_te = y_arr[test_idx]
        if not np.array_equal(expected_y_te, art["y_te"]):
            raise ValueError(
                f"Fold {fold_idx}: y_te mismatch. The data used for this fold "
                f"differs from the current data. Ensure the same input data is used."
            )

        y_oof[test_idx] = art["y_pred_fold"]
        y_true[test_idx] = art["y_te"]
        fold_results.append(art["fold_result"])
        modality_scores.extend(art["modality_scores_fold"])
        if art["fold_perm_importance"] is not None:
            fold_perm_importances.append(art["fold_perm_importance"])

    # 6. Create pipeline instance for post-CV steps.  The fingerprint is
    #    taken over the data this merge was handed, so the hash check below
    #    compares it against the data the folds were computed on — the
    #    feature-side counterpart to the per-fold `y_te` check above.
    pipeline = PCCMultimodalPipeline(
        output_dir=output_dir,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        orthogonalize=orthogonalize,
        random_state=random_state,
        n_jobs=n_jobs,
        meta_permutation_iterations=meta_permutation_iterations,
        n_subsamples=n_subsamples,
        n_bootstrap_eval=n_bootstrap_eval,
        cohort=cohort,
        stack_variant=stack_variant,
        target=target,
        clean_controls=clean_controls,
        split_mh_submodalities=split_mh_submodalities,
        amendment_features=amendment_features,
        hyperparam_iterations=hyperparam_iterations,
        provenance_filename="provenance_merge.json",
        data_fingerprint=fingerprint_analysis_data(y, X_dict),
    )

    # 7. Config hash consistency between the folds and the merge itself.
    #    Checking the artifacts only against each other passes a merge
    #    invoked with the fold runs' flags left off — every fold agrees, and
    #    the meta-learner below is then fitted, and ``config.csv`` written,
    #    under whatever the defaults happen to be.
    merge_hash = pipeline._config_hash()
    if merge_hash != fold_hash:
        # The artifacts store the hash and not the values it was taken over,
        # so the differing field cannot be pointed at directly; naming the
        # fields and what the merge holds them at is what lets the operator
        # find it.
        merge_config = dict(pipeline._get_config())
        recorded = "\n".join(
            f"  {field}={merge_config[field]!r}"
            for field in sorted(merge_config)
            if field != "timestamp"
        )
        raise ValueError(
            f"Config mismatch between the folds and this merge: the folds "
            f"agree on {fold_hash}, the merge would record {merge_hash}. "
            f"One of the fields below differs from the fold-subset runs; "
            f"they are shown as this merge was invoked.\n"
            f"{recorded}\n"
            f"Re-run the merge with the arguments the fold-subset runs were "
            f"given, from the same environment."
        )

    # 8. Save CSV/plots that _nested_cv would have produced
    pd.DataFrame(fold_results).to_csv(
        pipeline.output_dir / "cv_fold_results.csv", index=False
    )
    mod_df = pd.DataFrame(modality_scores)
    mod_df.to_csv(pipeline.output_dir / "modality_scores_cv.csv", index=False)

    if not mod_df.empty:
        fig = plot_modality_contribution_comparison(mod_df)
        save_figure_for_latex(fig, pipeline.output_dir / "modality_contributions")

    # Aggregate permutation importances
    perm_importance_df = None
    if fold_perm_importances:
        feature_names = fold_perm_importances[0]["feature_names"]
        all_means = np.array([f["importances_mean"] for f in fold_perm_importances])
        perm_importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": all_means.mean(axis=0),
                "importance_std": all_means.std(axis=0),
            }
        ).sort_values("importance_mean", ascending=False)
        perm_importance_df.to_csv(
            pipeline.output_dir / "feature_importance_permutation.csv", index=False
        )

    cv_results: CVResults = {
        "y_true": y_true,
        "y_pred_proba": y_oof,
        "fold_results": fold_results,
        "modality_scores": modality_scores,
        "feature_importance_permutation": perm_importance_df,
    }

    # 9. Build pipelines and run post-CV steps
    factory = ModalityPipelineFactory(
        random_state=random_state,
        orthogonalize=orthogonalize,
        cv=n_inner_folds,
        n_jobs=n_jobs,
        n_subsamples=n_subsamples,
    )
    pipelines = pipeline._create_pipelines(factory, list(X_dict.keys()))

    return pipeline._run_post_cv_steps(
        cv_results,
        X_dict,
        y_arr,
        pipelines,
        confounders,
    )
