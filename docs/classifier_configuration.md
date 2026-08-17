# Classifier Configuration

This document describes the `LogisticRegressionCV` configuration for each
modality pipeline, including the choice of scoring metric and modality-specific
hyperparameter ranges.

---

## Scoring metric: `average_precision`

The default `LogisticRegressionCV` scoring (`log_loss`) selects regularisation
strength to minimise cross-entropy.  With low-prevalence outcomes (~11% PCC
positive), this favours very high regularisation (e.g. C=0.0001 with L1),
which shrinks all coefficients to zero and produces a null model
(ROC-AUC ~ 0.5).

Using `scoring='average_precision'` instead selects the regularisation strength
that maximises the area under the precision-recall curve.  This metric is
sensitive to the model's ability to rank positive cases above negative ones,
which is the relevant objective for downstream stacking.

Empirical validation: switching from `log_loss` to `average_precision` improved
out-of-fold ROC-AUC from 0.509 to 0.605 on the demographics modality, matching
the expected ~0.614 from the reference paper.

---

## Common settings

All modality classifiers share (unless overridden per modality):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `solver` | `saga` | Supports L1, L2, and elastic net penalties |
| `cv` | 3 | Inner CV for hyperparameter selection |
| `max_iter` | 10000 | Convergence for high-dimensional modalities at low regularisation |
| `tol` | 1e-4 | Standard convergence tolerance |
| `class_weight` | `balanced` | Upweights minority class (PCC+) |
| `penalty` | `elasticnet` | Implicit via `l1_ratios` parameter |

---

## Modality-specific hyperparameters

### Demographics (3 features)

| Parameter | Value |
|-----------|-------|
| `l1_ratios` | `[0.0]` (pure L2) |
| `Cs` | `logspace(-2, 2, 5)` |
| `solver` | `lbfgs` |

**Rationale:** Only 3 features (age, sex, study center).  No sparsity needed;
L2 regularisation is sufficient.  Uses `lbfgs` instead of the default `saga`
because only L2 is needed and `lbfgs` converges reliably on small feature sets,
whereas `saga` struggles with `class_weight="balanced"` on one-hot-encoded
centre dummies (~20 features).

### SES (~75 features after one-hot encoding)

| Parameter | Value |
|-----------|-------|
| `l1_ratios` | `[0.0, 1.0]` (L2, L1) |
| `Cs` | `logspace(-3, 2, 5)` |
| `max_iter` | `20000` |

**Rationale:** After one-hot encoding of `isco_major` (10 categories) and
`isco_submajor` (43 categories), plus other features, the feature space is
large.  L2 handles grouped categorical features, L1 provides sparsity.
Elastic net (0.5) dropped — it rarely outperforms pure L1 or L2 here.
The higher `max_iter` is needed because the many one-hot dummies from
`isco_submajor` slow SAGA convergence.

### Cognitive (~8 features)

| Parameter | Value |
|-----------|-------|
| `l1_ratios` | `[0.5, 1.0]` (elastic net, L1) |
| `Cs` | `logspace(-3, 2, 5)` |

**Rationale:** L1 sparsity focuses the model on the most predictive
cognitive measures. The stability-selection stage upstream does not reduce the
set at the configured threshold (see below), so this penalty is where feature
selection actually happens.

### Physical Activity (~17 features)

| Parameter | Value |
|-----------|-------|
| `l1_ratios` | `[0.5, 1.0]` (elastic net, L1) |
| `Cs` | `logspace(-3, 2, 5)` |

**Rationale:** Same as cognitive — the elastic-net penalty is where
sparsification happens.

### Medical History (~29 features)

| Parameter | Value |
|-----------|-------|
| `l1_ratios` | `[0.5, 1.0]` (elastic net, L1) |
| `Cs` | `logspace(-3, 2, 5)` |
| `max_iter` | `20000` |

**Rationale:** Many binary disease and medication flags, and the L1/elastic
net penalty is what selects among them. The higher `max_iter` (20000 vs. 10000 default) is needed because the SAGA
solver requires more iterations to converge with ~29 weakly regularised
binary features.

### Lab Values (~20 biomarkers)

| Parameter | Value |
|-----------|-------|
| `l1_ratios` | `[0.5, 1.0]` (elastic net, L1) |
| `Cs` | `logspace(-3, 2, 5)` |

**Rationale:** Continuous biomarker features post-stability-selection.  L1
sparsity selects the most discriminative biomarkers.

### Cardiovascular (~7 features)

| Parameter | Value |
|-----------|-------|
| `l1_ratios` | `[0.0]` (pure L2) |
| `Cs` | `logspace(-1, 3, 5)` |
| `solver` | `lbfgs` |

**Rationale:** Small feature set with clinically distinct measures (blood
pressure, heart rate, vascular stiffness).  No sparsity needed — all features
are potentially informative.  Uses `lbfgs` (pure L2, no L1 needed).  Extended
upper C bound (1000) because CV selected C at the previous upper boundary on
real data, indicating weak regularisation is optimal for this compact,
low-correlation feature set.

### Lung Function (~4 features)

| Parameter | Value |
|-----------|-------|
| `l1_ratios` | `[0.0]` (pure L2) |
| `Cs` | `logspace(-1, 2, 5)` |
| `solver` | `lbfgs` |

**Rationale:** Small feature set covering reference-normalised values and
flow parameters.  L2 regularisation handles mild collinearity.  Uses `lbfgs`
(pure L2, no L1 needed).

### MRI (all 6 atlases)

| Parameter | Value |
|-----------|-------|
| `l1_ratios` | `[0.5, 1.0]` (elastic net, L1) |
| `Cs` | `logspace(-3, 1, 5)` |

**Rationale:** Brain MRI features are highly correlated (neighbouring regions
share volume patterns).  All atlases with >20 features (Desikan, Destrieux,
Julich, Subcortical, Yeo) receive PCA (95 % variance) before the classifier;
Cerebellar (12 features) skips PCA.  Stability selection is not used — elastic
net in the classifier handles both feature selection and correlated-feature
grouping.  Pure L2 (Ridge) is excluded because it retains all features and
risks overfitting on correlated post-PCA components.  The narrower C range
(upper bound 10 instead of 100) enforces stronger regularisation to prevent
overfitting to neuroimaging features.

---

## Validation on real NAKO data

C-ranges were validated by fitting LogisticRegressionCV on all 14 modalities
with the full NAKO sample (N=19,240), applying lightweight preprocessing
(imputation + scaling, no orthogonalisation or stability selection) and
reporting the selected C, l1-ratio, and non-zero coefficient count per
modality. The primary validation criterion is that no modality produces a
**null model**
(all coefficients zero), which would provide no signal to the meta-learner.

Some modalities select C at the boundary of their grid.  This is expected and
acceptable in a stacking architecture: base learner predictions are recalibrated
by the XGBoost meta-learner, so the exact C value is of secondary importance as
long as the model produces non-degenerate predictions.  Extending grids to chase
interior optima would add computation without improving downstream performance.

Medical history receives `max_iter=20000` (vs. 10000 default) because its
combination of ~29 binary features and weak regularisation stresses the SAGA
solver, causing `ConvergenceWarning` at 10000 iterations.

---

## Stability selection lambda grids

Stability selection as Meinshausen & Bühlmann (2010) define it takes, for
each feature, the **maximum** selection probability over the regularisation
path, and keeps features above a threshold. The implementation here averages
over the path instead, which is a different and more permissive statistic.

**What that means in practice, measured rather than assumed:** at the
configured `threshold=0.6` and the lambda grids below, the stage selects
everything. In the production run all 396 calls kept 100 % of their features,
in every modality and every fold. On an outcome of pure coin flips, with no
feature carrying signal, it still keeps all of them
(`tests/test_stability_selection.py` pins this).

Read the stage as an elastic-net pre-selection that is currently a
pass-through, not as a dimensionality reduction. The per-modality feature
counts above are the counts the base learner receives, before its own penalty
selects among them. Raising the threshold or moving the grids into the sparse
region would change every number the pipeline reports, so it is not a change
to make casually.

All modalities use an explicit 4–5 point grid spanning the relevant range:

| Modality | Lambda grid | Rationale |
|----------|-------------|-----------|
| Cognitive | `[0.001, 0.01, 0.1, 1.0]` | 4 points, narrow range for small feature set |
| Lab values | `[0.01, 0.1, 1.0, 10.0]` | 4 points, shifted range for continuous biomarkers |
| Physical activity | `[0.001, 0.01, 0.1, 1.0, 10.0]` | 5 points, broad range for mixed feature types |
| Medical history | `[0.001, 0.01, 0.1, 1.0, 10.0]` | 5 points, broad range for binary disease flags |

MRI atlases do not use stability selection (see MRI section above).

The default grid (`logspace(-3, 1, 20)`) uses 20 points, which provides no
meaningful improvement in selection stability but increases computation by 4×.
With `n_bootstrap=100`, a 5-point grid yields 500 fits per stability selection
(vs. 2000 with 20 points).
