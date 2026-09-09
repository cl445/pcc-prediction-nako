"""Does baseline mental health predict measured olfaction, or only reported olfaction?

Post-infection loss of smell is the one post-COVID symptom with an
objective counterpart in NAKO: the Level-2 follow-up administers the
12-item Sniffin' Sticks identification screening, which NAKO also reads
into a normosmia / hyposmia / anosmia classification. That makes the
chemosensory domain the only place in this study where a self-reported
outcome can be held against an instrument reading of the same construct,
which is what this script does.

The contrast is paired. Both outcomes are fitted on the same
participants, with the same composite baseline-mental-health exposure and
the same age, sex and centre adjustment as every other sensitivity here,
so the two odds ratios differ in what is measured and in nothing else. A
comparison across separate samples would leave open that the samples, not
the measurements, carry the difference.

Timing decides what each outcome can mean. ``fu1_year_reconstructed``
dates the olfactory examination to the calendar year and
``first_infection_months`` dates the reported infection, so participants
split into those tested before their infection and those tested after it:

For participants tested before their infection, the reading cannot
contain post-COVID anosmia; the infection had not happened yet. It
measures the olfactory function they brought into the pandemic. If
depressed participants had worse olfaction all along, this is where it
shows, and the later self-report would be tracking a deficit that
predates the infection. If instead mental health predicts the
self-report while leaving the pre-infection reading untouched, the
self-report is tracking disposition rather than the nose.

For those tested after their infection, the instrument could register
post-COVID anosmia. That subgroup is small, so it works as a convergence
check on the first rather than as a result of its own, and its interval
is wide enough to say so.

Two further panels bound the reading. A linear-probability model
regresses the self-report on the instrument reading and the baseline
PHQ-9 together, which puts both on one scale and asks which of them the
self-report follows. And the never-infected serve as a reference for the
objective outcome, since a mental-health association with hyposmia that
exists independently of infection would change how the infected panels
read.

Participants reporting a cold within the six weeks before the test are
excluded, following the A5 pre-specification: nasal congestion depresses
the identification score for reasons unrelated to the exposure. The flag
is unrecorded for some participants, and unrecorded is not positive, so
those participants are kept and the flag is instead entered as a
covariate in a sensitivity panel.

Coverage limit: the reconstructed examination year exists only for
participants of the primary NAKO delivery, so these panels run on a
subset of the infected clean-controls sample rather than all of it. The
emitted JSON carries the counts.

Usage:
    uv run python scripts/supplementary/olfactory_objective_anchor.py
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from pcc_analysis.config import get_paper_constants_dir, get_processed_dir
from pcc_analysis.sensitivity import composite_mh_positive

if TYPE_CHECKING:
    from collections.abc import Sequence

console = Console()
logger = logging.getLogger(__name__)

# Months in ``first_infection_months`` count from December 2019, so month 1
# is January 2020. Both constants follow from that epoch.
INFECTION_EPOCH_YEAR = 2019
INFECTION_EPOCH_MONTH = 12

# Sniffin' Sticks identification sum runs 0..12 and the baseline PHQ-9 sum
# runs 0..27. Reporter-bias coefficients are reported over these spans so
# the two predictors can be compared at all.
OLF_SCALE_POINTS = 12
PHQ9_SCALE_POINTS = 27

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


class Estimate(TypedDict):
    """One odds ratio with the event count it rests on."""

    n: int
    n_events: int
    prevalence: float
    odds_ratio: float | None
    ci: list[float] | None
    p: float | None


class PairedPanel(TypedDict):
    n: int
    objective_hyposmia: Estimate
    self_reported_smell_loss: Estimate
    odds_ratio_ratio: float | None
    odds_ratio_ratio_ci: list[float] | None
    bootstrap_draws_identified: int | None


class ReporterBiasPanel(TypedDict):
    n: int
    objective_coef: float
    objective_coef_ci: list[float]
    objective_p: float
    objective_over_full_scale_pp: float
    phq9_coef: float
    phq9_coef_ci: list[float]
    phq9_p: float
    phq9_over_full_scale_pp: float
    r_squared: float


class Panels(TypedDict):
    pre_infection: PairedPanel
    post_infection: PairedPanel
    never_infected: Estimate
    reporter_bias: ReporterBiasPanel | None
    reporter_bias_cold_adjusted: ReporterBiasPanel | None


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------
def _base_frame() -> pd.DataFrame:
    """Demographics, Corona-2 items, baseline MH, olfactometry, visit timing."""
    processed = get_processed_dir()
    demo = pd.read_parquet(
        processed / "demographics.parquet",
        columns=["ID", "basis_age", "basis_sex", "basis_uort"],
    )
    corona = pd.read_parquet(processed / "corona2_pcc.parquet")
    mental = pd.read_parquet(processed / "mental_health.parquet")
    olf = pd.read_parquet(
        processed / "followup1_olfactometry.parquet",
        columns=[
            "ID",
            "olf_identification_sum",
            "olf_hyposmia_flag",
            "olf_cold_recent",
        ],
    )
    visit = pd.read_parquet(
        processed / "followup1_visit_meta.parquet",
        columns=["ID", "fu1_year_reconstructed"],
    )
    acute = pd.read_parquet(
        processed / "acute_infection.parquet", columns=["ID", "first_infection_months"]
    )

    mh_cols = ["ID", "phq9_sum", "gad7_sum"]
    if "mini_major_depression" in mental.columns:
        mh_cols.append("mini_major_depression")

    frame = (
        demo.merge(corona, on="ID", how="left")
        .merge(mental[mh_cols], on="ID", how="left")
        .merge(olf, on="ID", how="left")
        .merge(visit, on="ID", how="left")
        .merge(acute, on="ID", how="left")
    )
    frame["infection_year"] = _infection_year(frame["first_infection_months"])
    return frame


def _infection_year(months: pd.Series) -> pd.Series:
    """Calendar year of the first reported infection.

    ``first_infection_months`` counts months from December 2019, so the
    year advances every twelve months starting from that offset.
    """
    offset = INFECTION_EPOCH_MONTH - 1 + months.astype("Float64")
    return (INFECTION_EPOCH_YEAR + (offset // 12)).astype("Float64")


def _clean_controls(frame: pd.DataFrame) -> pd.DataFrame:
    """Infected participants under the primary clean-controls definition.

    Same filter as the published chemosensory analysis, so the self-report
    odds ratio here is readable next to the one already in the paper.
    """
    infected = (frame["had_covid"] == 1) & (frame["valid_symptoms"] == 1)
    sample = frame.loc[infected].copy()
    sub_threshold = (sample["d_co2_k0"] == 1) & (sample["bahmer_any_pcs"] == 0)
    return sample.loc[~sub_threshold].copy()


def _without_cold_on_test_day(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop participants who reported a cold in the weeks before the test.

    An unrecorded flag is not a positive one. Filling it with zero keeps
    those participants, which matters because the flag is missing for a
    large share of the cohort and dropping them would cost more sample
    than the confounder is worth.
    """
    cold = frame["olf_cold_recent"].fillna(0).astype(int)
    return frame.loc[cold != 1].copy()


def _timing_stratum(frame: pd.DataFrame, stratum: str) -> pd.DataFrame:
    """Participants whose olfactory test falls before or after their infection.

    The reconstruction resolves to the calendar year, so a test in the
    same year as the infection cannot be ordered against it and is
    excluded from both strata rather than assigned to one.
    """
    delta = frame["fu1_year_reconstructed"].astype("Float64") - frame["infection_year"]
    mask = delta < 0 if stratum == "pre_infection" else delta > 0
    return frame.loc[mask.fillna(False)].copy()


def _confounders(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["age"] = frame["basis_age"].astype(float)
    out["sex"] = frame["basis_sex"].astype("category").cat.codes.astype(float)
    centre = pd.get_dummies(
        frame["basis_uort"].astype("category"), prefix="c", drop_first=True, dtype=float
    )
    return pd.concat([out, centre], axis=1)


def _objective_hyposmia(frame: pd.DataFrame) -> pd.Series:
    return frame["olf_hyposmia_flag"].astype("Int8")


def _self_reported_smell_loss(frame: pd.DataFrame) -> pd.Series:
    return frame["symptom_loss_of_smell"].fillna(0).astype("Int8")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _fit_logit(y: pd.Series, design: pd.DataFrame) -> Any | None:
    """Logistic fit, or ``None`` when the model does not identify.

    Perfect separation warns rather than raising and hands back infinite
    standard errors, so the returned object is checked as well as the
    exceptions. A stratum too thin to carry the model is a fact about the
    data and is logged, not raised.
    """
    if y.empty or int(y.sum()) == 0:
        logger.warning("No events in a stratum of %d participants", len(y))
        return None
    try:
        result = sm.Logit(
            y.astype(float),
            sm.add_constant(design, has_constant="add").astype(float),
            missing="drop",
        ).fit(disp=False)
    except (LinAlgError, PerfectSeparationError, ValueError) as error:
        logger.warning("Model not estimable on %d participants: %s", len(y), error)
        return None
    if not np.all(np.isfinite(result.bse)):
        logger.warning("Model did not identify on %d participants", len(y))
        return None
    return result


def _estimate(y: pd.Series, exposure: pd.Series, conf: pd.DataFrame) -> Estimate:
    design = pd.concat([exposure.rename("mh_positive"), conf], axis=1)
    observed = y.notna()
    result = _fit_logit(y.loc[observed], design.loc[observed])
    events = int(y.sum()) if y.notna().any() else 0
    base: Estimate = {
        "n": int(observed.sum()),
        "n_events": events,
        "prevalence": float(y.mean()) if observed.any() else float("nan"),
        "odds_ratio": None,
        "ci": None,
        "p": None,
    }
    if result is None:
        return base
    lo, hi = result.conf_int().loc["mh_positive"]
    base["odds_ratio"] = float(np.exp(result.params["mh_positive"]))
    base["ci"] = [float(np.exp(lo)), float(np.exp(hi))]
    base["p"] = float(result.pvalues["mh_positive"])
    return base


def _bootstrap_odds_ratio_ratio(
    frame: pd.DataFrame, *, resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, list[float], int] | None:
    """Percentile interval for self-report OR divided by objective OR.

    The two odds ratios come from the same participants, so their
    intervals are dependent and cannot be compared by eye. Resampling
    participants and refitting both models on each draw carries that
    dependence through to the ratio.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ratios: list[float] = []
    index = np.arange(len(frame))
    for _ in range(resamples):
        draw = frame.iloc[rng.choice(index, size=len(index), replace=True)]
        conf = _confounders(draw)
        exposure = composite_mh_positive(draw)
        objective = _estimate(_objective_hyposmia(draw), exposure, conf)
        reported = _estimate(_self_reported_smell_loss(draw), exposure, conf)
        if objective["odds_ratio"] and reported["odds_ratio"]:
            ratios.append(reported["odds_ratio"] / objective["odds_ratio"])
    logger.info("Bootstrap identified in %d of %d draws", len(ratios), resamples)
    if len(ratios) < resamples // 2:
        logger.warning("Fewer than half the draws identified; interval withheld")
        return None
    return (
        float(np.median(ratios)),
        [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
        len(ratios),
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def paired_panel(frame: pd.DataFrame, *, bootstrap: bool) -> PairedPanel:
    """Both outcomes, same participants, same exposure and adjustment."""
    conf = _confounders(frame)
    exposure = composite_mh_positive(frame)
    objective = _estimate(_objective_hyposmia(frame), exposure, conf)
    reported = _estimate(_self_reported_smell_loss(frame), exposure, conf)

    ratio: float | None = None
    ratio_ci: list[float] | None = None
    draws: int | None = None
    if bootstrap and objective["odds_ratio"] and reported["odds_ratio"]:
        drawn = _bootstrap_odds_ratio_ratio(frame)
        if drawn is not None:
            ratio, ratio_ci, draws = drawn

    return {
        "n": len(frame),
        "objective_hyposmia": objective,
        "self_reported_smell_loss": reported,
        "odds_ratio_ratio": ratio,
        "odds_ratio_ratio_ci": ratio_ci,
        "bootstrap_draws_identified": draws,
    }


def reporter_bias_panel(
    frame: pd.DataFrame, *, extra_covariates: Sequence[str] = ()
) -> ReporterBiasPanel | None:
    """Self-report regressed on the instrument reading and the PHQ-9 together.

    A linear-probability model rather than a logistic one because the
    coefficients are read as percentage points of self-report per scale
    point, which is what makes the two predictors comparable.
    """
    columns = ["olf_identification_sum", "phq9_sum", *extra_covariates]
    sample = frame.dropna(subset=columns).copy()
    if sample.empty:
        logger.warning("No participants carry both the instrument reading and PHQ-9")
        return None

    design = pd.concat(
        [
            sample["olf_identification_sum"].astype(float).rename("objective"),
            sample["phq9_sum"].astype(float).rename("phq9"),
            *(sample[c].astype(float).rename(c) for c in extra_covariates),
            _confounders(sample),
        ],
        axis=1,
    )
    result = sm.OLS(
        _self_reported_smell_loss(sample).astype(float),
        sm.add_constant(design, has_constant="add").astype(float),
        missing="drop",
    ).fit()
    objective_ci = result.conf_int().loc["objective"]
    phq9_ci = result.conf_int().loc["phq9"]
    return {
        "n": int(result.nobs),
        "objective_coef": float(result.params["objective"]),
        "objective_coef_ci": [float(objective_ci[0]), float(objective_ci[1])],
        "objective_p": float(result.pvalues["objective"]),
        "objective_over_full_scale_pp": float(
            100 * OLF_SCALE_POINTS * result.params["objective"]
        ),
        "phq9_coef": float(result.params["phq9"]),
        "phq9_coef_ci": [float(phq9_ci[0]), float(phq9_ci[1])],
        "phq9_p": float(result.pvalues["phq9"]),
        "phq9_over_full_scale_pp": float(
            100 * PHQ9_SCALE_POINTS * result.params["phq9"]
        ),
        "r_squared": float(result.rsquared),
    }


def never_infected_reference(frame: pd.DataFrame) -> Estimate:
    """Baseline MH against objective hyposmia among the never-infected."""
    sample = _without_cold_on_test_day(frame.loc[frame["had_covid"] == 0].copy())
    sample = sample.dropna(subset=["olf_hyposmia_flag"])
    return _estimate(
        _objective_hyposmia(sample), composite_mh_positive(sample), _confounders(sample)
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(estimate: Estimate) -> str:
    if estimate["odds_ratio"] is None or estimate["ci"] is None:
        return f"not estimable (n = {estimate['n']:,})"
    lo, hi = estimate["ci"]
    return (
        f"{estimate['odds_ratio']:.2f} [{lo:.2f}, {hi:.2f}], "
        f"{estimate['n_events']:,} events / {estimate['n']:,}"
    )


def _render(panels: Panels) -> Table:
    table = Table(title="Olfactory objective anchor", show_header=True)
    table.add_column("Panel", style="bold")
    table.add_column("Baseline-MH odds ratio", justify="right")
    for name in ("pre_infection", "post_infection"):
        panel: PairedPanel = panels[name]  # type: ignore[literal-required]
        table.add_row(f"[{name}] objective hyposmia", _fmt(panel["objective_hyposmia"]))
        table.add_row(
            f"[{name}] self-reported smell loss",
            _fmt(panel["self_reported_smell_loss"]),
        )
        ratio, ci = panel["odds_ratio_ratio"], panel["odds_ratio_ratio_ci"]
        if ratio is not None and ci is not None:
            table.add_row(
                f"[{name}] ratio self-report / objective",
                f"{ratio:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
            )
    table.add_row("never infected, objective hyposmia", _fmt(panels["never_infected"]))
    bias = panels["reporter_bias"]
    if bias is not None:
        table.add_row(
            "self-report per full instrument scale",
            f"{bias['objective_over_full_scale_pp']:+.2f} pp",
        )
        table.add_row(
            "self-report per full PHQ-9 scale",
            f"{bias['phq9_over_full_scale_pp']:+.2f} pp",
        )
    return table


def _tex_constants(panels: Panels) -> str:
    pre = panels["pre_infection"]
    post = panels["post_infection"]
    never = panels["never_infected"]
    bias = panels["reporter_bias"]

    lines = [
        "% Generated by scripts/supplementary/olfactory_objective_anchor.py"
        " -- do not edit by hand.",
        f"\\newcommand{{\\resOlfObjPreN}}{{{pre['n']}}}",
    ]

    def emit(prefix: str, estimate: Estimate) -> None:
        if estimate["odds_ratio"] is None or estimate["ci"] is None:
            return
        lines.append(f"\\newcommand{{\\{prefix}}}{{{estimate['odds_ratio']:.2f}}}")
        lines.append(f"\\newcommand{{\\{prefix}lo}}{{{estimate['ci'][0]:.2f}}}")
        lines.append(f"\\newcommand{{\\{prefix}hi}}{{{estimate['ci'][1]:.2f}}}")

    emit("resOlfObjPreOrObjective", pre["objective_hyposmia"])
    emit("resOlfObjPreOrSelfReport", pre["self_reported_smell_loss"])
    emit("resOlfObjPostOrObjective", post["objective_hyposmia"])
    emit("resOlfObjNeverInfOr", never)

    lines.append(
        "\\newcommand{\\resOlfObjPreHyposmiaPct}"
        f"{{{100 * pre['objective_hyposmia']['prevalence']:.1f}}}"
    )
    lines.append(
        "\\newcommand{\\resOlfObjPostHyposmiaPct}"
        f"{{{100 * post['objective_hyposmia']['prevalence']:.1f}}}"
    )
    lines.append(f"\\newcommand{{\\resOlfObjPostN}}{{{post['n']}}}")

    ratio, ratio_ci = pre["odds_ratio_ratio"], pre["odds_ratio_ratio_ci"]
    if ratio is not None and ratio_ci is not None:
        lines.append(f"\\newcommand{{\\resOlfObjRatio}}{{{ratio:.2f}}}")
        lines.append(f"\\newcommand{{\\resOlfObjRatiolo}}{{{ratio_ci[0]:.2f}}}")
        lines.append(f"\\newcommand{{\\resOlfObjRatiohi}}{{{ratio_ci[1]:.2f}}}")

    if bias is not None:
        lines.append(f"\\newcommand{{\\resOlfObjBiasN}}{{{bias['n']}}}")
        lines.append(
            "\\newcommand{\\resOlfObjBiasObjectivePP}"
            f"{{{bias['objective_over_full_scale_pp']:.2f}}}"
        )
        lines.append(
            "\\newcommand{\\resOlfObjBiasPhqPP}"
            f"{{{bias['phq9_over_full_scale_pp']:.2f}}}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    console.rule("[bold]Olfactory objective anchor vs. self-report")

    frame = _base_frame()
    infected = _without_cold_on_test_day(_clean_controls(frame))
    infected = infected.dropna(subset=["olf_hyposmia_flag", "fu1_year_reconstructed"])

    pre = _timing_stratum(infected, "pre_infection")
    post = _timing_stratum(infected, "post_infection")
    logger.info(
        "Clean-controls sample with instrument reading and dated visit: %s "
        "(%s tested before infection, %s after)",
        f"{len(infected):,}",
        f"{len(pre):,}",
        f"{len(post):,}",
    )

    panels: Panels = {
        "pre_infection": paired_panel(pre, bootstrap=True),
        "post_infection": paired_panel(post, bootstrap=False),
        "never_infected": never_infected_reference(frame),
        "reporter_bias": reporter_bias_panel(pre),
        "reporter_bias_cold_adjusted": reporter_bias_panel(
            pre.dropna(subset=["olf_cold_recent"]),
            extra_covariates=("olf_cold_recent",),
        ),
    }

    summary = {
        "design": (
            "Paired contrast on identical participants: baseline mental "
            "health against objectively measured hyposmia (12-item Sniffin' "
            "Sticks, NAKO olf_kat classification) and against self-reported "
            "post-infection loss of smell, same exposure and same adjustment."
        ),
        "exposure_definition": (
            "Composite baseline mental-health positivity: PHQ-9 sum >= 10 "
            "OR GAD-7 sum >= 10 OR MINI major depression == 1 "
            "(missing components treated as non-positive)."
        ),
        "caveat": (
            "The examination year is reconstructed to the calendar year "
            "only, so participants tested in their year of infection cannot "
            "be ordered against it and enter neither stratum. The "
            "reconstruction covers the primary NAKO delivery, so these "
            "panels run on a subset of the infected clean-controls sample. "
            "Participants recorded as having a cold on the test day are "
            "excluded; an unrecorded flag counts as no cold and is entered "
            "as a covariate in the cold-adjusted panel instead."
        ),
        "panels": panels,
    }

    out_dir = get_paper_constants_dir()
    out_path = out_dir / "olfactory_objective_anchor.json"
    out_path.write_text(json.dumps(summary, indent=2))
    tex_path = out_dir / "olfactory_objective_anchor_tex.tex"
    tex_path.write_text(_tex_constants(panels))

    console.print(_render(panels))
    console.print(f"[green]Wrote {out_path} and {tex_path}")


if __name__ == "__main__":
    main()
