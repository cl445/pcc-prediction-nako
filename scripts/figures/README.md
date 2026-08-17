# Paper figure generation

Scripts that render the manuscript's main and supplementary figures from
pipeline results. They read the held-out predictions and modality scores under
`results/` and write `.pgf` files for inclusion in a LaTeX document.

## Run

From the `code/` directory, using its uv environment:

```bash
uv run python scripts/figures/regenerate_evaluation_curves.py
uv run python scripts/figures/regenerate_modality_contributions.py   # writes .pgf + _prauc.pgf
uv run python scripts/figures/regenerate_decision_curve.py
uv run python scripts/figures/regenerate_lean_dca.py
```

All paths are resolved relative to the script location, so the repository can
live anywhere. Shared matplotlib/PGF styling lives in `_style.py`.

**Inputs.** Each script reads one of the deterministic run directories that
`scripts/run_analysis.sh` writes under `results/runs/` — `primary` for the
evaluation curves, decision curve and modality contributions, `lean_mri` and
`lean_non_mri` for the Lean DCA panel. Run the analysis first; there is
nothing to plot before then.

**Output.** `results/figures/<name>.pgf`, plus a `<name>.pdf` preview. If a
`paper/figures/` directory exists alongside this repository the figures are
written there instead, so a manuscript checked out next to the code picks them
up directly; see `figure_output_dir()` in `_style.py`. Without a TeX
distribution the `.pgf` is skipped and only the `.pdf` is written.
