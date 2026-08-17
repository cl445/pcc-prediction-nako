# Smoke-test profiles

`default.json` is the distributional profile the synthetic-data generator
draws from. It is the one file in this repository derived from the real NAKO
data, so it is worth being precise about what it holds.

## What it contains

Per column, across 28 modalities and 928 variables:

- `dtype`, and for integers the pandas dtype, so the generator reproduces the
  column's domain rather than only its range
- `missing_frac` — the fraction of rows that are NA, to three decimals
- for numeric columns: `mean`, `std`, and the 1st, 25th, 50th, 75th and 99th
  percentiles, each to **three significant figures**
- for boolean columns: `true_frac`, to three decimals
- for categorical columns: the category proportions, to three decimals
- per modality: the fraction of rows missing that modality entirely
- which columns are derived from others, so the generator can recompute them
  instead of sampling them independently and producing rows that contradict
  themselves

## What it does not contain

- **No minima or maxima.** An observed extreme is one real participant's
  measured value. The 1st and 99th percentiles bound the synthetic draws just
  as well and describe no individual.
- **No cross-tabulations and no correlations.** Every statistic is marginal,
  computed one column at a time.
- **No participant-level value of any kind**, and no identifier.
- **No row or column counts.** The generator sizes its output from
  `--n-subjects` and takes the width from `columns`, so the size of the
  delivery this was profiled from is not needed to synthesise anything. It
  describes a particular NAKO delivery rather than a distribution, which is the
  one thing this file is for.
- No generation timestamp and no source path: the file describes the data, not
  the machine that read it.

On precision: three significant figures is coarser than the descriptive tables
a paper prints — baseline age is stored here as `48.8`. The generator needs a
plausible shape, not a faithful one, and nothing that ships with a repository
should be the most precise description of a cohort in existence.

The smallest surviving category proportion is 0.001, which on any modality here
is well over a hundred participants.

## Regenerating it

Needs access to the processed parquets, so it runs only where the real data
are configured:

```bash
uv run python -m smoke_test profile --input-dir data/processed --output smoke_test/profiles/default.json
```

Generating fixtures from it needs no data access at all:

```bash
uv run python -m smoke_test generate --output-dir smoke_data
```
