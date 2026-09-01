# Offline re-analysis

These take the per-trial dumps produced by
`scripts/colab/score_checkpoints.py --dump-per-trial` and recompute intervals
locally. **No GPU required** — the expensive part is the model evaluation, which
the dumps already contain.

| Script | Purpose |
|---|---|
| `dataset_intervals.py` | **the reportable numbers** — one conditional interval per dataset, plus the two population variants |
| `overall_interval.py` | a single dataset-balanced figure across all seven datasets |
| `participant_bands.py` | the same procedure scoped to one person: per-participant estimate and band |
| `band_variants.py` | recompute bands under each methodological choice (estimand, central tendency, grouping/balancing) |
| `variance_decomposition.py` | pairwise distances between checkpoints, to separate reproduction error from seed variance |
| `double_count_check.py` | simulation showing why only one level may be resampled |

Typical use:

```bash
python scripts/analysis/dataset_intervals.py --dumps <path>/per_trial_2026-08/uncapped
python scripts/analysis/overall_interval.py  --dumps <path>/per_trial_2026-08/uncapped
```

Both write into `docs/`. Each takes about a minute for all eight checkpoints.

**The one rule these implement:** resample exactly one level — the coarsest you
mean to generalise over. Trials within participant gives an interval conditional
on the people tested; participants carrying their trials whole gives a population
interval. Never both at once, and never a level finer than the trial: each trial's
metric is already a noisy realisation, so the spread across trials carries that
noise, and resampling inside a trial double-counts it. `double_count_check.py`
demonstrates the inflation. See `docs/VALIDATION.md` for the full argument.

The dumps from the 2026-08 reproduction study live in the project's Drive folder
under `per_trial_2026-08/` with a `MANIFEST.md`. Use `uncapped/` for organic data;
`capped/` undercounts participants on whacamole and tablet_writing.
