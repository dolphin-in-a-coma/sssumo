# Offline re-analysis

These take the per-trial dumps produced by
`scripts/colab/score_checkpoints.py --dump-per-trial` and recompute intervals
locally. **No GPU required** — the expensive part is the model evaluation, which
the dumps already contain.

| Script | Purpose |
|---|---|
| `band_variants.py` | recompute bands under each methodological choice (estimand, central tendency, grouping/balancing) |
| `variance_decomposition.py` | pairwise distances between checkpoints, to separate reproduction error from seed variance |

The dumps from the 2026-08 reproduction study live in the project's Drive folder
under `per_trial_2026-08/` with a `MANIFEST.md`. Use `uncapped/` for organic data;
`capped/` undercounts participants on whacamole and tablet_writing.
