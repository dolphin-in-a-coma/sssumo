---
name: sssumo-evaluation
description: Score SSSUMO checkpoints and report uncertainty correctly. Use when comparing checkpoints, reproducing released weights, choosing between bootstrap and t intervals, deciding the unit of analysis, or preparing evaluation numbers for the article.
---

# Evaluating SSSUMO checkpoints

Scoring is easy; reporting the uncertainty correctly is where the decisions are.
This records the choices that change conclusions, measured on this data.

## Tools

| Task | Tool |
|---|---|
| Each checkpoint's own score + interval | `scripts/colab/score_checkpoints.py` |
| Differences between checkpoints | `scripts/colab/compare_checkpoints.py` |
| Why two checkpoints differ on one dataset | `scripts/colab/diagnose_dataset.py` |
| Training-trajectory agreement | `scripts/colab/wandb_compare_runs.py` |

Always pass `--trials 0` (no cap) for anything reported: capping undercounts
participants on the large datasets and makes the count vary across noise levels
within one dataset.

`--dump-per-trial DIR` writes per-trial values plus sufficient statistics
(`n, Σy, Σy², SS_res, Σ|err|, Σ|y|, Σsmape`). Those reconstruct any *pooled*
metric exactly, so every interval variant can be recomputed later without a GPU.
Dump once, then explore methods locally.

## The unit of analysis decides the answer

Trials cluster within participant. On steering the intraclass correlation is
≈0.7, so a trial-level bootstrap makes the interval **three times too narrow**.
Do not bootstrap over trials for organic data.

The three defensible options, all implemented:

- `--organic-interval participant-t` — each participant contributes one score,
  interval is the ordinary t across participants. **Recommended default**: the
  degrees of freedom match the design and the sample size is visible in the width.
- `--organic-interval cluster-bootstrap` — the project's established method, and
  what the analysis notebook used. Note it is a *weighted-row approximation* of a
  cluster bootstrap, not the canonical one.
- `--organic-interval participant-spread` — median with 2.5/97.5 percentiles, or
  min/max below n = 40 where those tails are not estimable. Describes how much
  people differ, which is not the same question as how uncertain the mean is.

For "does this reproduction match", use `compare_checkpoints.py --paired-t`:
pairing is exact because all checkpoints see identically seeded trials, which
removes trial difficulty from the comparison entirely.

## Effect size and significance point in opposite directions here

Measured on the reproduction study:

- crank, 5 participants: Δ = −0.457, 95% CI [−1.098, +0.184], p = 0.12 — the
  largest effect in the study, **not** separable from zero.
- whacamole, 128 participants: Δ = −0.006, p < 0.001 — significant and
  practically meaningless.

Report both the effect and the n, and do not let either statistic stand alone.

## Central tendency changes what is visible

Pooled with the notebook's settings, released vs reproduced pretrained differ by
0.0033 under the **median** and 0.0669 under the **mean**. The median is robust to
exactly the catastrophic trials that constitute the failure mode. Neither is
wrong; only the mean can see a model that fails badly on a minority of trials.
State which one is being reported.

The synthetic estimand choice (per-trial mean vs metric recomputed over the pooled
signal) shifts every checkpoint by a near-constant −0.005 and leaves comparisons
intact — pick one, name it, move on.

## Small datasets

Test-split participants: whacamole 181, tablet_writing 44, Fitts 10, steering 9,
crank 5, object_moving 5, **pointing 2**. Percentile bootstraps undercover with
few clusters, and at n = 2 no interval is meaningful — report the two scores
individually or omit the dataset from interval-based claims.
