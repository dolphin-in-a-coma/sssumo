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

## First decide the target, then the unit

Two different questions get two different intervals, and mixing them up is the
main way to go wrong here.

**"Would this hold for other people?"** — a population claim. Participants are the
unit. Trials cluster within participant (intraclass correlation ≈0.7 on
steering), so a *pooled* trial bootstrap is about three times too narrow. Use:

- `--organic-interval participant-t` — each participant contributes one score,
  ordinary t across participants. **Default for a population claim**: the degrees
  of freedom match the design and the sample size shows up in the width.
- `--organic-interval cluster-bootstrap` — the project's established method and
  what the analysis notebook used. It is a *weighted-row approximation* of a
  cluster bootstrap, not the canonical one.
- `--organic-interval participant-spread` — median with 2.5/97.5 percentiles, or
  min/max below n = 40. Describes how much people differ, which is a different
  question from how uncertain the mean is.

**"How well does it work on our data?"** — a conditional claim, and what the
article actually needs, since 2–10 test participants cannot support the other
one. Trials are the unit, **stratified by participant** so the participant
composition is held fixed. `scripts/analysis/dataset_intervals.py`. The interval
is much narrower than the participant-t one, and that is correct — it excludes
participant variation on purpose, because we are not generalising over people.

Both come from `scripts/analysis/dataset_intervals.py` in one pass; it emits the
conditional interval, the participant t interval and the cluster bootstrap per
dataset, with `cluster_reliable` flagging n >= 20.

## Resample exactly one level: the coarsest you generalise over

Everything finer is already in the observed spread. Each trial's metric is itself
a noisy realisation, so the variation across trials already carries within-trial
noise; adding a sample-, chunk-, or block-level bootstrap on top **double-counts**
it. Verified by simulation at comparable variance levels: the trial bootstrap
alone recovers the analytic SE (0.00488 vs 0.00487), while adding a within-trial
level inflates it by 1.201 against a predicted double-count factor of 1.214.

This is why no per-sample dump is needed, and why chunking trials to get "finer"
error bars is wasted GPU time.

The same trap sits one rung up: for a population interval resample participants
**carrying their trials whole**, never participants *and* trials within them. A
participant's observed mean already contains their trial noise.

Sanity checks that catch a broken implementation:

- whacamole's conditional and population intervals must come out **identical** --
  with one trial per participant the two questions coincide.
- cluster bootstrap and participant t must **agree** at n = 44 and n = 181, and
  **diverge** below n ~ 20 (at n = 2 the cluster interval is ~11x too narrow,
  because two clusters admit only three distinct resamples).

Check any new implementation against the closed-form stratified SE,
`sqrt(Σ_p (n_p/n)² · s_p²/n_p)` — they should agree to a percent or two.

**whacamole has one trial per participant.** Stratified resampling has nothing to
vary there, so no conditional interval exists; it falls back to unstratified and
its interval carries participant variation. Anything that reports per-participant
uncertainty on whacamole is reporting zero-width bands.

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
