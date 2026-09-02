# Does the submovement primitive have to be minimum jerk?

SSSUMO renders every submovement with one fixed velocity pulse. This records what
happens when that pulse is swapped for three alternatives, and what a wrong choice
costs. Source pinned at `313f3ac` (analysis) on branch `feat/primitive-families`;
the epoch-20/22/24 matrices come from four continuous 25-epoch reruns at `f80eb17`.

**Conclusion: reconstruction quality is nearly blind to the assumed pulse family, so it
cannot be used to choose one.** Minimum jerk is not indicted as a *fit* — it is
indicted as something a fit metric cannot confirm. Detection separates the families
cleanly at every training depth measured, and the penalty for assuming the wrong pulse
*grows* with training.

> **Revised 2026-09-02.** An earlier version of this document claimed fit quality ranks
> the wrong family *first*. That held at epoch 12 but does not survive to the recipe's
> own endpoint — see Result 2. Every number below now comes from four arms trained
> continuously to epoch 24 in a single process each; the epoch-12 matrix that produced
> the original claim came from runs resumed at different points, with the optimiser
> state reset each time.

## The manipulation

Minimum jerk needs no separate implementation. `30 s^2 (1-s)^2` **is** Beta(3,3), and
`ContinuousPrimitive` at `beta_mean=(0.5,0)`, `beta_precision=(6,0)` reproduces it to
2.3e-8 on this grid for every duration in 5..60. So the shipped primitive was already a
frozen special case of a family.

| arm | family | parameters | peak | skew | sd |
|---|---|---|---|---|---|
| `minjerk` | beta | mean 0.50, precision 6 | 0.50 | 0.000 | 0.189 |
| `gaussian` | gaussian | half-width 3σ | 0.47 | 0.000 | 0.164 |
| `beta_asym` | beta | mean 0.40, precision 6 | 0.33 | +0.269 | 0.185 |
| `lgnb` | lgnb | mu −0.40, sigma 0.8 | 0.33 | +0.278 | 0.172 |

Two symmetric, two asymmetric. The asymmetric pair is matched to *each other* on peak
and skew, so a contrast between them isolates shape family from amount of asymmetry.
Pairwise L1 between unit-area pulses: symmetric pair 0.213, asymmetric pair 0.104,
cross-pair 0.387–0.499.

**What makes the comparison clean.** Submovement parameters and noise are drawn from an
RNG stream that never touches the primitive, and only the *rendering* goes through
`reconstruction_model`. With the seed fixed, all four generators emit **bit-identical
latent ground truth** (verified: `max|y − y| = 0`) and differ only in pulse shape.
Every cross-family difference is attributable to shape alone. Ground-truth submovement
rate is 2.66/s in every cell of every matrix below.

## Result 1 — the family changes how hard detection is

Epoch 12, identical targets, 1000 steps/epoch:

| arm | detection loss | onset dist | precision | amplitude loss | duration loss |
|---|---|---|---|---|---|
| min-jerk | 0.5116 | 1.090 | 0.808 | 153.1 | 26.4 |
| Gaussian | 0.5090 | 1.072 | 0.839 | 126.9 | 17.8 |
| Beta-asym | 0.3263 | 0.571 | 0.862 | 80.5 | 21.9 |
| LGNB | 0.3377 | 0.608 | 0.882 | 72.5 | 15.9 |

The split is by **symmetry class**. Within-class spread is 0.5% (symmetric) and 3.5%
(asymmetric) of detection loss; the between-class gap is 36% — an order of magnitude
larger, which is what makes n=1 per arm tolerable here.

The continuous reruns reproduce this at epoch 12 (detection loss 0.504 / 0.511
symmetric, 0.327 / 0.340 asymmetric) and it persists at epoch 20, where every arm has
improved but the classes stay separated by 39%:

| arm | detection loss | onset dist | precision |
|---|---|---|---|
| min-jerk | 0.3939 | 0.817 | 0.879 |
| Gaussian | 0.4000 | 0.807 | 0.903 |
| Beta-asym | 0.2412 | 0.409 | 0.905 |
| LGNB | 0.2484 | 0.427 | 0.923 |

Two distinct mechanisms, separable only because the design varies symmetry and width
independently:

- **Onset detection improves with asymmetry** (onset error nearly halved). It is *not*
  sharpness: the narrowest pulse, Gaussian, tracks min-jerk, while Beta-asym has
  essentially min-jerk's width and beats it decisively.
- **Duration estimation improves with narrowness** — duration loss orders LGNB <
  Gaussian < Beta-asym < min-jerk, tracking sd, not symmetry.

## Result 2 — fit quality cannot tell the families apart

144 cells: 4 decoders x 4 generators x 9 noise/overlap conditions, 256 trials of 1000
samples, repeated at epochs 20, 22 and 24. Median CI95 half-widths: R2 +/-0.0021, onset
F1 +/-0.0062, duration R2 +/-0.0136.

**Does reconstruction R2 rank the true family first?** Per generator column, the decoder
with the highest R2 should be the matched one:

| epoch | columns misranked | margin of each wrong call |
|---|---|---|
| 12 *(superseded)* | **3 / 4** | gaussian by 0.0133; lgnb by 0.0204; minjerk by 0.0057 |
| 20 | 1 / 4 | gaussian by **0.0010** |
| 22 | **0 / 4** | none |
| 24 | 1 / 4 | minjerk by **0.0031** |

At epoch 12 the misranking was systematic and the margins were several times the CI. At
the recipe's own depth it is at most one column, a *different* one at each epoch, by
0.001-0.003 — margins at or below the CI half-width. **A ranking that changes which
column it applies to between adjacent epochs, by a thousandth of an R2, is a tie, not a
finding.**

What survives is the weaker and more useful claim: across the four decoders,
reconstruction R2 varies by 0.045 relative (Result 4) — a mismatched decoder on
Gaussian data still scores 0.966-0.977 against the matched 0.976. The metric's dynamic
range across families is too small to discriminate them in either direction, which is
exactly what Result 3 predicts from shape alone.

Detection is the opposite. Onset F1 ranks the true family first in **4 of 4 columns at
every epoch measured (12, 20, 22, 24)**, and the gap widens with training:

| epoch | matched mean F1 | mismatched mean | cost of mismatch |
|---|---|---|---|
| 12 | 0.8865 | 0.6905 | 0.196 |
| 20 | 0.8779 | 0.6667 | 0.211 |
| 22 | 0.8776 | 0.6565 | 0.221 |
| 24 | 0.8692 | 0.6450 | 0.224 |

On Gaussian-generated data at epoch 20 the dissociation is visible in one column:

| decoder | recon R2 | rank | onset F1 | rank | submov/s (true 2.66) |
|---|---|---|---|---|---|
| **Gaussian (correct)** | 0.9764 | 2nd | **0.8644** | 1st | 3.05 |
| min-jerk | **0.9774** | 1st | 0.6566 | 2nd | 3.94 |
| Beta-asym | 0.9755 | 3rd | 0.5348 | 4th | 4.66 |
| LGNB | 0.9659 | 4th | 0.5788 | 3rd | 4.48 |

All four decoders sit within 0.012 R2 of each other while their onset F1 spans 0.33.
Fit ranks the correct family 2nd by one thousandth; recovery ranks it 1st by 0.21.

Mismatch penalty against each cell's own matched diagonal, epoch 20 (worst first):

| decoder | data | Δ recon R2 | Δ onset F1 | count ratio |
|---|---|---|---|---|
| Gaussian | Beta-asym | −0.067 | −0.368 | 1.42 |
| Beta-asym | Gaussian | −0.001 | −0.330 | 1.53 |
| Gaussian | LGNB | −0.060 | −0.311 | 1.51 |
| LGNB | Gaussian | −0.011 | −0.286 | 1.47 |
| Gaussian | min-jerk | −0.035 | −0.255 | 1.38 |
| min-jerk | Gaussian | **+0.001** | −0.208 | 1.29 |

Across the 12 off-diagonal cells, **1** shows a mismatched decoder scoring higher R2
than the matched one (it was 4 at epoch 12); **all 12** recover worse, and every one
inflates the submovement count — by 24% to 53% except within the skew-matched pair,
where the two asymmetric families are near-interchangeable (count ratio 0.98-1.05).

## Result 3 — the ceiling from shape alone

Decoder renders the *true* labels with its own family; no detector, no estimation error.
Model-free, and it reproduces inside the full matrix to 3 decimals — including across the epoch-20/22/24 reruns, which is the check that the two studies measure the same thing.

| decoder \ data | min-jerk | Gaussian | Beta-asym | LGNB |
|---|---|---|---|---|
| min-jerk | 1.000 | 0.960 | 0.842 | 0.867 |
| Gaussian | 0.952 | 1.000 | **0.748** | 0.805 |
| Beta-asym | 0.835 | 0.781 | 1.000 | **0.988** |
| LGNB | 0.850 | 0.816 | 0.987 | 1.000 |

Perfect detection cannot rescue a wrong shape (min-jerk caps at 0.84 on asymmetric
data), the skew-matched pair is near-interchangeable at 0.99, and even the worst
mismatch retains 0.75 — the metric's dynamic range across families is simply too small
to discriminate them. **The ceiling is overlap-independent** (0.8693 / 0.8696 / 0.8683),
so shape error superposes linearly.

## Result 4 — which quantities survive a change of basis

Relative spread across the four decoders, averaged over datasets:

| quantity | spread | |
|---|---|---|
| Reconstruction R² | 0.044 | most stable — hence useless for model selection |
| Submovements/s | 0.110 | robust · confirms the literature |
| Onset F1 | 0.197 | |
| Duration R² | 0.300 | fragile · confirms the literature |
| Amplitude R² | 0.386 | fragile · **contradicts** the literature |
| Onset distance | 0.439 | most sensitive · **contradicts** the literature |

For SSSUMO's reported statistics: **counts are reasonably safe; timing and amplitude are
not.** A cross-family sensitivity check belongs beside any per-dataset claim about
duration, amplitude or onset timing.

The *ordering* re-verified at epoch 20 unchanged — reconstruction R² most stable at 0.045
relative, onset distance most sensitive — so this result does not depend on training
depth. The magnitudes above are the epoch-12 normalisation and are not directly
comparable to a recomputation over the new matrices.

## Result 5 — clean data punishes the wrong basis hardest

Mismatch penalty (matched − mismatched onset F1):

| SNR \ overlap | 0.0–0.5 | 0.5–1.0 | 1.0–1.5 |
|---|---|---|---|
| ∞ | 0.160 | 0.354 | **0.363** |
| 20 | 0.096 | 0.259 | 0.259 |
| 10 | **0.041** | 0.177 | 0.193 |

*(epoch 20. The epoch-12 matrix gives the same pattern one step weaker: 0.154 / 0.294 /
0.270 at SNR ∞, 0.056 / 0.205 / 0.218 at SNR 10.)*

Noise does not reveal misspecification, it masks it — everything degrades toward the
same poor performance. Basis choice matters most where recordings are best. Overlap
sensitivity is entirely an **estimation** effect: the oracle ceiling above is
overlap-independent, so crowding only compounds shape error once it passes through a
detector.

## Limits

- **Training depth: resolved, and it cost one result.** The original study ran to epoch
  12 because Colab reclaimed three VMs mid-run. Four arms have since been retrained
  continuously to epoch 24 — one process each, zero optimiser resets — and the matrix
  recomputed at epochs 20, 22 and 24. Result 2's *direction* did not survive (see
  above); Results 1, 3, 4 and 5 did, the last two strengthening. Absolute numbers do
  shift: matched reconstruction R² rises from ~0.88 to ~0.98.
- **Epoch 24 is not convergence.** `lr_decay_start: 30` sits outside the 25-epoch
  budget *and* `lr_decay_total_change: 1` makes the scheduler a no-op even in range, so
  the learning rate is constant throughout — this is the released pretraining recipe,
  not a misconfiguration. Detection precision peaks at epoch 20, where
  `bn_dropout_freeze_start` fires, and declines monotonically after; onset F1 falls
  0.0185 on average across the 144 cells from epoch 20 to 24. Report epoch 20 and treat
  22/24 as the stability evidence. The conclusions are epoch-robust; the absolute
  numbers are not.
- **Shape is frozen and uniform.** Nothing is predicted per submovement, so this
  measures uniform mis-specification, not within-trial shape heterogeneity.
- **One run per arm**, unseeded weight init. The two-arms-per-class design bounds the
  nuisance at ~1/10 of the effect; formal replicates are not in. For scale, the minjerk
  arm differs from the released checkpoint by ~0.007 onset F1 — but differs *from
  itself* by 0.035 precision between epochs 22 and 24, so with a constant learning rate
  the choice of stopping epoch matters several times more than run-to-run variation.
- **Synthetic only.** Recovery is only measurable where ground truth exists.

## What is still open

The three arms that were in flight when this was first written have all landed.

- **`sssumo-ep22-beta-asym`** — completed the epoch-22 matrix; superseded by the full
  epoch-20/22/24 reruns.
- **`sssumo-learn-beta`** — **the shape is recoverable.** With the generator pinned to
  Beta(2.4, 3.6) and frozen, and the decoder starting from the symmetric minimum-jerk
  assumption, training moved `beta_mean` 0.500 → 0.403 against a truth of 0.400,
  closing 97% of the gap; the mode moved 0.500 → 0.352 against a truth of 0.350. Both
  duration slopes stayed at ~0, correctly. `beta_precision` started *at* the truth
  (6.0) and drifted 3.7% low, so one parameter was recovered and one slightly degraded.
  The shape sat frozen for epochs 0–4 and converged within a single epoch of epoch 5,
  which is `reconstruction_loss_start` — the reconstruction loss is the only gradient
  path to the primitive.
- **`sssumo-learn-lgnb`** — the same recovery test in the other family. **Still not
  run.** One family recovering its own shape is an anecdote.

Beyond those:

- **Replicates.** Still one run per arm; `--seed-offset` is the lever, since
  `config.seed` does not change the training data.
- **Per-submovement shape.** The detector emits `[onset, amplitude, duration]` only, so
  a run's pulses all share one shape. Adding shape channels collides with the dead
  `shape[1] == 4` reconstruction-mask path in `models.py` and `training.py`.
- **Organic data.** Recovery is only measurable against ground truth, so every result
  here is synthetic.
- **A learned-shape checkpoint cannot reproduce itself.** `torch.save` stored only
  `model.state_dict()`, and the primitive lives on the reconstructor — so the recovered
  shape survived only in the training log. Fixed in `a5d8169` (format-2 checkpoints
  carry optimiser, scheduler and primitive), but the `beta_learned` checkpoints predate
  it and need their shape read out of `train_log_beta_learned.txt`.

## Where the artefacts are

**`runs/0902-family-rerun/`** — the continuous 25-epoch reruns (gitignored):

- `matrix/cross_eval_epoch{20,22,24}.csv` — the three 144-cell matrices behind the
  revision above, produced on one L4 from `f80eb17`; **committed** as
  `docs/family_cross_eval_epoch{20,22,24}.csv`
- `<arm>/` — 25 checkpoints per arm (epochs 0–24) plus the training log, for
  `minjerk`, `gaussian`, `beta_asym`, `lgnb`
- also mirrored to the Nextcloud share under `0902-family-rerun/<arm>/`

The original epoch-12 study, `runs/0901-pulse-families/` (gitignored, on this machine):

- `results/cross_eval_epoch12.csv` — the 144-cell matrix behind Results 2–5
- `results/cross_eval_epoch2.csv` — the independent epoch-2 replication
- `results/oracle_ceiling.csv`, `results/training_metrics.csv`, `results/train_log_*.txt`
- `results/pulse_report.html` — published at
  https://claude.ai/code/artifact/46e97258-052e-44fe-82f1-ecaa3a5c7243
- `weights/` — epoch 12 for all four arms, epoch 22 for three, epoch 24 for lgnb

Off-machine backup: private Kaggle dataset `dolphininacoma/sssumo-family-ep12` holds the
pinned source plus the epoch-12 and epoch-22 checkpoints.

Reproduce the matrix with:

```
python scripts/colab/cross_evaluate_families.py \
  --root-dir runs/0901-pulse-families --epoch 12 --batch-size 256 \
  --out cross_eval_epoch12.csv
```

(needs `weights/` under `--root-dir`; ~100 min on CPU, minutes on a T4)
