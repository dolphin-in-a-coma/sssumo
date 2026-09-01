# Does the submovement primitive have to be minimum jerk?

SSSUMO renders every submovement with one fixed velocity pulse. This records what
happens when that pulse is swapped for three alternatives, and what a wrong choice
costs. Source pinned at `313f3ac` (analysis) on branch `feat/primitive-families`.

**Conclusion: reconstruction quality is nearly blind to the assumed pulse family, and
ranks the wrong one first.** Minimum jerk is not indicted as a *fit* — it is indicted
as something a fit metric can confirm.

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

Two distinct mechanisms, separable only because the design varies symmetry and width
independently:

- **Onset detection improves with asymmetry** (onset error nearly halved). It is *not*
  sharpness: the narrowest pulse, Gaussian, tracks min-jerk, while Beta-asym has
  essentially min-jerk's width and beats it decisively.
- **Duration estimation improves with narrowness** — duration loss orders LGNB <
  Gaussian < Beta-asym < min-jerk, tracking sd, not symmetry.

## Result 2 — fit quality ranks the wrong family first

144 cells: 4 decoders × 4 generators × 9 noise/overlap conditions, 256 trials of 1000
samples. Median CI95 half-widths: R² ±0.0021, onset F1 ±0.0062, duration R² ±0.0136.

On Gaussian-generated data:

| decoder | recon R² | rank | onset F1 | rank | submov/s (true 2.66) |
|---|---|---|---|---|---|
| **Gaussian (correct)** | 0.8577 | 3rd | **0.8796** | 1st | 2.46 |
| min-jerk | **0.8710** | 1st | 0.7068 | 2nd | 3.23 |
| Beta-asym | 0.8658 | 2nd | **0.5423** | 4th | 3.86 |
| LGNB | 0.8371 | 4th | 0.6147 | 3rd | 3.74 |

The correct family places **3rd of 4 on fit and 1st on recovery**; fit's second choice
is the worst recoverer and inflates counts 45%. The wrong decoder's 0.013 R² advantage
is ~6× the CI, so the ordering is real. Across the 12 off-diagonal cells, **4** show a
mismatched decoder scoring higher R² than the matched one; **all 12** recover worse.

Mismatch penalty against each cell's own matched diagonal (worst first):

| decoder | data | Δ recon R² | Δ onset F1 | count ratio |
|---|---|---|---|---|
| Gaussian | Beta-asym | −0.123 | −0.412 | 1.10 |
| Beta-asym | Gaussian | **+0.008** | −0.337 | 1.57 |
| Gaussian | LGNB | −0.091 | −0.332 | 1.17 |
| LGNB | Gaussian | −0.021 | −0.265 | 1.52 |
| Gaussian | min-jerk | −0.064 | −0.241 | 1.13 |
| min-jerk | Gaussian | **+0.013** | −0.173 | 1.31 |

## Result 3 — the ceiling from shape alone

Decoder renders the *true* labels with its own family; no detector, no estimation error.
Model-free, and it reproduces inside the full matrix to 3 decimals.

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

## Result 5 — clean data punishes the wrong basis hardest

Mismatch penalty (matched − mismatched onset F1):

| SNR \ overlap | 0.0–0.5 | 0.5–1.0 | 1.0–1.5 |
|---|---|---|---|
| ∞ | 0.154 | **0.294** | 0.270 |
| 20 | 0.105 | 0.242 | 0.218 |
| 10 | **0.056** | 0.205 | 0.218 |

Noise does not reveal misspecification, it masks it — everything degrades toward the
same poor performance. Basis choice matters most where recordings are best. Overlap
sensitivity is entirely an **estimation** effect: the oracle ceiling above is
overlap-independent, so crowding only compounds shape error once it passes through a
detector.

## Limits

- **Epoch 12, not 24.** Colab reclaimed three VMs mid-study; epoch 12 was the deepest
  checkpoint held for all four arms. Conclusions reproduce at epoch 2 (a separate
  complete 144-cell matrix), at epoch 12, and in the model-free oracle, so they are not
  an artefact of training depth — but absolute numbers would shift.
- **Shape is frozen and uniform.** Nothing is predicted per submovement, so this
  measures uniform mis-specification, not within-trial shape heterogeneity.
- **One run per arm**, unseeded weight init. The two-arms-per-class design bounds the
  nuisance at ~1/10 of the effect; formal replicates are not in.
- **Synthetic only.** Recovery is only measurable where ground truth exists.

## Where the artefacts are

`runs/0901-pulse-families/` (gitignored, on this machine):

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
