# Reproducing the released checkpoints

Can the two checkpoints in `checkpoints/` be reproduced from the configs in this
repository? This records the attempt, the evidence, and what did not match.

**Conclusion: both released checkpoints reproduce.** The residual differences are
the size of run-to-run variance. One dataset-specific discrepancy in the
*pretrained* checkpoint is characterised below; it does not survive fine-tuning.

## The artefacts under test

| Checkpoint | Stage | Config | Originating wandb run |
|---|---|---|---|
| `config-0423-ModGaussian_ampl_24.pth` | 1, pretraining | `config-0423-ModGaussian_ampl.yaml` | `bxkweckh` |
| `config-0425-tune_ModGauss_wo_writing_9.pth` | 2, fine-tuning | `config-0425-tune_ModGauss_wo_writing.yaml` | `jbj4im5v` |

**Provenance caveat.** `bxkweckh` is named *"DISCONNECTED BUT RUN WELL"* and is in
state `crashed` with only 8176 steps logged — about 8 of its 25 epochs. Training
continued after wandb logging stopped (checkpoint `_24.pth` was written 93 minutes
after the run started, consistent with the logged step rate), but **the wandb run
page does not contain the epochs that produced the released weights.** The full
history is in the training log alongside the checkpoint. `jbj4im5v` by contrast
finished cleanly with all 10000 steps.

## Method

Two independent lines of evidence:

1. **Trajectory** — compare per-epoch training metrics against the published run
   (`scripts/colab/wandb_compare_runs.py`). Catches divergence early and does not
   depend on the evaluation code.
2. **Behaviour** — re-evaluate the finished checkpoints on identical data
   (`scripts/colab/compare_checkpoints.py`). Paired: the RNGs are reseeded
   identically per checkpoint, so both see the same trials, the same organic
   subsample and the same noise. Without that, differences of the size we are
   looking for are indistinguishable from sampling variance.

Replicates with `--seed-offset` establish run-to-run variance, which is what makes
any observed gap interpretable. Note that changing `seed` in the YAML does *not*
work: `train()` sets `dataset.seed = epoch`, so `config.seed` never reaches the
data stream, and with `dropout_rate: 0` two runs of a fine-tuning config are
bit-identical.

## Runs

| Label | Stage | Base weights | Seed offset | Status |
|---|---|---|---|---|
| stage-1 rerun | 1 | from scratch | 0 | complete, 25 epochs |
| stage-1 replicate | 1 | from scratch | 1000 | complete, 25 epochs |
| A | 2 | published `_24` | 0 | complete |
| B | 2 | reproduced `_24` | 0 | complete |
| A2 | 2 | published `_24` | 1000 | complete |
| B2 | 2 | reproduced `_24` | 1000 | complete |

All six ran on Colab (L4 and T4) via `scripts/train.py` under
`scripts/colab/supervise.py`. Four VMs were reclaimed mid-run; each was resumed
from its last checkpoint with `--resume`, which is why some runs span two wandb
entries.

## Result: stage 1 reproduces

Synthetic evaluation with known ground truth, averaged over 9 noise × refractory
conditions, published checkpoint versus reproduction:

| Metric | Published | Reproduced | Δ |
|---|---:|---:|---:|
| Onset F1 | 0.8459 | 0.8497 | +0.0038 |
| Onset precision | 0.8035 | 0.8099 | +0.0064 |
| Onset recall | 0.8962 | 0.8962 | −0.0000 |
| Onset distance (samples) | 0.8238 | 0.8222 | −0.0016 |
| Reconstruction R² | 0.9842 | 0.9864 | +0.0023 |
| Amplitude R² | 0.8422 | 0.8400 | −0.0021 |
| Duration R² | 0.8143 | 0.8136 | −0.0007 |

The training trajectory agrees too: against the 8 epochs `bxkweckh` did log,
per-epoch onset precision differs by at most 0.006 after epoch 0, and detection
loss by at most 0.008.

### One discrepancy: crank at SNR ∞

Across all 24 organic dataset × noise cells, exactly one exceeds |ΔR²| = 0.02:

| | Published | Reproduced | Δ |
|---|---:|---:|---:|
| **crank, SNR ∞** | 0.6592 | 0.1990 | **−0.4603** |
| crank, SNR 20 | 0.6558 | 0.6312 | −0.0246 |
| crank, SNR 10 | 0.5853 | 0.5429 | −0.0424 |

Everything else sits within ±0.016. The gap is not sampling noise: at SNR ∞ no
noise is added and the trial subsample was seeded identically, so both models saw
identical input. The reproduction under-detects there (5.04 → 4.42 submovements/s).

**Per-trial diagnosis** (`scripts/colab/diagnose_dataset.py`, 64 trials) shows this
is not a uniform degradation but a small number of catastrophic failures:

| | Published | Reproduced |
|---|---:|---:|
| R² mean | 0.665 | 0.222 |
| R² median | 0.735 | 0.622 |
| R² min | −0.006 | −6.38 |
| trials with R² < 0 | 1.6% | 17.2% |

The median falls only 0.11; the 10 worst trials account for **78%** of the total
negative delta, and 12 of 64 are worse by more than 0.5.

![worst crank trials](crank_snr_inf_worst_trials.png)

The mechanism is visible in those trials: crank is continuous rotation, so the
clean signal is a sustained level with small ripples. The reproduced model
intermittently **stops emitting onsets** across such stretches, and because the
reconstruction is a sum of bells anchored at detected onsets, it collapses toward
zero in the gaps and recovers afterwards. That is detection dropout, not an
amplitude error, and it matches the lower detection rate (5.04 → 4.42
submovements/s).

Why it appears only at SNR ∞ is consistent with extrapolation: training samples
SNR from [10, 50], so a perfectly noiseless input is outside the training
distribution, and crank is the smoothest signal in the set. Adding mild noise
restores near-parity. This remains a hypothesis about the cause; the failure mode
itself is established.

### Putting the discrepancy in scale: a seed replicate

A second pretraining run differing only by `--seed-offset 1000` settles whether
the gap is meaningful. Synthetic metrics, mean absolute difference over the seven
ground-truth measures:

| Pair | mean \|Δ\| | max \|Δ\| |
|---|---:|---:|
| published vs rerun | **0.0024** | 0.0064 |
| published vs replicate | 0.0038 | 0.0084 |
| rerun vs replicate (seed only) | **0.0029** | 0.0082 |

**The distance from the published checkpoint to a reproduction is no larger than
the distance between two reproductions that differ only by seed.** On the measure
pretraining actually optimises, reproduction succeeds.

Crank remains the exception, and it is systematic rather than a fluke — both
reproductions under-detect there, and crank is the *only* dataset beyond
|ΔR²| = 0.02 for either (3 cells each, one per noise level):

| crank R² | Published | Rerun | Replicate |
|---|---:|---:|---:|
| SNR ∞ | 0.6592 | 0.1990 | 0.5875 |
| SNR 20 | 0.6558 | 0.6312 | 0.6318 |
| SNR 10 | 0.5853 | 0.5429 | 0.4864 |
| submovements/s at SNR ∞ | 5.04 | 4.42 | 4.51 |

So the noiseless-crank cell is both **systematically biased** (both reproductions
detect ~11% fewer submovements than the published checkpoint) and **extremely
high-variance** (−0.07 versus −0.46 between two seeds). That combination is what
you expect from an out-of-distribution operating point where a few trials tip
into catastrophic detection dropout: the direction is reproducible, the magnitude
is not. The released checkpoint sits on the fortunate side of it.

## Result: stage 2 reproduces

Run A (fine-tuned from the released pretrained checkpoint, as the config
specifies) against the released fine-tuned checkpoint.

**Trajectory.** Against the published run `jbj4im5v`, epoch-aligned by step:
onset precision differs by at most 0.0055, recall by 0.0095, detection loss by
0.0095, at every comparable epoch.

**Behaviour.** Synthetic evaluation, mean over 9 noise × refractory conditions:

| Metric | Released | Reproduced | Δ |
|---|---:|---:|---:|
| Onset precision | 0.8116 | 0.8017 | −0.0100 |
| Onset recall | 0.8811 | 0.8807 | −0.0004 |
| Onset F1 | 0.8440 | 0.8381 | −0.0059 |
| Onset distance | 0.8187 | 0.8160 | −0.0026 |
| Reconstruction R² | 0.9856 | 0.9849 | −0.0006 |
| Amplitude R² | 0.8455 | 0.8446 | −0.0009 |
| Duration R² | 0.8045 | 0.8039 | −0.0006 |

Organic test participants, mean R² across datasets: −0.0004 at SNR ∞, +0.0004 at
SNR 20, −0.0089 at SNR 10. Two of 24 dataset × noise cells exceed |ΔR²| = 0.02,
both at the noisiest setting (crank −0.042, object_moving −0.026).

Note that fine-tuning lifts organic reconstruction R² from ≈0.88 to ≈0.93.

### The stage-1 crank defect does not survive fine-tuning

Run B was fine-tuned from the *reproduced* pretrained checkpoint — the one
carrying the crank/SNR ∞ defect. It matches the released fine-tuned checkpoint at
least as well as run A does:

| | Released | A (from published) | B (from reproduced) |
|---|---:|---:|---:|
| crank, SNR ∞ | 0.8894 | 0.8871 | **0.9026** |
| crank, SNR 20 | 0.9246 | 0.9253 | 0.9177 |
| crank, SNR 10 | 0.8846 | 0.8425 | 0.8903 |
| organic mean R², SNR ∞ | 0.9330 | 0.9326 | 0.9326 |
| organic mean R², SNR 20 | 0.9386 | 0.9390 | 0.9380 |
| organic mean R², SNR 10 | 0.9109 | 0.9020 | 0.9093 |
| cells with \|ΔR²\| > 0.02 | — | 2 of 24 | **0 of 24** |

Synthetic deltas for run B are all ≤ 0.009. So the pretraining discrepancy is
fully repaired by the semi-supervised stage: crank/SNR ∞ goes from 0.199 in the
reproduced pretrained weights to 0.903 after fine-tuning, against 0.889 for the
released checkpoint.

### Reproduction error against run-to-run variance

With the 2×2 complete (base checkpoint × seed), the residuals can be attributed.
Mean absolute difference over the seven ground-truth synthetic metrics:

| Comparison | Isolates | mean \|Δ\| | max \|Δ\| |
|---|---|---:|---:|
| released vs A | reproduction | 0.0030 | 0.0100 |
| released vs B | reproduction | 0.0021 | 0.0087 |
| released vs A2 | reproduction | 0.0065 | 0.0156 |
| released vs B2 | reproduction | 0.0032 | 0.0123 |
| A vs A2 | **seed only** | 0.0036 | 0.0082 |
| B vs B2 | **seed only** | 0.0017 | 0.0036 |
| A vs B | **base only** | 0.0039 | 0.0113 |
| A2 vs B2 | **base only** | 0.0068 | 0.0196 |

Every quantity is the same order: reproduction error 0.002–0.007, seed variance
0.002–0.004, base-checkpoint effect 0.004–0.007. **The distance from the released
checkpoint to a reproduction is not distinguishable from the distance between two
reproductions.**

Organic mean R² across all five checkpoints spans 0.9324–0.9341 at SNR ∞,
0.9379–0.9396 at SNR 20 and 0.9020–0.9109 at SNR 10. Cells beyond |ΔR²| = 0.02
against the released weights: A has 2 of 24, B has 0, A2 has 0, B2 has 1 — and
crank appears only once, at the noisiest setting. The pretraining crank defect is
gone from every fine-tuned model.

### Confidence intervals on the reproduction gap

Point estimates alone cannot say whether a gap matters. `compare_checkpoints.py
--bootstrap N` bootstraps the **paired per-trial difference** against the released
checkpoint, resampling participants (2000 resamples, 95% intervals, one per
dataset × noise cell). Pairing is exact — every checkpoint sees identically seeded
trials — which removes trial-level variance from the interval.

Across 84 cells (21 dataset × noise, 4 reproductions), the delta in reconstruction
R² exceeds 0.01 in **two**:

| Checkpoint | Cell | Δ R² | 95% CI |
|---|---|---:|---|
| A | crank, SNR 10 | −0.0416 | [−0.062, −0.026] |
| B2 | crank, SNR ∞ | +0.0104 | [+0.002, +0.025] |

Everything else is below 0.005. Both outliers are crank, the dataset that also
carried the stage-1 discrepancy.

**Statistical significance is not the useful criterion here.** 16 of 84 intervals
exclude zero, but 14 of those have |Δ| < 0.005 — differences of 0.001 R² on values
near 0.93. Whacamole has 176 test participants, so its intervals are tight enough
to resolve a 0.0008 difference as "significant". Magnitude is what matters, and by
magnitude the reproductions are equivalent to the released weights everywhere
except crank.

**Caveat on participant counts.** The test split sizes are very uneven:

| Dataset | whacamole | tablet_writing | Fitts | steering | crank | object_moving | pointing |
|---|---:|---:|---:|---:|---:|---:|---:|
| test participants | 176 | 41 | 10 | 9 | 5 | 5 | **2** |

Participant-resampled intervals for pointing (n = 2), crank (n = 5) and
object_moving (n = 5) rest on very few clusters and should be read as indicative
only. That is a direct caveat on the crank finding, which is the one result this
whole exercise turned on: it is based on five test participants.

## Reproducing this

```bash
# stage 1
python scripts/train.py --config configs/config-0423-ModGaussian_ampl.yaml \
    --root-dir <root> --experiment-name my-pretrain

# stage 2, from the released pretrained checkpoint
python scripts/train.py --config configs/config-0425-tune_ModGauss_wo_writing.yaml \
    --root-dir <root> --experiment-name my-finetune

# compare against the released weights
python scripts/colab/compare_checkpoints.py --root-dir <root> \
    --config configs/config-0423-ModGaussian_ampl.yaml \
    --checkpoint published=config-0423-ModGaussian_ampl_24.pth \
    --checkpoint mine=my-pretrain_24.pth
```

See `scripts/colab/README.md` for driving these on Colab, including measured
timings and the failure modes worth knowing about.
