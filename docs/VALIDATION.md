# Reproducing the released checkpoints

Can the two checkpoints in `checkpoints/` be reproduced from the configs in this
repository? This records the attempt, the evidence, and what did not match.

Work in progress — sections marked *pending* are still running.

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
| stage-1 replicate | 1 | from scratch | 1000 | *pending* |
| A | 2 | published `_24` | 0 | *running* |
| B | 2 | reproduced `_24` | 0 | *running* |
| A2 | 2 | published `_24` | 1000 | *running* |
| B2 | 2 | reproduced `_24` | 1000 | *running* |

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

Candidate explanation, **not yet established**: training samples SNR from [10, 50],
so a noiseless input is extrapolation, and crank is the smoothest, most periodic
signal in the set — it was already the second-hardest dataset for the published
checkpoint. Adding mild noise restores near-parity. Whether this is within
pretraining run-to-run variance is what the stage-1 replicate is for.

## Result: stage 2 — *pending*

Interim, from the training trajectory (run A against published `jbj4im5v`,
epoch-aligned by step):

| Metric | Max |Δ| over comparable epochs |
|---|---|
| Onset precision | 0.0055 |
| Onset recall | 0.0095 |
| Detection loss | 0.0095 |

Final checkpoint comparison pending run completion.

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
