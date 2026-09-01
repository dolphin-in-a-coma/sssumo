# How SSSUMO works

What the code does and how the pieces fit. `README.md` covers installation and
citation; `AGENTS.md` lists the traps that have cost time; `docs/VALIDATION.md`
records the checkpoint reproduction study. This file is the map between them.

## The problem

A human reaching movement is not one smooth motion. It is a sequence of
overlapping **submovements** — discrete ballistic bursts the motor system issues
and blends together. Given only the **tangential velocity** (STV, the speed of
the hand or cursor regardless of direction, a single number per timestep),
SSSUMO recovers *when* each submovement started, *how long* it lasted and *how
large* it was.

The difficulty is that ground truth does not exist for real movement. Nobody can
label where a submovement began in a recording of a person turning a crank. So
the method is **semi-supervised**: it learns on synthetic signals, where the
labels are known by construction, and uses real recordings only through the
*statistics* of what it detects in them — never as targets.

Key terms used throughout the code:

| Term | Meaning |
|---|---|
| **organic** | real human recordings (seven datasets under `data/`) |
| **synthetic** | generated signals with known submovement labels |
| **onset** | the sample index where a submovement begins |
| **refractory** | the gap between consecutive onsets |
| **primitive** | the fixed velocity shape a single submovement contributes |
| **STV** | scalar tangential velocity, the model's only input |

## Two models, one forward pass

### The detector — `TDNNDetector` ([models.py:77](../sssumo/models.py#L77))

A time-delay neural network: nine 1-D convolutions, no pooling, no recurrence.
It maps a length-*T* velocity trace to a 3 × *T* output, one prediction per
input sample:

| Channel | Meaning | Activation |
|---|---|---|
| 0 | onset probability | sigmoid |
| 1 | amplitude | linear |
| 2 | duration | linear |

Channel assignment is at [training.py:263](../sssumo/training.py#L263).

The released configuration uses `channels: [1, 8, 16, 32, 64, 128, 128, 256, 256, 3]`
with `kernel_sizes: [13]*8 + [3]` and `dilations: 1`. Padding is
`(kernel_size - 1) * dilation // 2` — **centred**, not causal. Summing
`kernel_size - 1` over the nine layers gives a **receptive field of 99 samples**,
so each output looks 49 samples forward as well as back. The model is real-time
in throughput, not in latency.

### The reconstructor — `STEContinuousReconstructor` ([models.py:164](../sssumo/models.py#L164))

Turns the detector's three channels back into a velocity trace, differentiably,
so that reconstruction error can be a training signal.

1. **Binarise** the onset channel. `STEBinarizer` thresholds it in the forward
   pass but passes gradients straight through, so a hard decision stays
   trainable.
2. **Shape** each detected onset with a Beta-function primitive scaled by its
   predicted amplitude and duration.
3. **Sum** the shifted primitives.

The primitive is **frozen and symmetric** by default —
`primitive_beta_mean=[0.5, 0.0]`, `primitive_beta_precision=[6., 0.0]`
([models.py:165](../sssumo/models.py#L165)). The second element of each pair is a
duration-dependent slope that is never exercised.

**The reconstruction loss cannot train detection.** `gradient_for_detection`
defaults to `False` ([models.py:166](../sssumo/models.py#L166)), which detaches
the straight-through mask. Detection is trained by the cross-entropy terms alone;
reconstruction shapes amplitude and duration only.

## Training in two stages

### Stage 1 — synthetic pretraining

`configs/config-0423-ModGaussian_ampl.yaml`, 25 epochs. Purely synthetic: a
generator draws submovement onsets, durations and amplitudes from configured
distributions, sums the primitives, adds Gaussian noise at a sampled
signal-to-noise ratio, and hands over both the signal and its labels.

`use_reconstruction_loss: false` for the first 10 epochs
(`reconstruction_loss_start: 10`), and `bn_dropout_freeze_start: 20` freezes
batch-norm statistics and disables dropout for the last five.

Produces `config-0423-ModGaussian_ampl_24.pth`.

### Stage 2 — semi-supervised fine-tuning

`configs/config-0425-tune_ModGauss_wo_writing.yaml`, 10 epochs, starting from the
stage-1 checkpoint (`start_with_weights`). This is where organic data enters, and
it enters **only as statistics**
([training.py:98](../sssumo/training.py#L98)):

1. Run the current model over the *training* participants of each organic
   dataset.
2. Pool what it detected into per-dataset distributions of duration, amplitude
   and refractory period, fitted with `fastkde`.
3. Build a `SyntheticDataset` for each organic dataset that samples from those
   distributions — synthetic signals with real-looking statistics, and therefore
   with exact labels.
4. Mix them with the generic synthetic stream via `CombinedSyntheticDataset`
   using `proportions` (`0.07` each for six organic datasets, `0.42` generic).

Real signals are never regression targets. Step 1–3 repeat at step 0 and after
every epoch, which is why **stage-2 cost is dominated by statistics extraction,
not by gradient steps**.

`dropout_rate: 0` here, which matters — see the reproducibility note below.

### The loss

Four terms, summed ([training.py:366](../sssumo/training.py#L366)):

```
loss = detection + duration + amplitude [+ reconstruction]
```

- **detection** — binary cross-entropy, split into positive and negative parts.
  Positives are weighted by submovement amplitude when `weight_with_amplitude`
  is set; negatives by `negative_loss_multiplier`, which under `'adaptive'`
  scales by the ratio of predicted to true submovement counts.
- **duration**, **amplitude** — mean squared error at true onsets only.
- **reconstruction** — mean squared error between the reconstructed and clean
  signal, added from epoch `reconstruction_loss_start`.

Each auxiliary term is divided by its own running mean and multiplied by the
detection loss's running mean
([training.py:350](../sssumo/training.py#L350)), so all terms sit on a comparable
scale whatever their raw units. The consequence worth knowing: the *relative*
weight of the terms is fixed by construction, not tunable through the config.

## Directory map

```
sssumo/                package
  models.py            detector, reconstructor, primitives
  data.py              synthetic generator, organic loader, KDE samplers
  training.py          the training loop and its entry point train()
  utils.py             Config, metrics, evaluation, bootstrap
  dataset_reader.py    builds the STV CSVs from the original public datasets
  alternative_detectors.py, movement_decompose.py   baselines
configs/               YAML: one stage-1, ablations, and per-dataset holdouts
checkpoints/           released weights + provenance (checkpoints/README.md)
data/                  organic STV CSVs — gitignored, ~1.9 GB
notebooks/             Inference (maintained), Train, Analysis
scripts/
  train.py             CLI entry point
  colab/               remote GPU session tooling (scripts/colab/README.md)
  analysis/            offline re-analysis of per-trial dumps
docs/                  this file, VALIDATION.md, IMPROVEMENTS.md, RUN_INVENTORY.md
```

## Running it

### Install

```bash
pip install -e .
```

`data/` is gitignored and not shipped. Fetch it from the public archive the
inference notebook uses; there is a `curl` one-liner in
[scripts/colab/README.md](../scripts/colab/README.md).

### Train

```bash
python scripts/train.py --config configs/config-0423-ModGaussian_ampl.yaml --root-dir <root>
```

`--root-dir` is **not** the repo root. `Config` derives `<root>/data`,
`<root>/weights/<experiment>.pth`, `<root>/logs/<experiment>.txt` and
`<root>/TensorBoardLogs/<experiment>` from it
([utils.py:271](../sssumo/utils.py#L271)), so those three directories must exist
beside each other. The config *file* stays in `configs/`.

Useful flags ([scripts/train.py:32](../scripts/train.py#L32)):

| Flag | Effect |
|---|---|
| `--experiment-name` | names the run and therefore the checkpoint file |
| `--resume` | continue from the highest epoch checkpoint in `weights/` |
| `--seed-offset` | the only way to get a different training stream (below) |
| `--organic-eval-every`, `--synthetic-eval-every` | `0` disables |
| `--eval-datapoints` | cap trials per evaluation; `None` uses all |
| `--no-wandb` | disable logging; losses still go to `logs/<experiment>.txt` |

For a GPU session on Colab driven from the command line, see
[scripts/colab/README.md](../scripts/colab/README.md).

### Evaluate

`scripts/colab/score_checkpoints.py` scores checkpoints on both domains and can
dump per-trial results (`--dump-per-trial`), from which every interval in
`docs/VALIDATION.md` is recomputed offline with no GPU
([scripts/analysis/README.md](../scripts/analysis/README.md)).

## Configuration reference

`Config` ([utils.py:218](../sssumo/utils.py#L218)) loads a YAML file and
**flattens every section into plain attributes** — `general:`, `model:`,
`data:`, `training:`, `test:` are organisational only; nothing reads the section
names. Two derived values are set in the constructor: `device` from CUDA
availability (overriding whatever the file says) and `reconstruction_model`, an
`STEContinuousReconstructor` built from `gradient_for_detection` and the duration
range.

Parameters worth knowing:

| Key | Section | Meaning |
|---|---|---|
| `channels`, `kernel_sizes`, `num_layers`, `dilations` | model | detector shape; `channels` must end in `3` |
| `dropout_rate` | model | `0.2` in stage 1, `0` in stage 2 |
| `total_duration_distribution` | data | trial length in samples; evaluation overrides this to 1000 |
| `snr_distribution` | data | signal-to-noise range for added noise |
| `duration_distribution` | data | submovement duration in samples, `[5, 60]` |
| `refractory_distribution` | data | onset spacing, as percentages of duration |
| `combined_dataset`, `datasets`, `proportions` | data | the stage-2 mixture |
| `use_reconstruction_loss`, `reconstruction_loss_start` | training | when reconstruction enters |
| `bn_dropout_freeze_start` | training | epoch to freeze batch-norm and drop dropout |
| `start_with_weights` | training | `false`, a filename, or `true` for "highest epoch present" |
| `negative_loss_multiplier` | training | `'adaptive'`, `'balanced'`, or a number |

### Experiment naming

`Config` strips a leading `config-` when deriving the name from a filename, but
an explicit `experiment_name` in the YAML is kept verbatim. That is why the
released checkpoint is `config-0423-ModGaussian_ampl_24.pth` while its Weights &
Biases run is `0423-ModGaussian_ampl`. Checkpoints are written as
`<experiment_name>_<epoch>.pth`, so **reusing a config's own name will overwrite
released weights at the final epoch**.

## Reproducibility

**`config.seed` does not affect the training data.** `train()` sets
`dataset.seed = epoch + seed_offset` at the start of every epoch, and each sample
derives its randomness from `idx + seed * len(dataset)`. With `dropout_rate: 0`
in the fine-tuning configs, two runs of the same config are bit-identical. Use
`--seed-offset` for a genuine replicate — this is how `ft_A2` and `ft_B2` in the
validation study were produced.

Both released checkpoints have been reproduced from scratch; the reproduction
error is indistinguishable from seed-to-seed variance. Evidence, per-trial dumps
and the full interval methodology are in [VALIDATION.md](VALIDATION.md).

## Evaluation and reporting

Two domains, different ground truth:

- **Synthetic** — labels are known, so onset precision/recall/F1, onset distance,
  and amplitude/duration R² are all available.
- **Organic** — no labels, so only reconstruction quality (R², MASE, SMAPE) and
  the submovement rate can be measured.

Organic evaluation sweeps three noise conditions (signal-to-noise 10, 20, and
noiseless) across seven datasets.

**Train/test is an alternating participant split** — `participants[::2]` and
`[1::2]`. Test-set sizes are small on several datasets: crank 5,
object_moving 5, **pointing 2**.

For which interval to report and why the choice changes conclusions, see
[VALIDATION.md](VALIDATION.md) and the `sssumo-evaluation` skill under
`.agent/skills/`. The short version: resample exactly one level — the coarsest
you mean to generalise over — and report the conditional interval with the
participant and trial counts beside it.

## Licensing

Training on a mixture that includes `tablet_writing` produces weights covered by
that dataset's **research-only licence**. `config-0425-tune_ModGauss_wo_writing.yaml`
is the released fine-tuning config for exactly this reason. Stage-1 pretraining is
unaffected — it is purely synthetic, and `tablet_writing` is only ever read during
evaluation.

## Where to look next

| Question | File |
|---|---|
| What will trip me up in the code? | [`AGENTS.md`](../AGENTS.md) |
| Do the released checkpoints reproduce? | [`docs/VALIDATION.md`](VALIDATION.md) |
| Which checkpoint came from which run? | [`checkpoints/README.md`](../checkpoints/README.md) |
| How do I run this on a Colab GPU? | [`scripts/colab/README.md`](../scripts/colab/README.md) |
| How do I recompute intervals offline? | [`scripts/analysis/README.md`](../scripts/analysis/README.md) |
| What is known to be worth fixing? | [`docs/IMPROVEMENTS.md`](IMPROVEMENTS.md) |
| What runs exist and what did they cost? | [`docs/RUN_INVENTORY.md`](RUN_INVENTORY.md) |
