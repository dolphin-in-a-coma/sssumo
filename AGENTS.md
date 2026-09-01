# Working on SSSUMO

Semi-supervised submovement decomposition. Read `README.md` for what the method is;
this file records what is expensive to discover by reading the code.

## Layout

- `sssumo/` — the package. `models.py` (detector + differentiable reconstructor),
  `data.py` (synthetic generator, organic loader, KDE samplers), `training.py`
  (the loop), `utils.py` (Config, metrics, evaluation, bootstrap).
- `scripts/train.py` — CLI entrypoint. `scripts/colab/` — remote-session tooling;
  its README's **Running** section is the end-to-end recipe for a new GPU run,
  including the watcher and the artifact mirror that keep one from losing its
  checkpoints.
- `notebooks/` — `Inference.ipynb` is the maintained demo; `Train.ipynb` calls
  `sssumo.training.train`; `Analysis - organic and synth.ipynb` produced the
  article's numbers **and is where the bootstrap settings live**.
- `docs/VALIDATION.md` — the checkpoint reproduction study and its evidence.
`docs/PULSE_FAMILIES.md` — the primitive-shape study: minimum jerk vs Gaussian,
Beta and LGNB, and what a mismatched pulse costs. `docs/COLAB_RELIABILITY.md` —
why Colab sessions die at the one-hour mark, and the two fixes that shipped:
`scripts/colab/watch.py` (host-side) and `scripts/colab/apply_cli_patch.py`
(the CLI's own keep-alive daemon).

`data/` is gitignored (~1.9 GB). Fetch it from the public archive the inference
notebook uses; there is a `curl` one-liner in `scripts/colab/README.md`.

## Traps that have cost time

**`config.seed` does not affect the training data.** `train()` sets
`dataset.seed = epoch` every epoch, and every sample derives its randomness from
`idx + seed * len(dataset)`. Combined with `dropout_rate: 0` in the fine-tuning
configs, two runs of the same config are bit-identical. Use `--seed-offset` for a
replicate.

**`Config` flattens every YAML section** into plain attributes, then derives
`<root_dir>/data`, `<root_dir>/weights/<experiment>.pth`,
`<root_dir>/logs/<experiment>.txt`. The config *file* lives in `configs/`, but
`root_dir` needs `data/`, `weights/`, `logs/` beside it.

**Two paths produce two different experiment names.** `Config` strips the
`config-` prefix when deriving a name from a filename; explicit assignment keeps
it. That is why the released checkpoint is `config-0423-ModGaussian_ampl_24.pth`
while its wandb run is `0423-ModGaussian_ampl`.

**Train/test is an alternating participant split** — `participants[::2]` and
`[1::2]`. Halving Table 1's subject counts gives the test sizes: crank 5,
object_moving 5, **pointing 2**. Any per-dataset statistic on those rests on very
few clusters.

**Participant IDs are only unique within a dataset.** Five IDs repeat across the
seven organic datasets, so a global `groupby('Participant')` or
`nunique('Participant')` silently merges distinct people — it reports 256 test
participants as 251. Always group by `('Dataset', 'Participant')`, or scope the
frame to one dataset first, as `score_checkpoints.py` does.

**`--eval-datapoints` below ~32 can crash the evaluation.** `utils.py` fills NaNs
in pooled statistics by sampling each column's non-NaN values; with few trials
from an undertrained model a whole `next_*` column can be NaN and
`np.random.choice` raises. Use `None` (all trials) for anything reported.

**Stage-2 cost is dominated by statistics extraction, not training.** Each epoch
re-runs the model over every training participant and refits the `fastkde`
conditional distributions, at step 0 and after every epoch.

**`torchviz` is a module-scope import in `models.py` for dead code.** `make_dot` is
never called, but the import makes `torchviz` a hard dependency — and installing it
drags in `torch`, which on a managed GPU image (Kaggle, Colab) replaces the
GPU-matched build with a generic wheel and produces `CUDA error: no kernel image is
available for execution on the device`. On such a host, stub it onto `PYTHONPATH`
rather than installing it.

**Metrics come back as tensors still on the device.** `calculate_supervised_metrics`
computes the submovement counts from the mask, so they are CUDA tensors, and numpy
refuses to convert those. Anything that aggregates metric output must move to CPU
first. This passes silently on CPU, so it only surfaces on the GPU the script was
written to run on.

**`OrganicDataset` re-reads the whole CSV per noise condition** — 21 reads of up
to 567 MB per full organic evaluation where 7 would do.

## Model facts worth knowing before changing things

- The detector is fully convolutional with **centred** same-padding: each output
  depends on ±49 samples, so it is not causal.
- The reconstruction loss **cannot train the detection channel**:
  `gradient_for_detection` defaults to `False`, so the straight-through mask is
  detached. Detection is trained only by the BCE terms.
- The primitive shape is **a family, defaulting to minimum jerk**. `beta_mean=(0.5,
  0)`, `beta_precision=(6, 0)` is Beta(3,3), which *is* `30 s^2 (1-s)^2` — the two
  agree to 2.3e-8 on this grid, so minimum jerk is a frozen special case rather
  than a separate implementation. `ContinuousPrimitive(family=...)` also offers
  `gaussian` (truncated) and `lgnb` (support-bounded lognormal); every family is a
  unit-area density on normalised time, so the amplitude channel keeps meaning
  "area" across families. Set them from a `primitive:` YAML section. The second
  element of each parameter pair is a duration slope, 0 in every shipped config.
- **The shape is not predicted per submovement.** The detector emits
  `[onset, amplitude, duration]` and nothing else, so every pulse in a run has the
  same asymmetry and sharpness; duration sets width, which is scale, not shape.
  Per-submovement shape would need extra output channels — and note that
  `shape[1] == 4` already means "reconstruction mask" in `models.py` and
  `training.py`, so that dead path has to go first or the two collide.
- **Learning the shape needs two things beyond unfreezing it.** The optimiser is
  built from `model.parameters()`, and the primitive lives on the reconstructor, so
  `freeze_primitive_parameters: false` alone accumulates gradients that nothing
  steps. And `Config` hands the dataset the *same* reconstructor the loop decodes
  with unless a `generator_primitive_*` key asks for a separate frozen one — with a
  trainable shape that is degenerate, since training can cut the reconstruction
  loss by moving the ground truth instead of fitting it.
- The **4-channel reconstruction-mask path is dead code** — every config ends
  `channels: [..., 3]`, but `models.py` and the loop both carry `shape[1] == 4`
  branches.

## Licensing

Training on a mixture that includes `tablet_writing` yields weights covered by
that dataset's research-only licence. That is why
`config-0425-tune_ModGauss_wo_writing.yaml` is the released fine-tuning config.
Pretraining is unaffected — it is purely synthetic, and `tablet_writing` is only
read during evaluation.

## Conventions

- Keep notebook outputs out of commits (they have leaked an account name before).
- Never commit weights, logs, pulled-stats CSVs, or `*_output.ipynb`; `.gitignore`
  covers these.
- Credentials come from `~/.netrc` at run time. Never put a key in a config, in
  `argv`, or in the repo.
