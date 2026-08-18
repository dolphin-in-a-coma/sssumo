# Working on SSSUMO

Semi-supervised submovement decomposition. Read `README.md` for what the method is;
this file records what is expensive to discover by reading the code.

## Layout

- `sssumo/` — the package. `models.py` (detector + differentiable reconstructor),
  `data.py` (synthetic generator, organic loader, KDE samplers), `training.py`
  (the loop), `utils.py` (Config, metrics, evaluation, bootstrap).
- `scripts/train.py` — CLI entrypoint. `scripts/colab/` — remote-session tooling.
- `notebooks/` — `Inference.ipynb` is the maintained demo; `Train.ipynb` calls
  `sssumo.training.train`; `Analysis - organic and synth.ipynb` produced the
  article's numbers **and is where the bootstrap settings live**.
- `docs/VALIDATION.md` — the checkpoint reproduction study and its evidence.

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

**`--eval-datapoints` below ~32 can crash the evaluation.** `utils.py` fills NaNs
in pooled statistics by sampling each column's non-NaN values; with few trials
from an undertrained model a whole `next_*` column can be NaN and
`np.random.choice` raises. Use `None` (all trials) for anything reported.

**Stage-2 cost is dominated by statistics extraction, not training.** Each epoch
re-runs the model over every training participant and refits the `fastkde`
conditional distributions, at step 0 and after every epoch.

**`OrganicDataset` re-reads the whole CSV per noise condition** — 21 reads of up
to 567 MB per full organic evaluation where 7 would do.

## Model facts worth knowing before changing things

- The detector is fully convolutional with **centred** same-padding: each output
  depends on ±49 samples, so it is not causal.
- The reconstruction loss **cannot train the detection channel**:
  `gradient_for_detection` defaults to `False`, so the straight-through mask is
  detached. Detection is trained only by the BCE terms.
- The primitive shape is **frozen and symmetric** (`beta_mean=(0.5, 0)`,
  `beta_precision=(6, 0)`); the second element of each pair is a duration slope
  that is never used.
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
