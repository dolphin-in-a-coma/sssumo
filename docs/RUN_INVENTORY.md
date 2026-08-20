# Run inventory — reproduction study, 2026-08-17/19

Everything produced by the study in `VALIDATION.md`, and where it lives. Recorded
because none of it is recoverable from the code.

Repo state: branch `feat/colab-cli-training`, PR #2 (unmerged). All runs used
`scripts/train.py` under `scripts/colab/supervise.py` on Colab L4/T4.

## Training runs

wandb project `submovement_detector`. Names are the `--experiment-name` values;
checkpoints are `<name>_<epoch>.pth` in the project's Drive `weights/`.

| Run | Stage | Base weights | Seed offset | Epochs | Notes |
|---|---|---|---|---|---|
| `0817-ModGaussian_ampl_rerun` | 1 | scratch | 0 | 25 | spans two wandb runs — first VM died at epoch 6, resumed |
| `0817-ModGaussian_ampl_rerun_s1000` | 1 | scratch | 1000 | 25 | |
| `0817-tune_wo_writing` | 2 | released pretrained | 0 | 10 | first VM died at epoch 0, resumed |
| `0817-tune_wo_writing_from_rerun` | 2 | reproduced pretrained | 0 | 10 | |
| `0817-tune_wo_writing_s1000` | 2 | released pretrained | 1000 | 10 | T4 reclaimed twice, resumed from epochs 3 and 7 |
| `0817-tune_wo_writing_from_rerun_s1000` | 2 | reproduced pretrained | 1000 | 10 | T4 reclaimed, resumed |

Runs that span several wandb entries do so because a VM was reclaimed and the run
resumed with `--resume`; epoch numbering stays continuous in the checkpoints and
the text log, only the wandb view is split.

## Released checkpoints' originating runs

| Checkpoint | wandb run | Caveat |
|---|---|---|
| `config-0423-ModGaussian_ampl_24.pth` | `bxkweckh` | logged 8 of 25 epochs — the wandb page does **not** cover the epochs that produced the weights |
| `config-0425-tune_ModGauss_wo_writing_9.pth` | `jbj4im5v` | complete, all 10 000 steps |

See `checkpoints/README.md` for the full provenance note.

## Where the data lives

| Artefact | Location |
|---|---|
| All checkpoints (93 from this study) | project Drive `weights/` |
| Per-trial evaluation dumps | project Drive `per_trial_2026-08/` + `MANIFEST.md` |
| Derived tables | `docs/*.csv` in this repo |
| Reference pages | `docs/score_bands.html`, `docs/band_method_sensitivity.html` |

The per-trial dumps are the expensive artefact — about three GPU-hours. Every
interval variant in `VALIDATION.md` was recomputed from them offline with
`scripts/analysis/`.

## Measured costs (for planning)

| | T4 | L4 |
|---|---|---|
| Stage-1 epoch | ~11 min | ~4.8 min |
| Stage-2 epoch | — | ~8–13 min (extraction-dominated) |
| Organic eval, uncapped, 8 checkpoints | — | ~35 min per 4 checkpoints |

L4 quota was two concurrent; a third request returns `Precondition Failed`. T4s
were reclaimed after roughly 1–1.5 h mid-run; L4s completed multi-hour jobs.

## Published reference pages

Both were generated from the **capped** data and have not been regenerated from
the uncapped numbers:

- Score bands — https://claude.ai/code/artifact/e4e5fd33-aade-472d-bab8-17b627416fa3
- Band method sensitivity — https://claude.ai/code/artifact/18ffefed-9a26-4b78-847e-48e9c786de7e
