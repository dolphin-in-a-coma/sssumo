# Released checkpoints

| File | Stage | Config | wandb run |
|---|---|---|---|
| `config-0423-ModGaussian_ampl_24.pth` | 1 · pretraining | `configs/config-0423-ModGaussian_ampl.yaml` | `bxkweckh` |
| `config-0425-tune_ModGauss_wo_writing_9.pth` | 2 · fine-tuning | `configs/config-0425-tune_ModGauss_wo_writing.yaml` | `jbj4im5v` |

Fine-tuning starts from the pretrained checkpoint. `start_with_weights` resolves
against `<root_dir>/weights/`, so copy the file there before fine-tuning — it is
not read from `checkpoints/`.

## Provenance caveat for the pretrained checkpoint

`bxkweckh` is named *"DISCONNECTED BUT RUN WELL"* and is in state `crashed` with
**8176 of its 25 000 steps logged** — about 8 of 25 epochs. Training continued
after wandb logging stopped: `_24.pth` was written 93 minutes after the run
started, consistent with the logged step rate. **The wandb page therefore does not
contain the epochs that produced these weights.** The full history is in the
training log stored beside the checkpoint in the run's `root_dir/logs/`.

`jbj4im5v` finished cleanly with all 10 000 steps, so its wandb history is
complete.

Note the name mismatch is expected, not a bookkeeping error: `Config` strips the
`config-` prefix when deriving an experiment name from a filename, while explicit
assignment keeps it, so the run is `0423-ModGaussian_ampl` and the file is
`config-0423-ModGaussian_ampl_24.pth`.

## Licence

The fine-tuned checkpoint is trained on a mixture that **excludes** handwriting,
which is why it can be released under CC BY 4.0. Any checkpoint fine-tuned on a
mixture including `tablet_writing` inherits that dataset's research-only licence.

## Reproduction

Both reproduce; see `docs/VALIDATION.md` for the evidence and the one documented
discrepancy.
