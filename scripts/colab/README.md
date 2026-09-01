# Training on Colab from the command line

Drive a Colab GPU session with the [`colab` CLI](https://pypi.org/project/google-colab-cli/)
instead of the notebook UI. `notebooks/Train.ipynb` stays the readable reference;
`scripts/train.py` is the same loop as a script, so it can run unattended.

```bash
pip install google-colab-cli
```

## One-time setup

```bash
cp scripts/colab/run.example.json scripts/colab/run.local.json
```

Edit `run.local.json` — it is gitignored, and it is the only file that should
ever hold machine-specific paths or account names.

## Running

```bash
SESSION=my-run

# 1. allocate a GPU
colab --auth=oauth2 new -s $SESSION --gpu T4

# 2. (only for data_source "drive:") mount Drive -- needs a TTY, so run it yourself
colab --auth=oauth2 drivemount -s $SESSION

# 3. upload the config and the tools the VM needs
colab --auth=oauth2 upload -s $SESSION scripts/colab/run.local.json /content/run.json
colab --auth=oauth2 upload -s $SESSION scripts/train.py            /content/train.py
colab --auth=oauth2 upload -s $SESSION scripts/colab/supervise.py  /content/supervise.py

# 4. clone the pinned commit, install, stage data (~1 min from Drive, longer from the URL)
colab --auth=oauth2 exec -s $SESSION -f scripts/colab/vm_setup.py --timeout 1800

# 5. smoke test before committing hours of GPU time
#    (temporarily set num_samples 20, num_epochs 1, organic_eval_every 1,
#     eval_datapoints 64 in run.local.json and re-upload)
colab --auth=oauth2 exec -s $SESSION -f scripts/colab/vm_launch.py --timeout 180
colab --auth=oauth2 exec -s $SESSION -f scripts/colab/vm_wait.py   --timeout 1700

# 6. the real run
colab --auth=oauth2 exec -s $SESSION -f scripts/colab/vm_launch.py --timeout 180
colab --auth=oauth2 exec -s $SESSION -f scripts/colab/vm_status.py --timeout 180   # poll

# 7. always release the VM
colab --auth=oauth2 stop -s $SESSION
```

`colab exec -f` passes no `argv`, which is why the VM-side tools read
`/content/run.json` rather than taking flags.

## The wandb key

Never put the key in `run.local.json`, in a command argument, or in the repo.
Upload it as a file whose only content is the key:

```bash
python - <<'EOF'
import netrc, os
key = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")[2]
fd = os.open("/tmp/wandb_key", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(fd, key.encode()); os.close(fd)
EOF
colab --auth=oauth2 upload -s $SESSION /tmp/wandb_key /content/.wandb_key
shred -u /tmp/wandb_key
```

`vm_launch.py` falls back to `--no-wandb` when the key file is absent, so the
run still works without it — losses and onset metrics also go to
`<root_dir>/logs/<experiment>.txt`.

## Keeping outputs alive

`/content` dies with the VM, sometimes minutes after a run ends. Two options:

- **`drive:` source with `persist_to_drive: true`** — `weights/` and `logs/` are
  symlinked into Drive, so each epoch's checkpoint is durable as it is written.
  Data is still copied to local disk first, because reading 500 MB CSVs through
  the Drive FUSE layer during evaluation is slow.
- **`url:` source** — download each checkpoint from the polling loop as it
  appears (`colab download`); don't wait for the run to finish.

If a VM is lost, set `"resume": true` to continue from the highest epoch
checkpoint present. The configs' own `start_with_weights: false` would silently
restart from zero.

## Other tools here

| Script | Runs on | Purpose |
|---|---|---|
| `watch.py` | locally | keep the session reachable and pull every artifact as it appears |
| `remint.py` | locally | rebuild a pruned session record from a live assignment |
| `apply_cli_patch.py` | locally | give the CLI's keep-alive daemon a proxy-token refresh |
| `compare_checkpoints.py` | the VM | paired comparison of two checkpoints on identical data |
| `wandb_query.py` | locally | query the wandb API using `~/.netrc`, nothing installed |
| `wandb_compare_runs.py` | locally | per-epoch trajectory comparison between runs |
| `score_checkpoints.py` | the VM | absolute scores with intervals, both domains |
| `diagnose_dataset.py` | the VM | per-trial diagnosis of a gap between two checkpoints |

`watch.py`, `remint.py`, `supervise.py` and `apply_cli_patch.py` are vendored here so
the repo stands alone; the `colab-operator` skill carries the same four as its reusable
copies. They were identical when written — if you fix a bug in one, fix it in both.

For which interval to use and why it matters, see the `sssumo-evaluation` skill in
`.agent/skills/` — the unit of analysis changes conclusions on this data, not just
decimal places.

## Cost of a run

Measured 2026-08-17. Stage-1 pretraining (`config-0423-ModGaussian_ampl`):

| | T4 | L4 |
|---|---|---|
| Training | ~0.7 s/step, ~11 min/epoch | ~0.48 s/step, ~4.8 min/epoch |
| 25 epochs | ~4.3 h | ~2 h |

| Fixed costs | |
|---|---|
| Clone + `pip install .` | ~60 s |
| Data staging | 84 s from a Drive folder, ~150–180 s from the public zip |
| Synthetic evaluation | ~6 s/epoch on GPU (minutes on CPU — use `--synthetic-eval-every 0`) |
| Organic evaluation | ~7 s per dataset load, 21 loads per full evaluation |

**Stage-2 fine-tuning is dominated by statistics extraction, not training.** Each
epoch re-runs the model over the training participants of every dataset and
refits the `fastkde` conditional distributions. At `--eval-datapoints 128` one
extraction pass is ~50 s; uncapped it is several times that, and it happens once
at step 0 plus once per epoch.

## Losing the artifact you actually wanted

A completed run's VM is reclaimed within minutes, so the *final* checkpoint is the one
most likely to be lost, and the proxy token expires an hour into a run that may last
four. `docs/COLAB_RELIABILITY.md` has the diagnosis; `watch.py` is the answer, and
replaces the hand-rolled poll loop this study lost two final checkpoints to:

```bash
python scripts/colab/watch.py -s $SESSION --endpoint $ENDPOINT \
    --remote-dir /content/sssumo/weights --pattern "$EXPERIMENT"'_*.pth' \
    --state /content/$EXPERIMENT.state.json --local-dir runs/$EXPERIMENT
```

It re-mints the token from the JWT's own `exp` *before* it expires, polls every 150 s
(under the reclaim window), pulls every file whose local size does not match the
remote's rather than only the newest, and does the final pull the moment the supervisor
records a return code. Add `--config` when the run uses one, and `--stop-on-complete`
to release the VM once the last artifact is verified.

`--endpoint` is worth passing whenever more than one session is live: without it a
recovery binds whatever assignment happens to be up, which is a *different* run's VM
once yours is gone.

To fix the token expiry at the source instead, patch the CLI:

```bash
python scripts/colab/apply_cli_patch.py --apply
```

That gives the keep-alive daemon — which already runs on a timer and knows the
endpoint — a proxy-token refresh, so sessions created afterwards never go unreachable.
It is a local fork of a third-party package, so re-run `apply_cli_patch.py` with no
arguments after any `colab update` or reinstall; it exits non-zero when the patch is
gone. `watch.py` does not depend on it.

For unattended multi-hour work prefer a Kaggle batch kernel over a live session: it
persists `/kaggle/working` on completion, so there is no reclaim window to lose. See
the `kaggle-cli` skill for the GPU pinning and input-discovery rules that path needs.

## When a session "disappears"

`colab exec` returning 404/401 usually does **not** mean the VM is gone — the
CLI caches a proxy token that expires after about an hour, and on failure it
prunes the local record. Check before doing anything destructive:

```bash
colab --auth=oauth2 sessions          # queries the server; orphans show as [?]
```

If the endpoint is still listed, the VM is alive and whatever you launched
detached is still running. Recover the record instead of allocating a new VM:

```bash
<cli-python> scripts/colab/remint.py <session-name> [endpoint]
```

If the endpoint is absent, the VM really was reclaimed; resume from the last
checkpoint with `--resume`.

The usual cause is simply the token clock running out — on an unpatched CLI this
happens to every session at the one-hour mark, daemon and scopes healthy. A missing
`colaboratory` OAuth scope produces the same symptom by a different route: it stops
the keep-alive daemon, and then the VM really *is* reclaimed, mid-run. Tell them apart
by whether `colab sessions` still lists the endpoint. To check the scope:
`colab --auth=oauth2 whoami`; if `https://www.googleapis.com/auth/colaboratory` is not
listed, force a fresh consent flow:

```bash
rm ~/.config/colab-cli/token.json && colab --auth=oauth2 sessions
```

Back up `~/.config/colab-cli/sessions.json` *after* creating a session, not
before — `colab new` prunes stale records, and copying at the wrong moment
overwrites a good backup with an empty one.

## Two things that will bite you

**Name the run so it cannot overwrite a released checkpoint.** Checkpoints are
written as `<experiment_name>_<epoch>.pth`. Reusing a config's own name means the
final epoch overwrites the published weights of that name.

**`eval_datapoints` below ~32 can crash the evaluation.** `utils.py` fills NaNs in
the pooled statistics by sampling from each column's non-NaN values; with few
trials from an undertrained model, a whole `next_*` column can be NaN and
`np.random.choice` raises `ValueError: 'a' cannot be empty`. Use `null` (every
trial) for real runs.

## Before publishing a checkpoint

Training on a mixture that includes `tablet_writing` produces weights covered by
that dataset's **research-only licence** — those must not be released under
CC BY 4.0. This is why `config-0425-tune_ModGauss_wo_writing.yaml` is the
released fine-tuning config.

Stage-1 pretraining is unaffected: it trains purely on synthetic data, and
`tablet_writing` is only ever read during evaluation, where no gradient touches
it.
