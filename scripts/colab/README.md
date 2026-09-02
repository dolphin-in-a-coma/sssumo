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
EXPERIMENT=my-pretraining-run      # must equal experiment_name in run.local.json
ROOT=/content/sssumo_run           # must equal root_dir

# 0. once per machine: make the CLI refresh its own proxy token, so sessions
#    stop going unreachable at the one-hour mark. Any reinstall wipes it.
python scripts/colab/apply_cli_patch.py --apply

# 1. allocate a GPU, then record the endpoint -- watch.py and remint.py both
#    need it, and "the only live assignment" binds the wrong VM once yours dies
colab --auth=oauth2 new -s $SESSION --gpu L4
ENDPOINT=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser(
  '~/.config/colab-cli/sessions.json')))['$SESSION']['endpoint'])")
echo "$SESSION -> $ENDPOINT"

# 2. (only for data_source "drive:") mount Drive -- needs a TTY, so run it yourself
colab --auth=oauth2 drivemount -s $SESSION

# 3. upload the config and the tools the VM needs
colab --auth=oauth2 upload -s $SESSION scripts/colab/run.local.json /content/run.json
colab --auth=oauth2 upload -s $SESSION scripts/train.py            /content/train.py
colab --auth=oauth2 upload -s $SESSION scripts/colab/supervise.py  /content/supervise.py
colab --auth=oauth2 upload -s $SESSION scripts/colab/vm_persist.py /content/vm_persist.py
#    plus the secrets, as files -- see "The wandb key" and "The share token"
#    for an unpushed commit, upload a verified archive instead of cloning:
#      python ~/.agent/skills/remote-compute/scripts/package_git_source.py \
#          --repo . --ref HEAD --output /tmp/sssumo-source.tgz
#      colab ... upload -s $SESSION /tmp/sssumo-source.tgz{,.manifest.json} /content/...
#    then set "source_archive": "/content/sssumo-source.tgz" in run.local.json

# 4. clone the pinned commit, install, stage data (~1 min from Drive, longer from the URL)
colab --auth=oauth2 exec -s $SESSION -f scripts/colab/vm_setup.py --timeout 1800

# 5. smoke test before committing hours of GPU time
#    (temporarily set num_samples 20, num_epochs 1, organic_eval_every 1,
#     eval_datapoints 64 in run.local.json and re-upload)
colab --auth=oauth2 exec -s $SESSION -f scripts/colab/vm_launch.py --timeout 180
colab --auth=oauth2 exec -s $SESSION -f scripts/colab/vm_wait.py   --timeout 1700

# 6. the real run
colab --auth=oauth2 exec -s $SESSION -f scripts/colab/vm_launch.py --timeout 180

# 7. leave the watcher running: it keeps the session reachable, pulls every
#    checkpoint as it lands, and releases the VM once the last one is verified
python scripts/colab/watch.py -s $SESSION --endpoint $ENDPOINT \
    --remote-dir $ROOT/weights --remote-dir $ROOT/logs \
    --pattern "$EXPERIMENT"'_*.pth' --pattern "$EXPERIMENT"'.txt' \
    --state /content/$EXPERIMENT.state.json \
    --local-dir runs/$EXPERIMENT --hours 6 --stop-on-complete

# 8. if you did not use --stop-on-complete, always release the VM yourself
colab --auth=oauth2 stop -s $SESSION
```

For a human-readable snapshot while the watcher runs — epochs done, min/epoch,
projected finish — `colab --auth=oauth2 exec -s $SESSION -f scripts/colab/vm_status.py
--timeout 180`. The kernel serialises cells, but the watcher's calls are short, so
the two interleave fine.

### Running several at once

Give each run its own session file, and pass the **same** `--config` to every command
including `watch.py` — a recovery that rebuilds the record in the default store leaves
the session just as unreachable, silently:

```bash
export CFG=/tmp/colab-$SESSION.json
colab --auth=oauth2 --config $CFG new -s $SESSION --gpu L4
python scripts/colab/watch.py -s $SESSION --endpoint $ENDPOINT --config $CFG ...
```

L4 is capped at **two concurrent**; a third returns `Precondition Failed`. Capacity
blocks (`Service Unavailable` on both L4 and T4) do happen and last tens of minutes —
route to Kaggle rather than retrying, see `docs/COLAB_RELIABILITY.md`.

### What to do when it breaks

| symptom | it is | do |
|---|---|---|
| `exec` 404s, `colab sessions` still lists the endpoint | token expired, VM fine | the watcher re-mints; by hand, `<cli-python> scripts/colab/remint.py $SESSION $ENDPOINT` |
| the endpoint is gone from `colab sessions` | VM reclaimed | new session, `"resume": true` in run.local.json |
| watcher logs `WARNING: re-mint succeeded but ... no fresh token` | `--config` mismatch | pass the run's own `--config` to `watch.py` |

`"resume": true` continues from the highest epoch checkpoint present. The configs'
`start_with_weights: false` would otherwise silently restart from zero.

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

`/content` dies with the VM, sometimes minutes after a run ends. Three options,
in order of how little they leave to chance:

- **`persist:` in `run.local.json`** — `vm_persist.py` runs on the VM alongside
  training and mirrors each artifact to durable storage as it is written, so a
  reclaimed VM costs nothing. Works with any `data_source`. See
  **Mirroring artifacts off the VM** below.
- **`drive:` source with `persist_to_drive: true`** — `weights/` and `logs/` are
  symlinked into Drive, so each epoch's checkpoint is durable as it is written.
  Data is still copied to local disk first, because reading 500 MB CSVs through
  the Drive FUSE layer during evaluation is slow. Needs one interactive
  `colab drivemount`.
- **`watch.py` from the host** — pulls each checkpoint as it appears. Use it
  *as well*: it is also how you get artifacts onto your own machine and how you
  learn the run finished.

## Mirroring artifacts off the VM

`vm_persist.py` is started by `vm_launch.py` whenever `run.local.json` has a
`persist` section, detached and separate from training so a network failure
cannot touch the run. Each cycle it uploads every matching file whose size has
changed since the last upload, reads the stored size back, and records what
succeeded — so a restart does not re-send the whole run, and a truncated upload
is never mistaken for a good one.

Two backends, chosen by `persist.backend`:

- **`webdav`** — any Nextcloud/ownCloud public share, including the ones this
  project already uses for data. The share token is the WebDAV username with an
  empty password.
- **`wandb`** — one versioned `wandb.Artifact` per file, under a
  `<experiment>-artifacts` run. Reuses the key already uploaded for logging.
- **`both`** — belt and braces.

### The share token

Never put it in `run.local.json`, in a command argument, or in the repo — it is a
credential, exactly like the wandb key, and it is sent as HTTP basic auth so it
never appears in a URL, a redirect or a traceback.

```bash
python - <<'EOF'
import os
key = "PASTE_THE_SHARE_TOKEN"          # the /s/<token> part of the share URL
fd = os.open("/tmp/persist_token", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(fd, key.encode()); os.close(fd)
EOF
colab --auth=oauth2 upload -s $SESSION /tmp/persist_token /content/.persist_token
shred -u /tmp/persist_token
```

**Prefer a file-drop (upload-only) share.** A Nextcloud share link is a bearer
capability: whoever holds it gets whatever the share grants. A read/write share
(`sharePermissions` 31 = read + update + create + delete + share) means a leaked
token can also read and delete everything already uploaded. A file drop grants
create only, which is all the mirror needs.

Check what a share grants before trusting it to a rented VM:

```bash
curl -s "https://<host>/index.php/s/<token>" | grep -o 'sharePermissions" value="[^"]*"'
```

The value is base64; `31` is full access, `4` is create-only.

If a VM is lost, set `"resume": true` to continue from the highest epoch
checkpoint present. The configs' own `start_with_weights: false` would silently
restart from zero.

## Other tools here

| Script | Runs on | Purpose |
|---|---|---|
| `watch.py` | locally | keep the session reachable and pull every artifact as it appears |
| `vm_persist.py` | the VM | mirror artifacts to durable storage as they are written |
| `remint.py` | locally | rebuild a pruned session record from a live assignment |
| `apply_cli_patch.py` | locally | give the CLI's keep-alive daemon a proxy-token refresh |
| `compare_checkpoints.py` | the VM | paired comparison of two checkpoints on identical data |
| `wandb_query.py` | locally | query the wandb API using `~/.netrc`, nothing installed |
| `wandb_compare_runs.py` | locally | per-epoch trajectory comparison between runs |
| `score_checkpoints.py` | the VM | absolute scores with intervals, both domains |
| `diagnose_dataset.py` | the VM | per-trial diagnosis of a gap between two checkpoints |

`watch.py`, `remint.py`, `supervise.py`, `vm_persist.py` and `apply_cli_patch.py` are vendored here so
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
