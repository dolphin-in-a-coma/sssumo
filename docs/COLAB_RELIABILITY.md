# Why Colab runs keep losing sessions, and what to do about it

Diagnosis from the 2026-09-01 pulse-family study, which lost two of four runs' final
checkpoints and spent ~90 minutes on infrastructure. The root cause turned out to be
narrower than it looked, and fixable in the CLI itself. Both fixes have since shipped —
see **What shipped** below; the diagnosis is kept because the symptoms recur on any
machine where the patch is not applied.

## Root cause: the CLI never refreshes its proxy token

`SessionState.token` (`colab_cli/state.py`) is written once by `colab new` and **never
updated**. `colab_cli/runtime.py:36` stores it and every kernel request sends that cached
value (`:100`, `:108`, `:113` — `colab-runtime-proxy-token` /
`X-Colab-Runtime-Proxy-Token`). The keep-alive daemon calls
`client.keep_alive_assignment(endpoint)` (`colab_cli/client.py:286`), which keeps the *VM
assignment* alive and touches nothing else.

So a session runs on two independent clocks:

| | mechanism | lifetime |
|---|---|---|
| VM assignment | keep-alive daemon, refreshed | hours |
| Proxy token | cached at creation, **never refreshed** | ~1 h |

At ~60 minutes every kernel call 404s, the CLI auto-prunes the local record, and a
perfectly healthy VM reports "Session not found". In the study this hit all four sessions
at once.

**This is not the missing-scope failure the skill describes.** Verified during the
incident: the keep-alive daemon was running, and `colab whoami` showed the
`colaboratory` scope present. Check both before reaching for the scope explanation.

The token is a plain JWT — decode the payload for `exp` (and `aud`, which is the
endpoint), so remaining lifetime is readable without an API call:

```python
import base64, json, time
body = token.split(".")[1]; body += "=" * (-len(body) % 4)
claims = json.loads(base64.urlsafe_b64decode(body))
seconds_left = claims["exp"] - time.time()      # aud == endpoint
```

## What is already correct — do not change it

The **detached supervisor** (`scripts/colab/supervise.py`) is sound and proved itself:
training runs in a `start_new_session` process, so token expiry, kernel disconnects and
session pruning leave it untouched. During the incident one run advanced from epoch 15 to
19 *while completely unreachable*. Any fix belongs in the host-side watcher, not here.

## Second problem: artifact loss is a race, not a capability gap

`/content` dies with the VM, and a **completed** run is the first thing reclaimed — so
the final checkpoint, the one you actually want, is the likeliest casualty. A
completion-triggered download cannot win this race. Measured: a 5-minute poll on runs at
~3 min/epoch lost epochs 23 and 24 of a 25-epoch study.

## Third problem: capacity blocks are sustained

`Service Unavailable` on assign for **both** L4 and T4 across 8 attempts over ~15 minutes,
while an already-running VM continued fine. Not auth, not quota, not detectable in
advance. Only routable around.

## What shipped

### A — host-side watcher: `scripts/colab/watch.py`

Replaces the ad-hoc poll loop (and the old `checkpoint_puller.py`, now deleted).

- **Proactive re-mint** from the token's own `exp`, at `--refresh-at` of its life
  (default 0.8), rather than recovering after the first 404. The lifetime is *observed*
  at mint time — the JWT carries no `iat`, so it cannot be derived from the token.
- **Polls every 150 s** by default, under the reclaim window.
- **Pulls every file whose local size differs from the remote's**, across any number of
  `--remote-dir`/`--pattern` pairs. Size matching also rejects a checkpoint caught
  mid-write: the partial is deleted, not counted as held.
- **Final pull on the return code**, in the same cycle that detects it, then one more
  listing pass; anything still missing is named in the log.
- A failed download prints the size it got, the size it wanted, and the transport's own
  error. `subprocess.TimeoutExpired` is caught and retried next cycle.
- A re-mint that reports success without moving the token in the store we read is called
  out as a probable `--config` mismatch — the failure that made recovery a silent no-op
  for parallel runs.

Distinguishes the three failures that look alike: *token stale* (re-mint), *kernel busy*
(retry next cycle — a good re-mint plus a failed listing means a live VM), and
*assignment gone* (stop). Exercised against a stubbed CLI over all six paths.

No credential leaves the machine.

### B — the refresh in the CLI: `scripts/colab/apply_cli_patch.py`

`patches/0001-refresh-proxy-token.patch` gives the keep-alive daemon the refresh, since
it already runs on a 60 s timer per session and knows the endpoint. Two files:

- `state.py` gains `StateStore.update(name, **fields)` — read-modify-write under the
  existing exclusive lock. `add()` replaces the whole record, so refreshing through it
  would clobber a `kernel_id` an in-flight `exec` had just written.
- `commands/session.py` refreshes when the token has under 15 minutes left, *before* the
  keep-alive call: if keep-alive has started failing, a usable token is exactly what you
  need to rescue artifacts in the time the VM has left. An unreadable token counts as
  expired. A vanished assignment returns quietly — the keep-alive call is what reports a
  lost VM.

`apply_cli_patch.py` verifies sha256 against the version the patch was made for (0.6.0),
backs the originals up inside the package, applies, re-checks the hashes, clears the
bytecode, and then runs `patches/test_refresh.py` against a stubbed client — so
"applied" means the refresh *behaves*, not just that the bytes landed. Any failure
restores the originals. `--revert` undoes it; no arguments reports status and exits
non-zero when the patch is absent, which is the check to run after a `colab update`.

**The patch only covers sessions created after it is applied** — a running session keeps
the daemon it was started with. This is why A is not redundant with B.

### C — VM-side push: not done, deliberately

It would need a write credential on rented hardware, and A+B already cut the exposure to
one ~150 s poll interval with the final pull triggered by the return code rather than by
a timer. The credential-free ways to get the same guarantee are already available and
are the better answer here:

- **Drive** — `drive:` source with `persist_to_drive: true` symlinks `weights/` and
  `logs/` into Drive, so each checkpoint is durable as it is written. Needs one
  interactive `colab drivemount` (it wants a TTY and hangs under an agent).
- **Kaggle** — a batch kernel persists `/kaggle/working` on completion; there is no
  reclaim window to lose.

Revisit only if a run appears whose artifacts cannot tolerate a 150 s window, and then
with a narrowly-scoped, revocable, write-only destination — never a reusable credential.

### The one rule that survives all three

Always bind recovery to **the endpoint recorded at creation**. "The only live assignment"
silently re-points a session name at a *different* run's VM the moment the intended one
is gone, and everything downstream — the listing, the pulls, the epoch numbers — then
describes the wrong machine.

## Structural limits to plan around

- L4 capped at **2 concurrent**; a third returns `Precondition Failed`.
- T4s reclaimed at ~1–1.5 h mid-run; L4s survive multi-hour jobs. Checkpoint every epoch
  and always pass `--resume` so a reclaim costs one epoch.
- `colab exec -f` passes **no argv** — parameterise VM scripts with a JSON config
  uploaded to a fixed path.
- Isolate parallel runs with `--config <path>`; every recovery call must pass the same one.

## When Colab is the wrong tool

For unattended multi-hour work, a Kaggle batch kernel persists `/kaggle/working` on
completion and has no reclaim window to lose. In this study the arms that finished intact
were the two that ran on Kaggle. See the `kaggle-cli` skill for the `machine_shape` /
P100 trap and the input-discovery rules that path needs.
