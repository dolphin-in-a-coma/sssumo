# Why Colab runs keep losing sessions, and what to do about it

Diagnosis from the 2026-09-01 pulse-family study, which lost two of four runs' final
checkpoints and spent ~90 minutes on infrastructure. Written to be actionable: the root
cause is narrower than it looks and is fixable in the CLI.

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

## The plan

### A — host-side watcher (`scripts/colab/watch.py`)

Does not exist yet; was built ad hoc during the study. Requirements:

1. **Proactive re-mint** driven by the JWT `exp` — refresh at ~80% of lifetime instead of
   recovering after a 404. Reuses `scripts/colab/remint.py`.
2. **Poll under the reclaim window** (2–3 min, not 5+).
3. **Pull every artifact not already held**, not just the newest — a cycle spanning two
   epochs otherwise skips one permanently.
4. **Immediate final pull** the moment the state file shows a return code.
5. Catch `subprocess.TimeoutExpired`; never silence a failed `colab download` (stderr to
   `/dev/null` plus `&&` turns a lost artifact into "nothing new").

No credentials leave the machine. Fixes the token problem *and* the artifact race for any
caller of this repo.

### B — fix the refresh in the CLI

The real fix for the root cause: refresh the proxy token when `exp` is near, either in
`runtime.py` before a request or by having the keep-alive daemon own it (it already runs
on a timer per session and knows the endpoint). Removes re-minting from every caller.
Ships as a local patch or an upstream PR to `google-colab-cli`; a local fork needs a
re-apply check after `colab update`.

Working re-mint, for reference — note `state.config_path` **must** be set before
`state.store` is first touched, or the store binds the default path:

```python
# run with the CLI's own interpreter:
#   /Users/evgeruda/.local/share/uv/tools/google-colab-cli/bin/python
from colab_cli.common import state
from colab_cli.state import SessionState
state.config_path = "<same --config the run uses>"
a = next(a for a in state.client.list_assignments() if a.endpoint == TARGET)
pi = a.runtime_proxy_info                      # fresh token + url
state.store.add(SessionState(name=NAME, token=pi.token, url=pi.url, endpoint=TARGET,
                             variant=a.variant.name, accelerator=a.accelerator.name,
                             kernel_id=None))  # None -> new kernel, same VM disk
```

Always bind the endpoint recorded at creation. "The only live assignment" silently
re-points a session name at a *different* run's VM once the intended one is gone.

### C — push artifacts from inside the VM

Eliminates the reclaim race at the source rather than racing it, but needs a write
credential on a rented VM. Constraints found:

- `colab drivemount` and `colab auth` require a TTY and hang under an agent — not usable.
- The OAuth token does carry a `drive.file` scope, so a Drive-API writer is *possible*,
  but it means putting user credentials on rented hardware.
- Prefer a narrowly-scoped, revocable, **write-only** destination (e.g. a single GCS
  bucket via service account) over anything reusable.

Only worth it for runs long enough that a 2–3 minute exposure window is unacceptable.

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
