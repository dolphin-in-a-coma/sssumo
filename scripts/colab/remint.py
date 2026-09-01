"""Rebuild a local session record from a live Colab assignment.

Run with the colab CLI's own interpreter (the venv that has colab_cli installed;
`head -1 $(which colab)` prints it):
    <cli-python> scripts/colab/remint.py <session-name> [endpoint] [--config PATH]

Pass --config with the same path you pass to `colab --config`. Parallel runs each
keep their own session file, and without this the record is rebuilt in the default
store where those invocations will never look for it.

The CLI caches a proxy token that expires (~1 h). When it does, every kernel
request 404s and the CLI prunes the local record -- but the VM is usually still
alive and still running whatever was detached on it. This mints a fresh token
from the live assignment and re-adds the record, rather than losing the VM.

kernel_id=None means the next exec starts a new kernel on the same VM disk;
detached processes are unaffected.
"""
import sys

from colab_cli.common import state
from colab_cli.state import SessionState

argv = sys.argv[1:]
config_path = None
if "--config" in argv:
    i = argv.index("--config")
    config_path = argv[i + 1]
    del argv[i:i + 2]
# must be set before state.store is first touched, or the store binds the default
state.config_path = config_path

name = argv[0]
endpoint = argv[1] if len(argv) > 1 and argv[1] != "--any" else None
allow_any = "--any" in argv[1:]

assignments = list(state.client.list_assignments())
if not assignments:
    raise SystemExit("no live assignments -- the VM is gone, not just the token")

if endpoint:
    match = next((a for a in assignments if a.endpoint == endpoint), None)
elif allow_any and len(assignments) == 1:
    match = assignments[0]
else:
    # Binding whatever happens to be live is how a session name silently gets
    # re-pointed at a DIFFERENT run's VM: if the one you wanted was reclaimed and
    # another is still up, "the only live assignment" is the wrong machine.
    raise SystemExit(
        "pass the endpoint explicitly (or --any if you are certain there is only\n"
        "one run and it is the one you want). Live assignments:\n  "
        + "\n  ".join(f"{a.endpoint}  ({a.accelerator.name})" for a in assignments))

if match is None:
    raise SystemExit(f"endpoint {endpoint} is not live -- the VM is gone")

pi = match.runtime_proxy_info
state.store.add(SessionState(name=name, token=pi.token, url=pi.url,
                             endpoint=match.endpoint, variant=match.variant.name,
                             accelerator=match.accelerator.name, kernel_id=None))
print(f"re-minted '{name}' -> {match.endpoint} ({match.accelerator.name})"
      + (f" in {config_path}" if config_path else ""))
