"""Rebuild a local session record from a live Colab assignment.

Run with the colab CLI's own interpreter (the venv that has colab_cli installed;
`head -1 $(which colab)` prints it):
    <cli-python> scripts/colab/remint.py <session-name> [endpoint]

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

name = sys.argv[1]
endpoint = sys.argv[2] if len(sys.argv) > 2 else None

assignments = list(state.client.list_assignments())
if not assignments:
    raise SystemExit("no live assignments -- the VM is gone, not just the token")

if endpoint:
    match = next((a for a in assignments if a.endpoint == endpoint), None)
elif len(assignments) == 1:
    match = assignments[0]
else:
    raise SystemExit("several assignments live; pass the endpoint explicitly:\n  "
                     + "\n  ".join(a.endpoint for a in assignments))

if match is None:
    raise SystemExit(f"endpoint {endpoint} is not live -- the VM is gone")

pi = match.runtime_proxy_info
state.store.add(SessionState(name=name, token=pi.token, url=pi.url,
                             endpoint=match.endpoint, variant=match.variant.name,
                             accelerator=match.accelerator.name, kernel_id=None))
print(f"re-minted '{name}' -> {match.endpoint} ({match.accelerator.name})")
