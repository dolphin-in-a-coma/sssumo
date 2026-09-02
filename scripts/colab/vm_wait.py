"""Block on the VM until the supervised run finishes, then print the tail.

    colab --auth=oauth2 exec -s <session> -f scripts/colab/vm_wait.py --timeout 1700

Keep the exec timeout above max_wait_seconds. Training continues on the VM even
if the client times out first.
"""
import json
import os
import sys
import time

# `colab exec -f` sends this file's *text* to the kernel and sets no __file__,
# so the import path has to be the VM location vm_common.py is uploaded to,
# not this script's directory.
sys.path.insert(0, "/content")
from vm_common import alive, load, paths  # noqa: E402

cfg = load()
state_path, log_path = paths(cfg)
max_wait = cfg.get("max_wait_seconds", 1500)

t0, last = time.time(), 0
while time.time() - t0 < max_wait:
    state = json.load(open(state_path))
    if state["returncode"] is not None:
        break
    if not alive(state["supervisor_pid"]):
        print("supervisor died without recording a return code", flush=True)
        break
    size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    if size != last:
        print(f"[{time.time()-t0:.0f}s] log {size} bytes", flush=True)
        last = size
    time.sleep(15)

state = json.load(open(state_path))
elapsed = (state.get("completed_at") or time.time()) - state["started_at"]
print(f"\nreturncode={state['returncode']} elapsed={elapsed/60:.1f} min", flush=True)
with open(log_path, errors="replace") as f:
    print("--- log tail ---\n" + f.read()[-6000:], flush=True)
