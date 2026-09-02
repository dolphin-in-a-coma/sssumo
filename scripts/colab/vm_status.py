"""Report a supervised run's liveness, pace and checkpoints.

    colab --auth=oauth2 exec -s <session> -f scripts/colab/vm_status.py --timeout 180

A zero return code here is a transport signal only -- validate the downloaded
checkpoints before trusting the run.
"""
import json
import os
import re
import sys
import time

# `colab exec -f` sends this file's *text* to the kernel and sets no __file__,
# so the import path has to be the VM location vm_common.py is uploaded to,
# not this script's directory.
sys.path.insert(0, "/content")
from vm_common import alive, checkpoints, load, paths  # noqa: E402

cfg = load()
state_path, log_path = paths(cfg)

state = json.load(open(state_path))
live = alive(state["supervisor_pid"])
elapsed = (state.get("completed_at") or time.time()) - state["started_at"]
print(f"experiment: {cfg['experiment_name']}", flush=True)
print(f"supervisor_live={live} returncode={state['returncode']} "
      f"elapsed={elapsed/60:.1f} min", flush=True)

ck = checkpoints(cfg)
print(f"checkpoints ({len(ck)}): {ck}", flush=True)

log = open(log_path, errors="replace").read() if os.path.exists(log_path) else ""
done = [l for l in log.splitlines() if l.startswith("Epoch ") and "finished" in l]
clock = re.findall(r"^Time: ([\d.]+)$", log, re.M)
if done and clock:
    per = float(clock[-1]) / len(done)
    total = cfg.get("num_epochs") or 25
    print(f"{len(done)}/{total} epochs, ~{per/60:.1f} min/epoch, "
          f"~{per*total/3600:.1f} h projected", flush=True)

iters = [l for l in log.splitlines() if re.match(r"Epoch \d+, Iteration", l)]
print("last:", iters[-1] if iters else "no iterations logged yet", flush=True)

if state["returncode"] not in (None, 0):
    print("\n--- tail ---\n" + log[-3000:], flush=True)
