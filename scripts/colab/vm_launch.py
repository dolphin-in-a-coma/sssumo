"""Launch training detached, so short polling calls can watch it.

The kernel runs one cell at a time: a synchronous exec would block every
later status call for the whole run.

    colab --auth=oauth2 exec -s <session> -f scripts/colab/vm_launch.py --timeout 180
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vm_common import load, paths  # noqa: E402

cfg = load()
state, log = paths(cfg)
for p in (state, log):
    if os.path.exists(p):
        os.remove(p)

cmd = [
    sys.executable, "-u", "/content/train.py",
    "--config", cfg["config"],
    "--root-dir", cfg["root_dir"],
    "--experiment-name", cfg["experiment_name"],
    "--organic-eval-every", str(cfg["organic_eval_every"]),
    "--wandb-project", cfg["wandb_project"],
]
for key, flag in (("num_samples", "--num-samples"),
                  ("num_epochs", "--num-epochs"),
                  ("eval_datapoints", "--eval-datapoints")):
    if cfg.get(key) is not None:
        cmd += [flag, str(cfg[key])]
if cfg.get("resume"):
    cmd.append("--resume")
if cfg.get("wandb_key_file") and os.path.exists(cfg["wandb_key_file"]):
    cmd += ["--wandb-key-file", cfg["wandb_key_file"]]
else:
    cmd.append("--no-wandb")

print("cmd:", " ".join(cmd), flush=True)

p = subprocess.Popen(
    [sys.executable, "/content/supervise.py", "--state", state, "--log", log, "--"] + cmd,
    start_new_session=True,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("supervisor pid", p.pid, flush=True)
print("state:", state, "\nlog:", log, flush=True)

# Mirror artifacts to durable storage as they are written, so a reclaimed VM
# costs nothing. Detached and separate from training on purpose: a network
# failure here must not touch the run.
if (cfg.get("persist") or {}).get("backend"):
    mirror_log = f"/content/{cfg['experiment_name']}.persist.log"
    m = subprocess.Popen(
        [sys.executable, "-u", "/content/vm_persist.py"],
        start_new_session=True,
        stdout=open(mirror_log, "wb", buffering=0), stderr=subprocess.STDOUT,
    )
    print("mirror pid", m.pid, "->", mirror_log, flush=True)
else:
    print("no persist section: artifacts exist only on this VM until pulled",
          flush=True)
