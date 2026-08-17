"""Shared helpers for the VM-side tools.

`colab exec -f script.py` passes no argv, so every tool reads its parameters
from a JSON config uploaded to RUN_JSON instead of taking flags.
"""
import json
import os

RUN_JSON = "/content/run.json"


def load():
    with open(RUN_JSON) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def paths(cfg):
    """State and log files for the supervised run, namespaced by experiment."""
    name = cfg["experiment_name"]
    return f"/content/{name}.state.json", f"/content/{name}.log"


def alive(pid):
    """Liveness from the third field of /proc/<pid>/stat.

    A zombie keeps its /proc entry, so directory existence alone is not enough.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def checkpoints(cfg):
    weights = os.path.join(cfg["root_dir"], "weights")
    if not os.path.isdir(weights):
        return []
    return sorted(f for f in os.listdir(weights) if f.startswith(cfg["experiment_name"]))
