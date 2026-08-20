"""Run a command detached, streaming to a log, recording its return code atomically."""
import argparse, json, os, subprocess, sys, tempfile, time


def atomic_write(path, obj):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd

state = {"supervisor_pid": os.getpid(), "child_pid": None, "cmd": cmd,
         "started_at": time.time(), "returncode": None, "completed_at": None}
atomic_write(a.state, state)

env = dict(os.environ, PYTHONUNBUFFERED="1")
with open(a.log, "wb", buffering=0) as logf:
    child = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    state["child_pid"] = child.pid
    atomic_write(a.state, state)
    rc = child.wait()

state["returncode"] = rc
state["completed_at"] = time.time()
atomic_write(a.state, state)
