"""Pull checkpoints off an ephemeral Colab VM as they are written.

/content dies with the VM, so a long run without a Drive mount can lose
everything. This polls the session and downloads each new checkpoint, re-minting
the session token when it expires -- the VM normally outlives the token.

    python scripts/colab/checkpoint_puller.py SESSION PREFIX REMOTE_DIR LOCAL_DIR \
        [--final-suffix _9.pth] [--interval 240] [--hours 3.5]
"""
import argparse
import os
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("prefix", help="checkpoint filename prefix, e.g. my-run_")
ap.add_argument("remote_dir")
ap.add_argument("local_dir")
ap.add_argument("--final-suffix", default=None,
                help="stop once a checkpoint with this suffix is secured, e.g. _9.pth")
ap.add_argument("--interval", type=int, default=240)
ap.add_argument("--hours", type=float, default=3.5)
ap.add_argument("--endpoint", default=None,
                help="the session's endpoint; passed to remint so recovery cannot\n bind a different run's VM. Strongly recommended when several runs are live")
ap.add_argument("--cli-python", default=None,
                help="interpreter that has colab_cli installed; "
                     "defaults to reading the shebang of `which colab`")
a = ap.parse_args()

SESSION, PREFIX = a.session, a.prefix
REMOTE_DIR, LOCAL = a.remote_dir, os.path.expanduser(a.local_dir)
HERE = os.path.dirname(os.path.abspath(__file__))
INTERVAL = a.interval
DEADLINE = time.time() + a.hours * 3600

CLI_PY = a.cli_python
if CLI_PY is None:
    which = subprocess.run("command -v colab", shell=True, capture_output=True, text=True)
    with open(os.path.realpath(which.stdout.strip())) as f:
        CLI_PY = f.readline().lstrip("#!").strip()

os.makedirs(LOCAL, exist_ok=True)

LIST_SCRIPT = f"""
import os
print("CKPTS=" + ",".join(sorted(
    f for f in os.listdir({REMOTE_DIR!r}) if f.startswith({PREFIX!r}))))
"""
list_path = os.path.join(HERE, f"_list_{SESSION}.py")
with open(list_path, "w") as f:
    f.write(LIST_SCRIPT)


TRANSIENT = object()   # distinct from None ("VM gone")


class _Failed:
    """Stand-in for a call that never returned; keeps the poll loop alive."""
    returncode = 1
    stdout = stderr = ""


def sh(cmd, timeout=420):
    # A busy kernel can push `colab exec` past the wall clock. Letting
    # TimeoutExpired propagate kills the puller and silently stops protecting
    # the run, which is the opposite of this script's job.
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("  call timed out; will retry next cycle", flush=True)
        return _Failed()


def remint():
    target = f"{SESSION} {a.endpoint}" if a.endpoint else f"{SESSION} --any"
    r = sh(f"{CLI_PY} {HERE}/remint.py {target}")
    print(f"  remint: {(r.stdout or r.stderr).strip()[:90]}", flush=True)
    return r.returncode == 0


def remote_checkpoints():
    for attempt in (1, 2):
        r = sh(f"colab --auth=oauth2 exec -s {SESSION} -f {list_path} --timeout 300")
        for line in (r.stdout or "").splitlines():
            if line.startswith("CKPTS="):
                names = line[len("CKPTS="):].strip()
                return [n for n in names.split(",") if n]
        if attempt == 1:
            print("  list failed; re-minting", flush=True)
            if not remint():
                return None          # re-mint says the VM is gone
    # Re-mint succeeded but the listing still failed: the VM is alive and the
    # kernel is merely slow or busy. Treating this as "VM gone" is how the puller
    # used to abandon a live run.
    print("  listing still failing after a good re-mint; retrying next cycle", flush=True)
    return TRANSIENT


pulled = {f for f in os.listdir(LOCAL) if f.startswith(PREFIX)}
print(f"already local: {sorted(pulled)}", flush=True)

while time.time() < DEADLINE:
    remote = remote_checkpoints()
    if remote is None:
        print("VM is not in the live assignment list -- stopping", flush=True)
        break
    if remote is TRANSIENT:
        time.sleep(INTERVAL)
        continue
    new = [f for f in remote if f not in pulled]
    for name in sorted(new):
        r = sh(f"colab --auth=oauth2 download -s {SESSION} "
               f"{REMOTE_DIR}/{name} {LOCAL}/{name}")
        ok = os.path.exists(f"{LOCAL}/{name}") and os.path.getsize(f"{LOCAL}/{name}") > 1e6
        print(f"  pulled {name}: {ok}", flush=True)
        if ok:
            pulled.add(name)
    print(f"[{time.strftime('%H:%M:%S')}] remote={len(remote)} local={len(pulled)}", flush=True)
    if a.final_suffix and any(f.endswith(a.final_suffix) for f in pulled):
        print("final checkpoint secured", flush=True)
        break
    time.sleep(INTERVAL)

print("puller done", flush=True)
