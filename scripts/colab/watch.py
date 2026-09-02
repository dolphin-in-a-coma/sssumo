"""Watch a supervised Colab run: keep the session reachable, pull every artifact.

    python scripts/colab/watch.py -s SESSION --endpoint EP \
        --remote-dir /content/sssumo/weights --pattern 'my-run_*.pth' \
        --state /content/my-run.state.json --local-dir runs/my-run

Two things kill an unattended Colab run's results, and this handles both.

**The proxy token expires (~1 h) while the VM lives on.** The CLI caches it at
`colab new` and never refreshes it, so every kernel call starts 404ing and the
CLI prunes the local record -- on a perfectly healthy VM. This re-mints
*proactively*, from the JWT's own `exp`, at `--refresh-at` of the token's life,
instead of recovering after the first failure. (`scripts/colab/apply_cli_patch.py`
fixes this in the CLI itself; this stays correct either way, and is what you want
on a machine where the patch is not applied.)

**A finished run's VM is reclaimed within minutes**, so the last checkpoint is
the likeliest one to lose. Hence: poll under that window, pull *every* file not
already held rather than the newest, verify each against its remote size, and
pull immediately on seeing the supervisor's return code rather than on the next
cycle.

Nothing here puts a credential on the VM.
"""
import argparse
import base64
import fnmatch  # noqa: F401  (used inside the generated lister)
import json
import os
import subprocess
import sys
import tempfile
import time

NOMINAL_TOKEN_LIFETIME = 3600.0   # what the API reports as tokenExpiresInSeconds
TRANSIENT = object()              # a cycle failed; the VM is still there


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--session", required=True)
    ap.add_argument("--endpoint", help="the session's endpoint, from `colab sessions`. "
                    "Without it a re-mint falls back to --any, which binds whatever "
                    "assignment happens to be live -- a different run's VM, once yours "
                    "is gone. Pass it whenever more than one session exists")
    ap.add_argument("--config", help="the same --config the run uses; parallel runs each "
                    "keep their own session file")
    ap.add_argument("--remote-dir", action="append", required=True, metavar="DIR",
                    help="VM directory to watch; repeatable")
    ap.add_argument("--pattern", action="append", metavar="GLOB",
                    help="filename glob to pull; repeatable, default '*'")
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--state", help="supervise.py state file on the VM. Its returncode "
                    "is what triggers the final pull; without it the watch simply runs "
                    "to --hours")
    ap.add_argument("--interval", type=float, default=150,
                    help="seconds between cycles (default 150 -- under the reclaim window)")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--refresh-at", type=float, default=0.8, metavar="FRAC",
                    help="re-mint once this fraction of the token's life is spent")
    ap.add_argument("--exec-timeout", type=float, default=300)
    ap.add_argument("--stop-on-complete", action="store_true",
                    help="release the VM once the final pull is verified")
    ap.add_argument("--colab-bin", default="colab")
    ap.add_argument("--cli-python", help="interpreter that has colab_cli installed; "
                    "defaults to the shebang of `which colab`")
    a = ap.parse_args(argv)
    a.pattern = a.pattern or ["*"]
    if not 0 < a.refresh_at < 1:
        ap.error("--refresh-at must be between 0 and 1")
    return a


class _Failed:
    """A call that never came back. Keeps the loop alive instead of killing it."""
    returncode = 1
    stdout = stderr = ""


def sh(cmd, timeout):
    # A busy kernel routinely pushes `colab exec` past its wall clock. Letting
    # TimeoutExpired propagate kills the watcher, which silently ends the only
    # thing protecting the run.
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log("call timed out; retrying next cycle")
        return _Failed()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cli_python(explicit):
    if explicit:
        return explicit
    which = subprocess.run(["/bin/sh", "-c", "command -v colab"],
                           capture_output=True, text=True)
    path = which.stdout.strip()
    if not path:
        raise SystemExit("cannot find `colab` on PATH; pass --cli-python")
    with open(os.path.realpath(path)) as f:
        return f.readline().lstrip("#!").strip()


# --- the token -------------------------------------------------------------

def store_path(config):
    return config or os.path.expanduser("~/.config/colab-cli/sessions.json")


def token_seconds_left(config, session):
    """Remaining life of the cached proxy token, or None if there is no record.

    The token is a plain JWT: `exp` is readable without an API call, so the
    refresh decision costs nothing. It carries no `iat`, which is why the
    lifetime has to be observed at mint time rather than derived here.
    """
    try:
        with open(store_path(config)) as f:
            record = json.load(f).get(session)
    except (OSError, ValueError):
        return None
    if not record or not record.get("token"):
        return None
    body = record["token"].split(".")[1]
    body += "=" * (-len(body) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(body))
    except (ValueError, KeyError):
        return None
    return claims["exp"] - time.time()


class Session:
    """Keeps one session's local record fresh enough to be usable."""

    def __init__(self, args):
        self.a = args
        self.py = cli_python(args.cli_python)
        self.here = os.path.dirname(os.path.abspath(__file__))
        self.lifetime = NOMINAL_TOKEN_LIFETIME

    @property
    def base(self):
        cmd = [self.a.colab_bin, "--auth=oauth2"]
        if self.a.config:
            cmd += ["--config", self.a.config]
        return cmd

    def remint(self):
        cmd = [self.py, os.path.join(self.here, "remint.py"), self.a.session,
               self.a.endpoint or "--any"]
        if self.a.config:
            cmd += ["--config", self.a.config]
        r = sh(cmd, timeout=180)
        ok = r.returncode == 0
        said = (r.stdout + "\n" + r.stderr).strip().splitlines()
        log(f"  re-mint: {said[-1][:110]}" if said else "  re-mint: no output")
        if ok:
            # A token we just minted is by definition a full lifetime old, so
            # this is where the real lifetime becomes observable.
            left = token_seconds_left(self.a.config, self.a.session)
            if left is None or left < 60:
                # remint reported success but the record we read did not move:
                # it wrote to a different store. This is the --config mismatch
                # that makes recovery silently do nothing for parallel runs.
                log(f"  WARNING: re-mint succeeded but {store_path(self.a.config)} "
                    f"still has no fresh token for '{self.a.session}' -- check that "
                    f"--config matches the one the run uses")
            elif left > 60:
                self.lifetime = left
        return ok

    def ensure(self):
        """True if the session should be reachable; False if the VM is gone."""
        left = token_seconds_left(self.a.config, self.a.session)
        margin = (1 - self.a.refresh_at) * self.lifetime
        if left is None:
            log("  no local record (pruned) -- re-minting")
        elif left < margin:
            log(f"  token has {left/60:.1f} min left of {self.lifetime/60:.0f} "
                f"-- re-minting")
        else:
            return True
        return self.remint()


# --- the VM side -----------------------------------------------------------

LISTER = '''\
import fnmatch, json, os
files = []
for d in {dirs!r}:
    if not os.path.isdir(d):
        continue
    for name in sorted(os.listdir(d)):
        if any(fnmatch.fnmatch(name, p) for p in {pats!r}):
            path = os.path.join(d, name)
            try:
                files.append([path, os.path.getsize(path)])
            except OSError:
                pass
returncode, live = None, None
state_path = {state!r}
if state_path and os.path.exists(state_path):
    try:
        st = json.load(open(state_path))
        returncode = st.get("returncode")
        pid = st.get("supervisor_pid")
        if pid:
            try:
                # A zombie keeps its /proc entry, so the directory alone lies.
                with open("/proc/%d/stat" % pid) as f:
                    live = f.read().rsplit(")", 1)[1].split()[0] != "Z"
            except FileNotFoundError:
                live = False
    except ValueError:
        pass          # a torn read of the state file; next cycle sees it whole
print("WATCH=" + json.dumps(
    {{"files": files, "returncode": returncode, "supervisor_live": live}}))
'''


def probe(session, lister_path):
    """One `colab exec` for the file listing and the run's state.

    Returns the parsed dict, TRANSIENT when the call failed but the VM is alive,
    or None when the VM is gone.
    """
    a = session.a
    for attempt in (1, 2):
        r = sh(session.base + ["exec", "-s", a.session, "-f", lister_path,
                               "--timeout", str(int(a.exec_timeout))],
               timeout=a.exec_timeout + 120)
        for line in (r.stdout or "").splitlines():
            if line.startswith("WATCH="):
                return json.loads(line[len("WATCH="):])
        if attempt == 1:
            log("  listing failed; re-minting and retrying")
            if not session.remint():
                return None                      # the assignment is gone
    # A good re-mint plus a failed listing means a live VM with a busy or slow
    # kernel. Calling that "gone" is how a watcher abandons a healthy run.
    log("  listing still failing after a good re-mint; retrying next cycle")
    return TRANSIENT


def pull(session, files, held):
    """Download every listed file whose local copy does not match its remote size."""
    a = session.a
    for remote, size in files:
        name = os.path.basename(remote)
        local = os.path.join(a.local_dir, name)
        if held.get(name) == size:
            continue
        r = sh(session.base + ["download", "-s", a.session, remote, local],
               timeout=a.exec_timeout + 300)
        got = os.path.getsize(local) if os.path.exists(local) else -1
        if got == size:
            held[name] = size
            log(f"  pulled {name} ({size/1e6:.1f} MB)")
        elif got > size:
            # The file grew between the listing and the download. A training log
            # is appended to continuously, so it is *never* the same size twice
            # and an equality check can never pull it while the run is alive.
            # Record what actually arrived: the next listing reports a larger
            # remote, which pulls it again, and the final pull -- when nothing is
            # writing any more -- is the one that has to match exactly.
            held[name] = got
            log(f"  pulled {name} ({got/1e6:.1f} MB, grew during transfer)")
        else:
            # Never let a failed transfer read as "nothing new": that is exactly
            # how this study lost two final checkpoints.
            detail = (r.stderr or r.stdout or "").strip().splitlines()
            log(f"  FAILED {name}: got {got} of {size} bytes"
                + (f" -- {detail[-1][:140]}" if detail else ""))
            if got != -1:
                # A checkpoint still being written lists short; drop the partial
                # so a later cycle cannot mistake it for a complete file.
                os.remove(local)
    return held


def local_sizes(local_dir):
    return {f: os.path.getsize(os.path.join(local_dir, f))
            for f in os.listdir(local_dir)
            if os.path.isfile(os.path.join(local_dir, f))}


def main(argv=None):
    a = parse_args(argv)
    a.local_dir = os.path.expanduser(a.local_dir)
    os.makedirs(a.local_dir, exist_ok=True)
    session = Session(a)

    fd, lister_path = tempfile.mkstemp(prefix=f"watch_{a.session}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(LISTER.format(dirs=a.remote_dir, pats=a.pattern, state=a.state))

    held = local_sizes(a.local_dir)
    log(f"watching '{a.session}' -> {a.local_dir}; already held: {len(held)} file(s)")
    deadline = time.time() + a.hours * 3600
    completed = None

    try:
        while time.time() < deadline:
            if not session.ensure():
                log("no live assignment -- the VM is gone, not just the token")
                break

            info = probe(session, lister_path)
            if info is None:
                log("no live assignment -- the VM is gone, not just the token")
                break
            if info is TRANSIENT:
                time.sleep(a.interval)
                continue

            held = pull(session, info["files"], held)
            log(f"remote={len(info['files'])} held={len(held)} "
                f"rc={info['returncode']} live={info['supervisor_live']}")

            if info["returncode"] is not None:
                # Do not wait for the next cycle: the VM is now idle and is the
                # first thing the scheduler reclaims.
                log("run finished -- final pull")
                final = probe(session, lister_path)
                if isinstance(final, dict):
                    held = pull(session, final["files"], held)
                    missing = [os.path.basename(p) for p, s in final["files"]
                               if held.get(os.path.basename(p)) != s]
                    if missing:
                        log(f"STILL MISSING after the final pull: {missing}")
                completed = info["returncode"]
                break

            time.sleep(a.interval)
        else:
            log(f"--hours {a.hours} reached; the run may still be going")
    finally:
        os.unlink(lister_path)

    if completed is not None and a.stop_on_complete:
        r = sh(session.base + ["stop", "-s", a.session], timeout=180)
        log(f"  stop: {(r.stdout or r.stderr).strip()[:110]}")
    elif completed is not None:
        log(f"release the VM: {' '.join(session.base)} stop -s {a.session}")

    log(f"held {len(held)} file(s) in {a.local_dir}; run returncode {completed}")
    # A zero return code is a transport signal -- validate the artifacts before
    # trusting the run.
    return 0 if completed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
