"""Mirror a run's artifacts off the VM while it is still running.

    colab --auth=oauth2 exec -s <session> -f scripts/colab/vm_persist.py --timeout 120

`/content` dies with the VM, and an idle VM is reclaimed within minutes of a run
finishing -- so the final checkpoint is the likeliest one to lose. `watch.py`
races that window from the host; this removes it instead, by writing each
artifact to durable storage from inside the VM as it appears. The two are
complementary: run both and a lost VM costs nothing.

Configured by a `persist` section in /content/run.json (`colab exec -f` passes no
argv). Two sinks:

    "persist": {
      "backend": "webdav",
      "webdav_url": "https://datacloud.helsinki.fi/public.php/webdav",
      "webdav_token_file": "/content/.persist_token",
      "remote_dir": "my-run",
      "include": ["weights/*.pth", "logs/*.txt"],
      "interval": 60
    }

`backend` is "webdav", "wandb", or "both".

**No credential goes in run.json.** The share token is uploaded as a file whose
only content is the token, exactly like the wandb key, and is sent as HTTP basic
auth rather than in the URL -- so it cannot leak through a traceback, a redirect
or a log line.

A Nextcloud share link is a bearer capability: anyone holding it gets whatever
the share grants. Prefer a **file-drop** share (upload only) for this; a
read/write share also lets a leaked token delete everything already uploaded.
"""
import fnmatch
import json
import os
import sys
import time

# Standalone on purpose, like supervise.py: this runs detached, launched by
# path, so it cannot rely on vm_common.py being importable from wherever the
# kernel happens to be. The two helpers it needs are three lines each.
RUN_JSON = "/content/run.json"
SETTLE_SECONDS = 5      # don't upload a file torch.save is still writing
RETRIES = 4


def load():
    with open(RUN_JSON) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def paths(cfg):
    name = cfg["experiment_name"]
    return f"/content/{name}.state.json", f"/content/{name}.log"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class WebDAVSink:
    """A Nextcloud/ownCloud public share, addressed by its token.

    The token is the username with an empty password, which is how public
    shares authenticate. Keeping it in the auth tuple rather than the URL means
    it never reaches a log, a traceback or a redirect target.
    """

    name = "webdav"

    def __init__(self, url, token, remote_dir):
        import requests

        self.session = requests.Session()
        self.session.auth = (token, "")
        self.base = url.rstrip("/")
        self.remote_dir = remote_dir.strip("/")
        self.root = f"{self.base}/{self.remote_dir}" if self.remote_dir else self.base
        self._made = set()

    def prepare(self):
        self._mkcol_p(self.remote_dir)

    def _mkcol_p(self, path):
        """`mkdir -p` for WebDAV: MKCOL is not recursive and a PUT into a
        missing collection fails with 409, not with something that names the
        cause."""
        parts = [p for p in path.split("/") if p]
        for i in range(len(parts)):
            prefix = "/".join(parts[:i + 1])
            if prefix in self._made:
                continue
            r = self.session.request("MKCOL", f"{self.base}/{prefix}", timeout=60)
            # 201 created, 405 already there -- both fine.
            if r.status_code not in (201, 405):
                raise RuntimeError(f"MKCOL {prefix} -> {r.status_code}")
            self._made.add(prefix)

    def put(self, local_path, name):
        # An artifact name carries its own subdirectory (weights/foo.pth), and
        # that collection has to exist before the PUT.
        parent = os.path.dirname(name)
        if parent:
            self._mkcol_p(f"{self.remote_dir}/{parent}" if self.remote_dir else parent)
        size = os.path.getsize(local_path)
        with open(local_path, "rb") as f:
            r = self.session.put(f"{self.root}/{name}", data=f,
                                 timeout=max(120, size / 200_000))
        if r.status_code not in (200, 201, 204):
            raise RuntimeError(f"PUT {name} -> {r.status_code}")
        return self.remote_size(name)

    def remote_size(self, name):
        """Read the size back, so a truncated upload cannot pass as a success."""
        r = self.session.request("PROPFIND", f"{self.root}/{name}",
                                 headers={"Depth": "0"}, timeout=60)
        if r.status_code != 207:
            return -1
        marker = "<d:getcontentlength>"
        body = r.text
        if marker not in body:
            return -1
        return int(body.split(marker, 1)[1].split("<", 1)[0])


class WandbSink:
    """Each artifact as a versioned wandb artifact under one run."""

    name = "wandb"

    def __init__(self, project, run_name, key):
        import wandb

        wandb.login(key=key)
        self.wandb = wandb
        self.run = wandb.init(project=project, name=f"{run_name}-artifacts",
                              job_type="persist", reinit=True)

    def prepare(self):
        pass

    def put(self, local_path, name):
        art = self.wandb.Artifact(name.replace("/", "_").rsplit(".", 1)[0],
                                  type="checkpoint")
        art.add_file(local_path, name=name)
        self.run.log_artifact(art)
        # wandb uploads asynchronously and exposes no synchronous size to read
        # back, so trust the local size here rather than inventing a check.
        return os.path.getsize(local_path)


def read_secret(path):
    with open(path) as f:
        return f.read().strip()


def build_sinks(cfg, pcfg):
    backend = pcfg.get("backend", "webdav")
    sinks = []
    if backend in ("webdav", "both"):
        token_file = pcfg.get("webdav_token_file", "/content/.persist_token")
        if not os.path.exists(token_file):
            log(f"WARNING: no {token_file}; webdav sink disabled")
        else:
            sinks.append(WebDAVSink(pcfg["webdav_url"], read_secret(token_file),
                                    pcfg.get("remote_dir") or cfg["experiment_name"]))
    if backend in ("wandb", "both"):
        key_file = cfg.get("wandb_key_file")
        if not key_file or not os.path.exists(key_file):
            log("WARNING: no wandb key file; wandb sink disabled")
        else:
            sinks.append(WandbSink(cfg["wandb_project"], cfg["experiment_name"],
                                   read_secret(key_file)))
    return sinks


def candidates(cfg, patterns):
    """Every file matching an include pattern, relative to root_dir."""
    root = cfg["root_dir"]
    out = []
    for pattern in patterns:
        sub = os.path.dirname(pattern)
        directory = os.path.join(root, sub)
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            if fnmatch.fnmatch(entry, os.path.basename(pattern)):
                path = os.path.join(directory, entry)
                if os.path.isfile(path):
                    out.append((path, os.path.join(sub, entry) if sub else entry))
    return out


def main():
    cfg = load()
    pcfg = cfg.get("persist") or {}
    if not pcfg or not pcfg.get("backend"):
        log("no persist section in run.json -- nothing to do")
        return 0

    state_path, _ = paths(cfg)
    # Beside the run's own state file, so both move together if paths() does.
    mirror_state = state_path.replace(".state.json", ".persist.json")
    done = {}
    if os.path.exists(mirror_state):
        try:
            done = json.load(open(mirror_state))
        except ValueError:
            done = {}

    sinks = build_sinks(cfg, pcfg)
    if not sinks:
        log("no usable sink -- refusing to run and pretend artifacts are safe")
        return 1
    for sink in sinks:
        sink.prepare()
    log(f"mirroring to: {', '.join(s.name for s in sinks)}")

    patterns = pcfg.get("include") or ["weights/*.pth", "logs/*.txt"]
    interval = pcfg.get("interval", 60)
    finished_at = None

    while True:
        for path, name in candidates(cfg, patterns):
            size = os.path.getsize(path)
            key = f"{name}"
            # Skip a file still being written, and one already mirrored at this
            # exact size -- checkpoints are rewritten, logs grow.
            if time.time() - os.path.getmtime(path) < SETTLE_SECONDS:
                continue
            if done.get(key) == size:
                continue
            for attempt in range(1, RETRIES + 1):
                try:
                    for sink in sinks:
                        got = sink.put(path, name)
                        if got != size:
                            raise RuntimeError(
                                f"{sink.name} stored {got} of {size} bytes")
                    done[key] = size
                    tmp = mirror_state + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(done, f)
                    os.replace(tmp, mirror_state)
                    log(f"  mirrored {name} ({size/1e6:.1f} MB)")
                    break
                except Exception as e:                      # noqa: BLE001
                    # Never let a network blip take the mirror down: training is
                    # a separate process, but a dead mirror is a silent one.
                    wait = 2 ** attempt
                    log(f"  {name} attempt {attempt}/{RETRIES} failed: "
                        f"{type(e).__name__}: {str(e)[:120]}")
                    if attempt < RETRIES:
                        time.sleep(wait)

        # Keep going a little past the run so the last checkpoint is caught,
        # then stop -- the VM is about to be reclaimed anyway.
        if finished_at is None and os.path.exists(state_path):
            try:
                if json.load(open(state_path)).get("returncode") is not None:
                    finished_at = time.time()
                    log("run finished -- one more sweep")
            except ValueError:
                pass
        if finished_at is not None and time.time() - finished_at > interval:
            break
        time.sleep(interval)

    missing = [n for _, n in candidates(cfg, patterns) if n not in done]
    if missing:
        log(f"STILL NOT MIRRORED: {missing}")
        return 1
    log(f"mirrored {len(done)} file(s); nothing outstanding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
