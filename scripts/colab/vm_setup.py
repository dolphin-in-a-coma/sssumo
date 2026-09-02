"""Clone the pinned commit, install the package, and stage data on the VM.

Run once per session:
    colab --auth=oauth2 exec -s <session> -f scripts/colab/vm_setup.py --timeout 1800
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vm_common import load  # noqa: E402

CSVS = [
    "steering_tangential_velocity_data.csv",
    "crank_tangential_velocity_data.csv",
    "Fitts_tangential_velocity_data.csv",
    "whacamole_tangential_velocity_data.csv",
    "object_moving_tangential_velocity_data.csv",
    "pointing_tangential_velocity_data.csv",
    "tablet_writing_tangential_velocity_data.csv",
]


def run(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:], flush=True)
    return r.returncode


cfg = load()
root = cfg["root_dir"]

# --- source, pinned so results are attributable to an exact commit ------------
# Two transports. A clone is simplest when the commit is on a remote; an uploaded
# archive from package_git_source.py also carries commits that are not, so a run
# does not have to be published before it can be reproduced. The archive is
# checked against its manifest before anything is extracted -- a truncated upload
# would otherwise install silently and be attributed to the manifest's commit.
archive = cfg.get("source_archive")
if archive:
    if not os.path.isdir("/content/sssumo"):
        manifest = json.load(open(f"{archive}.manifest.json"))
        digest = hashlib.sha256()
        with open(archive, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        assert digest.hexdigest() == manifest["sha256"], "source archive checksum mismatch"
        with tarfile.open(archive) as tar:
            tar.extractall("/content", filter="data")
        os.rename(f"/content/{manifest['prefix'].rstrip('/')}", "/content/sssumo")
        print("verified source archive, commit", manifest["commit"], flush=True)
else:
    if not os.path.exists("/content/sssumo/.git"):
        assert run(f"git clone -q {cfg['repo']} /content/sssumo") == 0
    assert run(
        f"cd /content/sssumo && git checkout -q {cfg['commit']} && git rev-parse HEAD") == 0
run("cd /content/sssumo && pip install -q . 2>&1 | tail -5")

import torch  # noqa: E402  (only meaningful after the install above)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

os.makedirs(f"{root}/data", exist_ok=True)

# --- data ---------------------------------------------------------------------
source = cfg["data_source"]
t0 = time.time()

if source.startswith("drive:"):
    drive = source[len("drive:"):]
    assert os.path.isdir(drive), f"{drive} not found -- run 'colab drivemount' first"
    # Copy to VM-local disk: reading 500 MB CSVs through the Drive FUSE layer
    # during evaluation is far slower than the one-off copy.
    for name in CSVS:
        src, dst = f"{drive}/data/{name}", f"{root}/data/{name}"
        size = os.path.getsize(src)
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            print(f"  have {name} ({size/1e6:.0f} MB)", flush=True)
            continue
        shutil.copyfile(src, dst)
        assert os.path.getsize(dst) == size, f"size mismatch for {name}"
        print(f"  copied {name} ({size/1e6:.0f} MB)", flush=True)

    if cfg.get("persist_to_drive"):
        # weights/ and logs/ live in Drive so checkpoints outlive the VM
        for name in ("weights", "logs"):
            src, dst = f"{drive}/{name}", f"{root}/{name}"
            os.makedirs(src, exist_ok=True)
            if not os.path.islink(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                os.symlink(src, dst)
            print(f"{dst} -> {os.readlink(dst)}", flush=True)

elif source.startswith("url:"):
    url = source[len("url:"):]
    zip_path = f"{root}/data/dataset.zip"
    assert run(f'curl -sL "{url}" -o {zip_path}') == 0
    assert run(f"unzip -oq {zip_path} -d {root}/data/") == 0
    run(f"find {root}/data -mindepth 2 -name '*tangential_velocity_data.csv' "
        f"-exec mv {{}} {root}/data/ \\;")
    os.remove(zip_path)
else:
    raise ValueError(f"data_source must start with 'drive:' or 'url:', got {source!r}")

for name in ("weights", "logs"):
    os.makedirs(f"{root}/{name}", exist_ok=True)

present = sorted(f for f in os.listdir(f"{root}/data") if f.endswith(".csv"))
print(f"\ndata staged in {time.time()-t0:.0f}s: {len(present)} csv files", flush=True)
for f in present:
    print(f"  {f} ({os.path.getsize(f'{root}/data/{f}')/1e6:.0f} MB)", flush=True)
print("SETUP OK", flush=True)
