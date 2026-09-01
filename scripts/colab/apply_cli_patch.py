"""Apply (or check, or revert) the proxy-token refresh patch to the installed colab CLI.

    python scripts/colab/apply_cli_patch.py            # report status only
    python scripts/colab/apply_cli_patch.py --apply
    python scripts/colab/apply_cli_patch.py --revert

`google-colab-cli` mints the runtime proxy token once, at `colab new`, and never
refreshes it, while its keep-alive daemon renews only the *VM assignment*. The
two clocks diverge at about an hour: every kernel call 404s on a healthy VM and
the CLI prunes the local record. `patches/0001-refresh-proxy-token.patch` gives
the refresh to the daemon, which already runs on a timer and knows the endpoint.

This is a local fork of a third-party package, so it is undone by any reinstall
or `colab update`. Run this with no arguments after either -- it exits non-zero
when the patch is missing, so it also works as a check in a run script.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

PATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "patches", "0001-refresh-proxy-token.patch")
TARGETS = ("state.py", "commands/session.py")
MARKER = "def refresh_proxy_token("
TESTED_VERSION = "0.6.0"
# sha256 of each target before and after, for google-colab-cli 0.6.0. Anything
# else means the upstream file moved and the patch needs regenerating rather
# than forcing.
PRISTINE = {
    "state.py":
        "5b43ab294b38a7bd450678b9f0762741b0f43b51f036a6c634b385adbb86433e",
    "commands/session.py":
        "59d50c3fe04bfcd1eaa7e2441e79d5e2bed2d0201c469c37e5e614351bbe0c1c",
}
PATCHED = {
    "state.py":
        "943a575e232c24180bc9dd62d8c4e3d3177554cff2616cd9f96812abe26d902f",
    "commands/session.py":
        "fbf49b853ed2baf0059aef693a9a668ea4d6ea4514d5488aa79b5d3622c56de6",
}


def cli_python(explicit=None):
    if explicit:
        return explicit
    path = subprocess.run(["/bin/sh", "-c", "command -v colab"],
                          capture_output=True, text=True).stdout.strip()
    if not path:
        raise SystemExit("cannot find `colab` on PATH; pass --cli-python")
    with open(os.path.realpath(path)) as f:
        return f.readline().lstrip("#!").strip()


def locate(py):
    r = subprocess.run([py, "-c", "import colab_cli, importlib.metadata as m;"
                        "print(list(colab_cli.__path__)[0]);"
                        "print(m.version('google-colab-cli'))"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{py} cannot import colab_cli:\n{r.stderr.strip()}")
    pkg, version = r.stdout.split()
    return pkg, version


def digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def status(pkg):
    """'patched', 'pristine', or 'unknown' (upstream changed under us)."""
    have = {t: digest(os.path.join(pkg, t)) for t in TARGETS}
    if have == PATCHED:
        return "patched"
    if have == PRISTINE:
        return "pristine"
    if all(MARKER in open(os.path.join(pkg, t)).read() for t in ("commands/session.py",)):
        return "unknown-patched"
    return "unknown"


def backup_dir(pkg):
    return os.path.join(pkg, ".sssumo-patch-backup")


def apply(pkg, py, force):
    state = status(pkg)
    if state == "patched":
        print("already applied; nothing to do")
        return 0
    if state != "pristine" and not force:
        raise SystemExit(
            f"the installed files are neither pristine nor the version this patch\n"
            f"was made against ({state}). Regenerate the patch against the current\n"
            f"colab_cli rather than forcing it, or pass --force if you are sure.")

    backup = backup_dir(pkg)
    os.makedirs(os.path.join(backup, "commands"), exist_ok=True)
    for t in TARGETS:
        shutil.copy2(os.path.join(pkg, t), os.path.join(backup, t))

    parent = os.path.dirname(pkg)
    r = subprocess.run(["patch", "-p1", "-d", parent, "-i", PATCH],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        revert(pkg, quiet=True)
        raise SystemExit("patch failed; the originals have been restored")

    now = status(pkg)
    if now != "patched":
        revert(pkg, quiet=True)
        raise SystemExit(f"patched files do not match the expected hashes ({now});"
                         " originals restored")
    for t in TARGETS:                      # stale bytecode would mask the change
        pyc = os.path.join(pkg, os.path.dirname(t), "__pycache__")
        shutil.rmtree(pyc, ignore_errors=True)

    # Matching hashes only prove the bytes landed. Run the behavioural check
    # too, against a stubbed client, so "applied" means the refresh works.
    test = os.path.join(os.path.dirname(PATCH), "test_refresh.py")
    r = subprocess.run([py, test], capture_output=True, text=True)
    if r.returncode != 0:
        revert(pkg, quiet=True)
        raise SystemExit("the patched refresh failed its own checks; originals "
                         f"restored\n{r.stdout}{r.stderr}")
    print(r.stdout.strip())
    print("applied; the keep-alive daemon now refreshes the proxy token")
    print("NOTE: existing sessions keep the daemon they were started with -- "
          "the refresh only covers sessions created after this point")
    return 0


def revert(pkg, quiet=False):
    backup = backup_dir(pkg)
    missing = [t for t in TARGETS if not os.path.exists(os.path.join(backup, t))]
    if missing:
        raise SystemExit(f"no backup for {missing}; reinstall with "
                         "`uv tool install --reinstall google-colab-cli`")
    for t in TARGETS:
        shutil.copy2(os.path.join(backup, t), os.path.join(pkg, t))
        shutil.rmtree(os.path.join(pkg, os.path.dirname(t), "__pycache__"),
                      ignore_errors=True)
    if not quiet:
        print("reverted to the packaged files")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="apply even when the installed files are not the tested ones")
    ap.add_argument("--cli-python", help="interpreter that has colab_cli installed")
    a = ap.parse_args(argv)

    py = cli_python(a.cli_python)
    pkg, version = locate(py)
    print(f"google-colab-cli {version} at {pkg}")
    if version != TESTED_VERSION:
        print(f"  (patch was made against {TESTED_VERSION})")

    if a.apply:
        return apply(pkg, py, a.force)
    if a.revert:
        return revert(pkg)

    state = status(pkg)
    print(f"status: {state}")
    if state == "patched":
        return 0
    print("run with --apply to install the refresh, or use scripts/colab/watch.py,"
          "\nwhich re-mints from the host and needs no patch")
    return 1


if __name__ == "__main__":
    sys.exit(main())
