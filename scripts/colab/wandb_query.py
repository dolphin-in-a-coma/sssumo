"""Query the wandb GraphQL API using the credential already in ~/.netrc.

Standard library only -- nothing installed, and the key is never handled in code.
"""
import base64
import json
import netrc
import os
import sys
import urllib.request

ENTITY = os.environ.get("WANDB_ENTITY", "")
PROJECT = os.environ.get("WANDB_PROJECT", "submovement_detector")
if not ENTITY:
    raise SystemExit("set WANDB_ENTITY (your wandb user or team)")
auth = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")
basic = base64.b64encode(f"{auth[0]}:{auth[2]}".encode()).decode()


def gql(query):
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {basic}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    if "errors" in out:
        raise SystemExit(f"GraphQL errors: {out['errors']}")
    return out["data"]


RUN_FIELDS = "id name displayName state createdAt heartbeatAt config summaryMetrics"

run_id = sys.argv[1] if len(sys.argv) > 1 else "bxkweckh"
d = gql(f'{{ project(name: "{PROJECT}", entityName: "{ENTITY}") '
        f'{{ run(name: "{run_id}") {{ {RUN_FIELDS} }} }} }}')
run = d["project"]["run"]
if run is None:
    raise SystemExit(f"no run {run_id}")

cfg = json.loads(run["config"] or "{}")
cfg = {k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
       for k, v in cfg.items()}
summary = json.loads(run["summaryMetrics"] or "{}")

print(f"run:       {run['displayName']}")
print(f"id:        {run['name']}")
print(f"url:       https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run['name']}")
print(f"state:     {run['state']}")
print(f"created:   {run['createdAt']}Z")
print(f"last beat: {run['heartbeatAt']}Z")
print(f"runtime:   {summary.get('_runtime')}s   steps: {summary.get('_step')}")
print(f"_experiment_name in config: {cfg.get('_experiment_name')!r}")

import yaml  # noqa: E402  (pyyaml ships with the system python here)
yml = yaml.safe_load(open("configs/config-0423-ModGaussian_ampl.yaml"))
flat = {k: v for sec in yml.values() for k, v in sec.items()}

print("\nconfig vs configs/config-0423-ModGaussian_ampl.yaml:")
checked = diff = 0
for k, want in sorted(flat.items()):
    if k not in cfg:
        continue
    got = cfg[k]
    a = list(got) if isinstance(got, (list, tuple)) else got
    b = list(want) if isinstance(want, (list, tuple)) else want
    checked += 1
    if a == b:
        print(f"  OK   {k} = {a}")
    else:
        diff += 1
        print(f"  DIFF {k}: run={a}  yaml={b}")
print(f"\n  {checked} keys compared, {diff} differ")

print("\nepoch-level summary metrics:")
for k in sorted(summary):
    if k.startswith(("Onset_Epoch", "Loss_Epoch")):
        print(f"  {k}: {summary[k]}")
