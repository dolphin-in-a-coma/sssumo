"""Compare wandb runs on their logged metrics, for reproduction checks.

Answers "does my rerun match the published run?" from the training-time metrics,
which is independent of, and complementary to, re-evaluating the checkpoints.

Credentials come from ~/.netrc (machine api.wandb.ai) -- nothing is installed and
the key is never handled here.

    WANDB_ENTITY=me python scripts/colab/wandb_compare_runs.py \\
        --run published=jbj4im5v --run mine=9bj6nnyt \\
        --keys Onset_Epoch/Precision_Mean Loss_Epoch/Detection_Mean
"""

import argparse
import base64
import json
import netrc
import os
import urllib.request

API = "https://api.wandb.ai/graphql"

DEFAULT_KEYS = [
    "Loss_Epoch/Detection_Mean",
    "Loss_Epoch/Duration_Mean",
    "Loss_Epoch/Amplitude_Mean",
    "Loss_Epoch/Reconstruction_Mean",
    "Onset_Epoch/Precision_Mean",
    "Onset_Epoch/Recall_Mean",
    "Onset_Epoch/Distance_Mean",
]


def auth_header():
    creds = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")
    if not creds or not creds[2]:
        raise SystemExit("no api.wandb.ai entry in ~/.netrc; run `wandb login` first")
    return "Basic " + base64.b64encode(f"{creds[0]}:{creds[2]}".encode()).decode()


def gql(query, variables=None):
    req = urllib.request.Request(
        API, data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={"Content-Type": "application/json", "Authorization": auth_header()})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:              # surface the server's reason
        raise SystemExit(f"HTTP {exc.code}: {exc.read()[:600].decode(errors='replace')}")
    if "errors" in payload:
        raise SystemExit(f"GraphQL errors: {payload['errors']}")
    return payload["data"]


RUN_QUERY = """
query($p:String!,$e:String!,$r:String!,$k:[JSONString!]!){
  project(name:$p, entityName:$e){
    run(name:$r){
      name displayName state createdAt config summaryMetrics
      sampledHistory(specs:$k)
    }
  }
}"""


def fetch(entity, project, run_id, samples, keys):
    # sampledHistory takes one spec per requested series and returns one list each
    specs = [json.dumps({"keys": ["_step"] + list(keys), "samples": samples})]
    data = gql(RUN_QUERY, {"p": project, "e": entity, "r": run_id, "k": specs})
    run = data["project"]["run"]
    if run is None:
        raise SystemExit(f"no run {run_id} in {entity}/{project}")
    rows = []
    for group in run.get("sampledHistory") or []:
        for item in group or []:
            rows.append(json.loads(item) if isinstance(item, str) else item)
    return run, rows


def series(rows, key, steps_per_epoch=1000):
    """{epoch: value} for one key.

    Keyed by epoch derived from _step, never by row position: a resumed run
    starts at a later step, so position-aligned rows would compare different
    epochs against each other.
    """
    out = {}
    for r in rows:
        step, value = r.get("_step"), r.get(key)
        if step is None or not isinstance(value, (int, float)):
            continue
        epoch = round(step / steps_per_epoch) - 1   # logged at (epoch+1)*steps
        out[epoch] = value
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=RUNID")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT",
                                                        "submovement_detector"))
    ap.add_argument("--keys", nargs="*", default=DEFAULT_KEYS)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--steps-per-epoch", type=int, default=1000,
                    help="optimizer steps per epoch, to map _step -> epoch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.entity:
        raise SystemExit("set --entity or WANDB_ENTITY")

    runs = dict(spec.split("=", 1) for spec in args.run)
    fetched, collected = {}, {}
    for label, run_id in runs.items():
        run, rows = fetch(args.entity, args.project, run_id, args.samples, args.keys)
        fetched[label] = run
        collected[label] = rows
        summary = json.loads(run["summaryMetrics"] or "{}")
        print(f"{label:12s} {run['name']}  {run['state']:9s} "
              f"steps={summary.get('_step')}  {run['displayName']}", flush=True)

    labels = list(runs)
    print()
    for key in args.keys:
        table = {l: series(collected[l], key, args.steps_per_epoch) for l in labels}
        if not any(table.values()):
            continue
        print(f"--- {key} ---")
        epochs = sorted({e for v in table.values() for e in v})
        header = f"  {'epoch':>6s}" + "".join(f"{l:>16s}" for l in labels)
        if len(labels) == 2:
            header += f"{'delta':>12s}"
        print(header)
        for i in epochs:
            values = [table[l].get(i) for l in labels]
            row = f"  {i:6d}" + "".join(
                f"{v:16.4f}" if isinstance(v, float) else f"{'-':>16s}" for v in values)
            if len(labels) == 2 and all(isinstance(v, float) for v in values):
                row += f"{values[1]-values[0]:+12.4f}"
            print(row)
        print()

    if args.out:
        with open(args.out, "w") as f:
            json.dump({l: {"run": fetched[l]["name"], "rows": collected[l]}
                       for l in labels}, f, indent=1, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
