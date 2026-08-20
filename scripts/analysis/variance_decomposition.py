"""Decompose reproduction error against run-to-run variance.

The question is not "are the numbers identical" -- they never are -- but "is the
gap to the released checkpoint larger than the gap between two runs that differ
only by seed". Takes the JSON from compare_checkpoints.py.
"""
import json
import sys

LOWER_IS_BETTER = {"Onset_Distance", "Amplitude_MAE_Scaled", "Duration_MAE_Scaled",
                   "Reconstruction_MASE", "Reconstruction_SMAPE", "Reconstruction_MAE_Scaled"}
KEY_SYNTH = ["Onset_Precision", "Onset_Recall", "Onset_F1", "Onset_Distance",
             "Reconstruction_R2", "Amplitude_R2", "Duration_R2"]


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else float("nan")


def synth(results, label, key):
    conds = results[label]["synthetic"]
    return mean([conds[c].get(key) for c in conds])


def organic_mean(results, label, noise):
    cells = results[label]["organic"][noise]
    return mean([cells[d].get("Reconstruction_R2") for d in cells])


def main(path):
    r = json.load(open(path))
    labels = list(r)
    print(f"checkpoints: {labels}\n")

    print("=== SYNTHETIC (ground truth), mean over 9 conditions ===")
    header = f"  {'metric':22s}" + "".join(f"{l:>14s}" for l in labels)
    print(header)
    for key in KEY_SYNTH:
        vals = [synth(r, l, key) for l in labels]
        if all(v != v for v in vals):
            continue
        print(f"  {key:22s}" + "".join(f"{v:14.4f}" for v in vals))

    print("\n=== ORGANIC mean R2 by noise ===")
    for noise in r[labels[0]]["organic"]:
        vals = [organic_mean(r, l, noise) for l in labels]
        print(f"  SNR {noise:>4s}" + "".join(f"{v:14.4f}" for v in vals))

    # pairwise distance, summarised over the synthetic metrics
    print("\n=== pairwise |delta|, mean over synthetic metrics ===")
    print(f"  {'pair':28s} {'mean|d|':>10s} {'max|d|':>10s}")
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ds = [abs(synth(r, a, k) - synth(r, b, k)) for k in KEY_SYNTH]
            ds = [d for d in ds if d == d]
            print(f"  {a + ' vs ' + b:28s} {mean(ds):10.4f} {max(ds):10.4f}")

    print("\n=== organic cells beyond |delta R2| > 0.02 vs the first label ===")
    ref = labels[0]
    for other in labels[1:]:
        hits = []
        for noise in r[ref]["organic"]:
            for d in r[ref]["organic"][noise]:
                a = r[ref]["organic"][noise][d].get("Reconstruction_R2")
                b = r[other]["organic"][noise][d].get("Reconstruction_R2")
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(b - a) > 0.02:
                    hits.append(f"snr{noise}/{d} {a:.3f}->{b:.3f}")
        print(f"  {other:18s} {len(hits):2d} cells" + ("  " + "; ".join(hits) if hits else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "compare_final.json")
