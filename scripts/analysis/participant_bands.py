"""Per-participant bootstrap bands, aggregated to a dataset-level envelope.

Descriptive, conditional on the participants we actually tested. For each
participant the trials are resampled to give that participant's own uncertainty
band; the dataset aggregate is the *envelope* of those bands -- the lowest lower
bound and the highest upper bound. The claim it supports is "every participant we
tested lies in here", not "the population mean lies in here".

The mixture quantiles (all participants' bootstrap draws pooled, equal weight per
participant) are reported alongside as the narrower alternative. With five
participants each of its tails comes from a single participant's trial noise,
which is why the envelope is the headline number.

**The envelope widens with participant count** -- it is a max/min statistic, so its
width correlates with n at r = 0.92 across our datasets. It answers "does every
participant land in here", not "how variable is this dataset". The interquartile
range of the participant estimates is reported alongside for that, and is the
column to use when comparing datasets of different size.

No GPU: reads the uncapped per-trial dumps written by
`scripts/colab/score_checkpoints.py --dump-per-trial`.
"""
import argparse, glob, os
import numpy as np
import pandas as pd

B = 2000                     # bootstrap resamples per participant
METRICS = ["Reconstruction_R2", "Reconstruction_MASE",
           "Number_of_submovements_per_second"]
ORDER = ["pre_released", "pre_rerun", "pre_rerun_s1000",
         "ft_released", "ft_A", "ft_B", "ft_A2", "ft_B2"]


def participant_band(values, rng):
    """Point estimate and 2.5/97.5 band from resampling one participant's trials."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, None
    idx = rng.integers(0, len(values), size=(B, len(values)))
    draws = values[idx].mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi), draws


def main(dump_dir, out_participants, out_envelope):
    rng = np.random.default_rng(7)
    per_participant, envelope = [], []

    paths = sorted(glob.glob(os.path.join(dump_dir, "organic.*.csv")))
    if not paths:
        raise SystemExit(f"no organic.*.csv under {dump_dir}")

    for path in paths:
        ck = os.path.basename(path).split(".")[1]
        frame = pd.read_csv(path)
        frame["Noise_Condition"] = frame["Noise_Condition"].astype(str)

        for (dataset, noise), cell in frame.groupby(["Dataset", "Noise_Condition"]):
            for metric in METRICS:
                if metric not in cell.columns:
                    continue
                points, lows, highs, pooled = [], [], [], []

                for participant, g in cell.groupby("Participant"):
                    point, lo, hi, draws = participant_band(
                        g[metric].to_numpy(float), rng)
                    if draws is None:
                        continue
                    points.append(point); lows.append(lo); highs.append(hi)
                    pooled.append(draws)
                    per_participant.append(dict(
                        checkpoint=ck, dataset=dataset, snr=noise, metric=metric,
                        participant=participant, n_trials=int(g[metric].notna().sum()),
                        estimate=point, lower=lo, upper=hi))

                if not points:
                    continue
                pooled = np.concatenate(pooled)          # equal weight: same B each
                mix_lo, mix_hi = np.percentile(pooled, [2.5, 97.5])
                q25, q75 = np.percentile(points, [25, 75])
                envelope.append(dict(
                    checkpoint=ck, dataset=dataset, snr=noise, metric=metric,
                    n_participants=len(points),
                    median=float(np.median(points)),
                    q25=float(q25), q75=float(q75),
                    min_estimate=float(np.min(points)),
                    max_estimate=float(np.max(points)),
                    envelope_lo=float(np.min(lows)),
                    envelope_hi=float(np.max(highs)),
                    mixture_lo=float(mix_lo),
                    mixture_hi=float(mix_hi)))

    def ordered(rows):
        d = pd.DataFrame(rows)
        d["checkpoint"] = pd.Categorical(d.checkpoint, ORDER, ordered=True)
        by = ["checkpoint", "dataset", "snr", "metric"]
        return d.sort_values(by + (["participant"] if "participant" in d else []))

    ordered(per_participant).to_csv(out_participants, index=False)
    ordered(envelope).to_csv(out_envelope, index=False)
    print(f"wrote {out_participants} ({len(per_participant)} rows)")
    print(f"wrote {out_envelope} ({len(envelope)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", default="uncapped")
    p.add_argument("--out-participants", default="docs/organic_participant_bands.csv")
    p.add_argument("--out-envelope", default="docs/organic_envelope.csv")
    a = p.parse_args()
    main(a.dumps, a.out_participants, a.out_envelope)
