"""One number across all seven datasets, with a conditional interval.

Weighting is the whole question here. Pooling trials would make the "overall"
number mostly Fitts and tablet_writing -- together 75% of the trials -- so what
it really reports is those two tasks. The datasets are different *movement
tasks*, and the diversity of tasks is what an overall figure is claiming to
cover, so every dataset counts once:

    theta_overall = (1/D) * sum_d theta_d

where D = 7 and theta_d is dataset d's trial-weighted mean.

The interval comes from the same rule as everywhere else: resample trials within
participant, now nested inside dataset, holding both datasets and participants
fixed. Resampling *datasets* would be the claim that these seven stand in for
movement tasks in general -- with D = 7 that is not estimable, and no reader
would believe it, so it is not offered.

Datasets resample independently, so the overall bootstrap distribution is the
draw-by-draw mean of the per-dataset ones.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

B = 10000
CHUNK = 500
METRICS = ["Reconstruction_R2", "Reconstruction_MASE",
           "Number_of_submovements_per_second"]
ORDER = ["pre_released", "pre_rerun", "pre_rerun_s1000",
         "ft_released", "ft_A", "ft_B", "ft_A2", "ft_B2"]


def dataset_draws(values, participant, rng):
    """B bootstrap draws of one dataset's trial-weighted mean."""
    codes = pd.factorize(participant)[0]
    order = np.argsort(codes, kind="stable")
    values, codes = values[order], codes[order]
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(values)]))
    bounds = list(zip(starts, ends))
    if (ends - starts).max() == 1:          # one trial each: nothing to stratify
        bounds = [(0, len(values))]
    out = []
    for lo in range(0, B, CHUNK):
        size = min(CHUNK, B - lo)
        idx = np.empty((size, len(values)), dtype=np.int64)
        for s, e in bounds:
            idx[:, s:e] = rng.integers(s, e, (size, e - s))
        out.append(values[idx].mean(1))
    return np.concatenate(out)


def main(dump_dir, out_path):
    rng = np.random.default_rng(23)
    rows = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "organic.*.csv"))):
        ck = os.path.basename(path).split(".")[1]
        frame = pd.read_csv(path)
        frame["Noise_Condition"] = frame["Noise_Condition"].astype(str)
        for noise, cell in frame.groupby("Noise_Condition"):
            for metric in METRICS:
                if metric not in cell.columns:
                    continue
                per_ds, draws = {}, []
                for ds, g in cell.groupby("Dataset"):
                    g = g[np.isfinite(g[metric])]
                    if g.empty:
                        continue
                    d = dataset_draws(g[metric].to_numpy(float),
                                      g["Participant"].to_numpy(), rng)
                    per_ds[ds] = float(g[metric].mean())
                    draws.append(d)
                if not draws:
                    continue
                balanced = np.stack(draws).mean(0)      # each dataset counts once
                pooled_obs = float(cell[np.isfinite(cell[metric])][metric].mean())
                vals = np.array(list(per_ds.values()))
                rows.append(dict(
                    checkpoint=ck, snr=noise, metric=metric, n_datasets=len(per_ds),
                    est_balanced=float(vals.mean()),
                    lo=float(np.percentile(balanced, 2.5)),
                    hi=float(np.percentile(balanced, 97.5)),
                    est_trial_pooled=pooled_obs,
                    across_datasets_min=float(vals.min()),
                    across_datasets_max=float(vals.max()),
                    across_datasets_sd=float(vals.std(ddof=1))))
    out = pd.DataFrame(rows)
    out["checkpoint"] = pd.Categorical(out.checkpoint, ORDER, ordered=True)
    out.sort_values(["checkpoint", "snr", "metric"]).to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", default="uncapped")
    p.add_argument("--out", default="docs/overall_interval.csv")
    a = p.parse_args()
    main(a.dumps, a.out)
