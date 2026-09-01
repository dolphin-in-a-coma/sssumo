"""One interval per dataset, conditional on the participants we tested.

The reported number is the model's performance on a dataset. What makes it
uncertain is that the trials could have come out differently -- not that the
participants could have been different people. So exactly one level is
resampled: **trials, stratified by participant**. Stratification holds the
participant composition fixed, which is what makes this a statement about our
data rather than about a population.

Nothing finer is resampled, and nothing finer needs to be. Each trial's observed
metric is already a noisy realisation, so the spread across trials carries the
within-trial noise with it; adding a sample- or chunk-level resample on top
double-counts it (verified by simulation: the inflation factor matches
sqrt(v_b + 2*v_w)/sqrt(v_b + v_w) to three digits).

Two estimands are reported because they answer different questions:

  trial-weighted        mean over all trials -- matches the pooled figure the
                        article already reports, but a participant with 572
                        trials outweighs one with 5.
  participant-balanced  mean of the participant means -- every person counts
                        once, which is usually the fairer summary of "how well
                        does this work on these people".

The same rule, moved up one rung, gives the population interval instead:
resample **participants**, keeping each drawn participant's trials intact. Not
participants *and* trials within them -- each participant's observed mean already
carries their trial noise, so resampling inside a drawn participant double-counts
it exactly as a sample-level resample would one level down. One level, always.

  conditional  resample trials within participant   participants fixed
  population   resample participants, trials whole  generalises over people

The population interval is only meaningful with enough participants. Below about
20 clusters the percentile cluster bootstrap is anti-conservative, so the
participant-level t interval is reported alongside it -- its degrees of freedom
are honest at any n, and it is the one to quote when clusters are few.

Degenerate case: where a dataset has one trial per participant, participants and
trials are the same unit, stratified resampling returns the identical set every
time, and no conditional interval exists. Those datasets are resampled
unstratified and flagged -- their interval necessarily includes participant
variation. Their *population* interval is unaffected and remains valid.

No GPU: reads the uncapped per-trial dumps from
`scripts/colab/score_checkpoints.py --dump-per-trial`.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

B = 10000
CHUNK = 500                       # bootstrap draws held in memory at once
METRICS = ["Reconstruction_R2", "Reconstruction_MASE",
           "Number_of_submovements_per_second"]
ORDER = ["pre_released", "pre_rerun", "pre_rerun_s1000",
         "ft_released", "ft_A", "ft_B", "ft_A2", "ft_B2"]


def _draw(bounds, n_total, size, rng):
    """Index matrix (size, n_total): resample trials inside each participant."""
    out = np.empty((size, n_total), dtype=np.int64)
    for start, end in bounds:
        out[:, start:end] = rng.integers(start, end, (size, end - start))
    return out


def population_intervals(sizes, means, rng):
    """Resample participants, each carrying all of their trials.

    Only (n_p, mean_p) per participant is needed: the trial-weighted mean over a
    drawn multiset of participants is sum(n_p * m_p) / sum(n_p), so the trials
    themselves never have to be touched.
    """
    n = len(sizes)
    out = {}
    if n >= 2:
        m = means.mean()
        sd = means.std(ddof=1)
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        out.update(t_est=float(m), t_lo=float(m - half), t_hi=float(m + half))
    else:
        out.update(t_est=float(means.mean()), t_lo=np.nan, t_hi=np.nan)

    idx = rng.integers(0, n, (B, n))
    w = sizes[idx]
    tw = (w * means[idx]).sum(1) / w.sum(1)
    pb = means[idx].mean(1)
    out.update(
        clu_trial_lo=float(np.percentile(tw, 2.5)),
        clu_trial_hi=float(np.percentile(tw, 97.5)),
        clu_bal_lo=float(np.percentile(pb, 2.5)),
        clu_bal_hi=float(np.percentile(pb, 97.5)),
        cluster_reliable=bool(n >= 20))
    return out


def stratified_interval(values, participant, rng, stratify=True):
    """Percentile interval from resampling trials within participant.

    Returns both estimands and the flag saying whether stratification was
    possible at all.
    """
    codes = pd.factorize(participant)[0]        # participant labels are not always numeric
    order = np.argsort(codes, kind="stable")
    values = values[order]
    codes = codes[order]
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(values)]))
    bounds = list(zip(starts, ends))
    sizes = ends - starts
    n_total = len(values)

    degenerate = stratify and sizes.max() == 1
    if degenerate or not stratify:
        bounds = [(0, n_total)]           # one stratum: trials are the unit
        blocks = None
    else:
        blocks = bounds

    tw, pb = [], []
    for lo in range(0, B, CHUNK):
        size = min(CHUNK, B - lo)
        idx = _draw(bounds, n_total, size, rng)
        drawn = values[idx]
        tw.append(drawn.mean(1))
        if blocks is None:
            pb.append(drawn.mean(1))      # no participant structure to balance
        else:
            pb.append(np.stack([drawn[:, s:e].mean(1) for s, e in blocks], 1).mean(1))
    tw = np.concatenate(tw)
    pb = np.concatenate(pb)

    part_bounds = list(zip(starts, ends))
    sizes_all = np.array([e - s for s, e in part_bounds])
    means_all = np.array([values[s:e].mean() for s, e in part_bounds])
    per_part = np.array([values[s:e].mean() for s, e in
                         (blocks if blocks is not None else bounds)])
    pop = population_intervals(sizes_all, means_all, rng)
    return dict(
        n_trials=n_total, n_participants=len(sizes),
        min_trials=int(sizes.min()), max_trials=int(sizes.max()),
        est_trial=float(values.mean()),
        lo_trial=float(np.percentile(tw, 2.5)), hi_trial=float(np.percentile(tw, 97.5)),
        est_balanced=float(per_part.mean()),
        lo_balanced=float(np.percentile(pb, 2.5)),
        hi_balanced=float(np.percentile(pb, 97.5)),
        stratified=not degenerate and stratify,
        note="trials = participants; interval includes participant variation"
             if degenerate else "",
        **pop)


def main(dump_dir, out_path):
    rng = np.random.default_rng(11)
    rows = []
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
                good = cell[np.isfinite(cell[metric])]
                if good.empty:
                    continue
                r = stratified_interval(good[metric].to_numpy(float),
                                        good["Participant"].to_numpy(), rng)
                rows.append(dict(checkpoint=ck, dataset=dataset, snr=noise,
                                 metric=metric, **r))

    out = pd.DataFrame(rows)
    out["checkpoint"] = pd.Categorical(out.checkpoint, ORDER, ordered=True)
    out = out.sort_values(["checkpoint", "dataset", "snr", "metric"])
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", default="uncapped")
    p.add_argument("--out", default="docs/dataset_intervals.csv")
    a = p.parse_args()
    main(a.dumps, a.out)
