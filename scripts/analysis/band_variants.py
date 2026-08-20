"""Recompute score bands under each methodological choice, from per-trial dumps.

No GPU: everything is derived from the per-trial CSVs written by
`scripts/colab/score_checkpoints.py --dump-per-trial`. Run from a directory
holding `pt/` (or point the globs at the dumps in Drive).

Covers the choices that change conclusions: estimand (per-trial mean vs metric
recomputed over the pooled signal, reconstructed exactly from sufficient
statistics), central tendency (median vs mean), and dataset grouping/balancing.
See docs/VALIDATION.md for what each one did to the answer.
"""
import glob, json, os
import numpy as np, pandas as pd

RNG = np.random.default_rng(7)
S = 2000
ORDER = ["pre_released","pre_rerun","pre_rerun_s1000",
         "ft_released","ft_A","ft_B","ft_A2","ft_B2"]

def ci(vals, conf=.95):
    lo, hi = np.percentile(vals, [(1-conf)/2*100, (1+conf)/2*100])
    return float(np.mean(vals)), float(lo), float(hi)

# ---------- synthetic: two estimands, same i.i.d. trial bootstrap ----------
def pooled_r2(n, sy, sy2, ssr, idx):
    N = n[idx].sum(1)
    return 1 - ssr[idx].sum(1) / (sy2[idx].sum(1) - sy[idx].sum(1)**2 / N)

rows = []
for path in sorted(glob.glob("pt/synthetic.*.csv")):
    ck = os.path.basename(path).split(".")[1]
    d = pd.read_csv(path)
    for (noise, refr), g in [(("all","all"), d)] + list(d.groupby(["Noise","Refractory"])):
        group = "all" if noise == "all" else f"snr{noise}_refr{refr}"
        per = g["Reconstruction_R2"].to_numpy(float)
        n_, sy, sy2, ssr = (g[c].to_numpy(float) for c in ("n_points","sum_y","sum_y2","ss_res"))
        idx = RNG.integers(0, len(g), size=(S, len(g)))
        m, lo, hi = ci(per[idx].mean(1))
        rows.append(dict(checkpoint=ck, domain="synthetic", group=group,
                         variant="per-trial mean (mine)", mean=m, lower=lo, upper=hi, n=len(g)))
        m, lo, hi = ci(pooled_r2(n_, sy, sy2, ssr, idx))
        rows.append(dict(checkpoint=ck, domain="synthetic", group=group,
                         variant="pooled recompute (paper)", mean=m, lower=lo, upper=hi, n=len(g)))

# ---------- organic: faithful re-implementation of hierarchical_bootstrap_metrics ----------
def hierarchical(frame, metric, balance_datasets, central, n_sims=S):
    """Counter.__missing__ gives unsampled participants weight 0, as pandas .map does."""
    datasets = sorted(frame.Dataset.unique())
    per_ds = {ds: frame[frame.Dataset == ds].reset_index(drop=True) for ds in datasets}
    max_rows = max(len(v) for v in per_ds.values())
    out = np.empty(n_sims)
    for s in range(n_sims):
        pool = []
        for ds, dd in per_ds.items():
            parts = dd.Participant.unique()
            drawn = RNG.choice(parts, size=len(parts), replace=True)
            counts = pd.Series(drawn).value_counts()
            w = dd.Participant.map(counts).fillna(0).to_numpy(float)
            if w.sum() == 0:
                continue
            p = w / w.sum()
            k = max_rows if balance_datasets else len(dd)
            pool.append(RNG.choice(dd[metric].to_numpy(float), size=k, replace=True, p=p))
        vals = np.concatenate(pool)
        out[s] = np.median(vals) if central == "median" else np.mean(vals)
    return ci(out)

for path in sorted(glob.glob("pt/organic.*.csv")):
    ck = os.path.basename(path).split(".")[1]
    d = pd.read_csv(path)
    d = d[d.Dataset != "synthetic"]
    for noise, gn in d.groupby("Noise_Condition"):
        # (C) mine: per dataset, mean, unbalanced
        for ds, g in gn.groupby("Dataset"):
            m, lo, hi = hierarchical(g, "Reconstruction_R2", False, "mean")
            rows.append(dict(checkpoint=ck, domain="organic", group=f"{ds}/snr{noise}",
                             variant="per dataset, mean (mine)", mean=m, lower=lo, upper=hi,
                             n=g.Participant.nunique()))
        # (D) paper: datasets pooled, balanced, median   (E) same but mean
        for central, name in (("median","pooled+balanced, median (paper)"),
                              ("mean","pooled+balanced, mean")):
            m, lo, hi = hierarchical(gn, "Reconstruction_R2", True, central)
            rows.append(dict(checkpoint=ck, domain="organic", group=f"ALL/snr{noise}",
                             variant=name, mean=m, lower=lo, upper=hi,
                             n=gn.Participant.nunique()))

out = pd.DataFrame(rows)
out.to_csv("variants.csv", index=False)
print("rows:", len(out), "| variants:", sorted(out.variant.unique()))
