"""Pulse profiles from every arm, and how the two learned arms converge on their truth.

Profiles come from ContinuousPrimitive itself rather than a re-derivation, so what is
drawn is what the model actually renders.
"""
import ast, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from sssumo.models import ContinuousPrimitive

DUR = 60             # a real submovement duration: the training range is 5..60


def profile(family, **kw):
    """Unit-area pulse on normalised time, rendered at a duration of DUR samples.

    Two traps, both of which produced a wrong figure before this comment existed:

    `duration_range` defaults to (4, 30) and profile() renders onto a grid of
    duration_range[1] samples, so asking for a longer duration without widening the
    range silently returns a *clamped* pulse -- only the rising half of every family.

    And the second element of every parameter pair is a slope *multiplied by duration*.
    The learned arms have slopes of ~1e-3, which is negligible inside the 5..60 range
    the shape was fit for and is emphatically not negligible outside it: rendering at
    duration 400 inflates a learned beta_precision of 5.777 to 7.28, a 26% distortion
    that reads as the model overshooting its target when it has not.
    """
    p = ContinuousPrimitive(family=family, duration_range=(4, DUR), **kw)
    with torch.no_grad():
        v = p.profile(torch.tensor([[[float(DUR)]]])).squeeze().cpu().numpy()
    s = np.linspace(0, 1, len(v))
    return s, v * len(v)          # density on normalised time (unit area)


FIXED = {
    "min-jerk":  ("beta",  dict(beta_mean=(0.5, 0.), beta_precision=(6., 0.))),
    "Gaussian":  ("gaussian", dict(gaussian_centre=(0.5, 0.), gaussian_half_width=(2.5, 0.))),
    "Beta-asym": ("beta",  dict(beta_mean=(0.40, 0.), beta_precision=(6., 0.))),
    "LGNB":      ("lgnb",  dict(lgnb_mu=(-0.40, 0.), lgnb_sigma=(0.8, 0.))),
}
COL = {"min-jerk": "#2B3A67", "Gaussian": "#0B6E63",
       "Beta-asym": "#B5651D", "LGNB": "#8E3B7C"}


def trace(path, family):
    """Per-epoch primitive parameters, in log order."""
    out = []
    for line in open(path):
        m = re.match(rf"Primitive\[{family}\]: (\{{.*\}})", line.strip())
        if m:
            out.append(ast.literal_eval(m.group(1)))
    return out


BETA = trace("runs/0901-pulse-families/results/train_log_beta_learned.txt", "beta")
LGNB = trace("runs/0902-learn-lgnb/config-0901-family_lgnb_learned.txt", "lgnb")
print(f"epochs traced: beta {len(BETA)}, lgnb {len(LGNB)}")

ARMS = [
    dict(name="beta_learned", family="beta", tr=BETA, key="beta_mean",
         start=0.5, truth=0.40,
         mk=lambda d: dict(beta_mean=tuple(d["beta_mean"]), beta_precision=tuple(d["beta_precision"])),
         truth_kw=dict(beta_mean=(0.40, 0.), beta_precision=(6., 0.)),
         start_kw=dict(beta_mean=(0.50, 0.), beta_precision=(6., 0.))),
    dict(name="lgnb_learned", family="lgnb", tr=LGNB, key="lgnb_mu",
         start=0.0, truth=-0.40,
         mk=lambda d: dict(lgnb_mu=tuple(d["lgnb_mu"]), lgnb_sigma=tuple(d["lgnb_sigma"])),
         truth_kw=dict(lgnb_mu=(-0.40, 0.), lgnb_sigma=(0.8, 0.)),
         start_kw=dict(lgnb_mu=(0.0, 0.), lgnb_sigma=(0.8, 0.))),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 8.5, "figure.dpi": 110,
})
fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2))

# --- A: the four fixed families -------------------------------------------
a = ax[0, 0]
for name, (fam, kw) in FIXED.items():
    s, v = profile(fam, **kw)
    a.plot(s, v, lw=2.1, color=COL[name], label=name)
    a.axvline(s[int(np.argmax(v))], color=COL[name], lw=.8, ls=":", alpha=.55)
a.set_title("A · The four fixed pulse families", loc="left", fontweight="bold")
a.set_xlabel("normalised time within a submovement")
a.set_ylabel("velocity (unit area)")
a.legend(frameon=False, loc="upper right")
a.text(.015, .965, "dotted = peak position", transform=a.transAxes,
       fontsize=8, color="#666", va="top")

# --- B: convergence, as fraction of the gap closed -------------------------
b = ax[0, 1]
for arm, mark in zip(ARMS, ["o", "s"]):
    vals = [d[arm["key"]][0] for d in arm["tr"]]
    frac = [(arm["start"] - v) / (arm["start"] - arm["truth"]) for v in vals]
    c = COL["Beta-asym"] if arm["family"] == "beta" else COL["LGNB"]
    b.plot(range(len(frac)), frac, marker=mark, ms=4.2, lw=1.7, color=c,
           label=f'{arm["name"]}  ({arm["key"]})')
b.axhline(1.0, color="#444", lw=1, ls="--")
b.text(17.4, 1.005, "truth", fontsize=8.5, color="#444", ha="right", va="bottom")
b.axvspan(-.4, 4.4, color="#999", alpha=.13, lw=0)
b.text(2.0, .5, "primitive frozen:\nreconstruction loss\noff until epoch 5",
       fontsize=8.5, color="#555", ha="center", va="center")
b.set_ylim(-.08, 1.12)
b.set_title("B · Both arms close ~97% of the gap, in one epoch", loc="left", fontweight="bold")
b.set_xlabel("epoch"); b.set_ylabel("fraction of the gap to truth closed")
b.legend(frameon=False, loc="lower right")

# --- C, D: the profile trajectory per arm ---------------------------------
for j, arm in enumerate(ARMS):
    p = ax[1, j]
    cmap = cm.get_cmap("viridis")
    n = len(arm["tr"])
    post = []
    for i, d in enumerate(arm["tr"]):
        s, v = profile(arm["family"], **arm["mk"](d))
        p.plot(s, v, lw=1.15, color=cmap(i / max(n - 1, 1)), alpha=.85, zorder=2)
        if i >= 8:
            post.append(v)
    s, tv = profile(arm["family"], **arm["truth_kw"])
    s, sv = profile(arm["family"], **arm["start_kw"])
    p.plot(s, sv, lw=1.7, ls=":", color="#B00020", zorder=4, label="start (wrong shape)")
    p.plot(s, tv, lw=2.2, ls="--", color="#111", zorder=5, label="generator truth")
    post = np.array(post)
    p.fill_between(s, post.min(0), post.max(0), color="#0B6E63", alpha=.18, lw=0,
                   zorder=1, label="converged envelope (epochs 8–17)")
    p.set_title(f'{"C" if j == 0 else "D"} · {arm["name"]}: profile per epoch',
                loc="left", fontweight="bold")
    p.set_xlabel("normalised time"); p.set_ylabel("velocity (unit area)")
    p.legend(frameon=False, loc="upper right")
    sm = cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, n - 1))
    cb = fig.colorbar(sm, ax=p, pad=.015, fraction=.036)
    cb.set_label("epoch", fontsize=8.5); cb.ax.tick_params(labelsize=8)

fig.suptitle("SSSUMO pulse profiles: four fixed families, and two learned arms recovering their truth\n"
             "rendered at a submovement duration of 60 samples, the top of the 5-60 training range",
             fontsize=13, fontweight="bold", x=.008, ha="left", y=.988)
fig.tight_layout(rect=[0, 0, 1, .965])
out = "runs/0902-family-rerun/results/pulse_profiles.png"
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", out)

# residual spread, printed so the figure's envelope has a number beside it
for arm in ARMS:
    s, tv = profile(arm["family"], **arm["truth_kw"])
    res = [np.abs(profile(arm["family"], **arm["mk"](d))[1] - tv).max()
           for d in arm["tr"][8:]]
    print(f'{arm["name"]:14s} max |profile - truth| over epochs 8-17: '
          f'{min(res):.4f} to {max(res):.4f} (peak density ~{tv.max():.2f})')
