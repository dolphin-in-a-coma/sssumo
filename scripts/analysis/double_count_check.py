"""Does a trial-level bootstrap already carry within-trial noise?

theta_i(observed) = theta_i(true) + eps_i, so Var(observed across trials)
= Var(true) + Var(eps). A bootstrap over the observed per-trial values estimates
Var(observed)/n -- which already contains Var(eps)/n. Adding an explicit
sample-level resample on top should add it a SECOND time.

Parametrised so the two levels are comparable: n_eff = 10 independent blocks per
trial (1000 samples with an autocorrelation length of ~100), which is the regime
the real data is in.
"""
import numpy as np

rng = np.random.default_rng(0)
n_trials, n_eff = 200, 10          # 10 independent blocks per trial
tau, sigma = 0.05, 0.15            # between-trial SD, per-block SD

truth = rng.normal(0.9, tau, n_trials)
blocks = rng.normal(truth[:, None], sigma, (n_trials, n_eff))
observed = blocks.mean(1)

var_between_true = tau ** 2
var_within = sigma ** 2 / n_eff
analytic_se = np.sqrt(var_between_true + var_within) / np.sqrt(n_trials)

B = 4000
# (1) resample trials only
i = rng.integers(0, n_trials, (B, n_trials))
se_trial = observed[i].mean(1).std(ddof=1)

# (2) resample trials, then resample blocks within each drawn trial
picks = rng.integers(0, n_eff, (B, n_trials, n_eff))
drawn = blocks[i[:, :, None], picks]           # (B, n_trials, n_eff)
se_two = drawn.mean(2).mean(1).std(ddof=1)

print(f"  var between (true)      {var_between_true:.6f}")
print(f"  var within  (per trial) {var_within:.6f}   <- comparable, as intended")
print()
print(f"  analytic SE of the mean          {analytic_se:.5f}")
print(f"  (1) trial bootstrap only         {se_trial:.5f}")
print(f"  (2) trial + within-trial resample{se_two:.5f}")
print(f"      ratio (2)/(1)                {se_two / se_trial:.3f}")
print()
print(f"  expected ratio if double-counted "
      f"{np.sqrt(var_between_true + 2 * var_within) / np.sqrt(var_between_true + var_within):.3f}")
