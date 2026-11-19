"""
# Choosing a duration law

Each state of a hidden semi-Markov model needs a law for its thickness. `Geometric` is the law an
HMM implies, `EmpiricalDuration` puts free mass on every thickness up to `max_duration`, and the
parametric laws sit in between. Here the truth mixes a geometric state with an empirical one,
and `aic`, `bic` and `icl` weigh each fitted law against its parameter count. A brute-force check
over every state path of a short log closes the page.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import matplotlib.pyplot as plt
import numpy as np
import transiograms as tg
from common import ACCENT, GRAY, HIGHLIGHT, save

truth = tg.HiddenSemiMarkovModel(
    [tg.Gaussian([0.0], 1.0), tg.Gaussian([3.0], 1.0)],
    [tg.Geometric(0.25), tg.EmpiricalDuration([0.0, 0.1, 0.2, 0.4, 0.2, 0.1])],
    start=[0.5, 0.5],
    transition=[[0.0, 1.0], [1.0, 0.0]],
    max_duration=[30, 6],
    direction="down",
)
_, obs = truth.sample([300] * 6, seed=1)
print(len(obs), "logs of", obs[0].shape[0], "steps")

# %% [markdown]
# The same six logs fitted with each duration family. `score_samples` gives the log-likelihood of
# each log, and its sum is `score`. The empirical law spends one free parameter per thickness, so
# AIC, BIC and ICL show whether those parameters earn their keep.

# %%
fits = {}
for family in ("geometric", "poisson", "negative_binomial", "empirical"):
    fits[family] = tg.HiddenSemiMarkovModel(
        2, duration=family, max_duration=30, seed=0, direction="down", ordering="mean"
    ).fit(obs)
for family, m in fits.items():
    per_log = m.score_samples(obs)
    assert np.isclose(per_log.sum(), m.score(obs))
    print(
        f"{family:>17}: log L {m.score(obs):8.1f}  AIC {m.aic(obs):8.1f}  "
        f"BIC {m.bic(obs):8.1f}  ICL {m.icl(obs):8.1f}"
    )
print("per-log log L (empirical):", np.round(fits["empirical"].score_samples(obs), 1))

# %% [markdown]
# The empirical family reaches the highest log-likelihood, but it pays for 29 free thickness
# probabilities per state, so BIC and ICL rank it last. The negative binomial,
# two parameters per state, has the lowest AIC, BIC and ICL. In the figure the geometric family
# fits state 0 and misses the peaked thickness of state 1; the empirical family follows both shapes.

# %%
u = np.arange(1, 13)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained", sharey=True)
for j, ax in enumerate(axes):
    ax.bar(u, truth.durations[j].pmf(30)[:12], color=GRAY, label="true")
    ax.plot(u, fits["empirical"].durations[j].pmf(30)[:12], color=ACCENT, marker="o", ms=3, label="empirical")
    ax.plot(u, fits["geometric"].durations[j].pmf(30)[:12], color=HIGHLIGHT, label="geometric")
    ax.set(xlabel="thickness (steps)", title=f"state {j}")
axes[0].set_ylabel("probability")
axes[0].legend()
save(fig, "laws")

# %% [markdown]
# Theory check: on eight steps there are 2^8 = 256 state paths. `tg.checks.brute_force_hsmm`
# scores every one of them, with runs longer than `max_duration` excluded and the last run
# right-censored, and must agree with the forward-backward and Viterbi recursions.

# %%
x = obs[0][:8]
d = [law.pmf(int(m)) for law, m in zip(truth.durations, truth.max_duration)]
with np.errstate(divide="ignore"):
    log_d = [np.log(p) for p in d]
    log_survivor = [np.log(np.cumsum(p[::-1])[::-1]) for p in d]
log_b = np.column_stack([g.log_density(x) for g in truth.distributions])
lnl, best, _, gamma, _ = tg.checks.brute_force_hsmm(truth.start, truth.transition, log_d, log_survivor, log_b)
path, _ = truth.viterbi(x)
print("log-likelihood gap:", abs(truth.score(x) - lnl))
print("same Viterbi path:", np.array_equal(path, best))
print("largest posterior gap:", np.abs(truth.predict_proba(x) - gamma).max())
