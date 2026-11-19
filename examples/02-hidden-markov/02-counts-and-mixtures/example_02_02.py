"""
# Counts, mixtures and model choice

Gaussian emissions fit continuous logs. Two other families cover the rest: a Poisson emission for
counts per sample (fractures or veinlets per metre), and a Gaussian mixture for a facies whose
readings have two or more modes. This page fits both on seeded synthetic holes with known
parameters, scores each hole, picks the number of states with AIC, BIC and ICL, and saves a model
to JSON.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
import transiograms as tg
from common import ACCENT, GRAY, HIGHLIGHT, save

# %% [markdown]
# ## Poisson counts
#
# A two-state generator: sound rock with 1.5 fractures per metre and fractured zones with 8. Five
# holes of 300 one-metre samples each, all from seed 0.

# %%
truth = tg.HiddenMarkovModel(
    [tg.Poisson([1.5]), tg.Poisson([8.0])],
    start=[0.7, 0.3],
    transition=[[0.95, 0.05], [0.10, 0.90]],
)
true_states, counts = truth.sample([300] * 5, seed=0)
print("first counts of hole 0:", counts[0][:12, 0].astype(int))

# %% [markdown]
# `emission="poisson"` fits one rate per state. `ordering="mean"` puts the sound state first.

# %%
poisson = tg.HiddenMarkovModel(2, emission="poisson", n_init=3, seed=0, ordering="mean")
poisson.fit(counts)
print("rates:", [round(float(p.rate[0]), 2) for p in poisson.distributions])
print(poisson.transition.round(3))
decoded = poisson.predict(counts)
agree = np.mean(np.concatenate(decoded) == np.concatenate(true_states))
print(f"Viterbi agrees with the true states on {agree:.1%} of samples")

# %% [markdown]
# `score_samples` returns the log-likelihood of each hole; the holes are independent, so they sum
# to `score`. A hole with a low value per sample fits the model worst.

# %%
per_hole = poisson.score_samples(counts)
print("per hole:", per_hole.round(1))
print("per sample:", (per_hole / [len(c) for c in counts]).round(3))
print("sum minus score:", abs(per_hole.sum() - poisson.score(counts)))

# %% [markdown]
# Theory check: `Poisson.log_density` equals the closed form `k log(rate) - rate - log(k!)`.

# %%
rate = float(poisson.distributions[1].rate[0])
k = np.arange(15.0)
closed = np.array([v * math.log(rate) - rate - math.lgamma(v + 1) for v in k])
print("largest gap:", np.max(np.abs(poisson.distributions[1].log_density(k[:, None]) - closed)))

# %% [markdown]
# ## A bimodal facies
#
# The second generator has a clean sand with one GR mode and a heterolithic facies whose readings
# split between a sandy mode at 75 API and a shaly mode at 125 API. A single Gaussian per state
# would put the heterolithic mean between the modes, where almost no readings fall. Every state in a
# model shares one emission family, so the sand is a mixture with a single component.

# %%
heterolithic = tg.GaussianMixture([0.5, 0.5], [[75.0], [125.0]], [64.0, 64.0], covariance_type="spherical")
generator = tg.HiddenMarkovModel(
    [tg.GaussianMixture([1.0], [[20.0]], [25.0], covariance_type="spherical"), heterolithic],
    start=[0.5, 0.5],
    transition=[[0.97, 0.03], [0.04, 0.96]],
)
mix_states, gr = generator.sample([400] * 4, seed=1)

mixture = tg.HiddenMarkovModel(
    2, emission="mixture", n_mix=2, covariance_type="spherical", n_init=3, seed=0, ordering="mean"
)
mixture.fit(gr)
for j, m in enumerate(mixture.distributions):
    print(
        f"state {j}: weights {m.weights.round(4)}, means {m.means.ravel().round(1)}, variances {m.covariances.round(2)}"
    )

# %% [markdown]
# The heterolithic state recovers both modes and their equal weights. The sand state needs one
# component, so EM shrinks its spare one onto a single reading at 60.5 API, which
# shows as the narrow spike in the figure below. `covariance_prior` keeps that variance above zero.

# %% [markdown]
# Theory check: `GaussianMixture.log_density` equals the log of the weighted sum of normal
# densities, computed here by hand.

# %%
x = np.linspace(0.0, 170.0, 9)
w, mu, var = heterolithic.weights, heterolithic.means.ravel(), heterolithic.covariances
by_hand = np.log(
    sum(w[i] * np.exp(-((x - mu[i]) ** 2) / (2 * var[i])) / np.sqrt(2 * np.pi * var[i]) for i in range(2))
)
print("largest gap:", np.max(np.abs(heterolithic.log_density(x) - by_hand)))

# %% [markdown]
# ## How many states
#
# Fit one to four states with two mixture components each and compare the criteria (lower is
# better). ICL adds the entropy of the state paths to BIC, so it penalises states that overlap.

# %%
rows = []
for n_states in range(1, 5):
    m = tg.HiddenMarkovModel(
        n_states, emission="mixture", n_mix=2, covariance_type="spherical", n_init=3, seed=0
    ).fit(gr)
    rows.append((n_states, m.aic(gr), m.bic(gr), m.icl(gr)))
    print(f"{n_states} states: AIC {rows[-1][1]:9.1f}  BIC {rows[-1][2]:9.1f}  ICL {rows[-1][3]:9.1f}")
best = {
    name: rows[int(np.argmin([r[i] for r in rows]))][0] for i, name in enumerate(["AIC", "BIC", "ICL"], 1)
}
print("lowest:", best)

# %% [markdown]
# All three criteria pick two states, the number in the generator. The left panel overlays the
# fitted emission of each state on the readings of its true state.

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
grid = np.linspace(0.0, 170.0, 400)
flat = np.concatenate(gr).ravel()
labels = np.concatenate(mix_states)
for j, colour in enumerate([GRAY, ACCENT]):
    a.hist(flat[labels == j], bins=60, range=(0, 170), density=True, alpha=0.45, color=colour)
    a.plot(grid, np.exp(mixture.distributions[j].log_density(grid)), color=colour, lw=1.5)
a.set(xlabel="GR (API)", ylabel="density", title="Fitted emission per state")
n = [r[0] for r in rows]
for i, (name, colour) in enumerate([("AIC", GRAY), ("BIC", ACCENT), ("ICL", HIGHLIGHT)], 1):
    b.plot(n, [r[i] for r in rows], marker="o", label=name, color=colour)
b.set(xlabel="states", ylabel="criterion", title="Model choice", xticks=n)
b.legend()
save(fig, "mixtures")

# %% [markdown]
# ## Saving a model
#
# `to_json` writes the fitted parameters; `from_json` rebuilds a model that compares equal and
# scores the same data to the same value.

# %%
text = poisson.to_json()
again = tg.HiddenMarkovModel.from_json(text)
print(
    len(text),
    "characters; equal:",
    again == poisson,
    "; score gap:",
    abs(again.score(counts) - poisson.score(counts)),
)
