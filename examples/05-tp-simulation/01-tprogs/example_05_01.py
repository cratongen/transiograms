"""
# T-PROGS simulation

Transition-probability cokriging estimates the probability of each facies at a cell from the
nearest hard data, with the transition model in place of covariances. T-PROGS simulates cells
along a random path from those probabilities, then quenches each realization towards the model
transiograms. Hard cells never change.
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
from common import ACCENT, GRAY, save

d = tg.datasets.layered_drillholes(seed=0)
names = tg.Categories(["mud", "sand", "gravel"], colors=["#8c8c8c", "#e69f00", "#0072b2"])
grid = tg.Grid.from_json(d["grid"])
model = tg.TransitionModel.from_json(d["model"])
xyz = np.column_stack([d["X"], d["Y"], d["Z"]])
print(grid, len(d["CODE"]), "composites")

# %% [markdown]
# Cokriged probabilities and the most likely facies on a horizontal slice at 15.5 m, through the
# 25 holes.

# %%
pred = tg.tp_cokriging(model, grid, xyz, d["CODE"])
print(f"clipped and renormalised at {pred.corrected:.1%} of cells")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
tg.plot.grid_slice(pred.most_likely, grid, index=15, categories=names, ax=axes[0])
axes[0].plot(d["X"], d["Y"], "k.", ms=3)
axes[0].set_title("Most likely facies")
tg.plot.grid_slice(pred.probabilities[:, 2], grid, index=15, ax=axes[1], vmin=0, vmax=1)
axes[1].set_title("P(gravel)")
tg.plot.grid_slice(pred.entropy, grid, index=15, ax=axes[2], vmin=0, vmax=1)
axes[2].set_title("Normalised entropy")
save(fig, "cokriging")

# %% [markdown]
# Ten realizations. Each takes its own seeded path and draws, so the result does not depend on
# the number of threads.

# %%
real = tg.tprogs_simulate(model, grid, xyz, d["CODE"], n=10, seed=0)
print(f"{len(real)} realizations of {grid.n_cells} cells")
cells, codes = grid.snap(xyz, d["CODE"])
print("hard data honoured in every realization:", bool(np.all(real.realizations[:, cells] == codes)))
print("ensemble proportions:", real.proportions.mean(axis=0).round(3), "model:", model.proportions)
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), layout="constrained")
for r, ax in enumerate(axes[0]):
    tg.plot.grid_slice(real.realizations[r], grid, index=15, categories=names, ax=ax)
    ax.set_title(f"Realization {r + 1}, z = 15.5 m")
for r, ax in enumerate(axes[1]):
    tg.plot.grid_slice(real.realizations[r], grid, axis="y", index=15, categories=names, ax=ax)
    ax.set_title(f"Realization {r + 1}, section y = 155 m")
save(fig, "realizations")

# %% [markdown]
# Quenching never raises its objective, and the vertical transiograms of the realizations
# scatter around the model.

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
for o in real.objective:
    a.plot(o, marker="o", color=ACCENT, lw=0.8)
a.set(xlabel="sweep", ylabel="objective", title="Quenching")
lags = np.arange(1, 9)
for r in real.realizations:
    e = tg.grid_transiogram(r, grid, offset=(0, 0, 1), n_lags=8)
    b.plot(lags, e.probabilities[:, 1, 1], color=GRAY, lw=0.8)
b.plot(lags, model.transition(np.outer(lags, [0.0, 0.0, 1.0]))[:, 1, 1], color=ACCENT, lw=2, label="model")
b.set(xlabel="vertical lag (m)", ylabel="t_sand,sand", title="Sand auto-transiogram")
b.legend()
save(fig, "reproduction")
