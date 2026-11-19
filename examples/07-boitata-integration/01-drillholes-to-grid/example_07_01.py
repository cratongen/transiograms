"""
# From boitata drillholes to categorical grids

boitata desurveys and composites drillholes; transiograms segments the logs into facies, fits
transiograms and simulates facies on a grid; the realizations go back to boitata as block-model
columns for domain-controlled kriging. The two libraries share no code: they meet at NumPy arrays.

This page runs on transiograms's own seeded data, so it needs no boitata install. Each step names the
boitata call whose output or input the arrays stand for.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %% [markdown]
# ## Drillholes in
#
# A boitata composite table converts to a plain mapping with
# `{c: table[c] for c in table.column_names}`: hole ids, `FROM` and `TO` in depth, midpoint
# coordinates and a gamma-ray log. Here the mapping comes from `tg.datasets.layered_drillholes`,
# with a gamma-ray reading drawn from each composite's facies.

# %%
import matplotlib.pyplot as plt
import numpy as np
import transiograms as tg
from common import ACCENT, GRAY, save

source = tg.datasets.layered_drillholes(seed=0)
rng = np.random.default_rng(7)
composites = {
    "HOLE_ID": source["HOLE_ID"],
    "FROM": source["FROM"],
    "TO": source["TO"],
    "X": source["X"],
    "Y": source["Y"],
    "Z": source["Z"],
    "GR": np.array([30.0, 75.0, 120.0])[source["CODE"]] + rng.normal(0.0, 12.0, len(source["CODE"])),
}
print({k: v.dtype for k, v in composites.items()})

# %% [markdown]
# ## Lithology from the log
#
# A three-state Gaussian HMM decodes facies from gamma ray, hole by hole and top-down. The
# decoded codes play the role of logged lithology.

# %%
hmm = tg.HiddenMarkovModel(3, n_init=5, seed=0, direction="down", ordering="mean")
hmm.fit("GR", holes="HOLE_ID", data=composites)
composites["FACIES"] = hmm.predict("GR", holes="HOLE_ID", data=composites)
agreement = np.mean(composites["FACIES"] == source["CODE"])
print(f"decoded facies agree with the truth at {agreement:.1%} of composites")

# %% [markdown]
# ## Transiograms and the 3D model
#
# Vertical transiograms from the intervals, a Markov model with sills at the composite
# proportions, and lateral mean lengths ten times the vertical ones.

# %%
p = np.bincount(composites["FACIES"], minlength=3) / len(composites["FACIES"])
exp = tg.downhole_transiogram(
    "FACIES", holes="HOLE_ID", from_="FROM", to="TO", lags=np.arange(1.0, 9.0), data=composites
)
vertical = tg.TransiogramModel.fit_markov(exp, proportions=p)
model = tg.TransitionModel(p, 0.1 * vertical.rates, 0.1 * vertical.rates, vertical.rates)
print(model)

# %% [markdown]
# ## Categorical grids out
#
# `tg.Grid(origin, size, count)` is a boitata `Geometry` with zero rotation, cells x fastest.
# T-PROGS and MCRF both take the composite midpoints as hard data.

# %%
grid = tg.Grid.from_json(source["grid"])
xyz = np.column_stack([composites["X"], composites["Y"], composites["Z"]])
tprogs = tg.tprogs_simulate(model, grid, xyz, "FACIES", n=10, seed=0, data=composites)
mcrf = tg.mcrf_simulate(model, grid, xyz, "FACIES", n=10, seed=0, data=composites)
cells, codes = grid.snap(xyz, composites["FACIES"])
for name, r in (("T-PROGS", tprogs), ("MCRF", mcrf)):
    honoured = np.all(r.realizations[:, cells] == codes)
    gap = np.abs(r.proportions.mean(axis=0) - p).max()
    print(
        f"{name}: shape {r.realizations.shape}, hard data honoured {honoured}, largest proportion gap {gap:.3f}"
    )
print("realization as a 3D array:", tprogs.realizations[0].reshape(grid.shape).shape)

# %% [markdown]
# T-PROGS quenching keeps the ensemble proportions within 0.02 of the target, as the M7 check
# requires. MCRF reproduces the hard data and the transiograms but drifts further from the target
# proportions on this small grid.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
names = tg.Categories(["mud", "sand", "gravel"], colors=["#8c8c8c", "#e69f00", "#0072b2"])
tg.plot.grid_slice(source["field"], grid, axis="y", index=15, categories=names, ax=axes[0])
axes[0].set_title("True field, y = 155 m")
tg.plot.grid_slice(tprogs.realizations[0], grid, axis="y", index=15, categories=names, ax=axes[1])
axes[1].set_title("T-PROGS realization 1")
width = 0.35
axes[2].bar(np.arange(3) - width / 2, tprogs.proportions.mean(axis=0), width, color=ACCENT, label="T-PROGS")
axes[2].bar(np.arange(3) + width / 2, mcrf.proportions.mean(axis=0), width, color=GRAY, label="MCRF")
axes[2].scatter(np.arange(3), p, color="black", zorder=3, label="target")
axes[2].set_xticks(range(3), names.names)
axes[2].set(ylabel="proportion", title="Ensemble proportions")
axes[2].legend()
save(fig, "grids")

# %% [markdown]
# ## Back to boitata
#
# The arrays below are what a boitata block model takes: `with_column` accepts one int64 code per
# cell (-1 inactive), and `SIS(proportions=)` or kriging by domain accepts `(n_cells, k)`
# probabilities whose rows sum to one.

# %%
columns = {f"FACIES_{r + 1:02d}": tprogs.realizations[r] for r in range(len(tprogs))}
columns |= {f"P_{n.upper()}": tprogs.probabilities[:, j] for j, n in enumerate(names.names)}
columns["FACIES_MOST_LIKELY"] = tprogs.most_likely
print({k: (v.dtype.name, v.shape) for k, v in list(columns.items())[-4:]})
print("probability rows sum to one:", bool(np.allclose(tprogs.probabilities.sum(axis=1), 1.0)))
