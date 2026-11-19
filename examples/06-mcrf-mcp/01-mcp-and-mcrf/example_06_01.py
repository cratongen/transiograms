"""
# MCP and MCRF

Markov-type categorical prediction (Li and Zhang) treats the nearest datum in each octant as
conditionally independent given the facies at the cell: the probability of facies `k` is
`p_k prod_g t_{k, c_g}(h_g)`, normalised. A Markov chain random field simulates with the same
conditional along a random path, adding the cells it has already drawn.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import matplotlib.pyplot as plt
import numpy as np
import varvite as vv
from common import save

d = vv.datasets.layered_drillholes(seed=0)
names = vv.Categories(["mud", "sand", "gravel"], colors=["#8c8c8c", "#e69f00", "#0072b2"])
grid = vv.Grid.from_json(d["grid"])
model = vv.TransitionModel.from_json(d["model"])
xyz = np.column_stack([d["X"], d["Y"], d["Z"]])

# %% [markdown]
# The MCP conditional equals the maximum-entropy joint distribution of the cell and its four
# neighbours that honours each pairwise transiogram. `vv.checks.mcp_brute_force` builds that
# 243-cell table by iterative proportional fitting:

# %%
small = vv.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (21, 21, 1))
x0 = np.array([105.0, 105.0, 0.5])
near = x0 + np.array([[30.0, 10.0, 0.0], [-20.0, 40.0, 0.0], [-30.0, -20.0, 0.0], [10.0, -50.0, 0.0]])
codes = [0, 2, 1, 2]
cell = int(np.flatnonzero(np.all(small.coords == x0, axis=1))[0])
mcp = vv.mcp_predict(model, small, near, codes, closeness_limit=1e-6).probabilities[cell]
brute = vv.checks.mcp_brute_force(model.proportions, [model.transition(x - x0) for x in near], codes)
print("MCP:        ", mcp.round(6))
print("brute force:", brute.round(6))

# %% [markdown]
# MCP on the full grid: probabilities sum to one at every cell and equal one on the hard data's
# facies at conditioning cells.

# %%
pred = vv.mcp_predict(model, grid, xyz, d["CODE"])
cells, hard = grid.snap(xyz, d["CODE"])
print("rows sum to 1:", bool(np.allclose(pred.probabilities.sum(axis=1), 1.0, atol=1e-12)))
print("hard cells certain:", bool(np.all(pred.probabilities[cells, hard] == 1.0)))

# %% [markdown]
# Five MCRF realizations against the field the holes were drilled through.

# %%
real = vv.mcrf_simulate(model, grid, xyz, d["CODE"], n=5, seed=0)
print("hard data honoured in every realization:", bool(np.all(real.realizations[:, cells] == hard)))
print("proportions per realization:")
print(real.proportions.round(3))
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), layout="constrained")
panels = [(d["field"], "True field"), (pred.most_likely, "MCP most likely")]
panels += [(real.realizations[r], f"MCRF realization {r + 1}") for r in range(4)]
for ax, (values, title) in zip(axes.flat, panels):
    vv.plot.grid_slice(values, grid, axis="y", index=15, categories=names, ax=ax)
    ax.set_title(f"{title}, y = 155 m")
save(fig, "sections")
