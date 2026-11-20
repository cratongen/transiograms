# MCP and MCRF

Markov-type categorical prediction (Li and Zhang) takes the nearest datum in each quadrant (2D)
or octant (3D) and treats them as conditionally independent given the category at the cell:

`P(Z(x0) = k | data) ∝ p_k prod_g t_{k, c_g}(x_g - x0)`.

`tg.mcp_predict` returns these probabilities at every cell; `tg.mcrf_simulate` simulates along a
seeded random path, drawing each cell from the same conditional over the hard data and the cells
already drawn. Both take `neighbours="sectors"` or a number of nearest neighbours.

```python
import numpy as np
import transiograms as tg

d = tg.datasets.layered_drillholes(seed=0)
grid, model = tg.Grid.from_json(d["grid"]), tg.TransitionModel.from_json(d["model"])
xyz = np.column_stack([d["X"], d["Y"], d["Z"]])
pred = tg.mcp_predict(model, grid, xyz, d["CODE"])
real = tg.mcrf_simulate(model, grid, xyz, d["CODE"], n=10, seed=0)
```

When every candidate category has zero likelihood the method drops the least close neighbour
and retries, instead of flooring probabilities at an epsilon.

## Theory checks

The suite checks that the MCP conditional equals the maximum-entropy joint table that honours
each pairwise transiogram (`tg.checks.mcp_brute_force`, 243 cells for three categories and four
neighbours) within `1e-10`, that two neighbours on a line give the Markov bridge
`t_ak(h1) t_kb(h2) / t_ab(h1 + h2)`, that probabilities sum to 1, and that hard cells get
probability exactly 1 and appear in every realization.

On a single column of cells the nearest cell above and the nearest below screen every other
cell of a Markov chain, so the conditional over the two vertical sectors is exact. There MCRF
reproduces the model within sampling error: over 20 realizations of 20 000 cells, the largest
transiogram error is 0.006 (2.8 standard errors) and the largest proportion error 0.0005
(`crates/mcrf/tests/limits.rs::one_column_sectors_simulation_reproduces_the_markov_chain`).

## Fit quality and the limit of the method

`tg.fit_quality` compares the realizations with the model: the proportions of every realization
and of the ensemble, and the grid transiograms at lags 1 to `n_lags` along each direction, with
a standard error from the spread across realizations. `max_standard_errors` above about 4 means
the ensemble misses the model by more than sampling error.

```python
q = tg.fit_quality(model, grid, real)
print(q)
print("ensemble:", q.mean_proportions.round(3), "target:", q.target_proportions.round(3))
```

```text
FitQuality(n=10, k=3, offsets=24, proportion_error=0.0418, transiogram_error=0.3529, max_standard_errors=87.9)
ensemble: [0.469 0.29  0.242] target: [0.5 0.3 0.2]
```

In 2D and 3D the neighbours of a cell are not independent given its category. Conditional
independence counts the information they share more than once, so the conditional is too
confident in the category its neighbours agree on. Realizations come out too continuous and
drift from the target proportions. The gap is a property of the method, not of this code, and
it is largest without hard data, where nothing ties the realization back to the proportions.
Unconditional simulation of the M5 model on a 48 x 48 x 32 grid of 3 x 3 x 0.5 m cells, 10
realizations at seed 1, lags 1 to 6 along x, +z and -z:

```python
pi_z = [[0.0, 0.6, 0.4], [0.7, 0.0, 0.3], [0.2105263158, 0.7894736842, 0.0]]
lengths = [[12.0, 8.0, 6.0], [12.0, 8.0, 6.0], [2.0, 1.0, 0.8]]
m5 = tg.TransitionModel.from_geology([0.5, 0.3, 0.2], lengths, embedded_z=pi_z, background=2)
g5 = tg.Grid((0.0, 0.0, 0.0), (3.0, 3.0, 0.5), (48, 48, 32))
axes = [(1, 0, 0), (0, 0, 1), (0, 0, -1)]
for nb in ("sectors", 4):
    r = tg.mcrf_simulate(m5, g5, np.empty((0, 3)), [], n=10, seed=1, neighbours=nb)
    q = tg.fit_quality(m5, g5, r, directions=axes)
    up = q.mean[6, 1, 1]
    print(nb, q.mean_proportions.round(3), round(q.transiogram_error, 3), round(q.max_standard_errors, 1), up.round(3))
print("model t_11(1) along +z:", q.model[6, 1, 1].round(3))
```

```text
sectors [0.462 0.314 0.225] 0.19 31.2 0.765
4 [0.486 0.297 0.217] 0.088 22.4 0.671
model t_11(1) along +z: 0.643
```

| Neighbourhood | Proportion error | Largest transiogram error | Standard errors |
| --- | --- | --- | --- |
| `"sectors"` (default) | 0.038 | 0.190 | 31.2 |
| `1` | 0.001 | 0.393 | 374 |
| `2` | 0.002 | 0.276 | 101 |
| `4` | 0.017 | 0.088 | 22.4 |
| `8` | 0.054 | 0.262 | 56.8 |

The last column is the ensemble `t_11(1)` along +z (offset index 6). With the default sectors it
is 0.765 against 0.643 in the model: category 1 comes out too continuous. No
neighbourhood reproduces the transiograms within sampling error on this grid; four nearest
neighbours come closest. The sectors and four-neighbour rows are pinned by
`crates/mcrf/tests/limits.rs`, which fails if a change moves them out of their band, and the
other rows were measured with the same grid and seed.

Hard data reduce the drift. With 16 holes on the 96 x 96 x 32 M6 grid the ensemble proportions
fall within 0.02 of the target (`crates/mcrf/tests/acceptance.rs::m6_acceptance`). On the
example data above, with 750 points on a 32 x 32 x 30 grid, the proportion error is still 0.042;
T-PROGS on the same data and seed stays within 0.001 (see [T-PROGS](tp-simulation.md)).
Check `tg.fit_quality` on your own grid before you rely on MCRF proportions or continuity.

Worked page: [MCP and MCRF](../examples/06-mcrf-mcp/01-mcp-and-mcrf/README.md).
