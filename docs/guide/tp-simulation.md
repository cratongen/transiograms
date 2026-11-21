# T-PROGS simulation

Transition-probability cokriging (Carle and Fogg) estimates the probability of each category at
a cell from the nearest hard data, with the transition model in place of the covariances.
`tg.tp_cokriging` returns those probabilities without simulating; `tg.tprogs_simulate` draws
realizations by sequential indicator simulation and then quenches each one towards the model
transiograms.

```python
import numpy as np
import transiograms as tg

d = tg.datasets.layered_drillholes()
grid, model = tg.Grid.from_json(d["grid"]), tg.TransitionModel.from_json(d["model"])
xyz = np.column_stack([d["X"], d["Y"], d["Z"]])
real = tg.tprogs_simulate(model, grid, xyz, d["CODE"], n=20, seed=0)
real.realizations, real.proportions, real.probabilities
```

Hard data are points (composite midpoints); `Grid.snap` assigns each cell the point nearest its
centre. `mask` marks active cells. `max_data` (even) caps the data per cokriging system and
`closeness_limit` drops offsets whose transition matrix is close to the proportions. Quenching
runs up to `quench_iterations` zero-temperature sweeps over the lags with closeness above
`quench_limit` and stops when the objective gains less than `quench_tol`. Hard cells never change.

## Results

`Prediction` holds `probabilities` `(n_cells, k)`, `most_likely`, the normalised `entropy` and the
fraction of cells whose probabilities needed clipping. `Realizations` holds `realizations`
`(n, n_cells)` int64 with -1 outside the mask, per-realization `proportions`, ensemble
`probabilities`, and the quench `objective` per sweep. Cells run x fastest, so
`values.reshape(grid.shape)` indexes `[k, j, i]`.

## Memory

Simulation memory is sized from the available memory (`MemAvailable` on Linux, the available
physical memory on Windows, the installed memory on macOS), or from
`memory=` in bytes. A request that does not fit raises `tg.InvalidInput` naming the bytes it
needs before anything is allocated.

## Theory checks

The suite checks that a single datum gives exactly one row of `T(x0 - x1)`, that probabilities
sum to 1 within `1e-12`, that every realization honours every hard cell, that quenching never
raises its objective, and that results repeat for a seed. The Rust M5 test runs 20 realizations
and checks proportions within 0.02 and auto-transiograms within 0.05 RMSE. That RMSE bound is a
fixed tolerance, not a sampling bound, and it does not cover cross-transiograms; the next section
gives the measured gap.

## Fit quality and the limit of the method

`tg.fit_quality` compares the realizations with the model: the proportions of every realization
and of the ensemble, and the grid transiograms at lags 1 to `n_lags` along each direction, with
a standard error from the spread across realizations. `max_standard_errors` above about 4 means
the ensemble misses the model by more than sampling error.

Sequential indicator simulation from cokriging is an approximation: the indicator kriging
weights do not make the simulated field a Markov chain random field with the model
transiograms. The quench then fits only the lags it targets, those with closeness above
`quench_limit` (the adjacent cells with the default 0.4). Unconditional simulation of the M5
model on a 48 x 48 x 32 grid of 3 x 3 x 0.5 m cells, 10 realizations at seed 1, lags 1 to 6 along
x, +z and -z:

```python
pi_z = [[0.0, 0.6, 0.4], [0.7, 0.0, 0.3], [0.2105263158, 0.7894736842, 0.0]]
lengths = [[12.0, 8.0, 6.0], [12.0, 8.0, 6.0], [2.0, 1.0, 0.8]]
m5 = tg.TransitionModel.from_geology([0.5, 0.3, 0.2], lengths, embedded_z=pi_z, background=2)
g5 = tg.Grid((0.0, 0.0, 0.0), (3.0, 3.0, 0.5), (48, 48, 32))
r = tg.tprogs_simulate(m5, g5, np.empty((0, 3)), [], n=10, seed=1)
q = tg.fit_quality(m5, g5, r, directions=[(1, 0, 0), (0, 0, 1), (0, 0, -1)])
print(q)
gap = np.nanmax(np.abs(q.mean - q.model), axis=(1, 2))
print("largest gap per lag along x:", gap[:6].round(3))
```

```text
FitQuality(n=10, k=3, offsets=18, proportion_error=0.0000, transiogram_error=0.0260, max_standard_errors=15.4)
largest gap per lag along x: [0.    0.022 0.026 0.023 0.019 0.013]
```

The quench matches lag 1 and the proportions, but auto- and cross-transiograms miss the model by
up to 0.026 (13 to 15 standard errors on each axis) at lags 2 to 4. Minority categories come out
too continuous at mid lags: along x at lag 3, `t_22` is 0.350 against 0.324 and `t_20` 0.382
against 0.405; along +z at lag 3, `t_11` is 0.406 against 0.383. More quench sweeps change
nothing: each realization records two objective values, so the quench stops on `quench_tol`
long before `quench_iterations`. A lower `quench_limit` (13 lags at 0.1) gives 0.027. Without the quench
(`quench=False`) the gap is 0.075 (about 20 standard errors) and the proportions miss by 0.018.
`crates/tpsim/tests/limits.rs` pins these numbers and fails if a change moves them out of their
band. Check `tg.fit_quality` on your own grid before you rely on mid-lag transiograms.

Worked page: [T-PROGS](../examples/05-tp-simulation/01-tprogs/README.md).
