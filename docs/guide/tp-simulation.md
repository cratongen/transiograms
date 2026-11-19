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

Simulation memory is sized from the available memory (`MemAvailable` on Linux), or from
`memory=` in bytes. A request that does not fit raises `tg.InvalidInput` naming the bytes it
needs before anything is allocated.

## Theory checks

The suite checks that a single datum gives exactly one row of `T(x0 - x1)`, that probabilities
sum to 1 within `1e-12`, that every realization honours every hard cell, that quenching never
raises its objective, and that results repeat for a seed. The Rust M5 test runs 20 realizations
and checks proportions within 0.02 and auto-transiograms within 0.05 RMSE.

Worked page: [T-PROGS](../examples/05-tp-simulation/01-tprogs/README.md).
