# boitata integration

transiograms imports nothing from boitata. You pass boitata tables and grids in as NumPy arrays and take
int64 codes and probability rows back out. The [guide](../guide/boitata.md) lists each exchange and
the [integration page](../examples/07-boitata-integration/01-drillholes-to-grid/README.md) runs the
whole path.

| Step | transiograms call | Input | Output |
| --- | --- | --- | --- |
| Segment logs | `HiddenMarkovModel.fit` and `predict` with `holes=`, `data=` | composite columns | int64 facies per composite |
| Vertical transiograms | `downhole_transiogram` with `from_=`, `to=` | facies intervals | `ExperimentalTransiogram` |
| Fit and assemble | `TransiogramModel.fit_markov`, `TransitionModel` | experimental curves, proportions | 3D model |
| Simulate | `tprogs_simulate`, `mcrf_simulate` | `Grid`, `(n, 3)` midpoints, codes | `Realizations` |
| Back to boitata | `Realizations.realizations[r]`, `probabilities` | | `(n_cells,)` int64, `(n_cells, k)` rows summing to 1 |

## M7 check

`python/tests/test_integration.py` runs this path on `tg.datasets.layered_drillholes(seed=0)`
with a seeded gamma-ray log:

- the HMM recovers at least 97 percent of the true facies and their proportions within 0.02;
- T-PROGS and MCRF return an int64 `(20, 30, 32, 32)` cube that matches every hard-data cell;
- for T-PROGS, the test first asserts that three standard errors of the ensemble proportion stay
  below 0.02, then checks the ensemble proportions within 0.02 of the target.

MCRF misses the 0.02 proportion bound on this grid (largest gap about 0.04 over 20
realizations), so the test checks MCRF proportions only on the larger M6 grid in
`crates/mcrf/tests/acceptance.rs`.
