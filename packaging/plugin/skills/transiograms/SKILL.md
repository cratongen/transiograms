---
name: transiograms
description: How to write correct code with transiograms (`import transiograms as tg`): Markov chains, hidden Markov models, transiograms and categorical simulation. Use whenever code imports transiograms, or the user asks for facies, lithology or categorical simulation in Python and transiograms is installed or wanted.
---

# transiograms

transiograms is imported as `tg`. Install it with `pip install transiograms` (Python >= 3.11).
`pip install "transiograms[plot]"` adds matplotlib for `tg.plot`.

```python
import transiograms as tg
```

## Look it up before you write it

Do not write a transiograms call from memory. The installed package carries its own reference, and it matches the
installed version:

```sh
python -c "import transiograms, pathlib; print(transiograms.__version__, pathlib.Path(transiograms.__file__).with_name('_transiograms.pyi'))"
```

- Search that `_transiograms.pyi` stub for the class or function to get its signature.
- `python -c "import transiograms as tg; help(tg.MarkovChain.fit)"` prints the NumPy-style docstring, with every
  argument and its default.
- `tg.__all__` lists the public names. `tg.datasets`, `tg.plot` and `tg.checks` are modules and are kept off
  that list. `dir(tg.datasets)` and `dir(tg.plot)` list the datasets and the plots.
- The docs at https://cratongen.github.io/transiograms/ hold a worked page for each method, starting at
  https://cratongen.github.io/transiograms/guide/markov-chains/ and the API at
  https://cratongen.github.io/transiograms/api/core/.

If a name is not in `tg.__all__` and it is not `tg.datasets`, `tg.plot` or `tg.checks`, it does not exist. Say so
and offer the closest name that does.

## Kinds of object

Ask which kind you hold, and the calls follow.

| Kind | Examples | You call | Saves as |
| --- | --- | --- | --- |
| Value | `Categories`, `Grid` | properties, `to_json` | JSON |
| Sequence model | `MarkovChain`, `Sojourn` | `fit`, `simulate`, `sojourn` | JSON |
| Hidden model | `HiddenMarkovModel`, `HiddenSemiMarkovModel`, `Gaussian`, `Categorical`, `Poisson`, `GaussianMixture`, `Geometric`, `ShiftedPoisson`, `NegativeBinomial`, `EmpiricalDuration` | `fit`, `predict`, `sample` | JSON |
| Spatial model | `ExperimentalTransiogram`, `TransiogramModel`, `TransitionModel` | `downhole_transiogram`, `experimental_transiogram`, `fit_markov`, `from_geology` | JSON |
| Grid result | `Prediction`, `Realizations`, `FitQuality` | `tprogs_simulate`, `mcrf_simulate`, `mcp_predict`, `tp_cokriging`, `fit_quality` | JSON |
| Test | `TestResult` | `independence_test`, `order_test`, `homogeneity_test` | JSON |

One run through every kind:

```python
import numpy as np
import transiograms as tg

intervals = tg.datasets.varved_sequence(seed=0)
names = tg.Categories(intervals["names"])
chain = tg.MarkovChain().fit("CODE", holes="HOLE_ID", from_="FROM", to="TO", data=intervals)
print(chain.transition_matrix.round(2))
print(chain.independence_test().name)
beds = chain.simulate(10.0, n=2, seed=0)
print(beds["code"][:4])
stay = tg.sojourn("CODE", holes="HOLE_ID", from_="FROM", to="TO", data=intervals)
print(stay.mean_length.round(2))
order = tg.order_test("CODE", order=2, holes="HOLE_ID", from_="FROM", to="TO", data=intervals)
print(order.name, order.df)

logs = tg.datasets.electrofacies_logs(seed=0)
keep = np.isin(logs["HOLE_ID"], ["W01", "W02"])
obs = np.column_stack([logs["GR"][keep], logs["RHOB"][keep]])
hmm = tg.HiddenMarkovModel(3, n_init=1, seed=0, max_iter=30, direction="down", ordering="mean")
hmm.fit(obs, holes=logs["HOLE_ID"][keep])
states = hmm.predict(obs, holes=logs["HOLE_ID"][keep])
hsmm = tg.HiddenSemiMarkovModel.from_hmm(hmm, max_duration=6)
print(states.shape, hsmm.n_states)

exp = tg.downhole_transiogram(
    "CODE", lags=[0.5, 1.0, 2.0], holes="HOLE_ID", from_="FROM", to="TO", data=intervals
)
vertical = tg.TransiogramModel.fit_markov(exp, proportions=chain.proportions)
grid = tg.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 2.0), (4, 4, 3))
model = tg.TransitionModel.from_geology(
    chain.proportions, [[8.0, 6.0, 4.0], [8.0, 6.0, 4.0], [2.0, 1.5, 1.0]]
)
xyz = np.array([[15.0, 15.0, 3.0]])
hard = np.array([0])
real = tg.tprogs_simulate(model, grid, xyz, hard, n=2, seed=0)
mcrf = tg.mcrf_simulate(model, grid, xyz, hard, n=2, seed=0)
curves = tg.grid_transiogram(mcrf.realizations[0], grid, offset=(1, 0, 0), n_lags=2)
quality = tg.fit_quality(model, grid, mcrf, n_lags=2)
pred = tg.mcp_predict(model, grid, xyz, hard)
fig, ax = tg.plot.transition_matrix(chain, categories=names)
print(vertical.kind, real.realizations.shape, curves.lags.shape, pred.probabilities.shape)
print(round(quality.proportion_error, 4))
print(tg.MarkovChain.from_json(chain.to_json()).n_categories, grid.shape)
```

## Rules that hold everywhere

- **Options go on the constructor, data to `fit`, targets to `predict`, `simulate` or `sample`.** `fit` stores the
  result on the object and returns it, so calls chain.
- **Options are keyword-only.** `tg.MarkovChain("embedded", "down")` raises `TypeError`. Write
  `tg.MarkovChain("embedded", direction="down")`.
- **A string names a column of `data`.** `fit("CODE", holes="HOLE_ID", from_="FROM", to="TO", data=table)`.
- **Codes are int64, with -1 missing, or float with NaN.** Grid results use the same -1 outside the mask.
- **A grid array is in cell order, x fastest.** `values.reshape(grid.shape)` has shape `(nz, ny, nx)`.
- **`Grid` origin is the minimum corner of cell (0, 0, 0).** `count` is `(nx, ny, nz)`. The linear index is
  `ix + nx * (iy + ny * iz)`. A `Grid` has no rotation.
- **`TransitionModel.rotation` is `(azimuth, dip, rake)` in degrees.** Azimuth is clockwise from north. Dip 0
  leaves the principal +z' axis pointing up. `experimental_transiogram(..., direction=(azimuth, dip))` uses dip
  positive downward. Vertical up is dip -90.
- **`direction="up"` is the default on sequence models.** Depths in `data` run top-down, and `"up"` reads each
  hole from its base. `"down"` reads from the top.
- **Simulations take an integer `seed`.** The same seed returns the same arrays on any thread count.
- **These are not scikit-learn estimators.** They have no `get_params` and do not drop into a scikit-learn
  `Pipeline`.

## Missing values

Integer codes use -1. Float codes use NaN. `Categories.encode` writes -1 for `None` and for NaN.
`Categories.decode` writes `None` for -1. A gap larger than `gap` (default `1e-6`) splits a hole, and no
transition crosses that split. Cells outside `mask`, and cells a simulation did not fill, are -1.

## Errors

`tg.TransiogramsError` is the base. `tg.InvalidInput` is also a `ValueError`. `tg.ConvergenceError` is also a
`RuntimeError`. Read the message before changing the call. It names the argument at fault. transiograms reads
no files, so it has no file-error type.

## Saving

Parameters and results go to JSON: `chain.to_json()` and `tg.MarkovChain.from_json(text)`, and the same pair
on every Rust type in `tg.__all__`. pickle uses that JSON. There is no Parquet or CSV writer. Pass NumPy
arrays, or a mapping of column names to arrays.

## Data

`tg.datasets` builds every dataset from a seed, offline. Use one instead of inventing numbers.

| Function | What it returns |
| --- | --- |
| `varved_sequence(seed=0)` | 12 holes of sand, silt and clay: `HOLE_ID`, `FROM`, `TO`, `CODE`, `names`, `generator` |
| `electrofacies_logs(seed=0)` | 20 holes from a 3-state Gaussian HMM: `HOLE_ID`, `DEPTH`, `GR`, `RHOB`, `STATE`, `model` |
| `layered_drillholes(seed=0)` | 25 vertical holes through a 3D field, plus `field`, `model` and `grid` JSON |

`layered_drillholes` simulates a 32 x 32 x 30 grid before it returns. Use `varved_sequence` or
`electrofacies_logs` when you only need a log.

## Write it fast

Correct transiograms code still does the slow thing when it loops in Python around a call that already takes
the whole input. Name the slow path, then use the call that avoids it.

### One call for every hole

`MarkovChain.fit`, `sojourn`, `order_test`, `select_order`, `homogeneity_test`, `downhole_transiogram`,
`experimental_transiogram`, and the `fit`, `predict`, `score` and `sample` methods of `HiddenMarkovModel` and
`HiddenSemiMarkovModel`, plus `select_states`, take every hole in one call.

The slow path is a Python loop over hole ids that slices the columns and calls `fit` on each slice. `fit`
replaces what the object stored, so the loop keeps the last hole. Pass the full columns once:

```python
chain = tg.MarkovChain().fit("CODE", holes="HOLE_ID", from_="FROM", to="TO", data=intervals)
```

`select_states` compares state counts in that same one call. Do not loop `HiddenMarkovModel` over `n_states`
yourself when the user asked which count to keep.

### One call for every realization

`MarkovChain.simulate`, `HiddenMarkovModel.sample`, `HiddenSemiMarkovModel.sample`, `tprogs_simulate` and
`mcrf_simulate` take `n` and `seed`. Realization `r` of that call is drawn from its own stream, keyed by
`seed` and `r`.

The slow path is a Python loop of `n=1`. Each pass builds the neighbour table again, and the same `seed`
repeats realization 0. One call draws them all:

```python
real = tg.tprogs_simulate(model, grid, xyz, hard, n=20, seed=0)
mcrf = tg.mcrf_simulate(model, grid, xyz, hard, n=20, seed=0)
```

### The memory budget on a simulation grid

`tp_cokriging`, `tprogs_simulate`, `mcp_predict` and `mcrf_simulate` take `memory=` in bytes. Leave it at the
default unless the user set a budget. The default is the available memory: `MemAvailable` on Linux, the
available physical memory on Windows, the installed memory on macOS. A request that does not fit raises
`tg.InvalidInput` with the byte count before it allocates. Pass the user's number as `memory=` when they set
one. Do not pre-build the realization array in Python and do not invent a smaller budget.

### The shared neighbour table in T-PROGS and MCRF

`tprogs_simulate` builds one offset table and every realization searches that table. `mcrf_simulate` builds
one sector search and one transition table the same way. Both tables are built inside the single call with
`n=`. The loop of `n=1` above is what builds them again.

The other slow path is a Python walk of `grid.coords` that calls `TransitionModel.transition` once per cell.
Pass the `Grid` and the `TransitionModel` to `tprogs_simulate`, `mcrf_simulate`, `tp_cokriging` or
`mcp_predict`. When you do need the matrices, pass every lag in one call: `model.transition(lags)`.

Quenching is the `quench=True` default of `tprogs_simulate`. Leave it there. Do not rewrite the sweeps in
Python.

### Do not promise a speed

Give a time only after you ran the code and timed it. Otherwise say it was not measured. This repository
publishes no benchmark number.

## Hand it to a sibling

transiograms is one of a family of standalone libraries for mining and geoscience at https://github.com/cratongen.
None imports another. They exchange objects through the `__minetypes__` protocol, Arrow (`__arrow_c_stream__`) for
tables, and NumPy arrays.

| Library | Import | What it does |
| --- | --- | --- |
| boitata | `import boitata as bt` | Geostatistics: drill holes, compositing, block models, variograms, kriging, simulation |
| segyrs | `import segyrs` | Read, write and slice SEG-Y files, including files larger than memory |
| transiograms | `import transiograms as tg` | Markov chains, hidden Markov models, transiograms and categorical simulation |
| wireframes | `import wireframes as wf` | Solids, surfaces and strings: read, write, measure and combine |

<!-- Below this line: what transiograms hands to each sibling and takes from it, with one short example per neighbour.
Only calls that exist today in both libraries, checked against ../<sibling>/python/<pkg>/*.pyi. -->

### boitata

Sequence functions (`MarkovChain.fit`, `downhole_transiogram`, `sojourn`, `order_test`, `select_order`,
`homogeneity_test`) and hidden Markov models accept `drillholes`, `intervals` and `composites` protocol objects
directly: hole indices, `from_depth` and `to_depth` are read automatically without column plumbing. Spatial
functions (`experimental_transiogram`, `tp_cokriging`, `tprogs_simulate`, `mcp_predict`, `mcrf_simulate`,
`grid_transiogram`, `fit_quality`) accept `grid` and `block_model` targets and `point_set` and `composites` data.

`tg.Categories`, `tg.Grid` and `tg.Realizations` implement `__minetypes__()`. Category codes cross the protocol
boundary as `int32` with `-1` for missing values (`Realizations.__minetypes__()` produces `kind="grid"` with
`int32` realization columns `realization_0 .. realization_{n-1}`).

```python
import numpy as np
import transiograms as tg

intervals = tg.datasets.varved_sequence(seed=0)
chain = tg.MarkovChain().fit("CODE", holes="HOLE_ID", from_="FROM", to="TO", data=intervals)
grid = tg.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 2.0), (4, 4, 3))
model = tg.TransitionModel.from_geology(
    chain.proportions, [[8.0, 6.0, 4.0], [8.0, 6.0, 4.0], [2.0, 1.5, 1.0]]
)
real = tg.mcrf_simulate(
    model, grid, np.empty((0, 3)), np.empty(0, dtype=np.int64), n=2, seed=0
)
proto = real.__minetypes__()
assert proto["kind"] == "grid"
assert proto["columns"]["realization_0"].dtype == np.int32
```

### segyrs

`segyrs.create` writes a SEG-Y file from a sample array, a sample interval in microseconds and a sample
format. transiograms has no trace header and no SEG-Y reader. There is no handoff.

### wireframes

wireframes reads and writes `Mesh`, `Polylines` and `PointSet`. transiograms has no mesh type and no
function that takes one. There is no handoff.

## Before you report a result

Run the code. Then check what the method promises.

- A chain: the rows of `transition_matrix` sum to 1. Use `independence_test` or `symmetry_test` when the
  question is about the matrix, and `order_test` when the question is about the order.
- A simulation: `fit_quality` for the proportions and the transiograms. Hard cells from `grid.snap(xyz, hard)`
  equal `realizations[:, cells]` in every realization.
- An HMM or HSMM: `score`, and the agreement of `predict` with the labels when the user has labels.

`tg.checks` holds the NumPy oracles the tests compare against: `quasi_independence`, `kemeny_snell_std`,
`markov_bridge_marginal`, `brute_force_hmm`, `brute_force_hsmm` and `mcp_brute_force`. Use one for a small
case. They enumerate paths, so keep the case small. Report the numbers you got, in the units of the data.
