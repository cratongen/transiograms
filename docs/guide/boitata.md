# Working with boitata

transiograms is one of a family of standalone libraries for mining and geoscience: boitata,
segyrs, transiograms and wireframes. None depends on another; they exchange NumPy arrays.
With boitata the arrays are:

| Exchange | boitata | transiograms |
| --- | --- | --- |
| Category tables | `Categories.names`, `Categories.colors` | `tg.Categories(names, colors=colors)` |
| Intervals | composite table `HOLE_ID`, `FROM`, `TO`, codes | `holes=`, `from_=`, `to=`, `codes`, with `data=` |
| Points | composite midpoints `(n, 3)` | `coords` of the grid functions |
| Codes | `Categories.encode` output, float with NaN | int64 with -1, or float with NaN |
| Grids | `BlockModel(origin, size, count)`, zero rotation | `tg.Grid(origin, size, count)`, cells x fastest |
| Realizations | block-model `with_column` | `Realizations.realizations[r]`, int64 per cell |
| Probabilities | `SIS(proportions=)`, kriging by domain | `probabilities`, `(n_cells, k)` rows summing to 1 |

Pass names and colors into the constructor. boitata's category JSON also carries `mapping` and
`other`, and `tg.Categories.from_json` rejects those keys.

A rotated boitata block model works through coordinates in its local frame; anisotropy lives in
`TransitionModel.rotation`. Hard data are points: pass interval midpoints, which boitata's
`composite` produces.

Seeds follow boitata: `seed=s`, realization `i` uses the same `realization_seed(s, i)` key in
both libraries.

The [integration page](../examples/07-boitata-integration/01-drillholes-to-grid/README.md) runs
the whole path on transiograms's seeded data: a gamma-ray log segmented by an HMM, vertical
transiograms, a 3D model, T-PROGS and MCRF realizations, and the block-model columns that go back.
