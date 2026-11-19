# Working with boitata

varvite has no dependency on boitata. The two libraries exchange NumPy arrays:

| Exchange | boitata | varvite |
| --- | --- | --- |
| Category tables | `Categories` scheme JSON | `vv.Categories.from_json` (names and colors) |
| Intervals | composite table `HOLE_ID`, `FROM`, `TO`, codes | `holes=`, `from_=`, `to=`, `codes`, with `data=` |
| Points | composite midpoints `(n, 3)` | `coords` of the grid functions |
| Codes | `Categories.encode` output, float with NaN | int64 with -1, or float with NaN |
| Grids | `Geometry(origin, size, count)`, zero rotation | `vv.Grid(origin, size, count)`, cells x fastest |
| Realizations | block-model `with_column` | `Realizations.realizations[r]`, int64 per cell |
| Probabilities | `SIS(proportions=)`, kriging by domain | `probabilities`, `(n_cells, k)` rows summing to 1 |

A rotated boitata block model works through coordinates in its local frame; anisotropy lives in
`TransitionModel.rotation`. Hard data are points: pass interval midpoints, which boitata's
`composite` produces.

Seeds follow boitata: `seed=s`, realization `i` uses the same `realization_seed(s, i)` key in
both libraries.

The [integration page](../examples/07-boitata-integration/01-drillholes-to-grid/README.md) runs
the whole path on varvite's seeded data: a gamma-ray log segmented by an HMM, vertical
transiograms, a 3D model, T-PROGS and MCRF realizations, and the block-model columns that go back.
