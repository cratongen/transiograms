# Working with boitata

transiograms is one of a family of standalone libraries for mining and geoscience: boitata,
segyrs, transiograms and wireframes. None depends on another; they exchange objects through the
`__minetypes__` protocol and NumPy arrays.

Sequence functions (`MarkovChain.fit`, `downhole_transiogram`, `sojourn`, `order_test`,
`select_order`, `homogeneity_test`) and hidden Markov models accept `drillholes`, `intervals` and
`composites` protocol objects directly (hole index, `from_depth` and `to_depth` are read
automatically). Spatial functions (`experimental_transiogram`, `tp_cokriging`, `tprogs_simulate`,
`mcp_predict`, `mcrf_simulate`, `grid_transiogram`, `fit_quality`) accept `grid` and `block_model`
targets and `point_set` and `composites` data. `tg.Categories`, `tg.Grid` and `tg.Realizations`
implement `__minetypes__()`, with category codes crossing the protocol boundary as `int32`.

With boitata the exchanges are:

| Exchange | boitata | transiograms |
| --- | --- | --- |
| Category tables | `Categories` (`__minetypes__` kind `"categories"`) | `tg.Categories(cats)` or `categories=cats` |
| Intervals | `Drillholes`, `intervals`, `composites` protocol objects | `MarkovChain().fit(dh)` or `fit("CODE", data=dh)` |
| Points | `PointSet` or `composites` protocol objects, or `(n, 3)` midpoints | `coords` of the grid functions |
| Codes | `int32` protocol codes (`-1` missing), or float with NaN | `int32` at the protocol boundary; `int64` with -1 or float with NaN |
| Grids | `BlockModel(origin, size, count)`, zero rotation | `tg.Grid(block_model)` or `grid=block_model`, cells x fastest |
| Realizations | block-model `with_column` or protocol consumer | `Realizations.__minetypes__()` (`int32` columns) or `Realizations.realizations[r]` |
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
