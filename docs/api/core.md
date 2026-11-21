# Core

`tg.Categories(names, *, colors=None)` holds category names in code order. `encode(labels)` returns
int64 codes with -1 for missing labels; `decode(codes)` returns the names, None for -1.
`Categories.from_labels(labels)` sorts the unique labels, numerically when every label is an integer.

`tg.Grid(origin, size, count)` is an axis-aligned grid whose origin is the corner of the first cell.
Cells run x fastest, then y, then z, so `values.reshape(grid.shape)` indexes `[k, j, i]`.
`grid.coords` gives the centroids and `grid.snap(coords, codes, *, mask=None)` assigns hard data to
cells: the point nearest to a cell's centroid wins.

Every error derives from `tg.TransiogramsError`. `tg.InvalidInput` is also a `ValueError` and
`tg.ConvergenceError` a `RuntimeError`.

::: transiograms.Categories

::: transiograms.Grid

::: transiograms.TransiogramsError

::: transiograms.InvalidInput

::: transiograms.ConvergenceError
