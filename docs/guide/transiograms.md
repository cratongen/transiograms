# Transiograms

A transiogram `t_ij(h) = P(Z(x + h) = j | Z(x) = i)` plays the role of the variogram for
categories. Diagonal curves (auto-transiograms) fall from 1 to the proportion `p_j`;
off-diagonal curves (cross-transiograms) rise from 0 to `p_j`.

## Experimental transiograms

| Function | Data |
| --- | --- |
| `vv.downhole_transiogram(codes, lags=, from_=, to=)` | intervals along holes: exact overlap lengths, no lag tolerance |
| `vv.experimental_transiogram(coords, codes, direction=, lag=, n_lags=)` | points: lag classes with angle and bandwidth tolerances, `pairs="same_hole"` or `"different_hole"` |
| `vv.grid_transiogram(codes, grid, offset=, n_lags=)` | a categorical grid along one cell offset |

Each returns an `ExperimentalTransiogram` with `lags`, `mean_lags`, `counts` and `probabilities`
as `(n_lags, k, k)` arrays. A direction is `(azimuth, dip)` in degrees; dip -90 points up.

## Models

`vv.TransiogramModel` models one direction:

- `TransiogramModel.markov(rates)`: `T(h) = exp(R h)`; `fit_markov(experimental, proportions=)`
  fits `R` by weighted least squares, optionally with fixed sills.
- `TransiogramModel.entrywise(proportions, structures, ranges)`: one exponential, spherical or
  Gaussian curve per entry, with a complement entry per row that closes it to 1 (Li 2007);
  `fit_entrywise` fits the ranges. Construction raises if any entry goes negative, naming it.

`vv.TransitionModel(proportions, rates_x, rates_y, rates_z, rotation=)` is the 3D model the grid
simulations use: one generator per principal axis, interpolated in between (Carle and Fogg), and
rotated by azimuth, dip and rake. `TransitionModel.from_geology(proportions, mean_lengths)` builds
it from mean lengths per axis, with maximum-entropy juxtapositions unless you give `embedded_*`.

## Theory checks

At 200 lags up to 20 mean lengths and at `10^4` mean lengths, every model has non-negative
entries, rows summing to 1 within `1e-12`, `T(0) = I` and `T(h) -> 1 p^T`. The suite also checks
the reversal identity `p_i t_ij(h) = p_j t_ji(-h)`, `det T(h) = exp(h tr R)`, the mean-length
ellipsoid, the downhole estimator against exact interval overlaps, the point estimator against the
grid estimator, and that `fit_markov` recovers a known `R` within 5 percent.

Worked pages: [transiograms](../examples/04-transiograms/01-transiograms/README.md) and
[entrywise transiograms and closeness](../examples/04-transiograms/02-entrywise-and-closeness/README.md).
