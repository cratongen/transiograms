# Transiograms

A transiogram `t_ij(h) = P(Z(x + h) = j | Z(x) = i)` plays the role of the variogram for
categories. Diagonal curves (auto-transiograms) fall from 1 to the proportion `p_j`;
off-diagonal curves (cross-transiograms) rise from 0 to `p_j`.

## Experimental transiograms

| Function | Data |
| --- | --- |
| `tg.downhole_transiogram(codes, lags=, from_=, to=)` | intervals along holes: exact overlap lengths, no lag tolerance |
| `tg.experimental_transiogram(coords, codes, direction=, lag=, n_lags=)` | points: lag classes with angle and bandwidth tolerances, `pairs="same_hole"` or `"different_hole"` |
| `tg.grid_transiogram(codes, grid, offset=, n_lags=)` | a categorical grid along one cell offset |

Each returns an `ExperimentalTransiogram` with `lags`, `mean_lags`, `counts` and `probabilities`
as `(n_lags, k, k)` arrays. A direction is `(azimuth, dip)` in degrees; dip -90 points up.

## Models

`tg.TransiogramModel` models one direction:

- `TransiogramModel.markov(rates)`: `T(h) = exp(R h)`; `fit_markov(experimental, proportions=)`
  fits `R` by weighted least squares, optionally with fixed sills.
- `TransiogramModel.entrywise(proportions, structures, ranges)`: one exponential, spherical or
  Gaussian curve per entry, with a complement entry per row that closes it to 1 (Li 2007);
  `fit_entrywise` fits the ranges. Construction raises if any entry goes negative, naming it.

`tg.TransitionModel(proportions, rates_x, rates_y, rates_z, rotation=)` is the 3D model the grid
simulations use: one generator per principal axis, interpolated in between (Carle and Fogg), and
rotated by azimuth, dip and rake. `TransitionModel.from_geology(proportions, mean_lengths)` builds
it from mean lengths per axis, with maximum-entropy juxtapositions unless you give `embedded_*`.

## Fit quality

`TransiogramModel.residuals(experimental)` (Rust `Model1D::residuals`) returns experimental minus
model `t_ij` at every lag, `(n_lags, k, k)`: the misfit that `fit_markov` and `fit_entrywise`
minimise. A Markov model cannot follow a log whose transiogram oscillates, such as cyclic beds of
nearly constant thickness; the residuals show how far it falls short.

```python
import numpy as np
import transiograms as tg

d = tg.datasets.varved_sequence()
exp = tg.downhole_transiogram(d["CODE"], lags=[0.5, 1.0, 1.5, 2.0], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
fit = tg.TransiogramModel.fit_markov(exp)
print(f"varved: largest |residual| = {np.nanmax(np.abs(fit.residuals(exp))):.3f}")

rng = np.random.default_rng(7)
thickness = rng.integers(3, 6, size=20_000)
codes = np.repeat(np.arange(20_000) % 3, thickness)
cyclic = tg.downhole_transiogram(codes, lags=np.arange(1.0, 13.0), depth=np.arange(codes.size) + 0.5)
r = tg.TransiogramModel.fit_markov(cyclic).residuals(cyclic)
print(f"cyclic: largest |residual| = {np.nanmax(np.abs(r)):.3f} at lag {cyclic.lags[np.nanargmax(np.abs(r).max(axis=(1, 2)))]:.0f}")
```

```text
varved: largest |residual| = 0.046
cyclic: largest |residual| = 0.449 at lag 8
```

The varved logs come from a Markov chain, so their residuals are sampling noise. The cyclic log
(beds 0, 1, 2 in turn, 3 to 5 cells thick) is not Markov: the best Markov fit misses it by 0.45.
The seeded Rust test `markov_fit_residuals_quantify_a_cyclic_log` pins that gap between 0.40 and
0.50. The entrywise curves rise monotonically too, so neither family follows such a log; where
the residuals stay large, treat the model as an approximation at those lags.

## Theory checks

At 200 lags up to 20 mean lengths and at `10^4` mean lengths, every model has non-negative
entries, rows summing to 1 within `1e-12`, `T(0) = I` and `T(h) -> 1 p^T`. The suite also checks
the reversal identity `p_i t_ij(h) = p_j t_ji(-h)`, `det T(h) = exp(h tr R)`, the mean-length
ellipsoid, the downhole estimator against exact interval overlaps, the point estimator against the
grid estimator, and that `fit_markov` recovers a known `R` within 5 percent.

Worked pages: [transiograms](../examples/04-transiograms/01-transiograms/README.md) and
[entrywise transiograms and closeness](../examples/04-transiograms/02-entrywise-and-closeness/README.md).
