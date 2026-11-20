# transiograms-transiogram: algorithms

Signatures are in [spec.md](spec.md) section 4.5. Categories 0..K-1, proportions p, D = diag(p).
`t_ij(h) = P(Z(x + h) = j | Z(x) = i)`: tail at x, head at x + h, lag = head - tail.

## 1. Properties every model satisfies

`t_ij(h) >= 0`; `T(h) 1 = 1`; `T(0) = I`; `T(h) -> 1 p^T`; `p^T T(h) = p^T`; reversal identity
for any stationary field `p_i t_ij(h) = p_j t_ji(-h)`, i.e. `T(-h) = D^-1 T(h)^T D`.
Indicator cross-covariance `C_ij(h) = p_i (t_ij(h) - p_j)`.

## 2. Experimental transiograms

### 2.1 Points (count estimator, Li and Zhang 2019 Eq. 22, GAMEAS conventions)
Direction vector from (azimuth a clockwise from north, dip d positive downward):
`u = (sin a cos d, cos a cos d, -sin d)`. Vertical up is dip -90.

A pair (tail x, head y), `s = y - x`, is in lag class l (l = 1..n_lags, nominal `l * lag`) when:
- `| |s| - l lag | <= lag_tolerance`;
- directional: `s . u > 0` (directed) and the angle between s and u `<= angle_tolerance`;
  the horizontal distance of s from the u axis `<= bandwidth` and the vertical distance
  `<= vertical_bandwidth` (GSLIB gamv definitions);
- omnidirectional (`direction = None`): distance only;
- `pairs`: SameHole or DifferentHole filter on hole ids (requires `holes`).
Directed counts each ordered pair once with the tail at the origin. Undirected or
omnidirectional counts each unordered pair in both orientations, which forces
`p_i t_ij = p_j t_ji` and removes asymmetry; `Experimental.directed` records the mode.

`F_ij(l)` = pairs with tail code i, head code j. `t_ij(l) = F_ij / sum_k F_ik` (rows sum to 1
exactly; NaN rows when the tail total is 0). `mean_lags` = mean |s| per class. Never use the
`sum i_k i_k' / (pi_k N(h))` form (rows do not sum to 1).

Pair search: uniform hash grid with cell side `n_lags * lag + lag_tolerance`; tails processed in
fixed blocks of 1024 indices in parallel; per-block partial counts reduced in block order.

### 2.2 Downhole exact estimator (intervals)
For each run of `SequenceSet::runs(direction, gap)` and each lag h > 0:
- numerator `M_ij(h) = sum over intervals a (code i) and b (code j) of
  |[a0, a1] intersect [b0 - h, b1 - h] intersect [r0, r1 - h]|`, where `[r0, r1]` is the run;
- denominator `N_i(h) = sum_j M_ij(h)`.
Sum over runs before dividing. `counts` hold lengths; `mean_lags = lags`. Two-pointer sweep per
run (intervals sorted by position). This is exact at any h, including lags below the sample
length.

### 2.3 Grid estimator
Codes on a grid, offset `e` (integer vector), lags `l = 1..n_lags`: all ordered pairs
`(x, x + l e)` inside the grid with both ends active and non-missing. `lags = l |e * size|`.

## 3. Markov models (Carle and Fogg 1997)

`T(h) = exp(h R)` for h >= 0 with R a generator satisfying `p^T R = 0`; `T(-h) = D^-1 T(h)^T D`.
`R` diagonal `-1 / L_i`, `r_ij = pi_ij / L_i`, slope at 0 equals R, `L_i ~ Delta / (1 - t_ii(Delta))`
at the first lag. Compute with `RateMatrix::transition` (uniformization). Never use Sylvester's
spectral formula (fails on repeated eigenvalues; the single-rate model hits it).

### 3.1 fit_markov (weighted least squares)
Parameters: `ln r_ij` for i, j != q, i != j, and `ln(-r_ii)` for i != q, so (K - 1)^2 values;
row and column q come from the background fill (core.md section 6) with the given proportions
(default: stationary of the start). Objective
`sum_l sum_i N_i(l) sum_j (t_hat_ij(l) - t_ij(h_l))^2` over non-NaN rows; an invalid fill gives
`+inf`. Start: `start` if given, else `RateMatrix::from_transition(T_hat(first lag), lag)`
projected and refilled. `nelder_mead(step = 0.2, tol = 1e-10, max_eval = 2000 (K-1)^2)`; reaching `max_eval` before `tol`
raises `ConvergenceError`.

## 4. Entrywise models (Li 2007)

Structures `f(r)`: exponential `1 - exp(-3r)`, spherical `1.5r - 0.5r^3` for r < 1 else 1,
Gaussian `1 - exp(-3r^2)`. For each row i with complement column `c_i` (default i):
- cross entry j != i: `t_ij(h) = (n_ij + (1 - n_ij) f(h / a_ij)) p_j` for h > 0 (nugget n);
- auto entry (when not the complement): `t_ii(h) = 1 - (1 - p_i)(n_ii + (1 - n_ii) f(h / a_ii))`;
- complement: `1 - sum` of the others.
`T(0) = I`. Validity: evaluate at 1000 lags uniform on `(0, max_lag]` plus `10^4 max_lag`;
any entry below `-1e-15` raises `InvalidInput` naming `(i, j, h)`; values in `[-1e-15, 0)` clip
to 0 and the row renormalises. Negative h uses the reversal identity.
`fit_entrywise`: each non-complement entry independently, range by golden section on `ln a` in
`[ln(0.01 lag_1), ln(100 lag_max)]` minimising `sum_l N_i(l) (t_hat - t)^2`, nugget 0.

## 5. Three-dimensional TransitionModel

### 5.1 Rotation
Angles in degrees. Azimuth a turns the horizontal axes clockwise seen from above:
`x' = (cos a, -sin a, 0)`, `y' = (sin a, cos a, 0)`, `z' = (0, 0, 1)`.
Dip d turns about x', plunging y': `y'' = cos d y' - sin d z'`, `z'' = sin d y' + cos d z'`.
Rake r turns about y'': `x''' = cos r x' - sin r z''`, `z''' = sin r x' + cos r z''`.
The principal axes are `(x''', y'', z''')`; `to_principal(h) = (h . x''', h . y'', h . z''')`.
Rotation (0, 0, 0) is the identity.

### 5.2 Interpolation along a lag (T-PROGS Eq. 20, spMC)
For `h != 0`, principal components `h'`, `c_v = |h'_v| / |h|`. For each axis v use
`R_v` when `h'_v >= 0`, else `R_{-v} = D^-1 R_v^T D`. For i, j != q (diagonals included):
`|r_ij(phi)| = sqrt(sum_v (c_v r^(v)_ij)^2)`, sign negative on the diagonal, non-negative off it.
Then fill row and column q by the background fill. Validate (negative fill raises). Along a
principal axis this returns that axis's matrix exactly. Mean lengths lie on the ellipsoid
`1 / L_i(phi)^2 = sum_v (c_v / L_i,v)^2` (i != q).

`transition(h) = expm_rate(rates_along(h), |h|)`; `transition(0) = I`.

### 5.3 Closeness (T-PROGS)
`det T(h) = exp(|h| tr R_phi)`, so `closeness(h) = exp(|h| tr R_phi / (K - 1))`; no matrix
exponential needed. Anisotropic distance `-|h| tr R_phi`.

### 5.4 from_geology
Per axis v: `embedded[v]` given gives the partial `r_ij = pi_ij / L_i,v` (i, j != q) and the
background fill (`L_q,v` derived); `None` gives `RateMatrix::max_entropy(p, L_v)` (all K lengths
used, symmetric). The z' axis is up. `proportions` shared by all axes.

### 5.5 Table
Enumerate integer offsets d (excluding 0) by growing cubic shells `max(|dx|, |dy|, |dz|) = s`,
s = 1, 2, ...; keep offsets with `|d_v| < count[v]` and `closeness(grid.lag(d)) >= limit`;
stop after a shell that keeps none (closeness decreases along every ray). Estimate the bytes
(`n_offsets * (K^2 + 4) * 8`) per shell and call `memory::check` before allocating. Sort by
closeness descending, ties by `(dz, dy, dx)` ascending. Store `T(d)` for each offset, computing
only `d` with positive lexicographic sign and filling `-d` by the reversal identity.

## 6. Experimental checks used in tests and M4

- Lag grid: 200 lags from 0 to `20 max L` plus `10^4 max L`; assertions as spec section 7.
- On a grid, a 1D Markov chain sampled at Delta is exact (`T(n Delta) = T(Delta)^n`).

## 7. Fit-quality report

`fit_quality(model, grid, mask, realizations, directions, n_lags)` returns `FitQuality`. For
each realization: proportions over its active cells, and the grid estimator of section 2.3 at
offsets `l * d`, `l = 1..=n_lags`, for each direction `d`. Over realizations, per offset and
entry: the mean of the finite values, and the standard error `s / sqrt(c)` with `s` the sample
standard deviation of the `c` finite values (NaN when `c < 2`). `proportion_error` and
`transiogram_error` are the largest absolute gaps to the model; `max_standard_errors` divides
each gap by `max(se, 1e-3)`, so entries pinned exactly (hard data, quenched lags) do not
dominate. Realizations are processed in order, so the report does not depend on thread count.
Memory: `(n + 4)` matrices of `K^2` floats per offset, checked before allocating.

## Implementer decisions

(empty)
