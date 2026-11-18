# varvite-core: algorithms

Signatures are in [spec.md](spec.md) section 4.1. This file gives the steps.

## 1. Sequences

1. `from_intervals`: group rows by hole id in first-appearance order. Sort each hole's rows by
   `from` (stable). Validate `from < to`, finite values, and `to[i] <= from[i+1] + 1e-9`;
   otherwise raise `InvalidInput` naming the hole id and the depth.
2. `from_samples`: interval `[depth - step/2, depth + step/2]` per sample; `step > 0`.
3. `runs(direction, gap)`: walk each hole in input depth order and cut at a MISSING code or
   where `from[i+1] - to[i] > gap`. For `Up`, reverse each run and map depth to position
   `-depth` (so `start = -to`, `end = -from`). Runs are ordered by hole, then by the order in
   which they are met along the direction.
4. `beds(direction, gap, merge)`: per run, merge adjacent intervals with the same code when
   `merge` (sum of thicknesses, an internal gap below `gap` adds no thickness). The first bed of
   each run has `open_start = true`, the last `open_end = true` (a one-bed run has both).
5. `regular(direction, step, gap)`: per run, sample the code at positions
   `start_of_run + (i + 1/2) step` while the position is `< end_of_run`. Samples built by
   `from_samples` with the same step reproduce their codes exactly.

## 2. RNG streams

- `splitmix`, `realization_seed`: copy boitata `crates/core/src/rng.rs` exactly (spec section 3.3).
- `Stream::new(seed, keys)`: `key = keys.iter().fold(seed, |s, &k| realization_seed(s, k))`.
  State `s[i] = splitmix(key.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)))`
  for i = 0..4; if all four are 0 set `s[0] = 1`.
- xoshiro256++: `result = rotl(s0 + s3, 23) + s0; t = s1 << 17; s2 ^= s0; s3 ^= s1; s1 ^= s2;
  s0 ^= s3; s2 ^= t; s3 = rotl(s3, 45)` (wrapping arithmetic).
- `uniform = (next >> 11) * 2^-53`; `uniform_open = ((next >> 11) + 0.5) * 2^-53`.
- `below(n)`: Lemire: `m = x as u128 * n`; `l = m as u64`; if `l < n` then
  `t = n.wrapping_neg() % n`, redraw while `l < t`; return `m >> 64`.
- `normal`: `sqrt(-2 ln u1) cos(2 pi u2)` with `u1 = uniform_open`, `u2 = uniform`.
- `gamma(a)`: Marsaglia-Tsang for `a >= 1` (`d = a - 1/3`, `c = 1/sqrt(9d)`, loop: `x = normal`,
  `v = (1 + c x)^3`, reject `v <= 0`, accept if `ln u < x^2/2 + d - d v + d ln v`, return `d v`).
  For `a < 1`: `gamma(a + 1) * u^(1/a)`.
- `categorical(w)`: `total = sum w` (must be finite and > 0), `target = uniform * total`, return
  the first index with cumulative sum `> target`, skipping zero weights; if rounding exhausts
  the scan, return the last index with positive weight.

### Stream keys per routine (binding)

| Routine | Keys |
| --- | --- |
| chain `simulate`, realization r | `[r]` |
| chain `Sojourn::exponential_test`, replicate b | `[tag::BOOTSTRAP, b]` |
| hmm/hsmm fit, restart r | `[tag::INIT, r]` |
| hmm/hsmm `sample`, sequence i | `[tag::SAMPLE, i]` |
| hmm/hsmm `sample_posterior`, hole index i | `[tag::SAMPLE, i]` (n draws sequential) |
| select_states fold assignment | `[tag::FOLD]` |
| select_states fold f, K index s, restart r | `[tag::FOLD, f, s, r]`; full-data fits `[tag::INIT, s, r]` |
| tpsim and mcrf realization r | path `[r, tag::PATH]`, draws `[r, tag::DRAW]`, quench sweep s `[r, tag::QUENCH, s]` |

## 3. Numerics

### logsumexp
`m = max xs`; if `m == -inf` return `-inf`; if `m == +inf` return `+inf`; else
`m + ln(sum exp(x - m))`.

### Irreducibility (Tarjan)
Graph with an edge i -> j when `i != j` and `a_ij > 0`. Irreducible when there is a single
strongly connected component. A 1x1 matrix is irreducible.

### GTH stationary distribution
Input `A = P - I` or `Q` (only off-diagonals are read). Require irreducible, else `InvalidInput`
"reducible chain: classes {...}".

1. Copy off-diagonals into `A`. For `k` from `m-1` down to 1:
   `s = sum_{j<k} A[k, j]`; `A[i, k] /= s` for `i < k`;
   `A[i, j] += A[i, k] * A[k, j]` for `i, j < k`, `i != j`.
2. `x_0 = 1`; for `k = 1..m-1`: `x_k = sum_{j<k} x_j A[j, k]`.
3. Normalise `x` to sum 1.

GTH is subtraction-free and needs no aperiodicity (never use power iteration).

### expm_rate (uniformization)
1. `mu = max |q_jj|`; if `mu == 0` or `h == 0` return `I`.
2. `R = I + Q / mu` (non-negative, stochastic).
3. `s` = smallest integer `>= 0` with `mu h / 2^s <= 1`; `lam = mu h / 2^s`.
4. `T = sum_n w_n R^n`, `w_0 = e^-lam`, `w_n = w_{n-1} lam / n`, until the remaining Poisson
   tail `1 - sum w` is below `1e-17` (at most 30 terms for `lam <= 1`).
5. Square `T` `s` times. Divide each row by its sum (the deviation before division is below
   `1e-13`; assert it in debug builds).

### expm
`a.clone().exp()` from nalgebra (scaling and squaring with Pade, Higham 2005). Used only as a
cross-check and for general (non-generator) matrices.

### logm_stochastic (inverse scaling and squaring)
1. `X = P`, `s = 0`. While `||X - I||_1 > 0.25`: replace `X` by its principal square root by
   Denman-Beavers (`Y_0 = X`, `Z_0 = I`; `Y' = (Y + Z^-1)/2`, `Z' = (Z + Y^-1)/2` until
   `||Y' - Y||_1 < 1e-14 ||Y||_1`, at most 100 iterations), `s += 1`, at most 64 times.
   Non-convergence or a singular iterate raises `Convergence` "transition matrix has no real
   logarithm".
2. `E = X - I`; `L = sum_{n>=1} (-1)^(n+1) E^n / n` until the term's 1-norm `< 1e-17`.
3. Return `2^s L`.

### RateMatrix::from_transition (Israel, Rosenthal and Wei diagonal adjustment)
`Q0 = logm(P) / step`. Set negative off-diagonals to 0, set each diagonal to minus its row's
off-diagonal sum. `distance = max |Q - Q0|`, `exact = distance <= 1e-12 max|Q0|`.

### quasi_independence (IPF with a structural-zero diagonal)
Targets `rows` and `cols` (same total; zero-margin categories give zero rows and columns).
1. `E = 1` off the diagonal and where both margins are positive, else 0.
2. Repeat: scale each row to its target, then each column to its target.
3. Stop when `max |E_i. - rows_i|` and `max |E_.j - cols_j|` are both
   `<= tol * total` (default `tol = 1e-13`). Past `max_iter` (default 10000) raise
   `Convergence`.
4. m = 2 has the unique solution E = [[0, r0], [r1, 0]] only when `r0 = c1` and `r1 = c0`;
   otherwise raise `InvalidInput`.

### fundamental_matrix
`Z = (I - P + 1 pi^T)^-1` by LU; singular raises `InvalidInput`.

### golden_section, nelder_mead
Golden section: standard bracketing on `[a, b]`, stop when `b - a < tol (|x| + 1e-12)`.
Nelder-Mead: coefficients 1, 2, 0.5, 0.5; initial simplex `x0 + step e_i`; stop when the spread
of function values is below `tol (|f_best| + 1e-12)` or after `max_eval` evaluations; ties in
ordering broken by vertex index.

## 4. Special functions

- `ln_gamma`: Lanczos (g = 7, n = 9 coefficients) with reflection for `x < 0.5`.
- `digamma`: recurrence up to `x >= 6`, then the asymptotic series to `x^-12`.
- `gamma_q(a, x)`: series for `x < a + 1` (`P` then `1 - P`), Lentz continued fraction
  otherwise; `a > 0`, `x >= 0`.
- `chi2_sf(x, df) = gamma_q(df / 2, x / 2)`.

## 5. Memory

`available_bytes`: Linux parses `MemAvailable:` (kB) from `/proc/meminfo`; any failure or other
OS returns `None`. `check(needed, memory, what)`: `budget = memory.or(available_bytes())`;
`needed > budget` raises `InvalidInput`
"{what} needs {needed} bytes but the memory budget is {budget} bytes; pass memory= or use a
smaller grid".

## 6. Background fill (Carle 1997, T-PROGS chapter 6)

Input: `partial` k x k with entries read only for `i, j != q`; proportions `p > 0` summing to 1
(within 1e-12, else `InvalidInput`).

1. For `i != q`: `r_iq = -sum_{j != q} r_ij`.
2. For `j != q`: `r_qj = -(1 / p_q) sum_{i != q} p_i r_ij`.
3. `r_qq = -sum_{j != q} r_qj`.
4. Validate with `RateMatrix::new`; a negative `r_iq` or `r_qj` raises `InvalidInput`
   "rates incompatible with proportions: entry (i, j) = value".

Golden example (Carle): p = (0.5, 0.3, 0.2), L1 = 4, L2 = 2, pi_12 = 0.6, pi_21 = 0.7,
background 3 (index 2). Partial: r_00 = -0.25, r_01 = 0.6/4 = 0.15, r_10 = 0.7/2 = 0.35,
r_11 = -0.5. Result:
`Q = [[-0.25, 0.15, 0.10], [0.35, -0.5, 0.15], [0.1, 0.375, -0.475]]`, L3 = 2.105263157894737,
eigenvalues 0, -0.5, -0.725,
`Pi = [[0, 0.6, 0.4], [0.7, 0, 0.3], [0.2105263158, 0.7894736842, 0]]`,
`nu = (0.3378378378, 0.4054054054, 0.2567567568)`.

`T(0.5) = [[0.8894003915, 0.0663597651, 0.0442398434], [0.1474291504, 0.7899162352, 0.0626546144], [0.0553552955, 0.1492262345, 0.79541847]]`

`T(1) = [[0.8032653299, 0.1180408021, 0.0786938681], [0.2510484883, 0.6431007346, 0.1058507771], [0.115263943, 0.2402468928, 0.6444891642]]`

`T(5) = [[0.5410424993, 0.2753745004, 0.1835830003], [0.483595679, 0.3205022315, 0.1959020895], [0.4220002332, 0.3308104017, 0.2471893651]]`

`T(50)`: every row (0.5, 0.3, 0.2). This Q is not reversible.

## 7. Maximum entropy rates

Given `p` and `L` (all k, positive):
1. `phi_i = p_i / L_i`. Necessary condition: `phi_i <= sum_{j != i} phi_j` for all i, else
   `InvalidInput` naming i (no valid model exists).
2. `E = quasi_independence(phi, phi)`; `Pi = E` with rows divided by `phi`.
3. `R = diag(1 / L) (Pi - I)`. Then `p^T R = 0` holds by construction (column sums of E equal
   `phi`); assert it within `1e-12 max|r|`.

The result is reversible (E symmetric when row and column targets are equal), which T-PROGS
uses for lateral directions.

## 8. Closed forms used in tests

- Two states, `a = q_01`, `b = q_10`, `s = a + b`: `T00 = b/s + (a/s) e^{-sh}`,
  `T01 = (a/s)(1 - e^{-sh})`, `p = (b, a) / s`.
- Single rate: `R = lambda (1 p^T - I)` gives `T(h) = 1 p^T + (I - 1 p^T) e^{-lambda h}`,
  `L_i = 1 / (lambda (1 - p_i))`; eigenvalue `-lambda` has multiplicity k - 1.

## Implementer decisions

- `digamma` uses the recurrence up to `x >= 10`, not 6: at 6 the first omitted series term is about
  1e-12, above the 1e-13 relative accuracy section 4 of spec.md requires.
- `expm_rate` stops the Poisson series when the term weight drops below `1e-18` (at most 40 terms)
  instead of testing `1 - sum w < 1e-17`, which rounding can keep above `1e-17` forever.
- `RateMatrix::new` accepts `[[0]]` for a single category (the closeness rule "1 when k = 1" needs
  one-category models); every other diagonal must be negative.
- `RateMatrix::from_partial` clamps a filled entry in `[-1e-12 max|r|, 0)` to 0 and raises below that.
- `TransitionMatrix::reversed` raises when a reversed row sums farther than `1e-9` from 1 (pi not
  stationary) and renormalises rows within that tolerance; `RateMatrix::reversed` rescales
  off-diagonals so rows sum to 0.
- `RateMatrix::is_reversible(p, tol)` compares `|p_i q_ij - p_j q_ji|` with `tol * max|q|`.
- `numerics::components` (public) returns the strongly connected classes used by
  `is_irreducible` and the "reducible chain" message.
- `quasi_independence` raises `InvalidInput` when a positive margin has no allowed cell, and when
  row and column totals differ by more than `1e-9` relative.
- `SequenceSet::regular` returns one Vec per run, possibly empty, aligned with `runs`; a sample
  position inside an internal gap takes the code of the interval before it.
- `Categories` JSON uses `deny_unknown_fields`, so `from_json` names an unknown key.
- `serde_mat` also has `option` and `vec` submodules for `Option<Mat>` and `Vec<Mat>` fields.
- New module `synthetic`: seeded test data with known parameters for every crate (`markov_codes`,
  `markov_holes`, `rate_beds`, `rate_holes`, `hmm_sample`, `layered_field`, `drillholes`).
  Sequence or hole `i` uses `Stream::new(seed, &[tag::SAMPLE, i])`. The layered field is a
  laterally constant layer cake, so its exact transiogram is `exp(Q_z h_z)`.
- `crates/py/src/core.rs` exposes two private test hooks, `_sequence_runs` and `_sequences`, so
  `args::sequence_set` and `args::sequences` are tested from Python. `RowMap` has fields `rows`
  (input row per step, per sequence, in direction order), `n_rows`, `list` and `direction`.
- When `depth` is given, samples spaced wider than `step` leave a gap of `spacing - step`, which
  breaks the run when it exceeds `gap` (not a separate `1e-6 * step` rule).
- Stub crates carry `#![allow(unused_variables)]` in `lib.rs` and `crates/py/src/args.rs` carries
  `#![allow(dead_code)]`; owners remove them once their bodies are filled. `Hsmm::initialize` and
  `tpsim::quench` carry `#[allow(clippy::too_many_arguments)]` because the contract fixes eight
  arguments.
- The `lint` task checks `python examples` only: there is no `ci/` folder yet.
