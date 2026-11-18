# varvite-tpsim: algorithms

Signatures are in [spec.md](spec.md) section 4.6. T-PROGS TSIM (Carle 1999) with the choices
below. All lag lookups go through `TransitionModel::table`.

## 1. Transition-probability cokriging (TSIM step 1, transition basis)

Data `x_1..x_n` with codes `c_alpha`, target `x_0`. Unknown K x K blocks `W_alpha`:

`sum_beta T(x_beta - x_alpha) W_beta = T(x_0 - x_alpha)`, alpha = 1..n.

Block row alpha, block column beta holds `T(x_beta - x_alpha)` (head minus tail). Solve the
nK x nK system by SVD (nalgebra), zeroing singular values below `svd_cutoff * sigma_max`.
`Pr_k = sum_alpha W_alpha[c_alpha, k]`. These sum to 1 before correction (`T 1 = 1`); assert
within 1e-12. `correct`: clip to [0, 1] and renormalise; count corrected nodes.

Closed forms: one datum gives `Pr = row c_1 of T(x_0 - x_1)`. On a line with
`x_1 < x_2 < x_0`, `W_1 = 0` and `W_2 = T(x_0 - x_2)` (Chapman-Kolmogorov).

## 2. Search

Informed nodes: snapped hard data plus already simulated nodes. Scan the table offsets in order
(closeness descending) from the node and take the first `max_data` informed nodes; with
`max_per_octant > 0`, skip a node whose octant (sign pattern of the offset, `>= 0` versus `< 0`
per axis) already holds that many. No informed node gives `Pr = p`.

## 3. Sequential simulation (one realization r)

1. Frozen: hard cells. Active: mask (all cells when None).
2. Path: unfrozen active cells in index order, shuffled with stream `[r, PATH]`.
3. Per node: search, cokrige, correct, draw with `categorical` from stream `[r, DRAW]`, mark
   informed.
4. Quench (section 4) when enabled.

Realizations run in parallel (rayon over r) and collect in index order. Memory check before
allocating: `n * n_cells * 4` bytes of codes plus the table, plus `n * n_cells * 8` for the
Python int64 output in the binding.

`predict` (`vv.tp_cokriging`): the same cokriging at every active non-hard cell with hard data
only; hard cells get the one-hot probability of their code.

## 4. Simulated quenching (TSIM step 2, zero temperature)

Lags `quench_lags(model, grid, limit)`: the 13 lexicographically positive offsets of
`{-1, 0, 1}^3 \ {0}` (4 in 2D) with `closeness >= limit`, plus every unit axis offset whose axis
has more than one cell. Targets `t^MOD(l) = model.transition(grid.lag(l))`.

Objective
`O = sum_l sum_j sum_k (F_jk(l) / N_j(l) - t^MOD_jk(l))^2 + w_p sum_k (n_k / n - p_k)^2`,
rows with `N_j(l) = 0` contribute 0; F counts ordered in-grid active pairs `(x, x + l)`.

Sweep s (stream `[r, QUENCH, s]`): visit the unfrozen active cells in a fresh shuffled order. At
cell x with code a, for each b != a compute Delta O exactly:
- for each lag l: pair `(x, x + l)` (if in grid and active) moves one count from `F[a][c]` to
  `F[b][c]` and one from `N_a` to `N_b`; pair `(x - l, x)` with tail code e moves one count from
  `F[e][a]` to `F[e][b]`;
- recompute the contributions of the affected rows `{a, b, e}` for each l before and after;
- proportion term from `n_a - 1`, `n_b + 1`.
Accept the b with the most negative Delta O when it is `< 0` (lowest b on ties) and update the
counts. Stop after `max_iter` sweeps, after a sweep with no change, or when `O / O_0 < tol`.
Return the objective after each sweep (index 0 = before quenching). Quenching is sequential
within a realization (parallel node updates would conflict through the shared counts).

Invariants: O never increases; frozen cells never change; incremental Delta O equals the full
recompute; counts equal a recount after every sweep.

## 5. Proportion tolerance for acceptance tests

`Var(p_hat_k) ~ |D|^-2 sum_x sum_y C_kk(y - x)`, `C_kk(h) = p_k (t_kk(h) - p_k)`, summed over
the active cells (compute with the table lags, falling back to the full lag set on small
grids). The ensemble mean of n realizations has variance `Var / n` only when realizations are
independent; the tests assert `3 sqrt(Var) < 0.02` for a single realization, which bounds the
ensemble too.

## 6. M5 acceptance test

Grid 64 x 64 x 32, cell (1, 1, 0.25); model from `from_geology` with K = 3,
p = (0.5, 0.3, 0.2), mean lengths x' and y' (12, 8, 6), z' (2, 1, 0.8), embedded z from the
Carle golden example, lateral maximum entropy; hard data from 16 vertical holes simulated by the
model's z chain; n = 20, seed 0. Assert 100 percent hard agreement in each realization,
ensemble proportions within 0.02 (after the variance assertion), and per axis the RMSE between
the ensemble-mean grid auto-transiograms and `T(l Delta_v)` diagonals within 0.05 for
`l Delta_v <= 2 max_i L_i,v`. The implementer may change the sizes if the variance assertion
fails, and records the change under "Implementer decisions".

## Implementer decisions

- M5 sizes (section 6): cell (6, 6, 1) instead of (1, 1, 0.25), same 64 x 64 x 32 count. At
  the spec sizes the untruncated section 5 bound is 3 sigma = 0.194 for category 0, far above
  0.02. At (6, 6, 1) it is 0.0199, 0.0151 and 0.0124 for categories 0, 1 and 2 (test
  `implementer_m5_sizes_pass_the_untruncated_variance_assertion`, brute force over every lag).
  The x and y auto-transiogram RMSE then covers 4 lags (`2 * 12 / 6`).
- Cokriging null-space gauge (section 7 screening check): row `q = argmax p` is fixed at 0 in
  every `W_alpha` except that of the datum nearest the target (first one on ties). On a line
  x1 < x2 < x0 this gives literally `W1 = 0` and `W2 = T(x0 - x2)`. Probabilities do not depend
  on the gauge.
- `max_data` is capped at the search table length before any allocation, and the cokriging
  system of that size is checked against the memory budget, so a huge `max_data` returns
  `InvalidInput` or runs instead of overflowing.
- Non-finite datum or target coordinates in `cokrige` return `InvalidInput`.
- Memory: `predict` and `simulate` check every buffer they hold (codes or probabilities, hard
  and frozen flags, per-realization paths and quench state for the realizations in flight, the
  table and its index, and one cokriging system per thread) against one budget, and build the
  table within what the rest leaves. `cokrige` checks its system against available memory.
- Hard data with the same cell twice return `InvalidInput`.
