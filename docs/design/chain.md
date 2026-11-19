# transiograms-chain: algorithms

Signatures are in [spec.md](spec.md) section 4.2. m is the number of categories.

## 1. Tallies

Never count a transition across holes, gaps, missing codes or runs (use
`SequenceSet::runs` / `beds` / `regular` only).

- Embedded: `n_ij` = bed transitions i -> j inside each run, from
  `beds(direction, gap, merge = !self_transitions)`. Row sum `n_i.` = beds of i with a successor
  (= exits), column sum `n_.j` = beds of j with a predecessor. Keep both margins.
- Exposure `T_j` = total thickness of every merged bed of j, open beds included.
- Generator MLE: `q_jk = n_jk / T_j` for j != k over merged beds, `q_jj = -n_j. / T_j`;
  `mean_lengths_j = T_j / n_j.`. Censored (open-end) beds add thickness and no exit; never use
  the mean bed thickness. With `pseudocount a > 0`: `q_jk = (n_jk + a) / T_j`. A category with
  `n_j. = 0` and `a = 0` follows `on_empty_row` (Uniform: `q_jk = 1 / ((m - 1) T_j)`).
  A category absent from the data (`T_j = 0`) raises `InvalidInput` naming it.
- Fixed: `c_ij` over consecutive samples of `regular(direction, step, gap)`;
  `P_hat_ij = c_ij / c_i.` (Anderson and Goodman 1957).

`from_counts` rows: `(n_ij + a) / (n_i. + a * allowed_i)`; embedded rows have `pi_ii = 0`
exactly. Empty rows with `a = 0`: Raise names the category; Uniform spreads over allowed cells.

## 2. Derived quantities

- Embedded chain: `bed_frequencies` = stationary of Pi (GTH on `Pi - I`); periodic chains are
  fine. `proportions` = stationary of the generator; identity check
  `p_k = nu_k L_k / sum_j nu_j L_j`.
- Fixed chain: `mean_lengths_k = step / (1 - p_kk)` (geometric run length in steps,
  `P(L = s) = p_kk^(s-1) (1 - p_kk)`); embedded from fixed: `pi_ij = p_ij / (1 - p_ii)`; a row
  with `p_ii = 1` makes the chain reducible, which raises.
- Continuous (built): `L_j = -1 / q_jj`, `pi_jk = q_jk L_j`, `nu_j` proportional to `p_j / L_j`.
- `transition_at(h)`: generator chains use `RateMatrix::transition(h)`; fixed chains without a
  usable generator require `h = n step` (within `1e-9 step`) and return `P^n`, else
  `InvalidInput`.

## 3. Independence test (order 1 against independence)

- Embedded: quasi-independence `e_ij = a_i b_j` (i != j), `e_ii = 0`, margins `n_i.` and `n_.j`:
  `numerics::quasi_independence(row sums, column sums)`. Drop categories with a zero row or
  column margin first. `df = m'^2 - 3 m' + 1` for the `m'` remaining categories; `m' <= 2`
  raises `InvalidInput` "an embedded chain with 2 categories always alternates; no test exists".
  Never use `r_i c_j / N` or Gingerich's `n_i n_j / (N - n_i)`.
- Embedded with `self_transitions`: ordinary contingency, `e_ij = n_i. n_.j / N`,
  `df = (m' - 1)^2`.
- Fixed: `e_ij = c_i. c_.j / c`, `df = (m' - 1)^2`. The docstring warns that bed persistence
  makes this test reject almost always and says nothing about juxtaposition.
- Statistics over allowed cells with `E > 0`: Pearson `X2 = sum (O - E)^2 / E`;
  `G = 2 sum O ln(O / E)` with `0 ln 0 = 0`. `p = chi2_sf(stat, df)`.
  `small_expected` = fraction of allowed cells with `E < 5`.

Golden: `N = [[0,30,10,5],[8,0,25,12],[20,6,0,9],[10,14,8,0]]` gives
`E = [[0, 20.6805682797, 15.4762764461, 8.8431552742], [17.022383202, 0, 17.8042536869, 10.1733631111], [11.68497564, 16.3315427453, 0, 6.9834816147], [9.292641158, 12.987888975, 9.719469867, 0]]`,
`X2 = 29.298020503608846`, `G = 31.237967935478377`, `df = 5`, `p_X2 = 2.0266e-05`,
`p_G = 8.4064e-06`. Gingerich's first row would be `[0, 18.080, 14.063, 12.857]` (must differ).
3x3: `N = [[0,12,7],[5,0,14],[9,4,0]]` gives
`E = [[0, 8.919987347, 10.080012653], [8.080012653, 0, 10.919987347], [5.919987347, 7.080012653, 0]]`
and the cycle identity `E01 E12 E20 = E02 E21 E10`.

`difference_matrix = Pi_obs - E / rowsum(E)` (Miall 1973); positive entries are preferred paths.

## 4. Order tests and selection

- Fixed, order k against k - 1 (Anderson and Goodman): count `c_{s, j}` for k-histories s,
  dropping the first k states of every run. `G = 2 sum c_{s,j} ln(P_hat(j | s) / P_hat(j | s'))`,
  s' the last k - 1 states of s (k = 1: compare with the marginal). Pearson uses the expected
  counts `c_{s.} P_hat(j | s')`. `df = m^(k-1) (m - 1)^2`, minus `(m - 1)` for each k-history
  never observed and minus 1 per zero cell structurally impossible under the (k - 1) fit.
- Embedded order 2 against 1: triplets `n_ijk` with i != j, j != k (i = k allowed); null
  `e_ijk = n_ij. n_.jk / n_.j.`; `df = m (m - 2)^2`, reduced by the empty middle slices
  (`(m - 2)^2` each). `order > 2` for embedded raises `InvalidInput`.
- `select_order(max_order)` (fixed only): every order 0..=max_order is fitted on the same
  transitions (drop the first `max_order` states of every run). `LL_k = sum c ln P_hat`;
  `n_params = m^k (m - 1)`; `AIC = -2 LL + 2 n_params`; `BIC = -2 LL + n_params ln N`, N the
  number of transitions used.

## 5. Homogeneity across groups

`G = 2 sum_h sum_ij c^h_ij ln(P_hat^h_ij / P_hat_ij)`, P_hat pooled. `df = sum_i (H_i - 1)(k_i - 1)`
where `H_i` is the number of groups observing row i and `k_i` the number of pooled-positive
columns of row i (embedded: off-diagonal cells only). Full case: `(H - 1) m (m - 1)` fixed,
`(H - 1) m (m - 2)` embedded. Pearson: per-row H x k_i contingency, summed.

## 6. Symmetry, asymmetry, entropy

- Bowker: `X2 = sum_{i<j} (n_ij - n_ji)^2 / (n_ij + n_ji)` over pairs with `n_ij + n_ji > 0`,
  `df` = number of such pairs; requires counts.
- `asymmetry() = F - F^T`, `F_jk = p_j q_jk` when a generator exists, else `nu_j pi_jk`.
- Entropy (Hattori 1976), bits, `0 log 0 = 0`: `post_i = -sum_j pi_ij log2 pi_ij`;
  `pre_j = -sum_i r_ij log2 r_ij` with `r_ij = n_ij / n_.j` when counts exist, else
  `nu_i pi_ij / sum_i nu_i pi_ij`. `max = log2(m - 1)` for embedded (zero diagonal), `log2 m`
  otherwise; normalised = value / max. `chain = sum_i nu_i post_i`.

## 7. Sojourns

`sojourn` uses merged beds. `n_beds`, `n_open` (beds with either end open), `total` = T_j,
`mean_length = T_j / exits_j` (NaN with no exit). `survival(code)`: Kaplan-Meier on bed
thicknesses with open beds treated as right-censored at their observed thickness; returns the
distinct event thicknesses and S just after each. `exponential_test`: Kolmogorov-Smirnov D
between the closed beds (both ends closed) and the exponential with their mean; p value by
parametric bootstrap: draw the same number of exponential thicknesses with that mean from
stream `[BOOTSTRAP, b]`, refit the mean, recompute D;
`p = (1 + #{D_b >= D}) / (1 + n_bootstrap)`. Fewer than 2 closed beds raises.
The docstring notes the discretisation bias for sampled logs: a run of s samples spans between
`(s - 1) step` and `(s + 1) step`.

## 8. Proportion variance (Kemeny and Snell)

For a fixed or embedded transition matrix P with stationary pi:
`Z = (I - P + 1 pi^T)^-1`; `Var(p_hat_k) ~ pi_k (2 Z_kk - 1 - pi_k) / N`;
`proportion_std(N) = sqrt` of that. For a generator chain, use `P = T(step)` of the simulation
step. Golden (Q of core.md section 6, step 0.5, N = 10^5): predicted SD
(0.00448, 0.00346, 0.00335); binomial SD (0.00158, 0.00145, 0.00126). A correct seeded run gave
p_hat = (0.49276, 0.30497, 0.20227), a 4.6 binomial-sigma miss on category 0 and 1.6 Markov
sigma. At N = 5000 over 200 replicates the empirical SD was (0.0180, 0.0155, 0.0139) against
predicted (0.0201, 0.0155, 0.0150).

M1 acceptance test: build the chain from the golden Q, simulate 10^5 steps of `T(0.5)` from the
stationary law with seed 0, refit fixed with step 0.5: every `|P_hat_ij - P_ij| <=
3 sqrt(P_ij (1 - P_ij) / c_i.)` and every `|p_hat_k - p_k| <= 3 proportion_std(10^5)_k`.

## 9. Simulation

Sequences run along the chain's direction from position 0. Hard data (sorted, non-overlapping
intervals; a point has `from == to`) split the target into free segments.

### Discrete (fixed steps or embedded beds), transition P, stationary pi
- Unconditional: `x_0 ~ pi` (or the start code), `x_{t+1} ~ P[x_t, :]` via
  `Stream::categorical`.
- Bridge from `x_0 = a` to `x_n = b`:
  `P(x_{t+1} = j | x_t = i, x_n = b) = P_ij (P^(n-t-1))_jb / (P^(n-t))_ib`. Precompute `P^r`
  for `r = 0..n` (stop at the r where `max |P^r - 1 pi^T| < 1e-15` and reuse it beyond).
  `(P^n)_ab = 0` raises `InvalidInput` "hard data at positions .. are impossible under the chain".
- Before the first hard datum: simulate backward with the reversed chain
  `P*_ij = pi_j P_ji / pi_i` from the datum's code, then flip. After the last: forward.
- Embedded beds: same with `P = Pi`; consecutive equal codes are impossible and are rejected by
  the bridge.

### Continuous (generator Q, stationary p)
- Unconditional (Gillespie): hold `Exp(-q_kk)` (`exponential() / -q_kk`), jump with
  `Pi[k, :]`; start `x_0 ~ p` (memoryless, so exact); cut the last bed at `length`.
- Before the first hard interval: reversed generator `Q* = D^-1 Q^T D` from its code.
- Bridge over duration `T` from a to b (Hobolth and Stone 2009, uniformization):
  1. `mu = max |q_jj|`, `R = I + Q / mu`.
  2. Draw N with `P(N = n) = Pois(n; mu T) (R^n)_ab / T(T)_ab`, accumulating `R^n` row a
     incrementally until the cumulative probability exceeds the uniform draw.
  3. Draw N sorted uniform times on (0, T) (draw N uniforms, sort ascending).
  4. For r = 1..N: `P(x_r = j | x_{r-1} = i) proportional to R_ij (R^(N-r))_jb`.
  5. Drop virtual self-jumps; merge adjacent equal codes into beds.
  `T(T)_ab = 0` raises like the discrete case.
- Output beds per realization as `(from, to, code)`, merged across segment boundaries when the
  codes agree; hard intervals appear as given.
- Fixed `simulate(length)` uses steps with cells whose centres fall in hard intervals frozen.

Bridge marginal check: `P(x_t = j | a, b) = (P^t)_aj (P^(n-t))_jb / (P^n)_ab` (continuous:
`T(t)_aj T(T-t)_jb / T(T)_ab`).

## Implementer decisions

- Python binding: `MarkovChain(kind="fixed")` with `step=None` uses the sampling step of the data
  (`step=`, inferred from `depth`, or 1 for unit-step codes); intervals without either raise.
- `MarkovChain.simulate` returns fixed-chain codes along the direction (column 0 is the deepest
  sample for "up"), as the bed dict gives positions along the direction; reverse the columns for
  top-down rows.
- Fixed-chain hard intervals freeze the cells whose centres they cover; a point (`from == to`)
  freezes the cell that contains it, so a point on a cell boundary is never dropped.
