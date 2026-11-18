# varvite-hmm: algorithms

Signatures are in [spec.md](spec.md) section 4.3. K states, sequences n = 1..N of length T_n,
start pi, transition A, `lb_j(x_t) = log b_j(x_t)`. Formulas follow Rabiner (1989), checked by
the brute-force tests; trust the tests over this text if they disagree.

## 1. Emission log densities

Compute `lb` for all (t, j) once per E-step (T x K matrix), then add clamps:
`lb_j(x_t) = -inf` where `allowed[t, j]` is false.

- Categorical: `log B_j[x]`; missing gives 0.
- Poisson (independent channels): `sum_c (x_c ln lam_c - lam_c - ln_gamma(x_c + 1))` over observed
  channels; `lam` floored at `1e-300`.
- Gaussian: `-0.5 [d_o ln(2 pi) + ln|S_oo| + z^T z]`, `L z = x_o - mu_o`, L the Cholesky factor
  of the observed sub-block `S_oo`; `ln|S_oo| = 2 sum ln L_ii`. Never form an inverse. Cache the
  Cholesky per (state, missingness pattern) within an E-step.
- Gaussian mixture: `logsumexp_m (ln w_m + log N_m(x))`.
- All channels missing: 0 for every state (exact marginalisation).

## 2. Forward-backward

### Log space (reference, default)
- `la_1(j) = ln pi_j + lb_j(x_1)`; `la_t(j) = lb_j(x_t) + LSE_i(la_{t-1}(i) + ln a_ij)`.
- `lbeta_T = 0`; `lbeta_t(i) = LSE_j(ln a_ij + lb_j(x_{t+1}) + lbeta_{t+1}(j))`.
- `ln L = LSE_j la_T(j)`; `gamma_t = exp(la_t + lbeta_t - ln L)`;
  `xi_t(i, j) = exp(la_t(i) + ln a_ij + lb_j(x_{t+1}) + lbeta_{t+1}(j) - ln L)`.
- Compute `ln A` once per call. Accumulate `xi_sum` during the backward pass; never store xi.

### Scaled
- Per step `m_t = max_j lb_j(x_t)`, `b~_j = exp(lb_j - m_t)` (guard against underflow: one
  4-channel outlier underflows raw densities in f64).
- `alpha^_1 = pi * b~_1 / c_1`; `alpha~_t(j) = [sum_i alpha^_{t-1}(i) a_ij] b~_j(x_t)`,
  `c_t = sum_j alpha~_t(j)`, `alpha^_t = alpha~_t / c_t`; `ln L = sum_t (ln c_t + m_t)`.
- `beta^_T = 1`; `beta^_t(i) = sum_j a_ij b~_j(x_{t+1}) beta^_{t+1}(j) / c_{t+1}`.
- `gamma_t = alpha^_t * beta^_t` (do not renormalise);
  `xi_t(i, j) = alpha^_t(i) a_ij b~_j(x_{t+1}) beta^_{t+1}(j) / c_{t+1}`.

Either implementation: `ln L = -inf` (or `c_t = 0`) raises `InvalidInput`
"observation impossible under the model: sequence {n}, step {t}" (structural zero in A or a
clamp that contradicts A).

### Path entropy (for ICL)
`H(S | X) = ln L - [sum_j gamma_1(j) ln pi_j + sum_ij xi_sum(i, j) ln a_ij + sum_t sum_j gamma_t(j) lb_j(x_t)]`
with `0 * ln 0 = 0`; clamps excluded from `lb` (use the unclamped emission term with the clamped
posteriors).

## 3. Viterbi (always log space)

`delta_1(j) = ln pi_j + lb_j(x_1)`; `delta_t(j) = max_i [delta_{t-1}(i) + ln a_ij] + lb_j(x_t)`,
`psi_t(j)` = argmax scanning i upward with strict `>` (lowest index wins ties). End:
`s_T` = argmax with the same rule; backtrack. Return `max_j delta_T(j)` = `ln P(x, s*)`.
All `-inf` raises `InvalidInput`.

Decoders: `Map` = argmax of gamma (lowest index on ties). `PosteriorViterbi` (Fariselli 2005):
Viterbi with score `ln gamma_t(j)` and transition weight 0 where `a_ij > 0`, `-inf` otherwise.

Posterior sampling (forward filtering, backward sampling): draw `s_T ~ alpha_T`, then
`s_t ~ alpha_t(i) a_{i, s_{t+1}}` normalised over i, using normalised forward vectors.

## 4. Baum-Welch with many sequences

E-step per sequence in parallel (rayon `par_iter` over sequence index), each returning
sufficient statistics: `gamma_1`, `xi_sum`, emission accumulators, `ln L_n`. Collect into a
`Vec` in index order and reduce serially.

M-step:
- `pi_j = (sum_n gamma_1n(j) + alpha_s - 1) / (N + K (alpha_s - 1))`.
- `a_ij = (XI_ij + alpha_t - 1) / (sum_j' XI_ij' + K' (alpha_t - 1))` where `XI = sum_n xi_sum`
  and the sums run over cells not structurally zero (`a_ij == 0` stays 0: multiplicative updates
  keep structural zeros). Use the row sum of XI as the denominator so rows sum to 1 to rounding.
  A row with zero expected occupancy keeps its previous values.
- Sequences of length 1 contribute to pi and the emissions only.
- Parts not in `params`, and frozen distributions, are not updated.

Objective `= sum_n ln L_n + log_prior`. Stop when the gain per observed step is below `tol`
(converged) or after `max_iter`. A decrease larger than `1e-9 |objective|` raises `Convergence`
naming the iteration. `history` stores the objective after each E-step.

Clamped fitting maximises `ln P(x, labels)`; `score` uses the unclamped likelihood.

## 5. Emission M-steps (gamma-weighted, `G_j = sum gamma_t(j)` over observed steps)

Accumulate with weighted West / Chan updates (mean and co-moment per state, merged across
sequences in index order), never raw `E[x^2] - E[x]^2`.

- Categorical: `B_jk = (sum_t gamma_t(j)[x_t = k] + beta) / (G_j + M beta)`, beta the
  pseudocount.
- Poisson: `lam_jc = sum gamma x_c / G_jc` (per-channel observed weights).
- Gaussian mean: `mu_j = sum gamma x / G_j` (per channel over observed steps for diag and
  spherical).
- Covariance MAP. Prior kernel per state: `ln p(S) = -(1/2) ln|S| - (1/2) tr(S^-1 Psi)`,
  `Psi = eps diag(v)`, v the pooled per-channel variance of all observed data at fit start,
  eps = `covariance_prior` (0 disables the prior). Updates:
  - full: `S_j = (C_j + Psi) / (G_j + 1)` with `C_j = sum gamma (x - mu)(x - mu)^T`;
  - diag: `s2_jc = (C_jcc + eps v_c) / (G_jc + 1)`;
  - spherical: `s2_j = (sum_c C_jcc + eps sum_c v_c) / (d G_j + d)`.
  `log_prior` adds these kernels (diag and spherical use their diagonal forms) plus the
  Dirichlet terms `(alpha_s - 1) sum ln pi + (alpha_t - 1) sum ln a`.
- Full covariance with partially missing channels (exact E-step): for observed `o`, missing `m`:
  `E[x_m] = mu_m + S_mo S_oo^-1 (x_o - mu_o)`, `Cov[x_m] = S_mm - S_mo S_oo^-1 S_om`. Fill x with
  the expectation and add `gamma * Cov` into the m-block of the co-moment.
- Gaussian mixture: responsibilities `gamma_t(j, m) = gamma_t(j) exp(ln w_jm + ln N_jm(x_t) - lb_j(x_t))`;
  `w_jm = sum_t gamma_t(j, m) / G_j`; component means and covariances as above with those weights.
- A Cholesky failure retries once with `1e-10 tr(S)/d` added to the diagonal, then raises
  `Convergence` naming the state.

## 6. Initialisation

Per restart r, stream `[tag::INIT, r]`.

- Continuous families: standardise the rows with every channel observed (pooled over sequences);
  k-means++ seeding then Lloyd until assignments stop changing or 300 iterations; an empty
  cluster is re-seeded with the point farthest from its centroid (lowest index on ties). Means
  from centroids (back-transformed); covariances from members plus `Psi`, falling back to the
  pooled covariance when a cluster has fewer than d + 1 members. Mixtures: k-means with `n_mix`
  clusters inside each state's members. Too few complete rows (fewer than K): impute each
  missing channel by its pooled mean for initialisation only.
- Categorical: each state's table drawn from Dirichlet(1) (normalised `gamma(1)` draws); with
  labels, the labelled steps' counts plus 1.
- Poisson: k-means on the counts as for Gaussian; rate = cluster mean.
- `pi`: first-step label frequencies plus 1, normalised. `A`: transitions of the label sequence
  along each sequence plus 1, normalised (geology gives strongly diagonal A).
- When distributions are given and `t` is in `init_params`, labels are the per-step argmax of
  `lb` (lowest index on ties).
- Restarts run in parallel; keep the best final objective, lowest restart index on ties.

## 7. Ordering

After fitting: `Mean` sorts states by the first channel's mean ascending (Gaussian and mixture
families; mixture uses the weighted mean); `Occupancy` by stationary probability descending;
ties by current index. Categorical with `Mean` raises `InvalidInput`.

## 8. Free parameters

`s` in params: `(number of positive start entries) - 1`. `t`: `sum_rows (positive entries - 1)`.
`e`, per non-frozen state: categorical `M - 1`; Poisson `d`; Gaussian full `d + d(d+1)/2`,
diag `2d`, spherical `d + 1`; mixture `(m - 1) + m d + m cov(d)`. Frozen parts and letters
outside `params` count 0.

## 9. Sampling

`sample(lengths, seed)`: sequence i uses stream `[SAMPLE, i]`: `s_1 ~ pi`, `s_{t+1} ~ A[s_t]`,
`x_t ~ b_{s_t}`. Gaussian draws `mu + L z`, z standard normal; Poisson with `lam < 10` by
inverse CDF from 0 with log-space pmf terms `exp(k ln lam - lam - ln_gamma(k + 1))`, and with
`lam >= 10` by Hormann's (1993) PTRS transformed rejection (O(1) expected uniforms per draw, so a
large rate does not cost O(lam)); mixture picks the component first.

## 10. Brute-force oracle (tests)

Enumerate all K^T paths, `ln P(x, s) = ln pi_{s1} + sum ln a + sum lb`. Max and argmax (lowest
lexicographic path on ties) equal Viterbi; LSE equals `ln L`; normalised marginals equal gamma
and xi; `-sum P(s|x) ln P(s|x)` equals the entropy identity. Missing-row check: a NaN row equals
deleting it and using `A^2` across it.

## Implementer decisions

- Scaled forward-backward falls back to the log pass for a sequence when a filter entry that is
  positive in exact arithmetic drops below the smallest normal f64 (shifting by `m_t` cannot keep
  a path whose relative weight is below about `e^-708`, and later evidence may need that path),
  or when the backward pass yields a non-finite gamma or xi. Structural zeros in A are skipped in
  the backward sums so `0 * inf` never appears. Scaled and log therefore agree on every input.
- A non-finite EM objective raises `Convergence` naming the iteration.
- Sizes are checked before allocating: `t x d` and `t x K` use checked multiplication, a pass
  checks `6 t K` f64 against the memory budget, and `sample_posterior` checks
  `n (t * 4 + 24)` bytes; overflow or excess raises `InvalidInput`.
- `sample(lengths, seed)` returns `Vec` per the spec signature, so it cannot reject absurd
  lengths; callers must bound them.
