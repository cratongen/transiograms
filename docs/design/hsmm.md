# varvite-hsmm: algorithms

Signatures are in [spec.md](spec.md) section 4.4. Explicit-duration HSMM (Guedon 2003, Yu 2010)
with right censoring. Emissions, clamps, priors, restarts, ordering and sampling helpers come
from varvite-hmm.

## 1. Model

- `pi_j` start, embedded transitions `p_ij` with `p_jj = 0` and rows summing to 1 over i != j
  (K = 1: no transitions, the only path is one censored segment).
- Duration pmf `d_j(u)` on `u = 1..M_j`, renormalised on that support; survivor
  `D_j(u) = sum_{v = u}^{M_j} d_j(v)`.
- `M_j = max_duration[j]`; default (`None`) is the longest sequence length.
- The last segment of a sequence is right-censored: weight `D_j(u)` instead of `d_j(u)`.
- `w_j(u, t) = d_j(u)` for `t < T`, `D_j(u)` for `t = T`.
- `E_j(a, b) = sum_{s=a}^{b} lb_j(x_s)`, built incrementally inside the u loop (add
  `lb_j(x_{t-u+1})` as u grows); never as prefix-sum differences.

## 2. Forward (log space)

- `S_0(j) = ln pi_j`.
- For t = 1..T: `F_t(j) = LSE_{u=1}^{min(t, M_j)} [S_{t-u}(j) + ln w_j(u, t) + E_j(t-u+1, t)]`.
- For t < T: `S_t(j) = LSE_{i != j} [F_t(i) + ln p_ij]`.
- `ln L = LSE_j F_T(j)`.

## 3. Backward

- `G_T(j) = 0`.
- For t = T-1 down to 0:
  `H_t(k) = LSE_{u=1}^{min(T-t, M_k)} [ln w_k(u, t+u) + E_k(t+1, t+u) + G_{t+u}(k)]`;
  `G_t(j) = LSE_{k != j} [ln p_jk + H_t(k)]` (for t >= 1).
- Check: `ln L = LSE_j [ln pi_j + H_0(j)]` within 1e-12.

## 4. Posteriors (accumulate during the backward pass; memory O(T K))

- Segment of j on `[t-u+1, t]`: `eta = exp(S_{t-u}(j) + ln w_j(u, t) + E_j(t-u+1, t) + G_t(j) - ln L)`.
  Complete segments (t < T) add `eta` to `N_j(u)`; the censored final segment of observed length
  v adds `eta * d_j(u) / D_j(v)` to `N_j(u)` for every `u >= v` (Guedon's E-step).
- Start: `P(segment of j starts at 1) = exp(ln pi_j + H_0(j) - ln L)`.
- Transitions, t < T: `xi_t(i, j) = exp(F_t(i) + ln p_ij + H_t(j) - ln L)`, summed into `xi_sum`.
- Occupancy: `Start_t(j) = exp(S_{t-1}(j) + H_{t-1}(j) - ln L)`,
  `End_t(j) = exp(F_t(j) + G_t(j) - ln L)`, `gamma_t = gamma_{t-1} + Start_t - End_{t-1}`
  (`gamma_0 = 0`, `End_0 = 0`); clamp to [0, 1] and assert `sum_j gamma_t(j) = 1` within 1e-10.
  The test oracle computes gamma as the sum of eta over segments covering t.
- `ln L = -inf` raises `InvalidInput` naming sequence and step, as in hmm.
- Path entropy: `H = ln L - [sum_j start_j ln pi_j + sum_ij xi_sum ln p_ij + sum over segments
  eta ln w + sum_t sum_j gamma_t(j) lb_j(x_t)]`, `0 ln 0 = 0`.

## 5. Viterbi

- `W_0(j) = ln pi_j`; `V_t(j) = max_u [W_{t-u}(j) + ln w_j(u, t) + E_j(t-u+1, t)]`;
  `W_t(j) = max_{i != j} [V_t(i) + ln p_ij]`.
- Store argmax u (smallest u on ties, scanning upward with strict `>`) and argmax i (lowest
  index); final state argmax over `V_T` (lowest index); backtrack over segments.

## 6. M-step

- `pi_j = (sum_n start_n(j) + alpha_s - 1) / (N + K (alpha_s - 1))`.
- `p_ij = (XI_ij + alpha_t - 1) / sum_{k != i} (XI_ik + alpha_t - 1)`, i != j; structural zeros
  kept.
- Emissions: as hmm with these gamma.
- Durations: `Duration::fit(N_j, pseudocount)` on the expected counts, then renormalise on
  `1..M_j`:

| Law | Weighted MLE on counts `N(u)` |
| --- | --- |
| Geometric | `p = sum N / sum u N` |
| Poisson (shifted) | `rate = sum (u - 1) N / sum N` |
| Negative binomial (shifted) | profile over r: `p(r) = r / (r + mbar - 1)`, `mbar = sum u N / sum N`; maximise `sum N(u) ln d(u)` over `ln r` in `[ln 1e-3, ln 1e4]` by golden section (tol 1e-10). `mbar <= 1` gives the point mass at 1 (geometric p = 1). |
| Empirical | `(N(u) + pseudocount) / sum (N + pseudocount)` on `1..M` |

  `truncated_mass` reports the untruncated parametric law's mass above `M_j`; the docstring
  says to raise `max_duration` when it exceeds `1e-6`.
- EM objective, monotonicity check, `tol`, `history`, restarts: as hmm.md section 4.

## 7. Initialisation

As hmm.md section 6 for emissions; the label sequence's run lengths per state (right-censored
last runs ignored) fit each duration law by section 6 with pseudocount 1 for empirical;
embedded transitions from the label runs' transitions plus 1 off the diagonal.

## 8. Geometric equivalence (`from_hmm`)

`d_j(u) = (1 - a_jj) a_jj^(u - 1)`, `p_ij = a_ij / (1 - a_jj)` (i != j), same pi and
emissions, `M_j = max_duration`. `a_jj = 1` raises `InvalidInput`; `a_jj = 0` is the point mass
at 1. HSMM log L equals the HMM's within `1e-10` when `a_jj^M < 1e-14`.

## 9. Model selection

- Free parameters: HMM as hmm.md section 8. HSMM: start `K - 1`, transitions `K (K - 2)` for
  K >= 2 (0 for K <= 2), durations 1 (geometric, Poisson), 2 (NB), `M_j - 1` (empirical),
  emissions as hmm; structural zeros, frozen parts and letters outside `params` subtracted.
- n = `n_observed` (steps with at least one observed channel).
- `AIC = -2 ln L + 2k`; `BIC = -2 ln L + k ln n`; `ICL = BIC + 2 H(S | X)` with the exact path
  entropy summed over sequences (never the per-step marginal entropies). Lower is better.
- `select_states`: for each K in `states` (index s), fit with `n_init` restarts
  (streams `[INIT, s, r]`), unclamped `ln L` on all data. Cross-validation: `cv_folds = None`
  leave one hole out; `Some(k)` assigns holes to k folds by a permutation drawn from `[FOLD]`
  (fold of hole h = position of h in the permutation mod k); fold f fits on the other folds with
  streams `[FOLD, f, s, r]` and scores the held-out holes unclamped. `cv_log_likelihood` is the
  sum over folds (-inf stays -inf); `cv_per_step = cv_log_likelihood / n`. Fits over (K, fold)
  pairs run in parallel and reduce in index order.
- The docstring notes that depth samples are autocorrelated, so BIC tends to over-select K on
  dense logs and the cross-validated score is the honest check.

M3 BIC test: `datasets.electrofacies_logs(seed=0)` (20 holes x 300 steps, 3 well-separated
Gaussian states, 2 channels), `n_init = 5`, seed 0, K = 1..6, diag covariance: BIC argmin = 3.

## 10. Brute-force oracle

K = 3, T = 8: enumerate 3^8 state paths, run-length encode, reject runs longer than `M_j`,
weight `pi * prod p (between runs) * prod d (non-final runs) * D (final run) * prod b`.
LSE equals `ln L`, max equals Viterbi, run-level marginals equal eta, gamma, xi and start.

## Implementer decisions

(empty)
