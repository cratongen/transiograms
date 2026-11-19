# transiograms-mcrf: algorithms

Signatures are in [spec.md](spec.md) section 4.7. Li (2007), Li and Zhang (2019), Allard, D'Or
and Froidevaux (2011).

## 1. Conditional probability

Neighbours g = 1..m at `x_g` with codes `c_g`, target `x_0`, lags `h_g = x_g - x_0`
(tail at x_0).

Eq. 14 (MCP, the one used): `P(k | data) proportional to p_k prod_g t_{k, c_g}(h_g)`.
Evaluate in log space: `s_k = ln p_k + sum_g ln t_{k, c_g}(h_g)`, then
`P = exp(s - logsumexp(s))`. If every `s_k = -inf`, return None.

Eq. 13 (oracle): with the chain coming from neighbour 1 at `x_1` (lag `h_10 = x_0 - x_1`, tail at
x_1): `P(k) proportional to t_{c_1, k}(h_10) prod_{g >= 2} t_{k, c_g}(h_g)`. Equals Eq. 14 for
every reversal-consistent model (`p_i t_ij(h) = p_j t_ji(-h)`), which every `TransitionModel`
is.

Zero-likelihood policy: when `conditional` returns None, drop the neighbour with the lowest
closeness (ties: the last found) and retry; with no neighbour left use p.

A neighbour at lag 0 makes the posterior one-hot (`T(0) = I`); hard cells report exactly
probability 1 on their code.

## 2. Neighbourhood

Scan the table offsets in order (closeness descending) from the node.
- Sectors: 2D quadrants by `(dx >= 0, dy >= 0)`, 3D octants by `(dx >= 0, dy >= 0, dz >= 0)` in
  grid axes, half-open as written; keep the first informed node per sector; stop when every
  sector is filled or the table ends.
- Nearest(n): keep the first n informed nodes.
No informed node within the table gives `P = p`.

## 3. MCP prediction

Informed nodes: hard data only. Every active non-hard cell gets Eq. 14 from its neighbourhood;
cells processed in parallel blocks of 4096 indices (pure function of the cell, so order does
not matter). Output `Prediction::from_probabilities` (argmax lowest index on ties, entropy
`-sum p ln p / ln K`). `corrected = 0`.

## 4. Sequential MCRF simulation (Li and Zhang 2007)

Per realization r: hard cells frozen and informed; path = unfrozen active cells shuffled with
stream `[r, PATH]`; at each node compute Eq. 14 from informed neighbours (hard and simulated),
draw with `categorical` from stream `[r, DRAW]`, mark informed. Realizations in parallel,
collected in index order. Memory check as in tpsim.

## 5. Theory checks

- Eq. 14 = Eq. 13 within 1e-12 for an asymmetric vertical model (Carle golden Q on z').
- 1D, `u_1 < u_0 < u_2`: Eq. 14 equals the Markov bridge
  `t_{c1, k}(h_10) t_{k, c2}(h_02) / t_{c1, c2}(h_10 + h_02)` within 1e-12.
- MCP brute force: K = 3, 4 neighbours: IPF on the full `3^5 = 243`-cell joint table with the
  star constraints `P(Z_0 = k, Z_g = j) = p_k t_kj(h_g)` (start uniform, 10^4 sweeps or
  max change below 1e-15) gives the maximum-entropy joint; its conditional at the observed codes
  equals Eq. 14 within 1e-10.
- Layer cake: a 2D field constant in y made from a 1D chain in x, MCP with neighbours along x
  only reproduces the 1D bridge.
- M6 test: grid 64 x 64 x 32 as tpsim.md section 6 with the same model and holes: every
  probability row `>= 0` summing to 1 within 1e-12; hard cells exactly one-hot; MCRF n = 20,
  seed 0, ensemble proportions within 0.02 after the variance assertion; hard data in every
  realization.

## Implementer decisions

- M6 grid. On the spec grid (64 x 64 x 32, cell 1 x 1 x 0.25) the tpsim.md section 5 variance
  assertion fails: 3 sigma = 0.180 / 0.141 / 0.118 for categories 0 / 1 / 2, against the 0.02
  required. The proportion check therefore runs on 96 x 96 x 32 with cell 4 x 4 x 1, where it
  holds. The spec grid still runs every other M6 assertion (probability rows, one-hot hard cells,
  hard data in each of the 20 realizations), and a test asserts that it fails the variance bound
  (`crates/mcrf/tests/acceptance.rs`).
- Memory check. `simulate` checks `n * n_cells * 4` bytes of codes before building the table,
  then codes plus the table (`(K^2 + 4) * 8` bytes per offset) against the same budget. Building
  the table runs its own check (`TransitionModel::table`, which adds working space for the build)
  against the same budget.
