# Changelog

transiograms follows [semantic versioning](https://semver.org/). Each release lists what changed
for someone who uses the Python package.

## 0.1.0 (unreleased)

First public release.

- Markov chains along drillholes and logs (`tg.MarkovChain`): embedded and fixed-step transition
  matrices, continuous-lag generators with `P(h) = exp(Q h)`, independence, order, symmetry and
  homogeneity tests, entropy, sojourn statistics, and seeded 1D simulation of logs and beds,
  unconditional or bridged between known endpoints.
- Hidden Markov models (`tg.HiddenMarkovModel`) with categorical, Gaussian, Poisson and
  Gaussian-mixture emissions: forward-backward, Viterbi, Baum-Welch with seeded restarts, clamped
  labels, and AIC, BIC and ICL.
- Hidden semi-Markov models (`tg.HiddenSemiMarkovModel`) with geometric, shifted Poisson, negative
  binomial and empirical duration laws, and `tg.select_states` with cross-validated likelihood.
- Transiograms: experimental transiograms down holes, between points and on grids; Markov and
  entrywise 1D models whose rows sum to one at every lag; the 3D `tg.TransitionModel` from rates
  or mean lengths.
- Grid simulation: transition-probability cokriging and T-PROGS simulation with quenching
  (`tg.tp_cokriging`, `tg.tprogs_simulate`); MCP prediction and MCRF simulation
  (`tg.mcp_predict`, `tg.mcrf_simulate`), conditioned to drillholes.
- Fit-quality reports: `tg.fit_quality` compares realizations with their model, and
  `TransiogramModel.residuals` gives the misfit of a 1D fit. The measured gaps of MCRF and
  T-PROGS to their model are documented limits of the methods, pinned by seeded tests.
- `tg.plot` (with the `plot` extra), seeded `tg.datasets`, and NumPy reference checks in
  `tg.checks`.
- Every stochastic routine takes a `seed` and gives the same bits on any number of threads; grid
  work is sized from the available memory and checked before it allocates.
- One abi3 wheel per platform for CPython 3.11 and newer; NumPy is the only runtime dependency.
  Licensed under MIT OR Apache-2.0.

Known limits of this release: the API can change between minor versions, `tg.Categories` and
`tg.Grid` will move to a shared types package, performance is not measured, and the Rust crates
are not on crates.io.
