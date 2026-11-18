# varvite

`varvite` is a Rust-core, Python-exposed library for Markov modelling in geostatistics (`import varvite as vv`).

Varvite is a sedimentary rock formed in proglacial lakes, where alternating summer and winter deposition creates rhythmic pairs of coarse and fine laminae (varves). The library takes its name from those layered sequences and focuses on transition-based models of geological categories along drillholes and across 3D grids.

The API draws inspiration from the compositional design of the original `pomegranate` library (explicit distributions, transition graphs, fit/predict/sample methods) while staying scoped to geostatistical sequences and spatial facies modelling. `varvite` is standalone and has zero dependency on `boitata`. The two libraries exchange drillhole intervals, category codes, coordinates, and block-model grids through NumPy arrays, and `varvite` follows `boitata`'s Python conventions (flat namespace, Rust-backed value types with JSON and pickle serialization, `.pyi` stubs, `vv.plot`, and `vv.datasets`).

## Stack

- **Core**: Rust workspace, `rayon` for thread-count-independent parallelism, `proptest` for property tests, `criterion` for benchmarks.
- **Python bindings**: `pyo3` (`abi3-py311`) built with `maturin`.
- **Runtime dependencies**: `numpy` as the sole required runtime dependency; `matplotlib` as the optional `plot` extra (`pip install "varvite[plot]"`).
- **Tooling**: `mise` for task running and tool pinning, `uv` for Python environments, `mkdocs` for documentation.

## Scope

### Stage 1: Sequences along drillholes and logs

1. **Markov chains**:
   - Tallying and fitting embedded transition matrices (bed-to-bed transitions between distinct lithologies) and continuous-lag transition matrices (fixed depth step $\Delta z$ along regular or composited logs).
   - Continuous-time transition rate matrices (generator matrix $\mathbf{Q}$) and matrix-exponential transition probabilities $\mathbf{P}(h) = \exp(\mathbf{Q}h)$.
   - Chi-square and likelihood-ratio tests for the Markov property (first-order vs independent, first-order vs second-order) and stationarity across drillholes or depth zones.
   - Seeded 1D simulation of categorical logs and bed-thickness sequences, unconditional or conditioned at interval endpoints.
2. **Hidden Markov and hidden semi-Markov models**:
   - Hidden Markov models (HMM) with Gaussian (single and multivariate log/assay responses), categorical, and Poisson emissions for lithology or electrofacies segmentation along drillholes.
   - Scaled forward-backward posterior state probabilities, Viterbi most-likely state sequence, and Baum-Welch (EM) parameter estimation with optional semi-supervised clamping at logged core intervals.
   - Hidden semi-Markov models (HSMM) with explicit state dwell-time (thickness) distributions (geometric, Poisson, negative binomial, empirical) to respect bed-thickness histograms that geometric HMM transitions distort.
   - Model selection via AIC, BIC, and cross-validated log-likelihood across drillholes.

### Stage 2: Space (2D and 3D grids)

1. **Transiograms**:
   - Experimental directional and omnidirectional transiograms $p_{ij}(\mathbf{h})$ from drillhole samples and regular grids.
   - Analytical transiogram models (exponential, spherical, Gaussian, and matrix-exponential linear Markov models) enforcing non-negativity and row-sum-to-one constraints $\sum_j p_{ij}(\mathbf{h}) = 1$ at every lag $\mathbf{h}$, with sill values anchored to target category proportions.
2. **Transition-probability simulation (T-PROGS style)**:
   - 3D directional Markov chain assembly from strike, dip, and vertical transition rates or hydrofacies/lithofacies mean lengths and juxtapositions.
   - Sequential indicator conditional simulation on 2D and 3D grids using a transition-probability co-kriging approximation, followed by simulated annealing quenching to honour directional transiograms and hard drillhole conditioning.
3. **Markov chain random fields (MCRF) and Markov-type categorical prediction (MCP)**:
   - Conditional probability estimation of categories at unsampled grid nodes from nearest directional neighbours under Bayesian Markov conditional independence.
   - Deterministic MCP most-probable category maps, classification entropy, and seeded sequential MCRF categorical simulation conditioned to drillholes.

## Non-goals

- General Bayesian networks or arbitrary factor graphs.
- Unstructured mixture models without sequence or spatial transition structure.
- Neural sequence models.
- Continuous-variable variography, kriging, or Gaussian simulation (those live in `boitata`).

## Roadmap

### M1: Markov chains on sequences

- **Scope**: Rust workspace and `pyo3` skeleton (`crates/core`, `crates/chain`, `crates/py`), `vv.MarkovChain` (embedded and fixed-step transition counting, smoothing, generator matrix fitting, stationary distribution, chi-square and G-test for Markov order and stationarity, seeded 1D simulation), `.pyi` stubs, and `vv.plot.transition_matrix`.
- **Acceptance check**: For any fitted `MarkovChain`, every row of the transition matrix sums to 1 within `1e-12`, and a seeded simulated chain of $10^5$ steps reproduces the input transition matrix and stationary proportions within a three-sigma binomial tolerance.

### M2: Hidden Markov models (HMM)

- **Scope**: `crates/hmm` and `vv.HiddenMarkovModel` with Gaussian (diagonal and full covariance) and categorical emissions, log-domain forward-backward, Viterbi decoding, Baum-Welch fitting with optional clamped states on logged intervals, and `vv.plot.strip_log`.
- **Acceptance check**: On a 4-state, 8-step test sequence, `hmm.viterbi(obs)` matches an exhaustive brute-force enumeration of all $4^8$ state paths in both path and log-probability (`atol=1e-12`), forward-backward state posteriors sum to 1 at every depth step, and log-likelihood is non-decreasing at every Baum-Welch iteration.

### M3: Hidden semi-Markov models and model selection

- **Scope**: `vv.HiddenSemiMarkovModel` with explicit dwell-time distributions (geometric, Poisson, negative binomial, empirical), forward-backward and Viterbi duration decoding, EM fitting, and `vv.select_order` / `vv.select_states` returning AIC, BIC, and leave-one-hole-out log-likelihood tables.
- **Acceptance check**: An HSMM configured with geometric dwell-time distributions produces the same log-likelihood (`atol=1e-10`) and Viterbi state sequence as the equivalent standard HMM on the same observations, and BIC recovers the true number of states $K=3$ on a seeded synthetic drillhole dataset.

### M4: Transiograms

- **Scope**: `crates/transiogram`, `vv.experimental_transiogram` (1D downhole and 3D directional lag tolerances), `vv.TransiogramModel` (matrix-exponential and coupled analytical models with row-sum-to-one projection), and `vv.plot.transiogram`.
- **Acceptance check**: At every lag $h \ge 0$ and for every source category $i$, the modelled transition probabilities satisfy $p_{ij}(h) \ge 0$ and $\sum_j p_{ij}(h) = 1$ within `1e-12`, equal the Kronecker delta $\delta_{ij}$ at $h = 0$, and converge to the target category proportions as $h \to \infty$.

### M5: Transition-probability simulation on grids

- **Scope**: `crates/tpsim` T-PROGS engine (`vv.tprogs_simulate`), 3D anisotropy from principal Markov rates or mean lens dimensions, sequential transition-probability indicator simulation, and simulated annealing quenching on 2D/3D grids conditioned to point or interval samples.
- **Acceptance check**: Every realization reproduces all hard conditioning grid nodes without exception, and the ensemble average over 20 seeded realizations reproduces input category proportions within `0.02` and input directional auto-transiograms within `0.05` RMSE up to two correlation lengths.

### M6: MCRF and MCP on grids

- **Scope**: `vv.mcp_predict` (deterministic conditional probability cubes, most-probable category, and Shannon entropy) and `vv.mcrf_simulate` (sequential Markov chain random field realizations on 2D/3D grids conditioned to drillholes).
- **Acceptance check**: Predicted MCP probabilities at every grid node are non-negative and sum to 1 across categories within `1e-12`, hard conditioning nodes return a degenerate probability of 1 on the observed category, and seeded MCRF realizations reproduce target category proportions within `0.02`.

### M7: `boitata` integration example

- **Scope**: Worked end-to-end documentation example (`examples/07-boitata-integration/`) and test showing how `boitata` desurveyed/composited drillhole intervals pass as NumPy arrays into `varvite` for HMM lithology segmentation, 3D transiogram fitting, and MCRF/T-PROGS categorical grid simulation, and how the resulting categorical array feeds back into `boitata` for domain-controlled kriging.
- **Acceptance check**: The standalone integration check passes NumPy drillhole arrays into `varvite` and returns a 3D integer category grid matching the target grid shape, with 100% hard-data agreement at conditioning cells and category proportions within `0.02` of target proportions.

## Performance

Every performance number in this repository comes from a reproducible benchmark we run on the same machine and in the same session, using interleaved rounds and reporting medians alongside machine load. We benchmark Rust kernels with `criterion` and compare end-to-end Python calls against `hmmlearn` and `pomegranate`.

We have not run benchmarks yet (this repository holds the charter and roadmap only):

| Workload | `varvite` | `hmmlearn` | `pomegranate` |
| --- | --- | --- | --- |
| Markov chain transition tally ($10^6$ samples, 8 categories) | not measured | not measured | not measured |
| HMM forward-backward ($10^4$ steps, 6 states, 4 Gaussian channels) | not measured | not measured | not measured |
| HMM Viterbi decoding ($10^4$ steps, 6 states, 4 Gaussian channels) | not measured | not measured | not measured |
| HMM Baum-Welch 20 iterations (100 drillholes $\times$ 500 steps) | not measured | not measured | not measured |
| 3D T-PROGS / MCRF grid simulation ($100 \times 100 \times 50$, 4 facies) | not measured | not measured | not measured |

## License

MIT
