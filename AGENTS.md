# varvite

## Purpose

varvite is a Rust-core, Python-exposed library (`import varvite as vv`) for
Markov modelling in geostatistics: 1D sequence models along drillholes and
logs (Markov chains, hidden Markov models, hidden semi-Markov models) and 3D
spatial categorical models on grids (transiograms, transition-probability
simulation, Markov chain random fields, Markov-type categorical prediction).
It has zero dependency on boitata and exchanges drillhole intervals, category
codes, and grids with boitata through NumPy arrays.

## Commands

Tasks live in `mise.toml` (Python, uv, pixi, micromamba, git-cliff) and
`rust-toolchain.toml` (Rust):

```sh
mise install        # one-time setup
mise run build      # cargo build --workspace
mise run test       # rust tests, maturin develop, pytest
mise run lint       # rustfmt --check, clippy -D warnings, ruff check + format --check
mise run verify     # lint + test (must pass before a PR)
mise run bench      # cargo bench (no benches yet)
mise run examples   # rerun examples/*/*/example_*.py and rewrite their README.md and figures
mise run docs       # mkdocs serve
mise run docs-build # mkdocs build --strict
mise run changelog  # git-cliff -o CHANGELOG.md
mise run compat     # install the wheel via pip, uv, pixi and conda (mode: all, venv, uv, pixi, conda)
```

A single crate: `mise exec -- cargo test -p varvite-<short>`.

## Layout

The API contract is `docs/design/spec.md`, with one algorithm file per crate beside it.

- `crates/` - Rust workspace, directory `crates/<short>`, package `varvite-<short>`:
  - `core` - category codes, sequences and beds, transition and rate matrices, grids, seeded
    streams, numerics (GTH, matrix exponential and logarithm, IPF), special functions, memory
    budget, and `synthetic` (seeded test data with known parameters)
  - `chain` - embedded, fixed-step and continuous-lag Markov chains, tests, entropy, sojourns, 1D simulation
  - `hmm` - hidden Markov models: emissions, forward-backward, Viterbi, Baum-Welch
  - `hsmm` - hidden semi-Markov models and state-count selection (depends on `hmm`)
  - `transiogram` - experimental transiograms, 1D models and the 3D `TransitionModel`
  - `tpsim` - T-PROGS cokriging, sequential simulation and quenching (depends on `transiogram`)
  - `mcrf` - MCRF simulation and MCP prediction (depends on `transiogram`)
  - `py` - pyo3 (`abi3-py311`) binding crate `_varvite`, one module per area
- `python/varvite/` - Python package (`__init__.py`, `_varvite.pyi` stub, `plot.py`, `datasets.py`, `checks.py`, `errors.py`).
- `python/tests/` - pytest suite (`test_<area>.py`).
- `examples/` - numbered narrative scripts rendered into docs (`mise run examples`).
- `docs/` - mkdocs site (`index.md`, `install.md`, `api/`, `design/`).

New Rust algorithms go into the matching `crates/` subdomain, their pyo3 bindings
go into `crates/py/src/<area>.rs`, and the Python wrapper and `.pyi` stub go into `python/varvite/`.
Shared files carry `// >>> <area>` / `# >>> <area>` markers; edit only inside your area's markers.

## Conventions

- Conventional Commits (`feat:`, `fix:`, `docs:`, `bench:`, `test:`) and one logical change per PR.
- Every feature ships together: Rust implementation, pyo3 binding, `.pyi` stub,
  NumPy-style docstring, runnable example, and a rerunnable theory check (for example:
  transition matrix rows sum to 1, Viterbi matches brute-force search on a small
  sequence, simulated grids reproduce input proportions and transiograms within tolerance).
- Follow boitata's Python API style: flat top-level namespace (`import varvite as vv`),
  keyword-only optional arguments, `to_json` / `from_json` and `copyreg` pickle support on
  Rust types, error hierarchy rooted at `VarviteError` (`InvalidInput`, `ConvergenceError`),
  NumPy arrays as the universal exchange format, and `vv.plot` / `vv.datasets` submodules.
- Seeded, reproducible randomness everywhere: every stochastic routine takes an explicit
  integer `seed` and produces bit-identical output across runs.
- Results must not depend on Rayon thread count. Partition per-node or per-sequence RNG
  streams by index rather than thread arrival order.
- No fixed RAM caps in simulation or grid buffers. Size memory budgets from available
  system RAM with an optional user override, and raise `InvalidInput` before allocating
  when a request exceeds the budget.
- Run performance benchmarks with `criterion` and interleaved Python comparison rounds,
  reporting medians and machine load. Write "not measured" until you run a benchmark.
- Write comments only for non-obvious invariants, numerical stability guards (such as
  log-space scaling in forward-backward), or literature equations.

## Domain glossary

- **Embedded transition matrix**: transition probabilities between successive distinct beds or intervals, ignoring bed thickness (diagonal entries are zero).
- **Continuous-lag transition matrix**: transition probabilities at a fixed sampling step $\Delta z$ along a drillhole or log, including self-transitions.
- **Emission distribution**: probability distribution $P(y_t \mid s_t = k)$ of observed log or assay values given hidden geological state $k$ (Gaussian or categorical).
- **Dwell-time distribution**: explicit probability distribution of state run length (thickness in steps) in a hidden semi-Markov model (HSMM).
- **Transiogram**: spatial transition probability $p_{ij}(\mathbf{h}) = P(Z(\mathbf{x}+\mathbf{h}) = j \mid Z(\mathbf{x}) = i)$ as a function of lag vector $\mathbf{h}$; diagonal curves are auto-transiograms and off-diagonal curves are cross-transiograms.
- **T-PROGS**: transition-probability geostatistical simulation on a grid using a 3D Markov chain Cokriging / sequential indicator approximation and simulated annealing quenching.
- **MCRF / MCP**: Markov chain random field and Markov-type categorical prediction, estimating conditional category probabilities at an unsampled grid node from its nearest conditioning neighbours via directional transiograms under conditional independence.

## Never do

- Never add AI attribution or co-author trailers (`Co-Authored-By`, "Generated with Claude Code") to commits or PRs.
- Never add `boitata` as a build or runtime dependency; exchange data through NumPy arrays.
- Never commit proprietary data, client drillholes, or files over 1 MB; use `vv.datasets` or seeded synthetic sequences and grids.
- Never introduce thread-count-dependent outputs or unseeded randomness.
- Never hardcode fixed RAM ceilings (such as a static 1 GiB or 8 GiB limit) on simulation grids.
- Never estimate or invent benchmark numbers; write "not measured" until measured.
- Never push or force-push without asking first, and never call a code task done without running `mise run verify`.

## Definition of done

- `mise run verify` passes (`lint` + `test`).
- New or changed Rust functionality includes its pyo3 binding, `.pyi` stub, NumPy docstring, example, and theory check.
- `mise run docs-build` succeeds when documentation or examples change.
