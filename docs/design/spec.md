# transiograms API contract

This file is the single source that every implementer follows. Names, signatures, defaults,
array shapes and error behaviour written here are binding. Algorithms, formulas and golden
values live in the sibling files, one per crate:

- [core.md](core.md): numerics, RNG streams, sequences, grids, memory budget.
- [chain.md](chain.md): Markov chains, tests, entropy, sojourns, 1D simulation.
- [hmm.md](hmm.md): hidden Markov models.
- [hsmm.md](hsmm.md): hidden semi-Markov models and model selection.
- [transiogram.md](transiogram.md): experimental transiograms and transition models.
- [tpsim.md](tpsim.md): transition-probability cokriging, sequential simulation, quenching.
- [mcrf.md](mcrf.md): MCRF simulation and MCP prediction.

If something an implementer needs is missing here, they choose the simplest option that fits
the rules below, write it in their crate's design file under a heading "Implementer decisions",
and report it. They never change a name or signature in this file on their own.

## 1. Decisions that differ from the charter or the research

Each line: the decision, then why.

1. Crates follow the fixed eight-crate architecture (core, chain, hmm, hsmm, transiogram, tpsim,
   mcrf, py), not the six crates in the README layout: the split lets seven implementers work in
   parallel without sharing a crate. The core owner updates the README "Layout" section.
2. Directories are `crates/<short>` and packages are `transiograms-<short>` (as boitata does with
   `crates/core` and `boitata-core`): short paths, unique package names.
3. "Continuous-lag" means the Carle and Fogg generator chain `T(h) = exp(Qh)`; the step-`dz`
   chain is called "fixed". The README glossary uses "continuous-lag" for the step chain; Carle
   and Fogg's usage is the one the T-PROGS literature and the transiogram crate share.
4. There is no separate "continuous" fitting kind. `MarkovChain.fit(kind="embedded")` always
   returns the generator MLE `q_jk = n_jk / T_j` as `.generator`: it uses exactly the embedded
   tallies, so a third kind would duplicate it.
5. M1 acceptance: transition probabilities use the binomial three-sigma tolerance, but stationary
   proportions use the Kemeny and Snell variance (chain.md section 8). The binomial tolerance
   fails a correct implementation (4.6 binomial sigma on a verified example).
6. M5 and M6 proportion tolerance stays `0.02`, but each acceptance test first asserts that three
   standard deviations from the grid-variance formula (tpsim.md section 7) are below `0.02` for
   its grid and model, so the test cannot pass or fail by luck.
7. Own RNG (xoshiro256++ seeded through SplitMix64) instead of `rand`: twenty lines, and bit
   identity no longer depends on a third-party crate's algorithm choices across versions.
8. Seed derivation copies boitata's `splitmix` and `realization_seed` bit for bit, so `seed=s`,
   realization `i` names the same key in both libraries.
9. `nalgebra` is the one linear-algebra dependency (Cholesky, LU, SVD, symmetric eigen, Pade
   `exp`): none of these is a few lines.
10. Special functions (log-gamma, digamma, regularized incomplete gamma) are written in core
    (about 80 lines) instead of adding `statrs`.
11. Gaussian emissions support `full`, `diag` and `spherical`, not hmmlearn's `tied`: a tied
    covariance couples states and breaks the one-distribution-object-per-state design.
12. Poisson emissions are included (the README Stage 1 scope lists them) although the fixed
    architecture names only four families.
13. Covariance regularisation is a proper MAP prior (hmm.md section 5), not a variance floor:
    EM stays monotone in log L + log prior, which the M2 check needs.
14. HSMM has right censoring only. Left censoring (equilibrium first segment) is not built: it
    has no closed-form duration M-step and the first-bed bias is small next to right censoring.
15. HSMM `max_duration=None` means the longest sequence length: exact (no truncation) and sized
    from data, never from a fixed cap.
16. The TP cokriging SVD cutoff defaults to `1e-8` relative, not TSIM's `1e-3`: the
    transition-basis system is nonsingular when data are distinct, and `1e-3` discards real
    information and breaks the exact single-datum and screening checks.
17. Only the transition-probability basis is built for cokriging; the covariance basis is not.
18. MCRF and MCP both use Li and Zhang Eq. 14 (= MCP). Eq. 13 equals it for every
    reversal-consistent model, and every `TransitionModel` is reversal-consistent; Eq. 13 is a
    test oracle only.
19. MCP and MCRF zero-likelihood policy: drop the neighbour with the lowest closeness and retry,
    never floor `t` at an epsilon (a floor changes every probability silently).
20. Hard data are snapped as points (one code per cell, nearest point to the centroid wins).
    Intervals are passed as composited midpoints, which boitata's `composite` already produces.
21. Grids are axis-aligned. A rotated boitata block model is handled by passing coordinates in
    its local frame. Anisotropy rotation lives in `TransitionModel.rotation`.
22. Memory budget: Linux reads `MemAvailable` from `/proc/meminfo`, Windows the available physical
    memory and commit charge from `GlobalMemoryStatusEx`, macOS the installed memory from
    `sysctlbyname("hw.memsize")` (an upper bound). Other platforms have no automatic budget and
    only `memory=` is enforced.
23. Free test functions are named `order_test`, `homogeneity_test` and methods
    `independence_test`, `symmetry_test`: names starting with `test_` would be collected by pytest
    when imported into a test module.
24. No parametric-bootstrap p-values on the Markov tests, no reversible fast path for `exp(Qh)`,
    no multigrid path, no quenching by graph colouring: none is needed for any milestone.
25. All tests run in `mise run test`; there is no `slow` marker. `[profile.test]` uses
    `opt-level = 3` so the acceptance runs fit in normal test time.
26. Experimental transiograms are returned as plain arrays in a value type, with the counts.
    No Arrow, no `Table` type: transiograms's only runtime dependency is NumPy.
27. `direction` defaults to `"up"` (older to younger, +z up, T-PROGS usage) in every sequence
    model, HMM and HSMM included. Inputs are always given top-down by depth; outputs are always
    returned aligned with input rows.

## 2. Workspace and tooling

### 2.1 Crates

| Directory | Package | Lib name | Depends on (path) |
| --- | --- | --- | --- |
| `crates/core` | `transiograms-core` | `transiograms_core` | none |
| `crates/chain` | `transiograms-chain` | `transiograms_chain` | core |
| `crates/hmm` | `transiograms-hmm` | `transiograms_hmm` | core |
| `crates/hsmm` | `transiograms-hsmm` | `transiograms_hsmm` | core, hmm |
| `crates/transiogram` | `transiograms-transiogram` | `transiograms_transiogram` | core |
| `crates/tpsim` | `transiograms-tpsim` | `transiograms_tpsim` | core, transiogram |
| `crates/mcrf` | `transiograms-mcrf` | `transiograms_mcrf` | core, transiogram |
| `crates/py` | `transiograms-py` | `_transiograms` (cdylib) | all of the above |

### 2.2 Root `Cargo.toml`

```toml
[workspace]
members = ["crates/*"]
resolver = "3"

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.97"
license = "MIT OR Apache-2.0"
publish = false
repository = "https://github.com/cratongen/transiograms"
homepage = "https://cratongen.github.io/transiograms/"

[workspace.dependencies]
nalgebra = "0.33"
rayon = "1.10"
serde = { version = "1", features = ["derive"] }
serde_json = { version = "1", features = ["float_roundtrip"] }
thiserror = "1"
pyo3 = { version = "0.29", features = ["abi3-py311"] }
numpy = "0.29"
proptest = "1"
criterion = "0.8"
transiograms-core = { version = "0.1.0", path = "crates/core" }
transiograms-chain = { version = "0.1.0", path = "crates/chain" }
transiograms-hmm = { version = "0.1.0", path = "crates/hmm" }
transiograms-hsmm = { version = "0.1.0", path = "crates/hsmm" }
transiograms-transiogram = { version = "0.1.0", path = "crates/transiogram" }
transiograms-tpsim = { version = "0.1.0", path = "crates/tpsim" }
transiograms-mcrf = { version = "0.1.0", path = "crates/mcrf" }

[workspace.lints.clippy]
needless_range_loop = "allow"
type_complexity = "allow"

[profile.test]
opt-level = 3

[profile.dev.package."*"]
opt-level = 3
```

No other dependency may be added without changing this file. Every algorithm crate uses
`transiograms-core`, `nalgebra`, `serde`, `rayon` as needed, `thiserror` only in core, and `proptest`
plus `criterion` as dev-dependencies. `serde_json` is used by core (JSON helpers) and py.

### 2.3 Toolchain and tasks

- `rust-toolchain.toml`: channel `1.97.1`, components `rustfmt`, `clippy`. `rustfmt.toml`:
  `newline_style = "Unix"`.
- `mise.toml` tools: `python = "3.13"`, `uv = "0.12"`, `pixi = "0.81"`, `micromamba = "2.9"`,
  `git-cliff = "2.13"`. Env: `_.path = ["~/.cargo/bin"]`, `UV_PYTHON = "3.13"`.
- Tasks:

| Task | Runs |
| --- | --- |
| `build` | `cargo build --workspace` |
| `py:sync` | `uv sync -q --inexact` |
| `py:build` (depends `py:sync`) | `uv run --no-sync maturin develop -q --uv --release` |
| `test` (depends `py:build`) | `cargo test --workspace --exclude transiograms-py`, then `uv run --no-sync pytest -q` |
| `lint` (depends `py:sync`) | `cargo fmt --all --check`; `cargo clippy --workspace --all-targets -- -D warnings`; `uv run --no-sync ruff check python examples ci`; `uv run --no-sync ruff format --check python examples ci` |
| `verify` | depends `lint`, `test` |
| `bench` | `cargo bench --workspace --exclude transiograms-py` |
| `examples` (depends `py:build`) | `uv run --no-sync python examples/render.py` |
| `docs` / `docs-build` (depend `py:build`) | `mkdocs serve` / `mkdocs build --strict` via `uv run --no-sync` |
| `changelog` | `git-cliff -o CHANGELOG.md` |
| `compat` | `uv run --no-sync python ci/compat.py {{mode}}` (all, venv, uv, pixi, conda) |

A single crate is tested with `mise exec -- cargo test -p transiograms-<short>`.

### 2.4 `pyproject.toml`

maturin `>=1.10,<2` backend; `[tool.maturin] manifest-path = "crates/py/Cargo.toml"`,
`module-name = "transiograms._transiograms"`, `python-source = "python"`, `locked = true`.
`requires-python = ">=3.11"`, `dependencies = ["numpy>=1.26"]`, extras `plot = ["matplotlib>=3.8"]`,
`all = ["matplotlib>=3.8"]`. Dev group: `maturin>=1.10,<2`, `pytest>=8`, `ruff>=0.6`, `scipy>=1.13`
(tests only), `matplotlib>=3.8`, `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-gallery`.
`[tool.uv] package = false`. `[tool.pytest.ini_options] testpaths = ["python/tests"]`.
`[tool.ruff] line-length = 110`.

## 3. Conventions shared by every crate

### 3.1 Codes, missing values, categories

- A category code is `u32` in Rust (`transiograms_core::Code`), `0..k-1`. Missing is
  `transiograms_core::MISSING = u32::MAX`.
- Python gives and receives codes as `int64` with `-1` for missing. Float input is accepted when
  every finite value is an integer; `NaN` means missing (boitata's `Categories.encode` output).
  Any other value raises `InvalidInput` ("codes must be integers in 0..k-1, with -1 or NaN for missing"; text labels add a hint to encode them with `Categories.from_labels(labels).encode(labels)`).
- `k` (number of categories) is the `categories` argument when given (an `int` or a
  `Categories`), otherwise `max(code) + 1`.
- A `Categories` value is a code table: `names` and optional `colors`. Its JSON form
  `{"names": [...], "colors": [...]}` is the subset of boitata's scheme JSON; `from_json` rejects
  any other key with `InvalidInput` naming it.

### 3.2 Positions and direction

- Every drillhole input is top-down by depth: `from_ < to`, depth increasing downward. Users with
  elevations pass `depth = -elevation`.
- `Direction::Up` (Python `"up"`, the default) runs from the deepest row to the shallowest,
  older to younger, +z up. `Direction::Down` runs in input order.
- Inside the engines a run is described by a position along the direction:
  `position = depth` for Down and `position = -depth` for Up, so position always increases along
  the direction of travel.
- Every result aligned with rows (posteriors, decoded paths, simulated codes at given rows) is
  returned in input row order whatever the direction.
- Grids: x east, y north, z up. Cell index runs x fastest, then y, then z:
  `index = i + nx * (j + ny * k)` (boitata `Geometry` order).
- Transiogram lag `h = head - tail`; `t_ij(h) = P(Z(x + h) = j | Z(x) = i)`.

### 3.3 Seeds, streams and determinism

- Every stochastic routine takes `seed: u64` (Python `seed: int = 0`, `0 <= seed < 2**64`).
- `transiograms_core::rng::realization_seed(seed, k) = splitmix(splitmix(seed) ^ k)`, identical to
  boitata. A stream is `Stream::new(seed, &[k1, k2, ...])`, whose key is
  `keys.fold(seed, realization_seed)`.
- Key layouts are fixed per routine (core.md section 3 lists them). No routine draws from a
  stream shared across rayon tasks.
- Reductions over sequences, holes, folds, restarts, blocks or realizations collect results in a
  `Vec` indexed by the item and reduce serially in index order. Parallel blocks have a fixed size
  that never depends on the thread count.
- Same inputs and seed give bit-identical outputs for any `RAYON_NUM_THREADS`. Every crate has a
  test that runs with 1 and 4 threads (via a local `rayon::ThreadPoolBuilder`) and compares bits.

### 3.4 Memory

- `transiograms_core::memory::check(needed: u64, memory: Option<u64>, what: &str) -> Result<()>`.
  The budget is `memory` when given, else `available_bytes()` (Linux, Windows and macOS), else
  no check. A request above
  the budget raises `InvalidInput` before allocating, naming `what`, the bytes needed and the
  budget.
- Python functions that allocate per cell or per realization take `memory: int | None = None`
  (bytes).

### 3.5 Errors

Rust (`transiograms_core::error`):

```rust
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Convergence(String),
}
pub type Result<T> = std::result::Result<T, Error>;
pub fn invalid<T>(message: impl Into<String>) -> Result<T>;
pub fn convergence<T>(message: impl Into<String>) -> Result<T>;
```

Python (`python/transiograms/errors.py`):

```python
class TransiogramsError(Exception): ...


class InvalidInput(TransiogramsError, ValueError): ...


class ConvergenceError(TransiogramsError, RuntimeError): ...
```

`Error::InvalidInput` maps to `InvalidInput` and `Error::Convergence` to `ConvergenceError`
(through `crates/py/src/lib.rs::error(kind, msg)` as in boitata). Argument coercion failures also
raise `InvalidInput`. Messages say what to give, name the offending item (hole id and depth,
entry `(i, j)`, state, iteration), and never end with a period.

### 3.6 Numerical contract

- `transiograms_core::ROW_TOL = 1e-12`. Every transition matrix has entries in `[0, 1]` and row sums
  within `ROW_TOL` of 1. Every generator has off-diagonals `>= 0`, diagonals `< 0` and row sums
  within `ROW_TOL * max|q|` of 0. Constructors validate and raise `InvalidInput`.
- Probabilities in log space use `-inf` for zero; `logsumexp` of all `-inf` is `-inf`, never NaN.
- `0 * ln 0 = 0` in every entropy, G statistic and ICL term.

### 3.7 Serialisation

- Every public value type derives `serde::Serialize` and `Deserialize`. Matrices serialise as
  row-major nested lists through `transiograms_core::serde_mat` (`#[serde(with = "...")]`).
- Python classes have `to_json() -> str`, `from_json(text) -> Self` (staticmethod), `__eq__`, a
  one-line `__repr__`, and pickle through `copyreg` registered in `__init__.py` on
  `to_json` / `from_json`.

## 4. Rust API

`Mat` is `nalgebra::DMatrix<f64>`, re-exported as `transiograms_core::Mat`. Vectors are `Vec<f64>` or
slices. All functions return `Result<T>` from core unless the signature says otherwise.

### 4.1 transiograms-core

Modules: `error.rs`, `codes.rs`, `sequence.rs`, `matrix.rs`, `grid.rs`, `rng.rs`,
`numerics.rs`, `special.rs`, `memory.rs`, `serde_mat.rs`, `lib.rs` (re-exports below).

```rust
pub use nalgebra;
pub type Mat = nalgebra::DMatrix<f64>;
pub const ROW_TOL: f64 = 1e-12;
pub use error::{Error, Result, invalid, convergence};
pub use codes::{Code, MISSING, Categories};
pub use sequence::{Direction, Hole, SequenceSet, Run, Bed};
pub use matrix::{TransitionMatrix, RateMatrix, EmptyRow, Projection};
pub use grid::{Grid, Snapped, Prediction, Realizations};
pub use rng::Stream;
```

`codes.rs`

```rust
pub type Code = u32;
pub const MISSING: Code = u32::MAX;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Categories {
    pub names: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colors: Option<Vec<String>>,
}
impl Categories {
    /// Non-empty, unique names, at most 65535; colors, when given, one per name.
    pub fn new(names: Vec<String>, colors: Option<Vec<String>>) -> Result<Self>;
    /// Sorted unique non-null labels; numeric sort when every label parses as i64.
    pub fn from_labels<'a>(labels: impl IntoIterator<Item = Option<&'a str>>) -> Result<Self>;
    pub fn len(&self) -> usize;
    pub fn code(&self, name: &str) -> Option<Code>;
    pub fn name(&self, code: Code) -> Option<&str>;
    /// None -> MISSING; unknown labels -> InvalidInput listing up to 10 with their counts.
    pub fn encode<'a>(&self, labels: impl IntoIterator<Item = Option<&'a str>>) -> Result<Vec<Code>>;
}
/// Every code < k or MISSING, else InvalidInput naming the first bad row.
pub fn check_codes(codes: &[Code], k: usize) -> Result<()>;
/// max(code) + 1 over non-missing codes; 0 when all are missing.
pub fn n_categories(codes: &[Code]) -> usize;
```

`sequence.rs`

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction { Up, Down }

/// One drillhole, rows sorted by `from`, intervals non-overlapping (to[i] <= from[i+1] + 1e-9).
#[derive(Clone, Debug, PartialEq)]
pub struct Hole { pub id: String, pub from: Vec<f64>, pub to: Vec<f64>, pub codes: Vec<Code>, pub rows: Vec<usize> }

#[derive(Clone, Debug, PartialEq)]
pub struct SequenceSet { pub holes: Vec<Hole>, pub n_categories: usize }

/// A maximal stretch of one hole with no missing code and no gap, in direction order.
/// `start[i] < end[i]` are positions along the direction (section 3.2).
#[derive(Clone, Debug, PartialEq)]
pub struct Run { pub hole: usize, pub start: Vec<f64>, pub end: Vec<f64>, pub codes: Vec<Code>, pub rows: Vec<usize> }

/// A bed: adjacent same-code intervals of one run merged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bed { pub code: Code, pub thickness: f64, pub open_start: bool, pub open_end: bool }

impl SequenceSet {
    /// Rows grouped by hole id in first-appearance order; `rows` keeps input row indices.
    pub fn from_intervals(holes: &[String], from: &[f64], to: &[f64], codes: &[Code], k: usize) -> Result<Self>;
    /// Each sample becomes [depth - step/2, depth + step/2].
    pub fn from_samples(holes: &[String], depth: &[f64], codes: &[Code], step: f64, k: usize) -> Result<Self>;
    /// Each inner Vec is one hole with unit step; hole ids are "0", "1", ...
    pub fn from_regular(codes: &[Vec<Code>], k: usize) -> Result<Self>;
    /// Split at MISSING codes and at gaps larger than `gap`.
    pub fn runs(&self, direction: Direction, gap: f64) -> Vec<Run>;
    /// Per run; when `merge` is false, adjacent same-code intervals stay separate beds
    /// (self-transitions). First bed of a run has open_start, last has open_end.
    pub fn beds(&self, direction: Direction, gap: f64, merge: bool) -> Vec<Vec<Bed>>;
    /// Point-sampled codes per run at positions start + (i + 1/2) step < end of run.
    pub fn regular(&self, direction: Direction, step: f64, gap: f64) -> Result<Vec<Vec<Code>>>;
}
```

`matrix.rs`

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmptyRow { Raise, Uniform }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionMatrix { #[serde(with = "crate::serde_mat")] p: Mat }
impl TransitionMatrix {
    pub fn new(p: Mat) -> Result<Self>;
    /// (n_ij + a) / (n_i. + a * allowed_i); allowed cells exclude the diagonal when `embedded`.
    /// A row with no count and a = 0 follows `on_empty` (Uniform: uniform on allowed cells).
    pub fn from_counts(counts: &Mat, pseudocount: f64, embedded: bool, on_empty: EmptyRow) -> Result<Self>;
    pub fn k(&self) -> usize;
    pub fn matrix(&self) -> &Mat;
    pub fn is_embedded(&self) -> bool;          // every diagonal exactly 0
    pub fn stationary(&self) -> Result<Vec<f64>>; // GTH on P - I; InvalidInput if reducible
    pub fn power(&self, n: u64) -> Mat;         // repeated squaring
    pub fn reversed(&self, pi: &[f64]) -> Result<TransitionMatrix>; // P*_ij = pi_j P_ji / pi_i
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Projection { pub distance: f64, pub exact: bool } // max |change| of the generator

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RateMatrix { #[serde(with = "crate::serde_mat")] q: Mat }
impl RateMatrix {
    pub fn new(q: Mat) -> Result<Self>;
    /// Q = diag(1 / L) (Pi - I); Pi embedded, L > 0 finite.
    pub fn from_embedded(pi: &TransitionMatrix, mean_lengths: &[f64]) -> Result<Self>;
    /// logm(P) / step, projected to the nearest generator (core.md section 6).
    pub fn from_transition(p: &TransitionMatrix, step: f64) -> Result<(Self, Projection)>;
    /// Carle background fill: rates given for all i, j != q (diagonals included); row and
    /// column q filled from row sums and p^T Q = 0. InvalidInput names a negative filled entry.
    pub fn from_partial(partial: &Mat, proportions: &[f64], background: usize) -> Result<Self>;
    /// Maximum-entropy (quasi-independent) embedded chain from p and L (core.md section 7).
    pub fn max_entropy(proportions: &[f64], mean_lengths: &[f64]) -> Result<Self>;
    pub fn k(&self) -> usize;
    pub fn matrix(&self) -> &Mat;
    pub fn mean_lengths(&self) -> Vec<f64>;        // -1 / q_jj
    pub fn embedded(&self) -> TransitionMatrix;    // q_jk / -q_jj, zero diagonal
    pub fn stationary(&self) -> Result<Vec<f64>>;  // GTH on Q
    pub fn transition(&self, h: f64) -> TransitionMatrix; // h >= 0, uniformization
    pub fn reversed(&self, p: &[f64]) -> RateMatrix;      // D^-1 Q^T D
    pub fn is_reversible(&self, p: &[f64], tol: f64) -> bool;
}
```

`grid.rs`

```rust
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grid { pub origin: [f64; 3], pub size: [f64; 3], pub count: [usize; 3] }
impl Grid {
    pub fn new(origin: [f64; 3], size: [f64; 3], count: [usize; 3]) -> Result<Self>;
    pub fn n_cells(&self) -> usize;
    pub fn is_2d(&self) -> bool;                       // count[2] == 1
    pub fn index(&self, ijk: [usize; 3]) -> usize;
    pub fn ijk(&self, index: usize) -> [usize; 3];
    pub fn centroid(&self, index: usize) -> [f64; 3];
    /// Cells are [o + i s, o + (i + 1) s); the far face of the last cell is included.
    pub fn locate(&self, xyz: [f64; 3]) -> Option<usize>;
    pub fn offset(&self, index: usize, d: [i64; 3]) -> Option<usize>;
    pub fn lag(&self, d: [i64; 3]) -> [f64; 3];        // d * size
    pub fn snap(&self, coords: &[[f64; 3]], codes: &[Code], mask: Option<&[bool]>) -> Result<Snapped>;
}
/// Points outside the grid or mask, or with MISSING codes, are skipped and counted in `outside`.
/// Several points in one cell: nearest to the centroid wins, ties to the lowest input index;
/// `conflicts` counts cells whose points disagree. `cells` ascending.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapped { pub cells: Vec<usize>, pub codes: Vec<Code>, pub conflicts: usize, pub outside: usize }

/// Per-cell probabilities (n_cells * k, row per cell); inactive cells hold NaN, most_likely MISSING.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction { pub k: usize, pub probabilities: Vec<f64>, pub most_likely: Vec<Code>, pub entropy: Vec<f64>, pub corrected: f64 }

/// codes is n * n_cells, realization-major; inactive cells hold MISSING.
#[derive(Clone, Debug, PartialEq)]
pub struct Realizations { pub n: usize, pub n_cells: usize, pub k: usize, pub codes: Vec<Code>, pub corrected: Vec<f64>, pub objective: Vec<Vec<f64>> }
impl Prediction { pub fn from_probabilities(k: usize, probabilities: Vec<f64>, corrected: f64) -> Self; } // argmax lowest index on ties; entropy -sum p ln p / ln k (0 when k = 1)
```

`rng.rs`

```rust
pub fn splitmix(z: u64) -> u64;                  // boitata, bit for bit
pub fn realization_seed(seed: u64, k: u64) -> u64; // splitmix(splitmix(seed) ^ k)
pub mod tag { pub const PATH: u64 = 1; pub const DRAW: u64 = 2; pub const QUENCH: u64 = 3;
              pub const INIT: u64 = 4; pub const FOLD: u64 = 5; pub const SAMPLE: u64 = 6;
              pub const BOOTSTRAP: u64 = 7; }
pub struct Stream { /* xoshiro256++ state */ }
impl Stream {
    pub fn new(seed: u64, keys: &[u64]) -> Self;
    pub fn next_u64(&mut self) -> u64;
    pub fn uniform(&mut self) -> f64;        // [0, 1), 53 bits
    pub fn uniform_open(&mut self) -> f64;   // (0, 1)
    pub fn below(&mut self, n: u64) -> u64;  // Lemire with rejection, n >= 1
    pub fn normal(&mut self) -> f64;         // Box-Muller, cosine branch only
    pub fn exponential(&mut self) -> f64;    // -ln(uniform_open)
    pub fn gamma(&mut self, shape: f64) -> f64; // Marsaglia-Tsang; shape < 1 by boosting
    /// Inverse CDF over unnormalised non-negative weights, linear scan from index 0.
    pub fn categorical(&mut self, weights: &[f64]) -> usize;
    pub fn shuffle<T>(&mut self, items: &mut [T]); // Fisher-Yates from the end
}
```

`numerics.rs`

```rust
pub fn logsumexp(xs: &[f64]) -> f64;
pub fn log_add(a: f64, b: f64) -> f64;
pub fn is_irreducible(a: &Mat) -> bool;               // graph of positive off-diagonals, Tarjan
pub fn gth_stationary(a: &Mat) -> Result<Vec<f64>>;   // a = P - I or Q
pub fn expm_rate(q: &Mat, h: f64) -> Mat;             // uniformization with scaling and squaring
pub fn expm(a: &Mat) -> Mat;                          // nalgebra Pade exp (general matrices)
pub fn logm_stochastic(p: &Mat) -> Result<Mat>;       // inverse scaling and squaring
pub fn quasi_independence(rows: &[f64], cols: &[f64], tol: f64, max_iter: usize) -> Result<Mat>;
pub fn fundamental_matrix(p: &Mat, pi: &[f64]) -> Result<Mat>; // (I - P + 1 pi^T)^-1
pub fn golden_section(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> (f64, f64); // minimiser
pub fn nelder_mead(f: impl Fn(&[f64]) -> f64, x0: &[f64], step: f64, tol: f64, max_eval: usize) -> (Vec<f64>, f64, bool);
```

`special.rs`: `ln_gamma(x)`, `digamma(x)`, `gamma_q(a, x)` (regularized upper incomplete gamma),
`chi2_sf(x, df)`, all `f64 -> f64`, accurate to `1e-13` relative on their tested ranges.

`memory.rs`: `available_bytes() -> Option<u64>`, `check(needed, memory, what) -> Result<()>`.

`serde_mat.rs`: `serialize` / `deserialize` for `Mat` as `Vec<Vec<f64>>` (row-major), for
`#[serde(with = "transiograms_core::serde_mat")]`; submodules `option`, `vec` and `values` (`Vec<f64>`).
NaN is written as JSON `null` and read back as NaN.

### 4.2 transiograms-chain

Modules: `fit.rs` (tallies and `MarkovChain`), `tests.rs` (statistical tests), `entropy.rs`,
`sojourn.rs`, `simulate.rs` (1D simulation and bridges), `lib.rs`.

```rust
pub enum Kind { Embedded, Fixed { step: f64 }, Continuous } // Continuous only for built chains

pub struct FitOptions {
    pub kind: Kind,            // Embedded or Fixed
    pub direction: Direction,  // Up
    pub gap: f64,              // 1e-6
    pub self_transitions: bool, // false; Embedded only
    pub pseudocount: f64,      // 0.0
    pub on_empty_row: EmptyRow, // Raise
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarkovChain {
    pub kind: Kind,
    pub direction: Direction,
    pub counts: Option<Mat>,        // Embedded: bed transitions; Fixed: step transitions
    pub exposure: Option<Vec<f64>>, // total thickness T_j of each category (Embedded)
    pub transition: TransitionMatrix, // Pi (Embedded, Continuous) or P(step) (Fixed)
    pub generator: Option<RateMatrix>,
    pub proportions: Vec<f64>,
    pub mean_lengths: Option<Vec<f64>>,
    pub bed_frequencies: Vec<f64>,
    pub projection: Option<Projection>, // Fixed: how far logm(P)/step moved to be a generator
}

pub enum Method { G, Pearson }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,          // "independence", "order 2 vs 1", "symmetry", "homogeneity", "exponential"
    pub method: String,        // "g", "pearson", "bowker", "ks"
    pub statistic: f64,
    pub df: usize,             // 0 for "ks"
    pub p_value: f64,
    pub expected: Option<Mat>, // null-hypothesis expected counts when tabular
    pub small_expected: f64,   // fraction of allowed cells with expected < 5
}

pub struct Entropy { pub post: Vec<f64>, pub pre: Vec<f64>, pub post_normalized: Vec<f64>,
                     pub pre_normalized: Vec<f64>, pub chain: f64, pub max: f64 } // bits

pub struct Hard { pub from: Vec<f64>, pub to: Vec<f64>, pub codes: Vec<Code> } // positions along direction
pub enum Start { Stationary, Code(Code) }
pub enum Target { Steps(usize), Beds(usize), Length(f64) }
pub enum Simulated { Codes(Vec<Vec<Code>>), Beds(Vec<(Vec<f64>, Vec<f64>, Vec<Code>)>) }

pub struct OrderTable { pub orders: Vec<usize>, pub log_likelihood: Vec<f64>, pub n_params: Vec<usize>,
                        pub aic: Vec<f64>, pub bic: Vec<f64>, pub n_transitions: usize }

pub struct Sojourn { pub n_beds: Vec<usize>, pub n_open: Vec<usize>, pub total: Vec<f64>,
                     pub mean_length: Vec<f64>, pub thickness: Vec<Vec<f64>>, pub open: Vec<Vec<bool>> }

impl MarkovChain {
    pub fn fit(seqs: &SequenceSet, opts: &FitOptions) -> Result<Self>;
    pub fn from_transition(p: TransitionMatrix, kind: Kind, mean_lengths: Option<Vec<f64>>, direction: Direction) -> Result<Self>;
    pub fn from_rates(q: RateMatrix, direction: Direction) -> Result<Self>; // kind Continuous
    /// Embedded given: RateMatrix::from_partial with partial_ij = pi_ij / L_i (i != q);
    /// symmetric: only i < j of `embedded` is read, the rest from p_i r_ij = p_j r_ji.
    /// Embedded None: RateMatrix::max_entropy(p, L).
    pub fn from_geology(p: &[f64], mean_lengths: &[f64], embedded: Option<&Mat>, background: usize, symmetric: bool, direction: Direction) -> Result<Self>;
    pub fn k(&self) -> usize;
    pub fn transition_at(&self, h: f64) -> Result<TransitionMatrix>; // generator, or P^n for Fixed when h = n step
    pub fn independence_test(&self, method: Method) -> Result<TestResult>;
    pub fn symmetry_test(&self) -> Result<TestResult>;    // Bowker on counts
    pub fn entropy(&self) -> Result<Entropy>;
    pub fn difference_matrix(&self) -> Result<Mat>;       // Pi_obs - Pi_expected (quasi-independence)
    pub fn asymmetry(&self) -> Result<Mat>;               // F - F^T, F_jk = p_j q_jk or nu_j pi_jk
    pub fn proportion_std(&self, n_steps: usize) -> Result<Vec<f64>>; // Kemeny-Snell
    pub fn simulate(&self, target: Target, hard: &Hard, start: Start, n: usize, seed: u64) -> Result<Simulated>;
}
pub fn order_test(seqs: &SequenceSet, opts: &FitOptions, order: usize, method: Method) -> Result<TestResult>;
pub fn select_order(seqs: &SequenceSet, opts: &FitOptions, max_order: usize) -> Result<OrderTable>; // Fixed only
pub fn homogeneity_test(groups: &[SequenceSet], opts: &FitOptions, method: Method) -> Result<TestResult>;
pub fn sojourn(seqs: &SequenceSet, direction: Direction, gap: f64) -> Sojourn;
impl Sojourn {
    pub fn survival(&self, code: Code) -> (Vec<f64>, Vec<f64>); // Kaplan-Meier, open beds censored
    pub fn exponential_test(&self, code: Code, n_bootstrap: usize, seed: u64) -> Result<TestResult>;
}
```

Fit semantics by kind (details in chain.md):

- Embedded: counts from `beds(direction, gap, merge = !self_transitions)`; `transition` from
  `from_counts(counts, pseudocount, embedded = !self_transitions, on_empty_row)`; `exposure`,
  `mean_lengths = T_j / exits_j` and `generator = n_jk / T_j` from merged beds (distinct-code
  contacts only), `bed_frequencies` = stationary of `transition`, `proportions` = stationary of
  `generator`. Thickness of regular codes without depths is in steps.
- Fixed: `regular(direction, step, gap)` sequences, `transition = P(step)`, `proportions`
  stationary of P, `mean_lengths = step / (1 - p_kk)`, `bed_frequencies` stationary of
  `p_ij / (1 - p_ii)`, `generator` and `projection` from `RateMatrix::from_transition`, or `None`
  when `logm` fails (no real logarithm).
- Simulation targets: `Steps` needs Fixed (hard positions are step indices, inclusive);
  `Beds` needs a chain with an embedded `transition` (hard positions are bed indices);
  `Length` needs a `generator` (hard positions are lengths along the direction from 0).

### 4.3 transiograms-hmm

Modules: `sequence.rs` (observations and clamps), `emission.rs` (families, densities, M-step
accumulators), `forward_backward.rs` (scaled and log), `viterbi.rs`, `em.rs`, `init.rs` (k-means
and supervised initialisation), `model.rs` (`Hmm`), `lib.rs`.

```rust
/// t x d row-major; categorical codes stored as f64; NaN marks a missing channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Sequence { pub t: usize, pub d: usize, pub x: Vec<f64> }
/// t x K row-major; false forbids that state at that step.
#[derive(Clone, Debug, PartialEq)]
pub struct Clamp { pub k: usize, pub allowed: Vec<bool> }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CovarianceType { Full, Diag, Spherical }

pub struct Categorical { pub probabilities: Vec<f64>, pub pseudocount: f64, pub frozen: bool }
pub struct Gaussian { pub mean: Vec<f64>, pub covariance: Mat, pub covariance_type: CovarianceType, pub frozen: bool } // covariance always d x d
pub struct Poisson { pub rate: Vec<f64>, pub frozen: bool }  // independent channels
pub struct GaussianMixture { pub weights: Vec<f64>, pub components: Vec<Gaussian>, pub frozen: bool }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "lowercase")]
pub enum Emission { Categorical(Categorical), Gaussian(Gaussian), Poisson(Poisson), Mixture(GaussianMixture) }
impl Emission {
    pub fn dim(&self) -> usize;                 // 1 for categorical
    pub fn log_density(&self, x: &[f64]) -> f64; // missing channels marginalised; all missing -> 0
    pub fn n_params(&self) -> usize;            // 0 when frozen
    pub fn sample(&self, rng: &mut Stream) -> Vec<f64>;
    pub fn validate(&self) -> Result<()>;
    pub fn frozen(&self) -> bool;
}

pub enum Family { Categorical { n_symbols: usize }, Gaussian { covariance_type: CovarianceType },
                  Poisson, Mixture { n_mix: usize, covariance_type: CovarianceType } }

/// Letters: s start, t transition, e emissions, d durations (HSMM only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params { pub start: bool, pub transition: bool, pub emission: bool, pub duration: bool }
impl Params { pub fn parse(letters: &str) -> Result<Self>; }

pub enum Implementation { Log, Scaled }
pub enum Ordering { None, Mean, Occupancy }
pub enum Decoder { Viterbi, Map, PosteriorViterbi }

pub struct Priors { pub start: f64, pub transition: f64, pub covariance: f64 } // 1.0, 1.0, 1e-6

pub struct FitOptions {
    pub max_iter: usize,   // 100
    pub tol: f64,          // 1e-6, objective gain per observed step
    pub n_init: usize,     // 1
    pub seed: u64,         // 0
    pub params: Params,    // "ste"
    pub init_params: Params,
    pub priors: Priors,
    pub implementation: Implementation, // Log
    pub ordering: Ordering, // None
}

pub struct FitReport { pub history: Vec<f64>, pub converged: bool, pub n_iter: usize, pub restart: usize }

pub struct Posterior {
    pub log_likelihood: f64,
    pub gamma: Mat,      // t x K
    pub xi_sum: Mat,     // K x K, sum over t of xi_t
    pub entropy: f64,    // H(S | X) of the whole path
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hmm { pub start: Vec<f64>, #[serde(with = "transiograms_core::serde_mat")] pub transition: Mat, pub emissions: Vec<Emission> }
impl Hmm {
    pub fn new(start: Vec<f64>, transition: Mat, emissions: Vec<Emission>) -> Result<Self>;
    pub fn k(&self) -> usize;
    /// k-means (continuous) or Dirichlet / supervised (categorical) start; hmm.md section 6.
    pub fn initialize(k: usize, family: &Family, data: &[Sequence], clamps: Option<&[Clamp]>, priors: &Priors, stream: &mut Stream) -> Result<Self>;
    /// Restarts are fit in parallel; restart r uses Stream::new(seed, &[tag::INIT, r]).
    pub fn fit(&mut self, family: &Family, data: &[Sequence], clamps: Option<&[Clamp]>, opts: &FitOptions) -> Result<FitReport>;
    pub fn log_likelihood(&self, seq: &Sequence, clamp: Option<&Clamp>, imp: Implementation) -> Result<f64>;
    pub fn posterior(&self, seq: &Sequence, clamp: Option<&Clamp>, imp: Implementation) -> Result<Posterior>;
    pub fn viterbi(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<(Vec<u32>, f64)>;
    pub fn decode(&self, seq: &Sequence, clamp: Option<&Clamp>, decoder: Decoder) -> Result<Vec<u32>>;
    pub fn sample(&self, lengths: &[usize], seed: u64) -> Vec<(Vec<u32>, Sequence)>; // stream (seed, [SAMPLE, i])
    pub fn sample_posterior(&self, seq: &Sequence, clamp: Option<&Clamp>, n: usize, seed: u64, index: u64) -> Result<Vec<Vec<u32>>>; // stream (seed, [SAMPLE, index])
    pub fn n_params(&self, params: Params) -> usize;
    pub fn log_prior(&self, priors: &Priors, pooled_variance: &[f64]) -> f64;
}
/// Over all sequences, index-order reduction.
pub fn n_observed(data: &[Sequence]) -> usize; // steps with at least one observed channel
/// Shared with hsmm: per-channel (mean, population variance) of observed values; the weighted
/// emission M-step (`w[n][t]` occupancy of sequence n at step t); one state's emission log prior.
pub fn pooled_moments(data: &[Sequence]) -> (Vec<f64>, Vec<f64>);           // sequence.rs
pub fn update_emission(e: &Emission, state: usize, data: &[Sequence], w: &[Vec<f64>], eps: f64, v: &[f64]) -> Result<Emission>; // em.rs
pub fn emission_log_prior(e: &Emission, eps: f64, pooled_variance: &[f64]) -> f64; // model.rs
pub fn kmeans(points: &Mat, k: usize, stream: &mut Stream) -> Vec<usize>;
```

`fit` errors: `InvalidInput` for impossible observations or clamps (naming sequence index and
step), `Convergence` for a Cholesky failure after one jitter retry (naming the state) and for an
objective decrease larger than `1e-9 * |objective|` (naming the iteration).

### 4.4 transiograms-hsmm

Modules: `duration.rs`, `forward_backward.rs`, `viterbi.rs`, `em.rs`, `model.rs` (`Hsmm`),
`selection.rs`, `lib.rs`.

```rust
/// Shifted laws on u = 1, 2, ...; truncated and renormalised on 1..=max_duration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "law", rename_all = "snake_case")]
pub enum Duration {
    Geometric { p: f64 },                 // p (1 - p)^(u - 1), 0 < p <= 1
    Poisson { rate: f64 },                // e^-rate rate^(u-1) / (u-1)!, rate >= 0
    NegativeBinomial { r: f64, p: f64 },  // C(u + r - 2, u - 1) p^r (1 - p)^(u - 1)
    Empirical { pmf: Vec<f64> },          // pmf[u - 1]; support 1..=pmf.len()
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DurationDist { pub law: Duration, pub frozen: bool, pub pseudocount: f64 }
impl Duration {
    pub fn log_pmf(&self, m: usize) -> Vec<f64>;       // index u - 1, renormalised on 1..=m
    pub fn log_survivor(&self, m: usize) -> Vec<f64>;  // log D(u) = log sum_{v >= u} d(v)
    pub fn mean(&self, m: usize) -> f64;
    pub fn n_params(&self, m: usize) -> usize;         // 1, 1, 2, m - 1
    pub fn fit(&self, counts: &[f64], pseudocount: f64) -> Result<Duration>; // weighted MLE, counts[u - 1]
}
pub enum DurationFamily { Geometric, Poisson, NegativeBinomial, Empirical }

pub struct SegmentPosterior { pub log_likelihood: f64, pub gamma: Mat, pub xi_sum: Mat,
                              pub start: Vec<f64>, pub durations: Vec<Vec<f64>>, pub entropy: f64 }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hsmm {
    pub start: Vec<f64>,
    #[serde(with = "transiograms_core::serde_mat")] pub transition: Mat, // zero diagonal; K = 1 allowed ([[0]])
    pub durations: Vec<DurationDist>,
    pub emissions: Vec<Emission>,
    pub max_duration: Vec<usize>,
}
impl Hsmm {
    pub fn new(start: Vec<f64>, transition: Mat, durations: Vec<DurationDist>, emissions: Vec<Emission>, max_duration: Vec<usize>) -> Result<Self>;
    /// Geometric equivalence: d_j geometric with p = 1 - a_jj, p_ij = a_ij / (1 - a_jj).
    pub fn from_hmm(hmm: &Hmm, max_duration: usize) -> Result<Self>;
    pub fn initialize(k: usize, family: &Family, duration: DurationFamily, data: &[Sequence], clamps: Option<&[Clamp]>, priors: &Priors, max_duration: &[usize], stream: &mut Stream) -> Result<Self>;
    pub fn fit(&mut self, family: &Family, duration: DurationFamily, data: &[Sequence], clamps: Option<&[Clamp]>, opts: &FitOptions) -> Result<FitReport>;
    pub fn log_likelihood(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<f64>;
    pub fn posterior(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<SegmentPosterior>;
    pub fn viterbi(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<(Vec<u32>, f64)>;
    pub fn sample(&self, lengths: &[usize], seed: u64) -> Vec<(Vec<u32>, Sequence)>;
    pub fn sample_posterior(&self, seq: &Sequence, clamp: Option<&Clamp>, n: usize, seed: u64, index: u64) -> Result<Vec<Vec<u32>>>;
    pub fn n_params(&self, params: Params) -> usize;
    pub fn truncated_mass(&self) -> Vec<f64>; // untruncated law's mass above max_duration
}

pub enum Model { Hmm(Hmm), Hsmm(Hsmm) }
pub enum ModelKind { Hmm, Hsmm { duration: DurationFamily, max_duration: Option<usize> } }
pub struct SelectionSpec { pub states: Vec<usize>, pub family: Family, pub kind: ModelKind,
                           pub fit: FitOptions, pub cv_folds: Option<usize> } // None = leave one hole out, Some(0) = no CV
pub struct SelectionTable { pub states: Vec<usize>, pub log_likelihood: Vec<f64>, pub n_params: Vec<usize>,
                            pub n_observations: usize, pub aic: Vec<f64>, pub bic: Vec<f64>, pub icl: Vec<f64>,
                            pub cv_log_likelihood: Vec<f64>, pub cv_per_step: Vec<f64> }
pub fn aic(log_likelihood: f64, n_params: usize) -> f64;
pub fn bic(log_likelihood: f64, n_params: usize, n_observations: usize) -> f64;
pub fn icl(log_likelihood: f64, n_params: usize, n_observations: usize, entropy: f64) -> f64;
pub fn select_states(data: &[Sequence], clamps: Option<&[Clamp]>, spec: &SelectionSpec) -> Result<SelectionTable>;
```

### 4.5 transiograms-transiogram

Modules: `experimental.rs` (points, downhole, grid), `model1d.rs`, `model3d.rs`
(`TransitionModel`, rotation, interpolation, closeness), `table.rs` (offset templates),
`fit.rs`, `lib.rs`.

```rust
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Experimental {
    pub lags: Vec<f64>,          // nominal lag l * Delta (or the given lags)
    pub mean_lags: Vec<f64>,     // mean |h| of the pairs (NaN when none)
    pub counts: Vec<Mat>,        // F_ij per lag; lengths for the downhole estimator
    pub probabilities: Vec<Mat>, // F_ij / sum_k F_ik; NaN row when the tail total is 0
    pub tail_totals: Vec<Vec<f64>>,
    pub directed: bool,
}

pub enum PairFilter { All, SameHole, DifferentHole }
pub struct PairSpec {
    pub direction: Option<[f64; 2]>, // (azimuth, dip) degrees; None = omnidirectional
    pub lag: f64, pub n_lags: usize,
    pub lag_tolerance: f64,          // default lag / 2
    pub angle_tolerance: f64,        // degrees, default 22.5
    pub bandwidth: f64,              // horizontal, default inf
    pub vertical_bandwidth: f64,     // default inf
    pub directed: bool,              // default true; false counts both orientations
    pub pairs: PairFilter,           // default All
}
pub fn experimental(coords: &[[f64; 3]], codes: &[Code], holes: Option<&[u32]>, k: usize, spec: &PairSpec) -> Result<Experimental>;
pub fn downhole(seqs: &SequenceSet, direction: Direction, lags: &[f64], gap: f64) -> Result<Experimental>;
pub fn on_grid(codes: &[Code], grid: &Grid, mask: Option<&[bool]>, offset: [i64; 3], n_lags: usize, k: usize) -> Result<Experimental>;

pub enum Structure { Exponential, Spherical, Gaussian }
pub struct Entry { pub structure: Structure, pub range: f64, pub nugget: f64 }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Model1D {
    Markov { proportions: Vec<f64>, rates: RateMatrix },
    /// entries: k * k, None at the complement entry of each row.
    Entrywise { proportions: Vec<f64>, entries: Vec<Option<Entry>>, complement: Vec<usize> },
}
impl Model1D {
    pub fn markov(rates: RateMatrix) -> Result<Self>;              // proportions = stationary
    pub fn entrywise(proportions: Vec<f64>, entries: Vec<Option<Entry>>, complement: Vec<usize>, max_lag: f64) -> Result<Self>; // validates on a lag grid
    pub fn transition(&self, h: f64) -> Mat;                       // h < 0 by the reversal identity
    pub fn proportions(&self) -> &[f64];
    pub fn fit_markov(exp: &Experimental, proportions: Option<&[f64]>, background: usize, start: Option<&RateMatrix>) -> Result<Self>;
    pub fn fit_entrywise(exp: &Experimental, proportions: &[f64], structures: &[Structure], complement: &[usize]) -> Result<Self>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionModel {
    pub proportions: Vec<f64>,
    pub rates: [RateMatrix; 3],   // principal x', y', z' (z' up), each the +direction generator
    pub rotation: [f64; 3],       // azimuth, dip, rake in degrees
    pub background: usize,
}
impl TransitionModel {
    /// Each rate matrix valid with p^T R = 0 within 1e-10 max|r|.
    pub fn new(proportions: Vec<f64>, rates: [RateMatrix; 3], rotation: [f64; 3], background: usize) -> Result<Self>;
    /// Per axis: embedded given -> background fill; None -> maximum entropy (symmetric).
    pub fn from_geology(proportions: &[f64], mean_lengths: [&[f64]; 3], embedded: [Option<&Mat>; 3], background: usize, rotation: [f64; 3]) -> Result<Self>;
    pub fn k(&self) -> usize;
    pub fn to_principal(&self, h: [f64; 3]) -> [f64; 3];
    pub fn rates_along(&self, h: [f64; 3]) -> Result<RateMatrix>; // h != 0
    pub fn transition(&self, h: [f64; 3]) -> Mat;                 // T(0) = I
    pub fn closeness(&self, h: [f64; 3]) -> f64;                  // det T(h)^(1/(k-1)); 1 when k = 1
    pub fn mean_lengths(&self, h: [f64; 3]) -> Vec<f64>;
    pub fn table(&self, grid: &Grid, limit: f64, memory: Option<u64>) -> Result<Table>;
}
/// Offsets with closeness >= limit, both signs, excluding 0, sorted by closeness descending,
/// ties by (dz, dy, dx) lexicographic ascending.
pub struct Table { pub offsets: Vec<[i64; 3]>, pub closeness: Vec<f64>, pub transition: Vec<Mat> }
impl Table { pub fn get(&self, d: [i64; 3]) -> Option<usize>; }
```

### 4.6 transiograms-tpsim

Modules: `cokriging.rs`, `search.rs`, `sis.rs`, `quench.rs`, `lib.rs`.

```rust
pub struct QuenchOptions { pub max_iter: usize, pub limit: f64, pub tol: f64, pub proportion_weight: f64 } // 3, 0.4, 1e-3, 1.0
pub struct Options {
    pub n_realizations: usize,   // 1
    pub seed: u64,               // 0
    pub max_data: usize,         // 12, even
    pub closeness_limit: f64,    // 0.01
    pub max_per_octant: usize,   // 0 = no cap
    pub svd_cutoff: f64,         // 1e-8 relative to the largest singular value
    pub quench: Option<QuenchOptions>, // Some(default)
    pub memory: Option<u64>,
}
/// Probabilities before order-relation correction, summing to 1.
pub fn cokrige(model: &TransitionModel, data: &[([f64; 3], Code)], target: [f64; 3], svd_cutoff: f64) -> Result<Vec<f64>>;
pub fn correct(probabilities: &mut [f64]) -> bool;  // clip to [0, 1], renormalise; true if changed
pub fn predict(model: &TransitionModel, grid: &Grid, mask: Option<&[bool]>, hard: &Snapped, opts: &Options) -> Result<Prediction>;
pub fn simulate(model: &TransitionModel, grid: &Grid, mask: Option<&[bool]>, hard: &Snapped, opts: &Options) -> Result<Realizations>;
/// Sequential, zero temperature; returns the objective after each sweep (index 0 = initial).
pub fn quench(codes: &mut [Code], frozen: &[bool], grid: &Grid, mask: Option<&[bool]>, model: &TransitionModel, q: &QuenchOptions, seed: u64, realization: u64) -> Result<Vec<f64>>;
pub fn objective(codes: &[Code], grid: &Grid, mask: Option<&[bool]>, model: &TransitionModel, lags: &[[i64; 3]], proportion_weight: f64) -> f64;
pub fn quench_lags(model: &TransitionModel, grid: &Grid, limit: f64) -> Vec<[i64; 3]>;
```

### 4.7 transiograms-mcrf

Modules: `conditional.rs`, `search.rs`, `predict.rs`, `simulate.rs`, `lib.rs`.

```rust
pub enum Neighbourhood { Sectors, Nearest(usize) }
pub struct Options {
    pub neighbourhood: Neighbourhood, // Sectors: quadrants in 2D, octants in 3D
    pub closeness_limit: f64,         // 0.01
    pub n_realizations: usize,        // 1
    pub seed: u64,                    // 0
    pub memory: Option<u64>,
}
/// Eq. 14: p_k prod_g t_{k, c_g}(h_g) normalised, h_g = x_g - x_0, in log space.
/// None when every candidate has zero likelihood.
pub fn conditional(model: &TransitionModel, neighbours: &[([f64; 3], Code)]) -> Option<Vec<f64>>;
/// Li and Zhang Eq. 13 with `first` the index of the neighbour the chain comes from; test oracle.
pub fn conditional_eq13(model: &TransitionModel, neighbours: &[([f64; 3], Code)], first: usize) -> Option<Vec<f64>>;
pub fn predict(model: &TransitionModel, grid: &Grid, mask: Option<&[bool]>, hard: &Snapped, opts: &Options) -> Result<Prediction>;
pub fn simulate(model: &TransitionModel, grid: &Grid, mask: Option<&[bool]>, hard: &Snapped, opts: &Options) -> Result<Realizations>;
```

### 4.8 transiograms-py

Modules under `crates/py/src/`: `lib.rs` (module init, `error`, `invalid`, `convergence`),
`args.rs` (coercion helpers below), `core.rs` (`Categories`, `Grid`), `chain.rs`, `hmm.rs`,
`hsmm.rs`, `transiogram.rs`, `tpsim.rs`, `mcrf.rs`. Each area module has
`pub fn register(m: &Bound<PyModule>) -> PyResult<()>`; `lib.rs` calls all of them.

`args.rs` helpers (signatures binding for all area modules):

```rust
pub fn column<'py>(data: Option<&Bound<'py, PyAny>>, arg: &Bound<'py, PyAny>, what: &str) -> PyResult<Bound<'py, PyAny>>; // str -> data[str]
pub fn codes(obj: &Bound<PyAny>, what: &str) -> PyResult<Vec<Code>>;          // int -1 or float NaN missing
pub fn categories_k(obj: Option<&Bound<PyAny>>, codes: &[Code]) -> PyResult<usize>;
pub fn hole_ids(obj: Option<&Bound<PyAny>>, n: usize) -> PyResult<Vec<String>>; // None -> one hole "0"
pub fn direction(s: &str) -> PyResult<Direction>;
pub fn sequence_set(py_args: SequenceArgs) -> PyResult<SequenceSet>;         // section 5.2
pub fn sequences(obj, holes, data, direction, d_expected: Option<usize>) -> PyResult<(Vec<Sequence>, RowMap)>; // HMM input
pub fn coords(obj: &Bound<PyAny>, what: &str) -> PyResult<Vec<[f64; 3]>>;    // (n,3), (n,2) gets z = 0
pub fn to_rust(e: transiograms_core::Error) -> PyErr;
```

`RowMap` remembers the hole split, row order and direction so outputs go back in input row order.

## 5. Python API

### 5.1 Package layout

```
python/transiograms/__init__.py     from transiograms._transiograms import *; errors; copyreg; __all__ from dir()
python/transiograms/_transiograms.pyi    one stub file with one section per area (section 8)
python/transiograms/py.typed
python/transiograms/errors.py       TransiogramsError, InvalidInput, ConvergenceError
python/transiograms/plot.py         tg.plot (not star-exported), lazy matplotlib import
python/transiograms/datasets.py     tg.datasets (not star-exported), seeded synthetic data
python/transiograms/checks.py       tg.checks (not star-exported), NumPy brute-force oracles
```

Users write `import transiograms as tg`. All options are keyword-only after `*`. Docstrings are NumPy
style written as Rust `///` comments on pymethods. Outputs are float64 arrays (NaN missing),
int64 codes and states (-1 missing or inactive), uint64 cell indices. `seed: int = 0`.

### 5.2 Sequence data arguments (shared block, "SEQUENCE ARGS")

Every function that reads categorical drillhole sequences takes `codes` positionally and these
keywords:

```python
holes: ArrayLike | str | None = None  # per-row hole id; None = one hole
from_: ArrayLike | str | None = None  # interval tops (depth)
to: ArrayLike | str | None = None  # interval bottoms (depth)
depth: ArrayLike | str | None = None  # regular sample depths
step: float | None = None  # sample spacing; inferred from depth when None
gap: float = 1e-6  # larger gaps break runs
categories: Categories | int | None = None
data: Mapping[str, ArrayLike] | None = None  # str arguments name its columns
```

Rules: `from_` and `to` together give intervals; `depth` gives regular samples (step = median
positive within-hole spacing when `step` is None; consecutive samples whose spacing differs from
`step` by more than `1e-6 * step` break the run); neither gives unit-step regular codes in row
order per hole. Giving both `depth` and `from_` raises `InvalidInput`.

### 5.3 Core

```python
class Categories:
    def __init__(self, names: Sequence[str], *, colors: Sequence[str] | None = None): ...
    @staticmethod
    def from_labels(labels: ArrayLike) -> Categories: ...
    names: list[str]; colors: list[str] | None
    def encode(self, labels: ArrayLike) -> NDArray[np.int64]: ...   # None / NaN -> -1
    def decode(self, codes: ArrayLike) -> NDArray[np.object_]: ...  # -1 -> None
    def __len__(self) -> int: ...
    def to_json(self) -> str: ...; @staticmethod def from_json(text: str) -> Categories: ...

class Grid:
    def __init__(self, origin: Sequence[float], size: Sequence[float], count: Sequence[int]): ...
    origin: tuple[float, float, float]; size: tuple[float, float, float]; count: tuple[int, int, int]
    n_cells: int; shape: tuple[int, int, int]     # (nz, ny, nx), so values.reshape(grid.shape)
    coords: NDArray[np.float64]                   # (n_cells, 3) centroids
    def snap(self, coords: ArrayLike, codes: ArrayLike, *, mask: ArrayLike | None = None) -> tuple[NDArray[np.uint64], NDArray[np.int64]]: ...
    def to_json(self) -> str: ...; @staticmethod def from_json(text: str) -> Grid: ...
```

### 5.4 Chain

```python
class TestResult:
    name: str; method: str; statistic: float; df: int; p_value: float
    expected: NDArray[np.float64] | None; small_expected: float

class Entropy:
    post: NDArray; pre: NDArray; post_normalized: NDArray; pre_normalized: NDArray; chain: float; max: float

class MarkovChain:
    def __init__(self, kind: Literal["embedded", "fixed"] = "embedded", *, direction: Literal["up", "down"] = "up",
                 step: float | None = None, self_transitions: bool = False, pseudocount: float = 0.0,
                 on_empty_row: Literal["raise", "uniform"] = "raise"): ...
    def fit(self, codes, *, <SEQUENCE ARGS>) -> MarkovChain: ...
    @staticmethod
    def from_matrix(matrix: ArrayLike, *, kind: Literal["embedded", "fixed"] = "fixed", step: float = 1.0,
                    mean_lengths: ArrayLike | None = None, direction="up") -> MarkovChain: ...
    @staticmethod
    def from_rates(rates: ArrayLike, *, direction="up") -> MarkovChain: ...
    @staticmethod
    def from_geology(proportions: ArrayLike, mean_lengths: ArrayLike, embedded: ArrayLike | None = None, *,
                     background: int = -1, symmetric: bool = False, direction="up") -> MarkovChain: ...
    kind: str; direction: str; step: float | None; n_categories: int
    counts: NDArray | None; transition_matrix: NDArray; generator: NDArray | None
    proportions: NDArray; mean_lengths: NDArray | None; bed_frequencies: NDArray
    projection_distance: float | None
    def transition_at(self, h: float | ArrayLike) -> NDArray: ...          # (k,k) or (n,k,k)
    def independence_test(self, *, method: Literal["g", "pearson"] = "g") -> TestResult: ...
    def symmetry_test(self) -> TestResult: ...
    def entropy(self) -> Entropy: ...
    def difference_matrix(self) -> NDArray: ...
    def asymmetry(self) -> NDArray: ...
    def proportion_std(self, n_steps: int) -> NDArray: ...
    def simulate(self, length: float, *, n: int = 1, seed: int = 0, hard_from: ArrayLike | None = None,
                 hard_to: ArrayLike | None = None, hard_codes: ArrayLike | None = None,
                 start: int | Literal["stationary"] = "stationary") -> NDArray[np.int64] | dict[str, NDArray]: ...
    def simulate_beds(self, n_beds: int, *, n: int = 1, seed: int = 0, hard_from=None, hard_to=None,
                      hard_codes=None, start="stationary") -> NDArray[np.int64]: ...
    def to_json(self) -> str: ...; @staticmethod def from_json(text: str) -> MarkovChain: ...

def order_test(codes, *, order: int = 2, kind: Literal["embedded", "fixed"] = "fixed", direction="up",
               method="g", <SEQUENCE ARGS>) -> TestResult: ...
def select_order(codes, *, max_order: int = 3, direction="up", <SEQUENCE ARGS>) -> dict[str, NDArray]: ...
    # keys: order, log_likelihood, n_params, aic, bic; plus "n_transitions" (int array of length 1)
def homogeneity_test(codes, groups: ArrayLike | str, *, kind="embedded", direction="up", method="g",
                     <SEQUENCE ARGS>) -> TestResult: ...
def sojourn(codes, *, direction="up", <SEQUENCE ARGS>) -> Sojourn: ...

class Sojourn:
    n_beds: NDArray[np.int64]; n_open: NDArray[np.int64]; total: NDArray; mean_length: NDArray
    def thickness(self, code: int) -> tuple[NDArray, NDArray[np.bool_]]: ...   # thickness, open
    def survival(self, code: int) -> tuple[NDArray, NDArray]: ...
    def exponential_test(self, code: int, *, n_bootstrap: int = 999, seed: int = 0) -> TestResult: ...
```

`MarkovChain.simulate` returns int64 `(n, steps)` for `kind="fixed"` (`steps = length / step`,
which must be an integer within `1e-9`) and, for chains with a generator, a dict of arrays
`{"realization", "from", "to", "code"}` with positions along the direction from 0 to `length`.
`fixed` hard positions are lengths converted to the cells whose centres they cover.
`background=-1` means the last category. `homogeneity_test` groups are per row; transitions are
counted only between rows of the same group.

### 5.5 HMM

```python
class Categorical:
    def __init__(self, probabilities: ArrayLike, *, pseudocount: float = 0.0, frozen: bool = False): ...
class Gaussian:
    def __init__(self, mean: ArrayLike, covariance: ArrayLike | float, *,
                 covariance_type: Literal["full", "diag", "spherical"] = "full", frozen: bool = False): ...
    # covariance: (d,d) full, (d,) diag, scalar spherical
class Poisson:
    def __init__(self, rate: ArrayLike, *, frozen: bool = False): ...
class GaussianMixture:
    def __init__(self, weights: ArrayLike, means: ArrayLike, covariances: ArrayLike, *,
                 covariance_type="full", frozen: bool = False): ...
# each: .log_density(x) -> (n,) ; .sample(n, *, seed=0) -> (n, d) or (n,) ; attributes as constructed; to_json/from_json

class HiddenMarkovModel:
    def __init__(self, distributions: int | Sequence[Distribution], *, start: ArrayLike | None = None,
                 transition: ArrayLike | None = None,
                 emission: Literal["gaussian", "categorical", "poisson", "mixture"] = "gaussian",
                 covariance_type: Literal["full", "diag", "spherical"] = "diag", n_symbols: int | None = None,
                 n_mix: int = 1, max_iter: int = 100, tol: float = 1e-6, n_init: int = 1, seed: int = 0,
                 params: str = "ste", init_params: str | None = None, start_prior: float = 1.0,
                 transition_prior: float = 1.0, covariance_prior: float = 1e-6, pseudocount: float = 0.0,
                 direction: Literal["up", "down"] = "up", implementation: Literal["log", "scaled"] = "log",
                 ordering: Literal["none", "mean", "occupancy"] = "none"): ...
    def fit(self, sequences, *, holes=None, labels=None, allowed=None, data=None) -> HiddenMarkovModel: ...
    def score(self, sequences, *, holes=None, data=None) -> float: ...
    def score_samples(self, sequences, *, holes=None, data=None) -> NDArray: ...   # per hole
    def predict(self, sequences, *, holes=None, labels=None, allowed=None, data=None,
                algorithm: Literal["viterbi", "map", "posterior_viterbi"] = "viterbi") -> Rows[np.int64]: ...
    def predict_proba(self, sequences, *, holes=None, labels=None, allowed=None, data=None) -> Rows[np.float64]: ...
    def viterbi(self, sequences, *, holes=None, labels=None, allowed=None, data=None) -> tuple[Rows[np.int64], NDArray]: ...
    def sample(self, lengths: int | Sequence[int], *, seed: int = 0) -> tuple[list[NDArray], list[NDArray]]: ...
    def sample_posterior(self, sequences, *, n: int = 100, seed: int = 0, holes=None, labels=None,
                         allowed=None, data=None) -> NDArray[np.int64]: ...   # (n, rows)
    def aic(self, sequences, *, holes=None, data=None) -> float: ...
    def bic(self, sequences, *, holes=None, data=None) -> float: ...
    def icl(self, sequences, *, holes=None, data=None) -> float: ...
    n_states: int; start: NDArray; transition: NDArray; distributions: list[Distribution]
    history: NDArray; converged: bool; n_iter: int
    def to_json(self) -> str: ...; @staticmethod def from_json(text: str) -> HiddenMarkovModel: ...
```

Sequence input: `sequences` is a list of arrays (one per hole, rows top-down, shape `(t,)` or
`(t, d)`) or one array plus `holes=` (per-row ids, grouped by first appearance, row order kept).
`Rows[...]` means: a list of arrays when the input was a list, else one array in input row order.
`labels` are per-row state indices with -1 for unlabelled; `allowed` is a per-row boolean
`(rows, K)` mask; both may be given and are combined with AND. Categorical observations are
codes (-1 or NaN missing); continuous observations use NaN for missing channels.
`init_params=None` means `"ste"` minus the letters of the parts given explicitly (`start`,
`transition`, distribution objects). `score`, `aic`, `bic` and `icl` are unclamped.

### 5.6 HSMM and selection

```python
class Geometric:        def __init__(self, p: float, *, frozen: bool = False): ...
class ShiftedPoisson:   def __init__(self, rate: float, *, frozen: bool = False): ...
class NegativeBinomial: def __init__(self, r: float, p: float, *, frozen: bool = False): ...
class EmpiricalDuration:def __init__(self, pmf: ArrayLike, *, pseudocount: float = 0.0, frozen: bool = False): ...
# each: .pmf(max_duration) -> (max_duration,) ; .mean(max_duration) -> float ; to_json/from_json

class HiddenSemiMarkovModel:
    def __init__(self, distributions: int | Sequence[Distribution], durations: Sequence[DurationLaw] | None = None, *,
                 start=None, transition=None, emission="gaussian",
                 duration: Literal["geometric", "poisson", "negative_binomial", "empirical"] = "geometric",
                 covariance_type="diag", n_symbols=None, n_mix=1, max_duration: int | Sequence[int] | None = None,
                 max_iter=100, tol=1e-6, n_init=1, seed=0, params: str = "sted", init_params: str | None = None,
                 start_prior=1.0, transition_prior=1.0, covariance_prior=1e-6, pseudocount=0.0,
                 direction="up", ordering="none"): ...
    @staticmethod
    def from_hmm(hmm: HiddenMarkovModel, *, max_duration: int) -> HiddenSemiMarkovModel: ...
    # fit, score, score_samples, predict (algorithm "viterbi" | "map"), predict_proba, viterbi, sample,
    # sample_posterior, aic, bic, icl: same signatures as HiddenMarkovModel
    durations: list[DurationLaw]; max_duration: NDArray[np.int64]; truncated_mass: NDArray

def select_states(sequences, *, holes=None, data=None, labels=None, allowed=None,
                  states: Sequence[int] = range(1, 7), model: Literal["hmm", "hsmm"] = "hmm",
                  emission="gaussian", covariance_type="diag", n_symbols=None, n_mix=1, duration="geometric",
                  max_duration=None, cv_folds: int | None = None, n_init: int = 5, seed: int = 0,
                  max_iter: int = 100, tol: float = 1e-6, direction="up") -> dict[str, NDArray]: ...
    # keys: states, log_likelihood, n_params, aic, bic, icl, cv_log_likelihood, cv_per_step
```

### 5.7 Transiograms

```python
class ExperimentalTransiogram:
    lags: NDArray; mean_lags: NDArray; counts: NDArray            # (n_lags, k, k)
    probabilities: NDArray; tail_totals: NDArray; directed: bool  # (n_lags, k, k), (n_lags, k)

def experimental_transiogram(coords, codes, *, holes=None, direction: tuple[float, float] | None = None,
                             lag: float, n_lags: int, lag_tolerance: float | None = None,
                             angle_tolerance: float = 22.5, bandwidth: float = inf, vertical_bandwidth: float = inf,
                             directed: bool = True, pairs: Literal["all", "same_hole", "different_hole"] = "all",
                             categories=None, data=None) -> ExperimentalTransiogram: ...
def downhole_transiogram(codes, *, lags: ArrayLike, direction="up", <SEQUENCE ARGS>) -> ExperimentalTransiogram: ...
def grid_transiogram(codes: ArrayLike, grid: Grid, *, offset: tuple[int, int, int], n_lags: int,
                     mask=None, categories=None) -> ExperimentalTransiogram: ...

class TransiogramModel:                 # one direction
    @staticmethod
    def markov(rates: ArrayLike | MarkovChain) -> TransiogramModel: ...
    @staticmethod
    def entrywise(proportions, structures: ArrayLike, ranges: ArrayLike, *, nuggets=None,
                  complement: ArrayLike | None = None, max_lag: float) -> TransiogramModel: ...
    @staticmethod
    def fit_markov(experimental: ExperimentalTransiogram, *, proportions=None, background: int = -1,
                   start: ArrayLike | None = None) -> TransiogramModel: ...
    @staticmethod
    def fit_entrywise(experimental, proportions, *, structures: str | Sequence[str] = "exponential",
                      complement=None) -> TransiogramModel: ...
    kind: str; proportions: NDArray; rates: NDArray | None
    def transition(self, h: float | ArrayLike) -> NDArray: ...   # (k,k) or (n,k,k)

class TransitionModel:                  # 3D
    def __init__(self, proportions, rates_x, rates_y, rates_z, *, rotation=(0.0, 0.0, 0.0), background: int = -1): ...
    @staticmethod
    def from_geology(proportions, mean_lengths: ArrayLike, *, embedded_x=None, embedded_y=None, embedded_z=None,
                     rotation=(0.0, 0.0, 0.0), background: int = -1) -> TransitionModel: ...
                     # mean_lengths shape (3, k): rows x', y', z'
    proportions: NDArray; rates_x: NDArray; rates_y: NDArray; rates_z: NDArray
    rotation: tuple[float, float, float]; background: int
    def transition(self, lags: ArrayLike) -> NDArray: ...     # (3,) -> (k,k); (n,3) -> (n,k,k)
    def rates(self, lag: ArrayLike) -> NDArray: ...           # interpolated generator along lag
    def closeness(self, lags: ArrayLike) -> NDArray: ...
    def mean_lengths(self, lag: ArrayLike) -> NDArray: ...
    def to_json(self) -> str: ...; @staticmethod def from_json(text: str) -> TransitionModel: ...
```

### 5.8 Grid simulation and prediction

```python
class Prediction:
    probabilities: NDArray  # (n_cells, k); NaN outside the mask
    most_likely: NDArray[np.int64]
    entropy: NDArray  # entropy / ln k, in [0, 1]
    corrected: float  # fraction of nodes whose probabilities were clipped


class Realizations:
    realizations: NDArray[np.int64]  # (n, n_cells), -1 outside the mask
    proportions: NDArray  # (n, k)
    probabilities: NDArray  # (n_cells, k) frequency over realizations
    most_likely: NDArray[np.int64]
    entropy: NDArray
    corrected: NDArray  # (n,) fraction clipped during SIS (zeros for MCRF)
    objective: list[NDArray]  # quench objective per sweep per realization (empty for MCRF)


def tp_cokriging(
    model: TransitionModel,
    grid: Grid,
    coords,
    codes,
    *,
    mask=None,
    max_data: int = 12,
    closeness_limit: float = 0.01,
    max_per_octant: int = 0,
    svd_cutoff: float = 1e-8,
    memory: int | None = None,
    data=None,
) -> Prediction: ...
def tprogs_simulate(
    model,
    grid,
    coords,
    codes,
    *,
    n: int = 1,
    seed: int = 0,
    mask=None,
    max_data: int = 12,
    closeness_limit: float = 0.01,
    max_per_octant: int = 0,
    svd_cutoff: float = 1e-8,
    quench: bool = True,
    quench_iterations: int = 3,
    quench_limit: float = 0.4,
    quench_tol: float = 1e-3,
    proportion_weight: float = 1.0,
    memory: int | None = None,
    data=None,
) -> Realizations: ...
def mcp_predict(
    model,
    grid,
    coords,
    codes,
    *,
    mask=None,
    neighbours: Literal["sectors"] | int = "sectors",
    closeness_limit: float = 0.01,
    memory: int | None = None,
    data=None,
) -> Prediction: ...
def mcrf_simulate(
    model,
    grid,
    coords,
    codes,
    *,
    n: int = 1,
    seed: int = 0,
    mask=None,
    neighbours="sectors",
    closeness_limit: float = 0.01,
    memory: int | None = None,
    data=None,
) -> Realizations: ...
```

`coords` and `codes` are the hard data, snapped with `Grid.snap`. `max_data` must be even.

### 5.9 Plot, datasets, checks

`tg.plot` (every function takes `ax=None`, returns `(fig, ax)` or `(fig, axes)`, forwards
`**kwargs` to the main matplotlib call, imports matplotlib lazily with the hint
`pip install 'transiograms[plot]' or conda install -c conda-forge matplotlib`):

| Function | Owner |
| --- | --- |
| `transition_matrix(matrix, *, categories=None, annotate=True, ax=None, **kwargs)` | chain |
| `sojourn(sojourn, code, *, exponential=True, ax=None, **kwargs)` | chain |
| `strip_log(codes, *, depth=None, from_=None, to=None, posterior=None, categories=None, ax=None, **kwargs)` | hmm |
| `selection(table, *, criteria=("aic", "bic", "icl"), ax=None, **kwargs)` | hsmm |
| `transiogram(experimental=None, model=None, *, h=None, categories=None, axes=None, **kwargs)` | transiogram |
| `grid_slice(values, grid, *, axis="z", index=0, categories=None, ax=None, **kwargs)` | tpsim |

`tg.datasets` (each `name(seed=0) -> dict` of NumPy arrays and model JSON strings, documented
keys):

| Function | Content | Owner |
| --- | --- | --- |
| `varved_sequence(seed=0)` | intervals HOLE_ID, FROM, TO, CODE from a fining-upward generator chain; key `generator` | chain |
| `electrofacies_logs(seed=0)` | 20 holes x 300 steps, 3-state Gaussian HMM with 2 channels; keys `HOLE_ID`, `DEPTH`, `GR`, `RHOB`, `STATE` | hmm |
| `layered_drillholes(seed=0)` | vertical holes from a known `TransitionModel` on a grid; keys `X`, `Y`, `Z`, `HOLE_ID`, `CODE`, `model`, `grid` | transiogram |

`tg.checks` (pure NumPy, used by tests and examples): `quasi_independence(counts)` and
`kemeny_snell_std(P, n)` (chain); `brute_force_hmm(start, transition, log_b)` returning
`(log_likelihood, best_path, best_log_probability, gamma, path_entropy)` (hmm);
`brute_force_hsmm(start, transition, log_d, log_survivor, log_b)` with the same returns (hsmm);
`markov_bridge_marginal(P, a, b, n, t)` (chain); `mcp_brute_force(p, T_list, codes)` (mcrf).

## 6. Exchange with boitata

- Codes: int64 `0..k-1`, -1 missing; float with NaN accepted. Category tables: `Categories`
  JSON `{names, colors}`.
- Intervals: `HOLE_ID`, `FROM`, `TO` and a code column map to `holes`, `from_`, `to`, `codes`
  with `data=` (boitata tables convert with `{c: t[c] for c in t.column_names}` or any mapping).
- Points: `(n, 3)` coordinates plus codes plus optional holes (composite midpoints).
- Grids: `tg.Grid(origin, size, count)` equals boitata `Geometry` with zero rotation; cell order
  x fastest; `Realizations.realizations` `(n, n_cells)` int64 and `Prediction.probabilities`
  `(n_cells, k)` rows summing to 1 are what boitata's `SIS(proportions=)` and block-model
  `with_column` accept.
- No boitata import anywhere in transiograms, tests included.

## 7. Invariants and theory checks

Every item is a standing test. Rust tests live in `crates/<c>/src` unit tests and
`crates/<c>/tests/theory.rs`; Python tests in `python/tests/test_<area>.py`. Tolerances are
binding. "Threads" means the 1-versus-4-thread bit-identity test of section 3.3.

### core
- `splitmix(0) == 0xE220_A839_7B1D_CDAF`; `realization_seed` equals boitata's on 1000 pairs
  (hard-coded values from boitata's formula); 100 seeds x 1000 keys give 100000 distinct keys.
- `Stream`: uniform mean and variance within 5 sigma over 10^6 draws; `below(n)` chi-square
  uniform; `gamma(a)` mean and variance for a in {0.3, 1, 5}; `normal` moments.
- `logsumexp([-inf, -inf]) == -inf`; no NaN on mixed infinities.
- GTH: golden Q (chain.md section 3) gives p = (0.5, 0.3, 0.2) within 1e-14; `pi P = pi`
  within 1e-12; reducible input raises; 2-state periodic embedded chain gives (0.5, 0.5).
- `expm_rate`: T(0) = I; rows sum to 1 within 1e-12; T >= 0; T(a + b) = T(a) T(b) within 1e-12;
  2-state closed form within 1e-14; single-rate model `R = lambda (1 p^T - I)` closed form within
  1e-13 (repeated eigenvalue); agrees with `expm` (Pade) within 1e-12; golden T(0.5), T(1), T(5)
  within 1e-9; mu h = 10^4 does not underflow.
- `logm_stochastic(expm_rate(Q, h)) / h == Q` within 1e-10 on the golden Q.
- `quasi_independence`: golden 4x4 and 3x3 E within 1e-8; margins within 1e-12; cross-product
  ratios 1 within 1e-10.
- `RateMatrix::from_partial`: golden Carle example reproduces Q within 1e-12; negative fill
  raises naming the entry. `max_entropy` satisfies `p^T R = 0` within 1e-12 and raises when
  `p_i / L_i > sum_{j != i} p_j / L_j`.
- Special functions against tabulated values (`ln_gamma`, `digamma`, `chi2_sf`) within 1e-12
  relative; Python cross-check against scipy in the test suite.
- `SequenceSet`: gaps and MISSING split runs; Up reverses; overlapping intervals raise naming
  the hole; rows map back to input order.
- `Grid`: `index(ijk(i)) == i`; `locate(centroid(i)) == i`; snap ties and conflicts.
- Serde round trip for every public type (proptest).

### chain
- Every fitted or built chain: rows sum to 1 within 1e-12, embedded diagonal exactly 0,
  generator rows sum to 0 within 1e-12 max|q| (proptest over random sequences).
- Independence test on golden 4x4 counts: X2 = 29.298020503608846, G = 31.237967935478377,
  df = 5, p values 2.0266e-05 and 8.4064e-06 (1e-4 relative); differs from the Gingerich row.
  m = 2 embedded raises.
- Never tallies across holes, gaps or missing codes (hand-built case).
- Fixed P recovers from a 10^5-step seeded simulation: every `|P_hat - P| <= 3 sqrt(P(1-P)/c_i.)`;
  proportions within 3 Kemeny-Snell sigma (M1 acceptance). Start from the stationary law.
- Embedded MLE: `T_j / exits_j` recovers L on censored simulated holes within 3 sigma.
- Proportion identity `p_k = nu_k L_k / sum nu_j L_j` within 1e-12.
- Order test df `m^(k-1) (m-1)^2`; order selection uses the same transitions for every order;
  a simulated second-order chain is detected and a first-order one gives uniform p values
  (Kolmogorov-Smirnov on 200 seeded replicates, p > 0.001).
- Homogeneity: df `(H-1) m(m-1)` full case; two groups from one chain do not reject at 0.001.
- Bowker on a symmetric count table gives statistic 0.
- Entropy: uniform embedded rows give `post = log2(m-1)`.
- Discrete bridge: exhaustive m = 3, n = 6 paths: sampler path probabilities equal
  `P(path) / P^n_ab` within 1e-12; Monte Carlo frequencies pass chi-square at 0.001; marginal
  at t matches `markov_bridge_marginal`. Impossible endpoints raise.
- Continuous bridge (Hobolth-Stone): marginal at t equals `T(t)_aj T(T-t)_jb / T(T)_ab` within
  3 sigma; hard intervals reproduced exactly.
- Re-fit of simulated beds recovers Q within 3 sigma. Threads.

### hmm
- M2: K = 4, T = 8 Gaussian and categorical cases: Viterbi path and log-probability equal the
  exhaustive search over 4^8 paths (atol 1e-12, lowest-index ties); forward log L equals the
  logsumexp over paths (1e-12); gamma and xi equal path marginals (1e-12); path entropy equals
  the identity (1e-10).
- Every step: `sum_j gamma = 1` within 1e-12; `sum_j xi(i, j) = gamma_t(i)`.
- Scaled and log implementations: gamma within 1e-12, log L within 1e-10 relative.
- EM: objective (log L + log prior when priors are active) non-decreasing every iteration
  within -1e-9 |objective|; transition rows sum to 1 within 1e-12.
- Fully clamped data: one EM iteration equals the supervised counting MLE; a second changes
  nothing within 1e-12.
- Missing rows: log L with a NaN row equals log L with the row removed and A replaced by A^2
  across it (1e-12). Full-covariance partial missing equals quadrature marginalisation (1e-8).
- Separate holes score as the sum of single-hole scores; concatenation differs.
- State permutation leaves log L unchanged. Emission densities match scipy (Python).
- Recovery: 10^5 seeded steps from known parameters, fit from the truth: transition entries
  within 3 binomial sigma. Threads.

### hsmm
- M3: geometric HSMM from `from_hmm` with `max_duration` such that `a_jj^M < 1e-14` gives the
  HMM log L within 1e-10 and the same Viterbi path.
- Brute force K = 3, T = 8 (6561 paths, run-length encoded): log L, Viterbi, gamma, xi and
  segment-start posteriors within 1e-12; runs longer than M_j excluded.
- Forward and backward agree: `LSE_j F_T(j) == LSE_j [log pi_j + H_0(j)]` within 1e-12.
- gamma from start/end differences equals gamma from segment sums within 1e-10.
- Duration M-step: geometric and Poisson equal closed-form means on observed segments; NB
  returned r is a profile maximum (r (1 +/- 1e-3) score no better).
- EM monotone as in hmm. ICL entropy equals brute force within 1e-10.
- M3 selection: `select_states` on `datasets.electrofacies_logs(seed=0)` with `n_init = 5`
  returns the BIC argmin K = 3 over K = 1..6. Threads (including CV folds).

### transiogram
- M4: every model at lags `h` in a grid of 200 values from 0 to 20 max L plus 10^4 max L:
  entries >= 0, rows sum to 1 within 1e-12, `T(0) = I` exactly, `T(h) -> 1 p^T` within 1e-10 at
  the largest lag.
- Reversal: `p_i t_ij(h) = p_j t_ji(-h)` within 1e-12; `det T(h) = exp(|h| tr R)` within 1e-12.
- Interpolation along a principal axis returns that axis's R exactly; mean lengths lie on the
  ellipsoid `1/L(phi)^2 = sum_v (c_v / L_v)^2` within 1e-12 (non-background categories).
- Rotation: rotating the model and the lag together leaves T unchanged.
- Downhole exact estimator on a two-category alternating log with known lengths matches the
  analytic interval-overlap fraction within 1e-12; counts sum consistently with tail totals.
- Point estimator on vertical holes with `pairs="same_hole"` equals the grid estimator on the
  same data placed on a grid.
- `fit_markov` on 10^5-step simulated logs recovers R within 5 percent per entry.
- Entrywise models that go negative raise naming `(i, j, h)`. Threads.

### tpsim
- Single datum: probabilities equal row `c` of `T(x0 - x1)` within 1e-12.
- 1D screening (x1 < x2 < x0 on a line): weights `W1 = 0`, `W2 = T(x0 - x2)` within 1e-10.
- Probabilities sum to 1 within 1e-12 before correction.
- Hard nodes unchanged by SIS and by quenching in every realization.
- Quench objective never increases between sweeps; incremental Delta O equals full recompute
  within 1e-12 (proptest); counts equal a recount after every sweep.
- M5: 20 realizations on the acceptance grid: 100 percent hard-data agreement; ensemble
  proportions within 0.02 (after asserting 3 sigma < 0.02); ensemble auto-transiograms along
  each axis within 0.05 RMSE up to two correlation lengths. Threads.

### mcrf
- Eq. 14 equals Eq. 13 within 1e-12 for an asymmetric, reversal-consistent model.
- 1D two-neighbour case equals the Markov bridge `t_{a k}(h1) t_{k b}(h2) / t_{a b}(h1 + h2)`
  within 1e-12.
- MCP brute force: K = 3, four neighbours, IPF maximum entropy on the 243-cell table equals the
  MCP conditional within 1e-10.
- Layer-cake 2D field with x-neighbours reproduces the 1D chain conditional exactly.
- M6: every node's probabilities >= 0 and sum to 1 within 1e-12; hard nodes give probability
  exactly 1 on their code; MCRF ensemble proportions within 0.02 (after asserting 3 sigma <
  0.02); hard data reproduced in every realization. Threads.

### py
- Every class round-trips through `to_json` / `from_json` and pickle.
- Outputs come back in input row order for `direction="up"` and `"down"` and for list and
  `holes=` input.
- Each Rust error maps to the right Python class; `InvalidInput` is a `ValueError`.

## 8. Ownership

Implementers work in parallel in this repository. Each edits only the files listed for its
area. Shared files have marked sections created by the core owner; an implementer edits only
inside its own markers (`# >>> <area>` and `# <<< <area>` in Python and YAML, `// >>> <area>`
and `// <<< <area>` in Rust), with exact-string edits, never rewriting the whole file.

| Area | Files it may create and edit |
| --- | --- |
| core (lands first) | `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `rustfmt.toml`, `mise.toml`, `pyproject.toml`, `uv.lock`, `.gitignore`, `cliff.toml`, `CHANGELOG.md`, `mkdocs.yml`, `pixi.toml`, `environment.yml`, `ci/**`, `.github/**`, `README.md` (Layout section only), `crates/core/**`, every `crates/<c>/Cargo.toml` and `crates/<c>/src/lib.rs` skeleton (before hand-over), `crates/py/Cargo.toml`, `crates/py/src/{lib.rs,args.rs,core.rs}`, `python/transiograms/{__init__.py,errors.py,py.typed}`, the skeletons with markers of `python/transiograms/{_transiograms.pyi,plot.py,datasets.py,checks.py}`, `python/tests/{conftest.py,test_core.py}`, `examples/{render.py,common.py}`, `examples/00-core/**`, `docs/{index.md,install.md}`, `docs/api/core.md` |
| chain | `crates/chain/**`, `crates/py/src/chain.rs`, chain sections of `_transiograms.pyi`, `plot.py`, `datasets.py`, `checks.py`, `python/tests/test_chain.py`, `examples/01-markov-chains/**`, `docs/api/chain.md` |
| hmm | `crates/hmm/**`, `crates/py/src/hmm.rs`, hmm sections, `python/tests/test_hmm.py`, `examples/02-hidden-markov/**`, `docs/api/hmm.md` |
| hsmm | `crates/hsmm/**`, `crates/py/src/hsmm.rs`, hsmm sections, `python/tests/test_hsmm.py`, `examples/03-semi-markov-and-selection/**`, `docs/api/hsmm.md` |
| transiogram | `crates/transiogram/**`, `crates/py/src/transiogram.rs`, transiogram sections, `python/tests/test_transiogram.py`, `examples/04-transiograms/**`, `docs/api/transiogram.md` |
| tpsim | `crates/tpsim/**`, `crates/py/src/tpsim.rs`, tpsim sections, `python/tests/test_tpsim.py`, `examples/05-tp-simulation/**`, `docs/api/tpsim.md` |
| mcrf | `crates/mcrf/**`, `crates/py/src/mcrf.rs`, mcrf sections, `python/tests/test_mcrf.py`, `examples/06-mcrf-mcp/**`, `docs/api/mcrf.md` |
| integration (after all) | `examples/07-boitata-integration/**`, `python/tests/test_integration.py`, `docs/api/integration.md` |

Rules:

- An area's `crates/<c>/Cargo.toml` may list only workspace dependencies from section 2.2 and
  path dependencies allowed by section 2.1. Needing anything else is a reported blocker.
- The core owner creates every crate skeleton (`Cargo.toml`, empty `lib.rs`), every
  `crates/py/src/<area>.rs` with an empty `register`, and all section markers, then hands over.
- Build order: core, then chain, hmm and transiogram in parallel; hsmm starts with hmm's
  `sequence.rs` and `emission.rs` landed (hmm delivers them first); tpsim and mcrf start once
  `TransitionModel` and `Table` compile (transiogram delivers `model3d.rs` and `table.rs` first).
- Design files under `docs/design/` are read-only for implementers except their own crate's
  "Implementer decisions" heading.
- `docs/design/spec.md` changes only by the contract owner.

## 9. Milestone map

| Milestone | Areas | Acceptance tests |
| --- | --- | --- |
| M1 | core, chain, py (chain) | chain recovery test, row-sum proptest |
| M2 | hmm | brute force, posterior sums, EM monotone |
| M3 | hsmm | geometric equivalence, BIC K = 3 |
| M4 | transiogram | M4 lag-grid validity test |
| M5 | tpsim | M5 test |
| M6 | mcrf | M6 test |
| M7 | integration | NumPy round trip with 100 percent hard data and proportions within 0.02 |
