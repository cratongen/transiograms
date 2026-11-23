<p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://cratongen.github.io/transiograms/brand/wordmark-white.svg">
    <img alt="transiograms" src="https://cratongen.github.io/transiograms/brand/wordmark-black.svg" width="372">
  </picture>
</p>

[![PyPI](https://img.shields.io/pypi/v/transiograms?color=black&label=PyPI)](https://pypi.org/project/transiograms/)
[![Docs](https://img.shields.io/badge/docs-cratongen.github.io-black)](https://cratongen.github.io/transiograms/)
[![Licence](https://img.shields.io/badge/licence-MIT%20OR%20Apache--2.0-black)](https://github.com/cratongen/transiograms/blob/main/LICENSE-MIT)

**Markov models of geological categories: fit facies and lithology sequences down drillholes,
model transiograms, and simulate categories on 3D grids. A Rust core with Python bindings.**

transiograms is for geologists, geostatisticians and resource modellers who log categories
(facies, lithology, alteration, ore type) and want their spatial structure as numbers they can
test. Along drillholes and logs it fits Markov chains, hidden Markov models and hidden semi-Markov
models. On grids it builds 3D transition models from transiograms, simulates them with T-PROGS or
Markov chain random fields (MCRF), and predicts them with MCP. The name comes from the
transiogram: the probability of finding category `j` at lag `h` from a point in category `i`, as a
function of `h`.

[Docs](https://cratongen.github.io/transiograms/) ·
[Guide](https://cratongen.github.io/transiograms/guide/markov-chains/) ·
[Examples](https://cratongen.github.io/transiograms/examples/01-markov-chains/01-embedded-chains/) ·
[API](https://cratongen.github.io/transiograms/api/core/) ·
[Status](https://cratongen.github.io/transiograms/status/) ·
[Changelog](https://github.com/cratongen/transiograms/blob/main/CHANGELOG.md)

## Install

```sh
pip install transiograms
```

With uv, `uv add transiograms` in a project or `uv pip install transiograms` in an environment.
transiograms needs CPython 3.11 or newer and ships one `abi3` wheel per platform: Linux (x86_64
and aarch64, each with glibc and with musl), macOS (Intel and Apple silicon) and Windows x86_64.
NumPy is its only runtime dependency; `pip install "transiograms[plot]"` adds matplotlib for
`tg.plot`. Other platforms build from the sdist, which needs Rust 1.97 or newer. The
[install page](https://cratongen.github.io/transiograms/install-and-release/) has the details.

## Quick start

```python
import numpy as np
import transiograms as tg

beds = tg.datasets.varved_sequence(seed=0)
chain = tg.MarkovChain("embedded").fit("CODE", holes="HOLE_ID", from_="FROM", to="TO", data=beds)
print(chain.transition_matrix.round(2))
print("mean thickness (m):", chain.mean_lengths.round(2))

holes = tg.datasets.layered_drillholes(seed=0)
lags = np.arange(1.0, 9.0)
exp = tg.downhole_transiogram("CODE", holes="HOLE_ID", from_="FROM", to="TO", lags=lags, data=holes)
p = np.bincount(holes["CODE"], minlength=3) / len(holes["CODE"])
vertical = tg.TransiogramModel.fit_markov(exp, proportions=p)
print("vertical mean lengths (m):", (-1 / np.diag(vertical.rates)).round(2))
print("largest residual:", np.nanmax(np.abs(vertical.residuals(exp))).round(3))
```

```text
[[0.   0.83 0.17]
 [0.12 0.   0.88]
 [0.84 0.16 0.  ]]
mean thickness (m): [1.2  0.83 0.46]
vertical mean lengths (m): [9.43 7.4  5.44]
largest residual: 0.136
```

The first part fits an embedded Markov chain to twelve seeded holes of sand, silt and clay beds:
sand passes up into silt 83% of the time and silt into clay 88% of the time. The second part
measures vertical transiograms down 25 seeded drillholes, fits a Markov model with its sills at
the sample proportions, and reports the misfit. The
[examples](https://cratongen.github.io/transiograms/examples/01-markov-chains/01-embedded-chains/)
go on to hidden Markov segmentation, 3D transition models and grid simulation.

## What it does

- **Markov chains**: embedded and fixed-step transition matrices, continuous-lag generators with
  `P(h) = exp(Q h)`, tests of the Markov property, order and homogeneity, sojourn statistics, and
  seeded 1D simulation.
- **Hidden Markov and semi-Markov models**: Gaussian, categorical, Poisson and Gaussian-mixture
  emissions; forward-backward, Viterbi and Baum-Welch; explicit duration laws (geometric, shifted
  Poisson, negative binomial, empirical); AIC, BIC, ICL and cross-validated state-count selection.
- **Transiograms**: experimental transiograms from drillholes, points and grids; Markov and
  entrywise models whose rows sum to one at every lag; 3D transition models from rates or mean
  lengths.
- **Grid simulation**: transition-probability cokriging and T-PROGS simulation with quenching;
  MCP prediction and MCRF simulation, conditioned to drillholes.
- **Fit-quality reports**: `tg.fit_quality` compares T-PROGS or MCRF realizations with their model
  (proportions, and grid transiograms per lag with standard errors), and
  `TransiogramModel.residuals` gives the misfit of a 1D fit. The report is the same from Rust
  and Python.

Every stochastic routine takes a `seed` and gives the same bits on any number of threads. Grid
work is sized from the available memory and checked before it allocates.

## Status and limits

Version 0.1.0 is the first public release and an alpha: the API can change between minor
versions. Its known gaps:

- **Both simulators keep a measured gap to their model.** This is a limit of the methods, and
  seeded tests pin each number. MCRF without hard data drifts 0.038 from the target proportions
  and misses the transiograms by up to 0.19. T-PROGS with the default quench misses mid-lag
  transiograms by up to 0.026. The
  [T-PROGS](https://cratongen.github.io/transiograms/guide/tp-simulation/) and
  [MCRF](https://cratongen.github.io/transiograms/guide/mcrf-mcp/) guides explain both, and
  `tg.fit_quality` reports the gap for your own model.
- **The container types will change.** `tg.Categories` and `tg.Grid` move to a shared types
  package in a later release, and category codes may change from `int64` to `int32`.
- **It reads no files.** You pass NumPy arrays, or a mapping of column names to arrays, and get
  NumPy arrays back.
- **Performance is not measured.** No benchmark has run yet.
- **PyPI only.** The Rust crates are not on crates.io, and there is no conda-forge package yet;
  `pip install transiograms` works inside a conda environment.

The [status page](https://cratongen.github.io/transiograms/status/) lists each milestone, its
acceptance test and what is still missing.

## Beside the sibling libraries

transiograms is one of a family of standalone libraries for mining and geoscience. None depends on
another; they exchange NumPy arrays.

| Library | Import | What it does |
| --- | --- | --- |
| [boitata](https://github.com/cratongen/boitata) | `import boitata as bt` | Geostatistics: drill holes, compositing, block models, variograms, kriging, simulation |
| [segyrs](https://github.com/cratongen/segyrs) | `import segyrs` | Read, write and slice SEG-Y files, including files larger than memory |
| [transiograms](https://github.com/cratongen/transiograms) | `import transiograms as tg` | Markov chains, hidden Markov models, transiograms and categorical simulation |
| [wireframes](https://github.com/cratongen/wireframes) | `import wireframes as wf` | Solids, surfaces and strings: read, write, measure and combine |

The [boitata guide](https://cratongen.github.io/transiograms/guide/boitata/) shows how intervals,
category codes and grids pass between boitata and transiograms.

## Build from source

The repository uses [mise](https://mise.jdx.dev/) for every tool and command. In a clone:

```sh
mise install          # the pinned Python, uv and helpers; rustup reads rust-toolchain.toml
mise run py:build     # build the extension into .venv
mise run verify       # lint, Rust tests and pytest
mise run docs-build   # the documentation site, built strict
```

## Contributing

Issues are open. A code contribution starts with an issue; see
[CONTRIBUTING.md](https://github.com/cratongen/transiograms/blob/main/CONTRIBUTING.md).

## Licence

transiograms is dual-licensed under MIT OR Apache-2.0; pick either: [LICENSE-MIT](https://github.com/cratongen/transiograms/blob/main/LICENSE-MIT),
[LICENSE-APACHE](https://github.com/cratongen/transiograms/blob/main/LICENSE-APACHE). Each wheel
also carries `THIRD-PARTY-LICENSES.txt`, the licences of the Rust crates linked into it.
