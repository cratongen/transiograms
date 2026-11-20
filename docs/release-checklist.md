# Release checklist

The first public version is 0.1.0, published to PyPI only. The repository stays private until the
owner says otherwise. Nothing in this repository uploads, tags or pushes.

## The one remaining gate: minetypes adoption

The public release waits for the adoption of the shared types package `minetypes`, a separate step
after this pass. `minetypes` is not a dependency yet. It will replace these public types of this
library (see `minetypes/docs/adoption.md`, section transiograms):

| transiograms type | Replaced by | Where |
| --- | --- | --- |
| `tg.Categories` | `mt.Categories` (`int32` codes, `-1` missing; `tg` returns `int64` today) | Python class in `crates/py` |
| `tg.Grid` | `mt.Grid` (`origin`, `size`, `count`, x fastest; `tg.Grid` has no `rotation`) | Python class in `crates/py` |
| `transiograms_core::Categories` | `minetypes::Categories` | Rust, `crates/core/src/codes.rs` |
| `transiograms_core::Grid` | `minetypes::Geometry` (a regular grid without attributes) | Rust, `crates/core/src/grid.rs` |
| `transiograms_core::SequenceSet` and `Hole` | `minetypes::IntervalsView` (hole index, from, to) | Rust, `crates/core/src/sequence.rs` |
| `transiograms_core::synthetic::Intervals` and `Drillholes` | `minetypes` `Intervals` and `Drillholes` | Rust test-data generators, `crates/core/src/synthetic.rs` |

The adoption plan converts at the boundary for the two Python classes first and calls the Rust
side optional; which Rust types are replaced rather than converted is decided in the adoption step.
No new container type that duplicates a `minetypes` type may be added before then.

## Packaging (checked by `python/tests/test_packaging.py`)

- [x] Licence `MIT OR Apache-2.0` in `pyproject.toml` and every `Cargo.toml`, with `LICENSE-MIT`
      and `LICENSE-APACHE` in the wheel and the sdist.
- [x] `THIRD-PARTY-LICENSES.txt` in the wheel's `.dist-info/licenses/` and in the sdist: the
      extension links its Rust dependencies statically (MIT, Apache-2.0, BSD-2-Clause, Zlib,
      Unicode-3.0 and others, all compatible with `MIT OR Apache-2.0`), and those licences ask for
      their notices in binary distributions. cargo-about writes the file from `Cargo.lock`,
      `about.toml` and `about.hbs` over the six release targets; run
      `mise run third-party-licenses` after any `Cargo.lock` change and commit the file. The
      packaging tests fail when a crate linked on any release target is missing from it. No
      vendored or ported third-party code or data is in the repository.
- [x] Project URLs (Homepage, Documentation, Repository, Issues, Changelog) and an author in the
      PyPI metadata; `site_url`, `repo_url` and `repo_name` in `mkdocs.yml`.
- [x] Classifiers: `3 - Alpha`, Linux, macOS and Windows, CPython 3.11 to 3.15.
- [x] One `abi3` wheel per platform (`cp311-abi3`), `requires-python = ">=3.11"`, pyo3
      `abi3-py311`; `mise.toml` pins 3.11, 3.13 and 3.15.
- [x] `mise run wheel`, `test-wheel`, `sdist`, `test-sdist` and `check-dist` pass locally on
      Linux x86_64 (2026-11-20).
- [x] Workspace path dependencies carry `version = "0.1.0"`, so every library crate packages.

## Before the first upload

- [ ] Adopt `minetypes` (the gate above).
- [ ] A release workflow that builds the wheels in manylinux and musllinux containers and on
      macOS and Windows, guarded by `if: github.repository == 'cratongen/transiograms'` as in
      boitata, over the six targets of `about.toml` (`RELEASE_TARGETS` in
      `python/tests/test_packaging.py`). None exists yet: `mise run wheel` builds a `manylinux_2_35` wheel for this machine
      only, which is not the tag to upload.
- [ ] A docs workflow that deploys `mise run docs-build` to <https://cratongen.github.io/transiograms/>
      once the repository is public.

## Later: crates.io

The workspace is `publish = false` (PyPI first); `transiograms-py` is `publish = false` on its own
and is never published. Before a crates.io release the owner decides:

- whether `transiograms-transiogram` keeps its name next to the plural top name;
- each crate's `readme`, `keywords`, `categories` and `documentation` (docs.rs);
- whether each crate ships its own copy of the two licence files.

With `publish = false`, `cargo package --workspace` stops at `transiograms-chain` because cargo
looks for `transiograms-core` on crates.io; with `publish = true` in a scratch copy all seven
library crates package (`cargo package --no-verify`, 2026-11-20).

## Later: conda-forge

No recipe yet; conda-forge comes after PyPI.
