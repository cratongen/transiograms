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
      `about.toml` and `about.hbs` over the seven release targets; run
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

## Documentation (checked by `python/tests/test_docs.py` and `test_packaging.py`)

- [x] The README (the PyPI description) opens with `pip install transiograms` and a quick start
      whose printed output the docs test compares; it keeps no roadmap and no text of the old name.
- [x] The install page gives the Python range, the wheel platforms, the sdist build and a conda
      note.
- [x] Every page's code runs in a clean environment with only the package: the guide, the example
      pages and each example's downloadable script.
- [x] The API pages render the docstrings with mkdocstrings (types and defaults from the stub);
      every public name appears once; `mise run docs-build` passes with `--strict`.
- [x] `CONTRIBUTING.md` (a code contribution starts with an issue) and the issue forms in
      `.github/ISSUE_TEMPLATE/`.
- [x] The site uses the family theme (`docs/stylesheets/transiograms.css`), light and dark.
- [x] `CHANGELOG.md` has a curated `0.1.0 (unreleased)` section. At release, replace
      `(unreleased)` with the date and tag `v0.1.0`, so that `mise run changelog` (git-cliff with
      `cliff.toml`) drafts later entries from the commits after that tag.

## Before the first upload

- [ ] Adopt `minetypes` (the gate above). This is the one remaining gate.
- [x] `.github/workflows/release.yml` builds the sdist and seven `cp311-abi3` wheels (manylinux
      and musllinux for x86_64 and aarch64, macOS x86_64 and arm64, Windows x86_64) over the
      targets of `about.toml`, tests each wheel in a clean environment and publishes to PyPI with
      Trusted Publishing (environment `pypi`, no token), only from a `v*` tag push in
      `cratongen/transiograms`. `mise run wheel` builds a `manylinux_2_35` wheel for this machine
      only, which is not the tag to upload.
- [x] `.github/workflows/ci.yml` runs `mise run lint`, `test`, `test-wheel` and `test-sdist` on
      Linux, macOS and Windows (see the [install and release page](install-and-release.md)).
- [x] `.github/workflows/docs.yml` builds with `mise run docs-build` and deploys to
      <https://cratongen.github.io/transiograms/> through the `github-pages` environment once the
      repository is public.
- [x] Every workflow job is guarded by `if: github.repository == 'cratongen/transiograms'` and
      every action is pinned to a commit SHA (copied from segyrs and confirmed with
      `git ls-remote` on 2026-11-20); the packaging tests check both. actionlint reports nothing.
- [ ] None of the three workflows has run on GitHub yet. The first CI run on `main` and a manual
      release run (build and test only) are the owner's dry run before the tag.
- [ ] The owner's one-time setup: the PyPI pending publisher and the `pypi` environment
      (see [One-time setup](install-and-release.md#one-time-setup-repository-owner)).

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
