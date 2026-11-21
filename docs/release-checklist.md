# Release checklist

**Verdict on 2026-11-20: NOT READY to publish.**

Two things hold the release:

1. **minetypes adoption.** By owner decision, the public release waits for the shared types
   package `minetypes` to replace the duplicate container types listed below. That step has not
   started, and `minetypes` is not a dependency yet. This is the one remaining gate.
2. **The workflows have never run on GitHub.** Every local check passes on Linux x86_64. The six
   other wheels (Linux aarch64, both musllinux wheels, both macOS wheels, Windows) have never been
   built or tested anywhere. The owner's dry run below (a CI run on `main` and a manual release
   run) must pass before the tag.

Every audit blocker of the final review is closed, and a test that runs in `mise run verify`
guards each one (see the table). The first public version is 0.1.0, on PyPI only. The repository
stays private until the owner says otherwise. Nothing in this repository uploads, tags or pushes.

## Go/no-go

All checks ran on 2026-11-20 on Linux x86_64, one command at a time, with
`CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=4`, on commit `be643e3` (the tree before this page was
rewritten; this page changes no code).

| Check | Command | Result on 2026-11-20 | Commit | Go |
| --- | --- | --- | --- | --- |
| Lint | `mise run -j 1 lint` | rustfmt, clippy `-D warnings`, ruff check and ruff format clean | `be643e3` | go |
| Workflow lint | `mise run -j 1 lint-ci` | actionlint reports nothing | `be643e3` | go |
| Tests | `mise run -j 1 test` | Rust 285 passed, 0 failed, 10 ignored; pytest 1125 passed | `be643e3` | go |
| Examples | `mise run -j 1 examples` | all examples re-rendered; `git status` shows no change, so figures and printed output are reproducible | `be643e3` | go |
| Wheel | `mise run -j 1 wheel` | `transiograms-0.1.0-cp311-abi3-manylinux_2_35_x86_64.whl` (1.5 MB) | `be643e3` | go |
| Wheel in a clean venv | `mise run -j 1 test-wheel` | Python 3.11, imported from site-packages; 1120 passed, 5 skipped (tools absent from the clean venv) | `be643e3` | go |
| sdist and a clean build from it | `mise run -j 1 test-sdist` | sdist built, compiled from source in a fresh 3.11 venv, smoke check printed `0.1.0` | `be643e3` | go |
| Documented code blocks | `mise exec -- uv run --no-sync pytest -q python/tests/test_docs.py python/tests/test_docstrings.py` | 813 passed: every guide, API and example page block, each example script alone in an empty folder, every docstring example | `be643e3` | go |
| Docs site | `mise run -j 1 docs-build` | `mkdocs build --strict` clean, theme check 6 passed | `be643e3` | go |
| Site colours | grep of `site/stylesheets/transiograms.css` | 9 colour literals, all grey: `#000000`, `#171717`, `#262626`, `#525252`, `#737373`, `#a3a3a3`, `#e5e5e5`, `#f5f5f5`, `#ffffff`; Material primary and accent are `custom`; no inline colour styles in the HTML | `be643e3` | go |
| Distribution metadata | `mise run -j 1 check-dist` | twine 6.2.0 `check --strict` PASSED for the wheel and the sdist; abi3audit `is_abi3: true`, baseline 3.11, no non-abi3 symbols | `be643e3` | go |
| Wheels for the other six platforms | `release.yml` on GitHub | never run | none | **no-go** |
| minetypes adoption | separate adoption step | not started | none | **no-go** |

### Audit blockers and the tests that close them

Each test below passed on 2026-11-20 at `be643e3` when run on its own.

| Audit blocker | Test |
| --- | --- |
| MCRF misses the target proportions and transiograms, undocumented | `crates/mcrf/tests/limits.rs` (3 tests: 1D exactness, Sectors gap, Nearest(4) gap) |
| T-PROGS misses the transiograms beyond sampling error, undocumented | `crates/tpsim/tests/limits.rs` (2 tests: quenched and unquenched gaps) |
| No fit-quality report in Rust or Python | `transiograms-transiogram` quality tests (4); `test_mcrf.py::test_fit_quality_matches_numpy_counts`, `test_fit_quality_rejects_a_grid_of_another_size`; `test_tpsim.py::test_fit_quality_shows_the_quench_closes_part_of_the_gap` |
| Licence MIT only | `test_packaging.py::test_licence_is_mit_or_apache_everywhere` |
| No third-party licence notices in the wheel | `test_third_party_licences_name_every_crate_the_extension_links`, `test_cargo_about_covers_the_release_targets_and_build_dependencies`, `test_built_wheel_metadata_names_urls_licence_and_classifiers`, `test_built_sdist_carries_the_licences_and_the_lockfile` |
| No GitHub Actions workflows | `test_every_workflow_job_acts_only_in_this_repository_and_pins_actions_to_a_sha`, `test_release_publishes_only_what_it_builds_and_tests`, `test_ci_and_docs_run_the_developer_tasks` |
| No project URLs or authors; no `site_url` or `repo_url` | `test_project_urls_and_authors_point_at_the_organization`, `test_mkdocs_site_and_repo_urls` |
| README etymology describes the rock of the old name; old-name leftovers | `test_no_shipped_or_documented_text_keeps_the_old_name` |
| README has no install command or quick start; long description is a roadmap | `test_readme_opens_with_install_and_a_quick_start_not_a_roadmap`, `test_readme_names_the_docs_and_links_work_on_pypi` |
| Install page covers a source build only | `test_docs.py` (the install page blocks) and the README tests above |
| CHANGELOG has no 0.1.0 section | `test_changelog_has_a_section_for_this_version` |
| No release checklist, CONTRIBUTING.md or issue templates | `test_release_checklist_names_the_minetypes_gate_and_the_types_it_replaces`, `test_contributing_and_issue_templates_say_code_starts_with_an_issue` |
| Example pages import `common` and do not run as written | `test_docs.py::test_full_script_download_runs_on_its_own` (11 pages) |
| API pages are pasted stub signatures | `test_docs.py::test_api_pages_render_docstrings_with_stub_types_and_defaults`, `test_stub.py::test_api_pages_render_every_public_name_from_its_docstring` |
| 124 public objects without docstrings; no Examples or Raises | `python/tests/test_docstrings.py` (every public object, every example run as a doctest) |
| `layered_drillholes` does not follow its stated model | `test_datasets.py::test_layered_drillholes_departs_from_its_model` (gap pinned and quoted in the docstring) |
| Docs site has no brand stylesheet | `test_docs_site_uses_the_family_theme_in_both_schemes`, `python/tests/test_docs_theme.py` |

## The one remaining gate: minetypes adoption

`minetypes` will replace these public types of this library (see `minetypes/docs/adoption.md`,
section transiograms):

| transiograms type | Replaced by | Where |
| --- | --- | --- |
| `tg.Categories` | `mt.Categories` (`int32` codes, `-1` missing; `tg` returns `int64` today) | Python class in `crates/py` |
| `tg.Grid` | `mt.Grid` (`origin`, `size`, `count`, x fastest; `tg.Grid` has no `rotation`) | Python class in `crates/py` |
| `transiograms_core::Categories` | `minetypes::Categories` | Rust, `crates/core/src/codes.rs` |
| `transiograms_core::Grid` | `minetypes::Geometry` (a regular grid without attributes) | Rust, `crates/core/src/grid.rs` |
| `transiograms_core::SequenceSet` and `Hole` | `minetypes::IntervalsView` (hole index, from, to) | Rust, `crates/core/src/sequence.rs` |
| `transiograms_core::synthetic::Intervals` and `Drillholes` | `minetypes` `Intervals` and `Drillholes` | Rust test-data generators, `crates/core/src/synthetic.rs` |

The adoption plan converts at the boundary for the two Python classes first and calls the Rust
side optional; the adoption step decides which Rust types it replaces and which it converts. Until
then, nobody adds a container type that duplicates a `minetypes` type. `tg.FitQuality` is a report,
not a container, and stays.

## Open non-blocking findings

- **Method limits, documented and pinned.** Three strict sampling-bound audit tests stay
  `#[ignore]` because the methods cannot meet them: two in `crates/mcrf/tests/audit.rs`
  (`category 0: 0.4616 vs 0.5`; `Sectors axis x: 31.2 standard errors, max error 0.1900`) and one
  in `crates/tpsim/tests/audit.rs` (`quench true: axis x: 14.7 standard errors, max error
  0.0260`). `limits.rs` in each crate pins the measured gaps instead.
- **Platforms not exercised.** Only the Linux x86_64 wheel exists locally; `mise run wheel` tags
  it `manylinux_2_35`, which is not an upload tag. On macOS and Windows only clippy of the core
  crate has run.
- **Release test risk.** The release tests install SciPy and matplotlib with
  `--only-binary=:all:`. Where no binary wheel exists (alpine aarch64 on 3.15, for example), that
  job fails instead of skipping.
- **NumPy floor.** Only the compat smoke check (`mise run compat uv`) runs on Python 3.11 with
  NumPy 1.26; the full pytest suite has not run against that floor.
- **`fit_markov` non-convergence.** The new `ConvergenceError` path is tested at the
  `nelder_mead` level only; no seeded transiogram exhausts the evaluation budget.
- **`test-wheel` on Windows.** The task uses POSIX venv paths, so CI does not run it on Windows;
  `release.yml` tests the Windows wheel instead.
- **Rendered site.** Nobody has looked at the built site in a browser, in either scheme. Run
  `mise run docs` and check it once.
- **Commit split.** Commit `6c6357d` also carries the 04-01 and 05-01 narrative changes that
  belong with `39d5e79`. The tree at HEAD is correct.
- **crates.io.** The crates are `publish = false`; names, `readme`, `keywords`, `categories` and
  docs.rs metadata need work before a crates.io release (see below).
- **conda-forge.** No recipe yet; it comes after PyPI.

## Decisions only the owner can take

1. When to adopt `minetypes`, and which Rust types it replaces rather than converts.
2. When to make the repository public.
3. The default MCRF neighbourhood. Today it is `Sectors` (largest transiogram error 0.190 on the
   seeded limit test); `Nearest(4)` comes closer to the model unconditionally (0.088).
4. Whether `tg.datasets.layered_drillholes` keeps MCRF as its generator (it departs from its own
   model, documented and pinned) or switches to one that follows the model. A switch changes every
   test and example that uses it.
5. Whether to fund a different algorithm for the strict audit targets: a combiner without
   conditional independence for MCRF, or quenching over more lags or annealing for T-PROGS.
6. Whether `THIRD-PARTY-LICENSES.txt` should leave out this repository's own crates (cargo-about
   lists them under MIT today; the licences are correct either way).
7. Whether every public method needs its own Examples section (about 110 more examples) or the
   class example that calls it is enough, as now.
8. Whether each repository gets its own mise-managed Rust. Today Rust comes from rustup through
   `rust-toolchain.toml` (1.97.1), because mise's Rust install on this machine points into segyrs.
9. The PyPI author field: "Gustavo Pretto Scholze" (as in boitata and `LICENSE-MIT`) or a
   generic "the transiograms authors" (as in segyrs).
10. The transiogram mark and the wordmark in `docs/brand/`: keep or replace.

## Publishing by hand

Do these steps in order, after the `minetypes` gate is closed and the tree passes every check
in the table again. The [install and release page](install-and-release.md#one-time-setup-repository-owner)
has the same setup in more detail.

1. **Check the name.** Confirm that `transiograms` is still free on <https://pypi.org>.
2. **Make the repository public.** GitHub, Settings, General, Danger Zone, Change visibility,
   Public. The `cratongen` organization is on the free plan, so Pages only serves public
   repositories.
3. **Enable Pages.** Settings, Pages, Build and deployment, Source: **GitHub Actions**.
4. **Create the `pypi` environment.** Settings, Environments, New environment, name `pypi`. Add
   yourself as a required reviewer, and under Deployment branches and tags allow only tags that
   match `v*`.
5. **Allow the actions.** Settings, Actions, General: allow GitHub Actions and the pinned actions
   the workflows use.
6. **Add the PyPI pending trusted publisher.** On pypi.org, Account settings, Publishing, Add a
   new pending publisher, GitHub:
   - PyPI project name: `transiograms`
   - Owner: `cratongen`
   - Repository name: `transiograms`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   Two-factor authentication must be on. No API token and no repository secret are needed.
7. **Dry run.** Push `main` and wait for `ci` and `docs` to pass. Then Actions, release, Run
   workflow on `main`: it builds the sdist and all seven wheels and tests each one, and uploads
   nothing. Fix anything red before going on.
8. **Date the release.** In `CHANGELOG.md`, replace `0.1.0 (unreleased)` with `0.1.0 (YYYY-MM-DD)`,
   run the checks of the table, and commit.
9. **Tag and push the tag.**

   ```sh
   git tag -a v0.1.0 -m "transiograms 0.1.0"
   git push origin v0.1.0
   ```

10. **Approve the upload.** The release run builds and tests, then waits on the `pypi`
    environment. Review the run and approve the deployment.

PyPI never accepts the same version twice: if the upload fails after some files land, release
0.1.1.

### After publishing, check

- <https://pypi.org/project/transiograms/> shows 0.1.0, the README renders, the project links
  work, and the file list holds the sdist and seven wheels: `manylinux_2_28` x86_64 and aarch64,
  `musllinux_1_2` x86_64 and aarch64, `macosx_10_15_x86_64`, `macosx_11_0_arm64`, `win_amd64`.
- In a clean environment on Python 3.11 and on 3.15:

  ```sh
  pip install transiograms
  python -c "import transiograms as tg; print(tg.__version__)"
  ```

  prints `0.1.0`, and the README quick start prints the output it shows.
- <https://cratongen.github.io/transiograms/> loads in light and dark, and the README badges
  resolve.
- The pending publisher on pypi.org has become a normal trusted publisher for the project.

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
