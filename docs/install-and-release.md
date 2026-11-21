# Install and release

The first half of this page is for users: how to install transiograms. The second half is for
maintainers: how the wheels are built, tested and published.

## Install

```sh
pip install transiograms
```

or, with uv, `uv pip install transiograms`. transiograms needs CPython 3.11 or newer. Its only
runtime dependency is NumPy; `pip install "transiograms[plot]"` adds matplotlib for `tg.plot`.

Each release builds one wheel per platform for Linux (x86_64 and aarch64, each with glibc and
with musl), macOS (x86_64 and Apple silicon) and Windows (x86_64). Other platforms build from the sdist
(see below). transiograms is on PyPI only for now; a conda-forge package comes later, and inside a
conda environment `pip install transiograms` works.

```python
import transiograms as tg

print(tg.__version__)
```

The source is at <https://github.com/cratongen/transiograms> and the documentation at
<https://cratongen.github.io/transiograms/>.

### From a clone

To work on transiograms itself, build it from source with Rust and maturin. In a clone of the
repository:

```sh
mise install
mise run py:build
```

This installs transiograms into the project's `.venv`.

### Wheels and Python versions

A release is one wheel per platform. The wheel is `abi3` (`cp311-abi3`), so the same file works on every
CPython from 3.11 up; the classifiers list 3.11 to 3.15. `mise.toml` pins the oldest supported
Python (3.11), the development Python (3.13) and the newest (3.15), and `test_packaging.py` fails
when the pins, `requires-python`, the classifiers and the pyo3 `abi3-py311` feature disagree.
PyPy and the free-threaded builds get no wheel, because pip does not install an `abi3` wheel on
them; they can build from the sdist.

### Building from the sdist

A platform with no wheel builds from the sdist, which needs Rust 1.97 or newer (the workspace
`rust-version`; edition 2024):

```sh
pip install transiograms --no-binary transiograms
```

The repository pins Rust 1.97.1 in `rust-toolchain.toml`, which rustup reads. That file is not in
the sdist, so outside the repository only `rust-version = "1.97"` applies.

## Build and check the release files locally

These tasks build and check the release files locally. None of them uploads anything.

```sh
mise run wheel             # the release abi3 wheel for this platform into dist/
mise run test-wheel        # fresh 3.11 venv: install the wheel, run pytest against it
mise run test-wheel 3.15   # the same on the newest pinned Python
mise run sdist             # the sdist into dist/
mise run test-sdist        # build the sdist from source in a fresh 3.11 venv and import it
mise run check-dist        # twine check --strict and abi3audit --strict of dist/
mise run lint-ci           # actionlint and shellcheck over .github/workflows
```

`test-wheel` sets `TG_REQUIRE_DIST=1`, so the wheel and sdist checks in `test_packaging.py` fail
instead of skipping when `dist/` is empty.

### Licences in the packages

The extension links its Rust dependencies statically, so the wheel carries their licences:
`THIRD-PARTY-LICENSES.txt` sits beside `LICENSE-MIT` and `LICENSE-APACHE` in `license-files` and
lands in the wheel's `.dist-info/licenses/` and in the sdist. cargo-about writes it from
`Cargo.lock`, `about.toml` and `about.hbs` over the seven release targets:

```sh
mise run third-party-licenses   # after any Cargo.lock change, then commit the file
```

`test_third_party_licences_name_every_crate_the_extension_links` fails when a crate linked on any
release target (from `cargo tree`) is missing from the file. Crates that ship no licence file of
their own get the standard licence text with their authors and source from their `Cargo.toml`.

## Continuous integration

Three GitHub Actions workflows live in `.github/workflows/`. Every job carries
`if: github.repository == 'cratongen/transiograms'`, so a fork runs nothing, and every action is
pinned to a full commit SHA with its version in a trailing comment.
`test_every_workflow_job_acts_only_in_this_repository_and_pins_actions_to_a_sha` checks both.

| Workflow | When | What it runs |
| --- | --- | --- |
| `ci.yml` | push to `main`, pull request, by hand | `mise run lint` and `lint-ci` (actionlint) on Linux; `mise run test` on Linux, macOS and Windows; `mise run test-wheel 3.11` and `test-wheel 3.15` on Linux and macOS; `mise run test-sdist` on Linux |
| `docs.yml` | push to `main`, by hand | `mise run docs-build`, then a deploy to GitHub Pages through the `github-pages` environment |
| `release.yml` | a `v*` tag, by hand | the sdist and seven wheels, a test of each wheel, then the upload to PyPI |

CI runs the same mise tasks as a developer, so a green `mise run verify` on your machine and a
green CI run check the same things. `test-wheel` uses POSIX paths, so on Windows CI runs the
development Python only; `release.yml` tests the Windows wheel on 3.11, 3.13 and 3.15.

### The release workflow

| Job | What it does |
| --- | --- |
| `sdist` | On a tag, checks that the tag equals the workspace version in `Cargo.toml`, then builds the sdist. |
| `wheels` | Builds one `cp311-abi3` wheel per platform with `PyO3/maturin-action`, `--locked` against the committed `Cargo.lock`, and fails if the wheel carries a sidecar library. |
| `test-linux` | Installs each Linux wheel with pytest, SciPy and matplotlib in the official `python:<version>-slim` (glibc) or `-alpine` (musl) image and runs `python/tests`. x86_64 covers 3.11 to 3.15; aarch64 runs on an arm64 runner on 3.11, 3.13 and 3.15. matplotlib has no musllinux aarch64 wheel, so that platform runs without it and without the plot and documented-code tests. |
| `test-other` | The same for the macOS and Windows wheels on 3.11, 3.13 and 3.15, in a uv-managed CPython. |
| `publish` | Waits for every job above, checks that `dist/` holds the sdist and exactly one wheel for each of the seven platform tags, then uploads with Trusted Publishing. |

| Platform | Rust target | Wheel tag |
| --- | --- | --- |
| Linux x86_64, glibc | `x86_64-unknown-linux-gnu` | `manylinux_2_28_x86_64` |
| Linux aarch64, glibc | `aarch64-unknown-linux-gnu` | `manylinux_2_28_aarch64` |
| Linux x86_64, musl | `x86_64-unknown-linux-musl` | `musllinux_1_2_x86_64` |
| Linux aarch64, musl | `aarch64-unknown-linux-musl` | `musllinux_1_2_aarch64` |
| macOS Intel | `x86_64-apple-darwin` | `macosx_10_15_x86_64` |
| macOS Apple silicon | `aarch64-apple-darwin` | `macosx_11_0_arm64` |
| Windows x64 | `x86_64-pc-windows-msvc` | `win_amd64` |

The `publish` job runs only on a tag push in `cratongen/transiograms`. A manual run, a branch, a
pull request or a fork builds and tests at most, and never uploads. The job holds no token: PyPI
trusts the `pypi` environment of this repository's `release.yml` and nothing else.
`test_release_publishes_only_what_it_builds_and_tests` fails when a built platform has no test
job, when the targets differ from `about.toml`, or when one of these guards goes missing.

## Cut a release

The [release checklist](release-checklist.md) lists what is done and what still blocks 0.1.0.

1. Set `version` under `[workspace.package]` in `Cargo.toml`; the Python package takes its version
   from there. Move the `CHANGELOG.md` section from `(unreleased)` to the release date.
2. Run `mise run verify`, `mise run docs-build`, `mise run test-wheel`, `mise run test-sdist` and
   `mise run check-dist`, then commit.
3. Optional dry run: in GitHub, open Actions, then release, then Run workflow on `main`. It builds
   and tests every wheel and uploads nothing.
4. Tag the commit and push the tag:

   ```sh
   git tag -a v0.1.0 -m "transiograms 0.1.0"
   git push origin v0.1.0
   ```

5. The workflow builds, tests and then waits for the `pypi` environment. Approve the deployment;
   the upload starts after that.

PyPI never accepts the same version twice, so a failed upload needs a new version number.

## One-time setup (repository owner)

Do these steps once, by hand, before the first release.

### PyPI

1. Check that the name `transiograms` is free on pypi.org.
2. Sign in to pypi.org, open Account settings, then Publishing, and add a pending publisher:
   - PyPI project name: `transiograms`
   - Owner: `cratongen`
   - Repository name: `transiograms`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. Turn on two-factor authentication if it is not on yet.

The first upload creates the project and turns the pending publisher into a normal one. There is
no API token and no repository secret.

### GitHub

1. Open Settings, then Environments, and create `pypi`. Add yourself as a required reviewer and
   limit deployments to tags that match `v*`.
2. Under Settings, then Actions, then General, allow GitHub Actions and the actions the workflows
   use.

While the repository is private, every job uses billed Actions minutes, and macOS minutes cost
more than Linux ones. A public repository uses none.

## Docs site

The site is MkDocs with the Material theme. The pages are `docs/`, the navigation is `mkdocs.yml`
and the theme is `docs/stylesheets/transiograms.css`.

```sh
mise run docs          # serve on http://127.0.0.1:8000 with live reload
mise run docs-build    # mkdocs build --strict into site/, then the theme check
```

The API pages read the docstrings of the built extension, so both tasks build it first.
`docs.yml` runs `mise run docs-build` on every push to `main` and uploads `site/`. The
`cratongen` organization is on the free plan, which serves Pages from public repositories only:
while the repository is private, the deploy job is skipped and the build still runs strictly.
Once the owner makes the repository public:

1. Open Settings, then Pages, and set Source to GitHub Actions.
2. Open Actions, then docs, then Run workflow on `main`. The site goes live at
   <https://cratongen.github.io/transiograms/>.
