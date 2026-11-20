# Install

```sh
pip install transiograms
```

Its only runtime dependency is NumPy; `pip install "transiograms[plot]"` adds matplotlib for `tg.plot`.

The source is at <https://github.com/cratongen/transiograms> and the documentation at
<https://cratongen.github.io/transiograms/>.

## From a clone

transiograms builds from source with Rust and maturin. In a clone of the repository:

```sh
mise install
mise run py:build
```

This installs transiograms into the project's `.venv`.

## Wheels and Python versions

A release is one wheel per platform. The wheel is `abi3` (`cp311-abi3`), so the same file works on every
CPython from 3.11 up; the classifiers list 3.11 to 3.15. `mise.toml` pins the oldest supported
Python (3.11), the development Python (3.13) and the newest (3.15), and `test_packaging.py` fails
when the pins, `requires-python`, the classifiers and the pyo3 `abi3-py311` feature disagree.
PyPy and the free-threaded builds get no wheel, because pip does not install an `abi3` wheel on
them; they can build from the sdist.

## Building from the sdist

A platform with no wheel builds from the sdist, which needs Rust 1.97 or newer (the workspace
`rust-version`; edition 2024):

```sh
pip install transiograms --no-binary transiograms
```

The repository pins Rust 1.97.1 in `rust-toolchain.toml`, which rustup reads. That file is not in
the sdist, so outside the repository only `rust-version = "1.97"` applies.

## Release mechanics

These tasks build and check the release files locally. None of them uploads anything.

```sh
mise run wheel             # the release abi3 wheel for this platform into dist/
mise run test-wheel        # fresh 3.11 venv: install the wheel, run pytest against it
mise run test-wheel 3.15   # the same on the newest pinned Python
mise run sdist             # the sdist into dist/
mise run test-sdist        # build the sdist from source in a fresh 3.11 venv and import it
mise run check-dist        # twine check --strict and abi3audit --strict of dist/
```

`test-wheel` sets `TG_REQUIRE_DIST=1`, so the wheel and sdist checks in `test_packaging.py` fail
instead of skipping when `dist/` is empty.

### Licences in the packages

The extension links its Rust dependencies statically, so the wheel carries their licences:
`THIRD-PARTY-LICENSES.txt` sits beside `LICENSE-MIT` and `LICENSE-APACHE` in `license-files` and
lands in the wheel's `.dist-info/licenses/` and in the sdist. cargo-about writes it from
`Cargo.lock`, `about.toml` and `about.hbs` over the six release targets:

```sh
mise run third-party-licenses   # after any Cargo.lock change, then commit the file
```

`test_third_party_licences_name_every_crate_the_extension_links` fails when a crate linked on any
release target (from `cargo tree`) is missing from the file. Crates that ship no licence file of
their own get the standard licence text with their authors and source from their `Cargo.toml`.
