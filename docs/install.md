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
