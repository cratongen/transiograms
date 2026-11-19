# Install

transiograms builds from source with Rust and maturin. In a clone of the repository:

```sh
mise install
mise run py:build
```

This installs transiograms into the project's `.venv`. Its only runtime dependency is NumPy;
`pip install "transiograms[plot]"` adds matplotlib for `tg.plot`.
