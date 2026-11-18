# Install

varvite builds from source with Rust and maturin. In a clone of the repository:

```sh
mise install
mise run py:build
```

This installs varvite into the project's `.venv`. Its only runtime dependency is NumPy;
`pip install "varvite[plot]"` adds matplotlib for `vv.plot`.
