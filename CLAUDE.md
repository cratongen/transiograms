@AGENTS.md

## Claude Code

- Do not append `Co-Authored-By` or "Generated with Claude Code" trailers to
  commits or PR descriptions in this repo.
- Run all build, test, lint, and benchmark commands through `mise run` or
  `mise exec` rather than bare `cargo`, `maturin`, `pytest`, or `uv`.
- Load `benchmark-checklist` before reporting any performance comparison and
  `stop-slop` before drafting documentation or example prose.
