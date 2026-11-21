# Contributing to transiograms

Issues are open to everyone. Bug reports, wrong or surprising results, questions about a method
and ideas are welcome as
[GitHub issues](https://github.com/cratongen/transiograms/issues/new/choose).

## A code contribution starts with an issue

Open an issue before you write code, and wait for an answer on it. Say what you want to change and
why. A pull request that does not link an accepted issue is closed with a pointer back here. This
keeps work from being done twice and keeps the API small.

When the issue is accepted:

1. Fork the repository and branch from `main`.
2. Install [mise](https://mise.jdx.dev/), then run `mise install` and `mise run py:build`. Every
   command is a mise task (`mise run test`, `mise run lint`, `mise run docs-build`); do not call
   `cargo`, `python`, `pip`, `maturin` or `uv` on their own.
3. Make one logical change per commit, with
   [Conventional Commits](https://www.conventionalcommits.org/) messages (`fix(hmm): ...`,
   `feat(transiogram): ...`, `docs: ...`). Add no AI or `Co-Authored-By` trailer.
4. A feature ships whole: the Rust code in its crate under `crates/`, the pyo3 binding, the
   `.pyi` stub, a NumPy-style docstring, an example page and a theory check (for example: rows sum
   to 1, Viterbi matches a brute-force search, simulations reproduce their model within a stated
   tolerance). A fix adds a test that fails without it.
5. Run `mise run verify` (lint and tests) and, when docs or examples change, `mise run examples`
   and `mise run docs-build`. Open the pull request with `Closes #N` for the issue.

## Rules that are not negotiable

- **Fix bugs, document limits.** A method that cannot reproduce its target (proportions or
  transition probabilities) is kept, and its gap is documented with a seeded test that pins the
  number, as in `crates/mcrf/tests/limits.rs`. Do not loosen a tolerance or skip a test to make it
  pass.
- **Seeded and thread-independent.** Every stochastic routine takes an explicit `seed` and gives
  the same bits on any number of threads.
- **Memory from the machine, checked first.** No fixed RAM caps; size work from the available
  memory and raise `InvalidInput` before allocating when a request does not fit.
- **No client data.** Tests and examples use `tg.datasets`, seeded synthetic data or small public
  files whose licence allows redistribution. A file over 1 MB does not go in git.
- **Measured numbers only.** A performance number in an issue, a commit or the docs comes from a
  benchmark you ran: before and after, same machine, interleaved rounds, medians, load noted.
  Otherwise write "not measured".
- **No new runtime dependency.** NumPy is the only one; matplotlib is the optional `plot` extra.
- **English only** in code, docs and commit messages.

By contributing you agree that your contribution is licensed under MIT OR Apache-2.0, the licence
of this project, without additional terms.
