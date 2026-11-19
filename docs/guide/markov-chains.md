# Markov chains

A Markov chain describes a drillhole as a sequence of categories in which the next one depends
only on the current one. transiograms fits three kinds:

| Kind | Counts | Gives |
| --- | --- | --- |
| `MarkovChain("embedded")` | transitions between successive beds (zero diagonal) | `transition_matrix`, `mean_lengths`, the generator `Q` |
| `MarkovChain("fixed")` | transitions between samples one `step` apart | `P(step)`, and `Q` when `P` has a real logarithm |
| `MarkovChain.from_rates(Q)` | nothing: you give `Q` | `P(h) = exp(Q h)` at any lag |

`MarkovChain.from_geology(proportions, mean_lengths, embedded)` builds `Q` the way T-PROGS does,
from volume proportions, mean bed thicknesses and optional embedded juxtapositions.

## Sequence arguments

Every function that reads drillhole categories takes `codes` and the same keywords: `holes`,
`from_` and `to` for intervals, or `depth` and `step` for regular samples, `gap`, `categories`
and `data`. Strings name columns of `data`, so a boitata table converts with
`{c: t[c] for c in t.column_names}`:

```python
import transiograms as tg

d = tg.datasets.varved_sequence()
chain = tg.MarkovChain().fit("CODE", holes="HOLE_ID", from_="FROM", to="TO", data=d)
chain.transition_matrix, chain.mean_lengths, chain.generator
```

Rows go in top-down by depth. `direction="up"`, the default, reads each hole from its base
(older to younger); a transition `i -> j` then means `j` lies on `i`. Missing codes (-1 or NaN)
and gaps larger than `gap` split a hole into runs, and no transition crosses a run boundary.

## Tests

- `chain.independence_test()`: quasi-independence for embedded chains (Goodman), ordinary
  contingency for fixed ones. `chain.difference_matrix()` shows the preferred paths.
- `tg.order_test(codes, order=2)` and `tg.select_order(codes, max_order=3)`: Markov order, every
  order fitted on the same transitions.
- `tg.homogeneity_test(codes, groups)`: one matrix for all groups of holes against one per group.
- `chain.symmetry_test()`: Bowker test of `n_ij = n_ji`.
- `tg.sojourn(codes)` and `Sojourn.exponential_test(code)`: bed thicknesses against the
  exponential law a chain implies, with open beds censored.

## Simulation

`chain.simulate(length, n=, seed=)` draws `n` logs: integer codes per step for fixed chains, beds
`(realization, from, to, code)` for chains with a generator. `hard_from`, `hard_to` and
`hard_codes` condition on logged intervals through Markov bridges (Hobolth and Stone for the
continuous case). Outputs run along the direction from position 0.

## Theory checks

The test suite asserts that every fitted matrix has rows summing to 1 within `1e-12`, that a
seeded `10^5`-step simulation reproduces `P` within three binomial sigma and the proportions
within three Kemeny and Snell sigma (`tg.checks.kemeny_snell_std`), that the independence test
matches `tg.checks.quasi_independence` and scipy, and that bridge marginals match
`tg.checks.markov_bridge_marginal`.

Worked pages: [embedded chains](../examples/01-markov-chains/01-embedded-chains/README.md) and
[tests and simulation](../examples/01-markov-chains/02-tests-and-simulation/README.md).
