# Semi-Markov models and selection

An HMM's state durations are geometric: a one-step bed is always the most likely. When beds have
a typical thickness, `vv.HiddenSemiMarkovModel` gives each state an explicit duration law:

| Law | Parameters | Mean |
| --- | --- | --- |
| `vv.Geometric(p)` | `0 < p <= 1` | `1 / p` |
| `vv.ShiftedPoisson(rate)` | `rate >= 0` | `1 + rate` |
| `vv.NegativeBinomial(r, p)` | `r > 0`, `0 < p <= 1` | `1 + r (1 - p) / p` |
| `vv.EmpiricalDuration(pmf)` | weights on `1..len(pmf)` | from the weights |

```python
import varvite as vv

hsmm = vv.HiddenSemiMarkovModel(3, duration="negative_binomial", max_duration=40, n_init=3)
hsmm.fit(sequences)
hsmm.durations, hsmm.truncated_mass
```

The transition matrix has a zero diagonal: a state ends when its duration runs out. The first and
last segment of each hole are right-censored. `max_duration=None` means the longest sequence, so
nothing is truncated. The methods and arguments match `HiddenMarkovModel`, with `params` letters
`s`, `t`, `e` and `d` (durations).

`HiddenSemiMarkovModel.from_hmm(hmm, max_duration=)` builds the equivalent geometric HSMM; the
suite checks that it reproduces the HMM's log-likelihood within `1e-10` and its Viterbi path.

## Choosing the number of states

`vv.select_states(sequences, states=range(1, 7), model="hmm" or "hsmm")` fits each K and returns
AIC, BIC, ICL and cross-validated log-likelihood. `cv_folds=None` leaves one hole out, `0` skips
cross-validation, and `k` uses `k` seeded folds. Samples along a log are autocorrelated, so BIC
can pick too many states on dense logs; the held-out likelihood is the honest check.
`vv.plot.selection(table)` draws the table.

On `vv.datasets.electrofacies_logs(seed=0)` with `n_init=5` the BIC minimum is K = 3, the number
of states that generated the data; the suite asserts it.

Worked page: [durations and selection](../examples/03-semi-markov-and-selection/01-durations-and-selection/README.md).
