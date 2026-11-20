# Semi-Markov models and selection

An HMM's state durations are geometric: a one-step bed is always the most likely. When beds have
a typical thickness, `tg.HiddenSemiMarkovModel` gives each state an explicit duration law:

| Law | Parameters | Mean |
| --- | --- | --- |
| `tg.Geometric(p)` | `0 < p <= 1` | `1 / p` |
| `tg.ShiftedPoisson(rate)` | `rate >= 0` | `1 + rate` |
| `tg.NegativeBinomial(r, p)` | `r > 0`, `0 < p <= 1` | `1 + r (1 - p) / p` |
| `tg.EmpiricalDuration(pmf)` | weights on `1..len(pmf)` | from the weights |

```python
import numpy as np
import transiograms as tg

d = tg.datasets.electrofacies_logs(seed=0)
holes = [d["HOLE_ID"] == h for h in np.unique(d["HOLE_ID"])]
sequences = [np.column_stack([d["GR"][m], d["RHOB"][m]]) for m in holes]
hsmm = tg.HiddenSemiMarkovModel(3, duration="negative_binomial", max_duration=150, n_init=3, seed=0)
hsmm.fit(sequences)
print("mean thickness (samples):", [round(law.mean(150), 1) for law in hsmm.durations])
print("truncated mass:", np.array2string(hsmm.truncated_mass, precision=1))
```

```text
mean thickness (samples): [21.8, 20.4, 16.3]
truncated mass: [3.4e-05 4.9e-04 1.5e-04]
```

The transition matrix has a zero diagonal: a state ends when its duration runs out. The first and
last segment of each hole are right-censored. `max_duration=None` means the longest sequence, so
nothing is truncated, but the forward-backward pass then costs time in proportion to
`T * K * D` with `D = T`, that is quadratic in the length `T` of the longest sequence (memory
stays `O(T * K)` and is checked before allocating). On long logs, set `max_duration` to a few
times the thickest bed you expect, in samples, and check `truncated_mass`, which says how much
probability the cap cut off: on these 300-sample logs a cap of 150 leaves up to `5e-4`, and a cap
of 40 leaves 0.2 to 1.0, a sign that the cap is too short for the beds. The runtime of the default on long logs is not measured.

The methods and arguments match `HiddenMarkovModel`, with `params` letters `s`, `t`, `e` and `d`
(durations).

`HiddenSemiMarkovModel.from_hmm(hmm, max_duration=)` builds the equivalent geometric HSMM; the
suite checks that it reproduces the HMM's log-likelihood within `1e-10` and its Viterbi path.

## Choosing the number of states

`tg.select_states(sequences, states=range(1, 7), model="hmm" or "hsmm")` fits each K and returns
AIC, BIC, ICL and cross-validated log-likelihood. `cv_folds=None` leaves one hole out, `0` skips
cross-validation, and `k` uses `k` seeded folds. Samples along a log are autocorrelated, so BIC
can pick too many states on dense logs; the held-out likelihood is the honest check.
`tg.plot.selection(table)` draws the table.

On `tg.datasets.electrofacies_logs(seed=0)` with `n_init=5` the BIC minimum is K = 3, the number
of states that generated the data; the suite asserts it.

Worked pages: [durations and selection](../examples/03-semi-markov-and-selection/01-durations-and-selection/README.md)
and [duration laws](../examples/03-semi-markov-and-selection/02-duration-laws/README.md).
