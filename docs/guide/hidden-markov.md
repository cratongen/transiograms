# Hidden Markov models

A hidden Markov model treats geology as a hidden state sequence and each log or assay reading
as a draw from its state's emission distribution. Use it to segment logs into electrofacies or
lithologies when no one logged them, or to fill gaps between logged core.

## Building a model

```python
import numpy as np
import varvite as vv

logs = vv.datasets.electrofacies_logs()
x = np.column_stack([logs["GR"], logs["RHOB"]])
hmm = vv.HiddenMarkovModel(3, covariance_type="diag", n_init=5, seed=0, direction="down")
hmm.fit(x, holes=logs["HOLE_ID"])
states = hmm.predict(x, holes=logs["HOLE_ID"])
```

`distributions` is the number of states, or one emission object per state:
`vv.Gaussian(mean, covariance, covariance_type=)`, `vv.Categorical(probabilities)`,
`vv.Poisson(rate)` or `vv.GaussianMixture(weights, means, covariances)`. Give `start` and
`transition` as well and the model is ready to use without fitting. `frozen=True` keeps a
distribution fixed during fitting; `params` and `init_params` choose which parts EM fits and
which it initialises.

## Sequences

`sequences` is a list of arrays, one per hole with rows top-down, or one array with `holes=`.
Outputs come back the same way, in input row order. NaN marks a missing channel; the emission
marginalises it.

## Inference

| Method | Returns |
| --- | --- |
| `score`, `score_samples` | log-likelihood, total or per hole |
| `predict(algorithm=)` | states per row: "viterbi", "map" or "posterior_viterbi" |
| `predict_proba` | posterior state probabilities per row |
| `viterbi` | best path and its log-probability per hole |
| `sample_posterior(n=, seed=)` | `n` state paths drawn from the posterior |
| `sample(lengths, seed=)` | simulated states and observations |
| `aic`, `bic`, `icl` | information criteria |

`labels` (state per row, -1 unknown) and `allowed` (boolean `(rows, K)`) clamp rows to known
states, for example logged core. Fitting with labels is semi-supervised Baum-Welch.

## Theory checks

On a 4-state, 8-step sequence the suite compares Viterbi, the forward log-likelihood, the
posteriors and the path entropy with `vv.checks.brute_force_hmm`, an exhaustive search over
`4^8` paths, within `1e-12`. It also asserts that EM never lowers its objective, that fully
labelled data give the counting MLE in one iteration, and that emission densities match scipy.

Worked page: [electrofacies](../examples/02-hidden-markov/01-electrofacies/README.md).
