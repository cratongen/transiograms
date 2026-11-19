# Plots, datasets, checks

These submodules are not star-exported: write `vv.plot.transiogram(...)`.

## vv.plot

Every function takes `ax=None` (or `axes=None`), returns `(fig, ax)` or `(fig, axes)`, forwards
`**kwargs` to its main matplotlib call and imports matplotlib only when called
(`pip install 'varvite[plot]'`).

| Function | Draws |
| --- | --- |
| `transition_matrix(matrix, *, categories=None, annotate=True)` | heat map of a matrix or a `MarkovChain` |
| `sojourn(sojourn, code, *, exponential=True)` | Kaplan-Meier thickness survival and the exponential law |
| `strip_log(codes, *, depth=None, from_=None, to=None, posterior=None, categories=None)` | a categorical log, with posterior bands |
| `selection(table, *, criteria=("aic", "bic", "icl"))` | a `select_states` table |
| `transiogram(experimental=None, model=None, *, h=None, categories=None)` | the `k x k` grid of transiograms |
| `grid_slice(values, grid, *, axis="z", index=0, categories=None)` | one slice of a grid property |

## vv.datasets

Seeded synthetic data with known parameters; each takes `seed=0` and returns a dict.

| Function | Content |
| --- | --- |
| `varved_sequence()` | 12 holes of sand, silt and clay intervals: `HOLE_ID`, `FROM`, `TO`, `CODE`, `names`, `generator` |
| `electrofacies_logs()` | 20 wells x 300 samples from a 3-state Gaussian HMM: `HOLE_ID`, `DEPTH`, `GR`, `RHOB`, `STATE`, `model` |
| `layered_drillholes()` | 25 vertical holes through an MCRF field: `X`, `Y`, `Z`, `FROM`, `TO`, `HOLE_ID`, `CODE`, `field`, `model`, `grid` |

## vv.checks

Pure NumPy reference implementations the tests compare against.

| Function | Computes |
| --- | --- |
| `quasi_independence(counts)` | expected embedded counts by iterative proportional fitting |
| `kemeny_snell_std(P, n)` | standard deviation of chain proportions over `n` steps |
| `markov_bridge_marginal(P, a, b, n, t)` | `P(x_t = j | x_0 = a, x_n = b)` |
| `brute_force_hmm(start, transition, log_b)` | log-likelihood, best path and its log-probability, posteriors and path entropy over all paths |
| `brute_force_hsmm(start, transition, log_d, log_survivor, log_b)` | the same for a semi-Markov model |
| `mcp_brute_force(p, T_list, codes)` | the MCP conditional from the maximum-entropy joint table |
