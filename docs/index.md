# varvite

varvite fits and simulates Markov models of geological categories. Along drillholes it fits
Markov chains, hidden Markov models and hidden semi-Markov models; on grids it builds transition
models from transiograms and simulates with T-PROGS or Markov chain random fields.

```python
import numpy as np
import varvite as vv

d = vv.datasets.layered_drillholes()
exp = vv.downhole_transiogram("CODE", holes="HOLE_ID", from_="FROM", to="TO", lags=np.arange(1.0, 9.0), data=d)
vertical = vv.TransiogramModel.fit_markov(exp)
model = vv.TransitionModel(vertical.proportions, 0.1 * vertical.rates, 0.1 * vertical.rates, vertical.rates)
grid = vv.Grid.from_json(d["grid"])
real = vv.tprogs_simulate(model, grid, np.column_stack([d["X"], d["Y"], d["Z"]]), d["CODE"], n=10, seed=0)
```

The core is Rust; NumPy is the only runtime dependency. Every stochastic routine takes a `seed`
and returns the same bits on any number of threads.

- The [guide](guide/markov-chains.md) explains each method and its theory checks.
- The [examples](examples/01-markov-chains/01-embedded-chains/README.md) run end to end on
  seeded synthetic data and draw their figures.
- The [API](api/core.md) lists every class and function; the [design contract](design/spec.md)
  fixes names, shapes and tolerances.
