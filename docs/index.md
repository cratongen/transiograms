# varvite

varvite fits and simulates Markov models of geological categories: Markov chains, hidden Markov
and hidden semi-Markov models along drillholes, and transiograms, transition-probability
simulation and Markov chain random fields on grids.

```python
import varvite as vv

grid = vv.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (50, 40, 30))
cells, codes = grid.snap(coords, codes)
```

Only the core types exist so far. The [design contract](design/spec.md) lists what each area will
provide.
