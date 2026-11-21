# transiograms

transiograms fits and simulates Markov models of geological categories such as facies and
lithology. Along drillholes and logs it fits Markov chains, hidden Markov models and hidden
semi-Markov models. In 3D it measures and models transiograms and simulates categorical grids with
T-PROGS and Markov chain random fields (MCRF). The core is Rust; the Python package needs only
NumPy.

The library is named after the transiogram: the probability of finding category `j` at lag `h`
from a point in category `i`, as a function of `h`.

Documentation: <https://cratongen.github.io/transiograms/>. Source and issues:
<https://github.com/cratongen/transiograms>.

## Install

```sh
pip install transiograms
```

transiograms supports CPython 3.11 and newer through one `abi3` wheel per platform. NumPy is the
only runtime dependency; `pip install "transiograms[plot]"` adds matplotlib for `tg.plot`. A
platform without a wheel builds from the sdist, which needs Rust 1.97 or newer. See the
[install page](https://cratongen.github.io/transiograms/install/) for details.

## Quick start

```python
import numpy as np
import transiograms as tg

beds = tg.datasets.varved_sequence(seed=0)
chain = tg.MarkovChain("embedded").fit("CODE", holes="HOLE_ID", from_="FROM", to="TO", data=beds)
print(chain.transition_matrix.round(2))
print("mean thickness (m):", chain.mean_lengths.round(2))

holes = tg.datasets.layered_drillholes(seed=0)
lags = np.arange(1.0, 9.0)
exp = tg.downhole_transiogram("CODE", holes="HOLE_ID", from_="FROM", to="TO", lags=lags, data=holes)
p = np.bincount(holes["CODE"], minlength=3) / len(holes["CODE"])
vertical = tg.TransiogramModel.fit_markov(exp, proportions=p)
print("vertical mean lengths (m):", (-1 / np.diag(vertical.rates)).round(2))
print("largest residual:", np.nanmax(np.abs(vertical.residuals(exp))).round(3))
```

```text
[[0.   0.83 0.17]
 [0.12 0.   0.88]
 [0.84 0.16 0.  ]]
mean thickness (m): [1.2  0.83 0.46]
vertical mean lengths (m): [9.43 7.4  5.44]
largest residual: 0.136
```

The first part fits an embedded Markov chain to twelve seeded holes of sand, silt and clay beds.
The second measures vertical transiograms down 25 seeded drillholes, fits a Markov model with its
sills at the sample proportions, and reports the misfit. The
[examples](https://cratongen.github.io/transiograms/examples/01-markov-chains/01-embedded-chains/)
go on to hidden Markov segmentation, 3D transition models and grid simulation.

## What it does

- **Markov chains**: embedded and fixed-step transition matrices, continuous-lag generators with
  `P(h) = exp(Q h)`, tests of the Markov property, order and homogeneity, sojourn statistics, and
  seeded 1D simulation.
- **Hidden Markov and semi-Markov models**: Gaussian, categorical, Poisson and Gaussian-mixture
  emissions; forward-backward, Viterbi and Baum-Welch; explicit duration laws (geometric, shifted
  Poisson, negative binomial, empirical); AIC, BIC, ICL and cross-validated state-count selection.
- **Transiograms**: experimental transiograms from drillholes, points and grids; Markov and
  entrywise models whose rows sum to one at every lag; 3D transition models from rates or mean
  lengths.
- **Grid simulation**: transition-probability cokriging and T-PROGS simulation with quenching;
  Markov-type categorical prediction (MCP) and MCRF simulation, conditioned to drillholes.

Every stochastic routine takes a `seed` and gives the same bits on any number of threads. Grid
work is sized from the available memory and checked before it allocates.

## Fit quality and known limits

`tg.fit_quality` compares T-PROGS or MCRF realizations with their model (proportions, and grid
transiograms per lag with standard errors), and `TransiogramModel.residuals` gives the misfit of a
1D fit. Both simulators keep a measured gap to their model, documented as a limit of the method:
MCRF without hard data drifts 0.038 from the target proportions and misses transiograms by up to
0.19; T-PROGS with the default quench misses mid-lag transiograms by up to 0.026. Seeded tests pin
these numbers; the
[T-PROGS](https://cratongen.github.io/transiograms/guide/tp-simulation/) and
[MCRF](https://cratongen.github.io/transiograms/guide/mcrf-mcp/) guides explain them.

## Status

Version 0.1.0 is an alpha release: the API can change between minor versions. Performance is not
measured yet. The
[status page](https://cratongen.github.io/transiograms/status/) lists each milestone, its
acceptance test and what is still missing.

## Contributing

Issues are open. A code contribution starts with an issue; see
[CONTRIBUTING.md](https://github.com/cratongen/transiograms/blob/main/CONTRIBUTING.md).

## License

Licensed under either of the [Apache License, Version 2.0](https://github.com/cratongen/transiograms/blob/main/LICENSE-APACHE) or the [MIT licence](https://github.com/cratongen/transiograms/blob/main/LICENSE-MIT), at your option.
