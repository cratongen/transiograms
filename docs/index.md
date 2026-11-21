---
title: transiograms
---

# transiograms

<div class="tg-hero" markdown>

![transiograms](brand/wordmark-black.svg#only-light){ width="372" }
![transiograms](brand/wordmark-white.svg#only-dark){ width="372" }

<p class="tg-pitch">Markov models of geological categories: fit facies and lithology sequences down drillholes, model transiograms, and simulate categories on 3D grids.</p>

[Install](install-and-release.md){ .md-button .md-button--primary }
[Guide](guide/markov-chains.md){ .md-button }
[Examples](examples/01-markov-chains/01-embedded-chains/README.md){ .md-button }

</div>

transiograms is for geologists, geostatisticians and resource modellers who work with logged
categories (facies, lithology, alteration, ore type) and want their spatial structure as numbers
they can test. Along drillholes and logs it fits Markov chains, hidden Markov models and hidden
semi-Markov models. On grids it builds 3D transition models from transiograms and simulates them
with T-PROGS or Markov chain random fields (MCRF), or predicts them with MCP.

The core is Rust and the Python package is `import transiograms as tg`. NumPy is the only runtime
dependency. Every stochastic routine takes a `seed` and returns the same bits on any number of
threads.

## Quick start

```bash
pip install transiograms
```

Fit an embedded Markov chain to twelve seeded drillholes of sand, silt and clay:

```python
import transiograms as tg

d = tg.datasets.varved_sequence(seed=0)
print(d["names"])
chain = tg.MarkovChain().fit("CODE", holes="HOLE_ID", from_="FROM", to="TO", data=d)
print(chain.transition_matrix.round(2))
print(chain.mean_lengths.round(2))
print(chain.independence_test())
```

```text
['sand', 'silt', 'clay']
[[0.   0.83 0.17]
 [0.12 0.   0.88]
 [0.84 0.16 0.  ]]
[1.2  0.83 0.46]
TestResult(name="independence", method="g", statistic=234.913759, df=1, p_value=5.0557e-53)
```

Rows are the bed below, columns the bed above: sand passes up into silt 83% of the time and silt
into clay 88% of the time, a fining-upward cycle. The independence test rejects random stacking.

## Where to go next

| Page | What it covers |
| --- | --- |
| [Install and release](install-and-release.md) | pip from PyPI, a build from a clone or the sdist, wheels and Python versions, the release procedure |
| [Guide](guide/markov-chains.md) | Each method, its arguments and the theory checks the tests run |
| [Examples](examples/01-markov-chains/01-embedded-chains/README.md) | Worked pages on seeded data, with figures and downloadable scripts |
| [API reference](api/core.md) | Every class and function, from the docstrings |
| [Design notes](status.md) | Status and known limits of each method, the API contract and the algorithms |
| [Changelog](changelog.md) | What changed in each release |
| [Contributing](contributing.md) | Issues first, then pull requests; the commands every change runs |
