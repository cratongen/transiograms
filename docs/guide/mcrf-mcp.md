# MCP and MCRF

Markov-type categorical prediction (Li and Zhang) takes the nearest datum in each quadrant (2D)
or octant (3D) and treats them as conditionally independent given the category at the cell:

`P(Z(x0) = k | data) ∝ p_k prod_g t_{k, c_g}(x_g - x0)`.

`vv.mcp_predict` returns these probabilities at every cell; `vv.mcrf_simulate` simulates along a
seeded random path, drawing each cell from the same conditional over the hard data and the cells
already drawn. Both take `neighbours="sectors"` or a number of nearest neighbours.

```python
pred = vv.mcp_predict(model, grid, xyz, codes)
real = vv.mcrf_simulate(model, grid, xyz, codes, n=10, seed=0)
```

When every candidate category has zero likelihood the method drops the least close neighbour
and retries, instead of flooring probabilities at an epsilon.

## Theory checks

The suite checks that the MCP conditional equals the maximum-entropy joint table that honours
each pairwise transiogram (`vv.checks.mcp_brute_force`, 243 cells for three categories and four
neighbours) within `1e-10`, that two neighbours on a line give the Markov bridge
`t_ak(h1) t_kb(h2) / t_ab(h1 + h2)`, that probabilities sum to 1, and that hard cells get
probability exactly 1 and appear in every realization.

MCRF reproduces target proportions within 0.02 on the Rust M6 grid. On the small grid of the
example pages the ensemble drifts about 0.04 from the target; T-PROGS, whose quenching includes a
proportion term, stays within 0.002 there.

Worked page: [MCP and MCRF](../examples/06-mcrf-mcp/01-mcp-and-mcrf/README.md).
