import numpy as np
import pytest
import varvite as vv

P = np.array([0.5, 0.3, 0.2])
EMBEDDED = np.array([[0.0, 0.7, 0.3], [0.6, 0.0, 0.4], [0.5, 0.5, 0.0]])


def model():
    lengths = [[15.0, 10.0, 6.0], [9.0, 6.0, 4.0], [2.0, 1.5, 1.0]]
    return vv.TransitionModel.from_geology(P, lengths, embedded_x=EMBEDDED)


GRID2D = vv.Grid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (21, 21, 1))


def test_mcp_equals_maximum_entropy_brute_force():
    m = model()
    x0 = np.array([10.5, 10.5, 0.5])
    xyz = x0 + np.array([[3.0, 1.0, 0.0], [-2.0, 4.0, 0.0], [-3.0, -2.0, 0.0], [1.0, -5.0, 0.0]])
    codes = [0, 2, 1, 2]
    pred = vv.mcp_predict(m, GRID2D, xyz, codes, closeness_limit=1e-6)
    cell = int(np.flatnonzero(np.all(GRID2D.coords == x0, axis=1))[0])
    want = vv.checks.mcp_brute_force(P, [m.transition(x - x0) for x in xyz], codes)
    np.testing.assert_allclose(pred.probabilities[cell], want, atol=1e-10)


def test_two_neighbours_on_a_line_give_the_markov_bridge():
    m = model()
    x0 = np.array([10.5, 10.5, 0.5])
    a, b = x0 - [4.0, 0.0, 0.0], x0 + [3.0, 0.0, 0.0]
    pred = vv.mcp_predict(m, GRID2D, [a, b], [1, 2], neighbours=2, closeness_limit=1e-6)
    cell = int(np.flatnonzero(np.all(GRID2D.coords == x0, axis=1))[0])
    t1, t2, t12 = m.transition(x0 - a), m.transition(b - x0), m.transition(b - a)
    np.testing.assert_allclose(pred.probabilities[cell], t1[1] * t2[:, 2] / t12[1, 2], atol=1e-12)


def test_m6_probabilities_and_hard_cells():
    d = vv.datasets.layered_drillholes()
    grid, m = vv.Grid.from_json(d["grid"]), vv.TransitionModel.from_json(d["model"])
    xyz = np.column_stack([d["X"], d["Y"], d["Z"]])
    pred = vv.mcp_predict(m, grid, xyz, d["CODE"])
    p = pred.probabilities
    assert np.all(p >= 0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    cells, codes = grid.snap(xyz, d["CODE"])
    assert np.all(p[cells, codes] == 1.0)


def test_mcrf_reproduces_hard_data_and_is_seeded():
    d = vv.datasets.layered_drillholes()
    grid, m = vv.Grid.from_json(d["grid"]), vv.TransitionModel.from_json(d["model"])
    xyz = np.column_stack([d["X"], d["Y"], d["Z"]])
    r = vv.mcrf_simulate(m, grid, xyz, d["CODE"], n=3, seed=2)
    cells, codes = grid.snap(xyz, d["CODE"])
    assert np.all(r.realizations[:, cells] == codes)
    np.testing.assert_array_equal(r.corrected, 0.0)
    assert r.objective == []
    again = vv.mcrf_simulate(m, grid, xyz, d["CODE"], n=3, seed=2)
    np.testing.assert_array_equal(r.realizations, again.realizations)
    assert len(r) == 3


def test_errors_and_budgets():
    m = model()
    with pytest.raises(vv.InvalidInput):
        vv.mcp_predict(m, GRID2D, np.zeros((1, 3)), [0], neighbours="all")
    with pytest.raises(vv.InvalidInput, match="bytes"):
        vv.mcrf_simulate(m, GRID2D, np.zeros((0, 3)), [], n=10, memory=100)
    with pytest.raises(vv.InvalidInput):
        vv.mcrf_simulate(m, GRID2D, np.zeros((0, 3)), [], n=2**62)
