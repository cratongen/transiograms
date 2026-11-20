import pickle

import numpy as np
import pytest
import transiograms as tg

P = np.array([0.5, 0.3, 0.2])
EMBEDDED = np.array([[0.0, 0.7, 0.3], [0.6, 0.0, 0.4], [0.5, 0.5, 0.0]])


def model():
    lengths = [[15.0, 10.0, 6.0], [9.0, 6.0, 4.0], [2.0, 1.5, 1.0]]
    return tg.TransitionModel.from_geology(P, lengths, embedded_x=EMBEDDED)


GRID2D = tg.Grid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (21, 21, 1))


def test_mcp_equals_maximum_entropy_brute_force():
    m = model()
    x0 = np.array([10.5, 10.5, 0.5])
    xyz = x0 + np.array([[3.0, 1.0, 0.0], [-2.0, 4.0, 0.0], [-3.0, -2.0, 0.0], [1.0, -5.0, 0.0]])
    codes = [0, 2, 1, 2]
    pred = tg.mcp_predict(m, GRID2D, xyz, codes, closeness_limit=1e-6)
    cell = int(np.flatnonzero(np.all(GRID2D.coords == x0, axis=1))[0])
    want = tg.checks.mcp_brute_force(P, [m.transition(x - x0) for x in xyz], codes)
    np.testing.assert_allclose(pred.probabilities[cell], want, atol=1e-10)


def test_two_neighbours_on_a_line_give_the_markov_bridge():
    m = model()
    x0 = np.array([10.5, 10.5, 0.5])
    a, b = x0 - [4.0, 0.0, 0.0], x0 + [3.0, 0.0, 0.0]
    pred = tg.mcp_predict(m, GRID2D, [a, b], [1, 2], neighbours=2, closeness_limit=1e-6)
    cell = int(np.flatnonzero(np.all(GRID2D.coords == x0, axis=1))[0])
    t1, t2, t12 = m.transition(x0 - a), m.transition(b - x0), m.transition(b - a)
    np.testing.assert_allclose(pred.probabilities[cell], t1[1] * t2[:, 2] / t12[1, 2], atol=1e-12)


def test_m6_probabilities_and_hard_cells():
    d = tg.datasets.layered_drillholes()
    grid, m = tg.Grid.from_json(d["grid"]), tg.TransitionModel.from_json(d["model"])
    xyz = np.column_stack([d["X"], d["Y"], d["Z"]])
    pred = tg.mcp_predict(m, grid, xyz, d["CODE"])
    p = pred.probabilities
    assert np.all(p >= 0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    cells, codes = grid.snap(xyz, d["CODE"])
    assert np.all(p[cells, codes] == 1.0)


def test_mcrf_reproduces_hard_data_and_is_seeded():
    d = tg.datasets.layered_drillholes()
    grid, m = tg.Grid.from_json(d["grid"]), tg.TransitionModel.from_json(d["model"])
    xyz = np.column_stack([d["X"], d["Y"], d["Z"]])
    r = tg.mcrf_simulate(m, grid, xyz, d["CODE"], n=3, seed=2)
    cells, codes = grid.snap(xyz, d["CODE"])
    assert np.all(r.realizations[:, cells] == codes)
    np.testing.assert_array_equal(r.corrected, 0.0)
    assert r.objective == []
    again = tg.mcrf_simulate(m, grid, xyz, d["CODE"], n=3, seed=2)
    np.testing.assert_array_equal(r.realizations, again.realizations)
    assert len(r) == 3


def test_errors_and_budgets():
    m = model()
    with pytest.raises(tg.InvalidInput):
        tg.mcp_predict(m, GRID2D, np.zeros((1, 3)), [0], neighbours="all")
    with pytest.raises(tg.InvalidInput, match="bytes"):
        tg.mcrf_simulate(m, GRID2D, np.zeros((0, 3)), [], n=10, memory=100)
    with pytest.raises(tg.InvalidInput):
        tg.mcrf_simulate(m, GRID2D, np.zeros((0, 3)), [], n=2**62)


def m5_model():
    pi_z = [[0.0, 0.6, 0.4], [0.7, 0.0, 0.3], [0.2105263158, 0.7894736842, 0.0]]
    lengths = [[12.0, 8.0, 6.0], [12.0, 8.0, 6.0], [2.0, 1.0, 0.8]]
    return tg.TransitionModel.from_geology(P, lengths, embedded_z=pi_z, background=2)


def test_fit_quality_matches_numpy_counts():
    m = m5_model()
    grid = tg.Grid((0.0, 0.0, 0.0), (3.0, 3.0, 0.5), (12, 10, 8))
    real = tg.mcrf_simulate(m, grid, np.empty((0, 3)), [], n=3, seed=4)
    q = tg.fit_quality(m, grid, real, directions=[(1, 0, 0), (0, 0, -1)], n_lags=2)
    np.testing.assert_allclose(q.proportions, real.proportions, atol=1e-15)
    np.testing.assert_array_equal(q.offsets, [[1, 0, 0], [2, 0, 0], [0, 0, -1], [0, 0, -2]])
    per = np.stack(
        [
            tg.grid_transiogram(r, grid, offset=(0, 0, -1), n_lags=2, categories=3).probabilities
            for r in real.realizations
        ]
    )
    np.testing.assert_allclose(q.mean[2:], per.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(q.standard_error[2:], per.std(axis=0, ddof=1) / np.sqrt(3), atol=1e-14)
    np.testing.assert_allclose(q.model[3], m.transition([0.0, 0.0, -1.0]), atol=1e-14)
    assert q.transiogram_error == pytest.approx(np.nanmax(np.abs(q.mean - q.model)))
    assert q.proportion_error == pytest.approx(np.max(np.abs(q.mean_proportions - P)))
    back = pickle.loads(pickle.dumps(q))
    assert back.to_json() == q.to_json()


def test_fit_quality_rejects_a_grid_of_another_size():
    m = m5_model()
    grid = tg.Grid((0.0, 0.0, 0.0), (3.0, 3.0, 0.5), (6, 6, 4))
    real = tg.mcrf_simulate(m, grid, np.empty((0, 3)), [], n=2, seed=0)
    other = tg.Grid((0.0, 0.0, 0.0), (3.0, 3.0, 0.5), (6, 6, 5))
    with pytest.raises(tg.InvalidInput):
        tg.fit_quality(m, other, real)
    with pytest.raises(tg.InvalidInput):
        tg.fit_quality(m, grid, real, n_lags=0)


def test_unconditional_mcrf_gap_is_reported():
    """The method limit documented in docs/guide/mcrf-mcp.md, same seed as the Rust test."""
    m = m5_model()
    grid = tg.Grid((0.0, 0.0, 0.0), (3.0, 3.0, 0.5), (48, 48, 32))
    real = tg.mcrf_simulate(m, grid, np.empty((0, 3)), [], n=10, seed=1)
    q = tg.fit_quality(m, grid, real, directions=[(1, 0, 0), (0, 0, 1), (0, 0, -1)])
    assert 0.03 < q.proportion_error < 0.05
    assert 0.17 < q.transiogram_error < 0.21
    assert q.max_standard_errors > 20
