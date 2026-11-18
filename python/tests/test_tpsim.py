import numpy as np
import pytest
import varvite as vv

P = [0.5, 0.3, 0.2]


def model():
    return vv.TransitionModel.from_geology(P, [[12.0, 8.0, 6.0], [12.0, 8.0, 6.0], [3.0, 2.0, 1.5]])


GRID = vv.Grid((0.0, 0.0, 0.0), (1.0, 1.0, 0.5), (16, 16, 12))


def hard_data():
    xyz = np.array(
        [[4.5, 4.5, z] for z in 0.25 + 0.5 * np.arange(12)] + [[11.5, 9.5, 2.75], [2.5, 13.5, 4.25]]
    )
    codes = np.r_[np.arange(12) % 3, 1, 2]
    return xyz, codes


def test_single_datum_gives_one_row_of_the_transition_matrix():
    m = model()
    x1, c = np.array([[3.5, 6.5, 2.25]]), 2
    pred = vv.tp_cokriging(m, GRID, x1, [c])
    for cell in (0, 1000, GRID.n_cells - 1):
        x0 = GRID.coords[cell]
        np.testing.assert_allclose(pred.probabilities[cell], m.transition(x0 - x1[0])[c], atol=1e-12)


def test_probabilities_sum_to_one_and_hard_cells_are_certain():
    xyz, codes = hard_data()
    pred = vv.tp_cokriging(model(), GRID, xyz, codes)
    np.testing.assert_allclose(pred.probabilities.sum(axis=1), 1.0, atol=1e-12)
    cells, hard = GRID.snap(xyz, codes)
    np.testing.assert_array_equal(pred.most_likely[cells], hard)
    assert np.all((pred.entropy >= 0) & (pred.entropy <= 1 + 1e-12))
    assert 0.0 <= pred.corrected <= 1.0


def test_simulation_honours_hard_data_and_quenching_never_increases_the_objective():
    xyz, codes = hard_data()
    r = vv.tprogs_simulate(model(), GRID, xyz, codes, n=3, seed=1)
    cells, hard = GRID.snap(xyz, codes)
    assert r.realizations.shape == (3, GRID.n_cells)
    assert np.all(r.realizations[:, cells] == hard)
    for o in r.objective:
        assert np.all(np.diff(o) <= 1e-12 * np.abs(o[:-1]))
    np.testing.assert_allclose(r.proportions.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(r.probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_simulation_is_seeded_and_mask_marks_inactive_cells():
    xyz, codes = hard_data()
    mask = GRID.coords[:, 0] < 12.0
    a = vv.tprogs_simulate(model(), GRID, xyz, codes, n=2, seed=4, mask=mask, quench=False)
    b = vv.tprogs_simulate(model(), GRID, xyz, codes, n=2, seed=4, mask=mask, quench=False)
    np.testing.assert_array_equal(a.realizations, b.realizations)
    assert np.all(a.realizations[:, ~mask] == -1) and np.all(a.realizations[:, mask] >= 0)
    assert a.objective == [] or all(len(o) <= 1 for o in a.objective)


def test_budgets_and_absurd_requests_raise_before_allocating():
    xyz, codes = hard_data()
    with pytest.raises(vv.InvalidInput, match="bytes"):
        vv.tprogs_simulate(model(), GRID, xyz, codes, n=4, memory=1000)
    huge = vv.Grid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (100_000, 100_000, 100_000))
    with pytest.raises(vv.InvalidInput):
        vv.tprogs_simulate(model(), huge, xyz, codes, n=2)
    with pytest.raises(vv.InvalidInput):
        vv.tp_cokriging(model(), GRID, xyz, codes, max_data=7)
    with pytest.raises(vv.InvalidInput, match="categories"):
        vv.tp_cokriging(model(), GRID, xyz, codes + 5)
