import numpy as np
import varvite as vv


def test_m7_drillhole_arrays_in_categorical_grid_out():
    d = vv.datasets.layered_drillholes(seed=0)
    grid = vv.Grid.from_json(d["grid"])
    p = np.bincount(d["CODE"], minlength=3) / len(d["CODE"])
    e = vv.downhole_transiogram(
        "CODE", holes="HOLE_ID", from_="FROM", to="TO", lags=np.arange(1.0, 9.0), data=d
    )
    vertical = vv.TransiogramModel.fit_markov(e, proportions=p)
    model = vv.TransitionModel(p, 0.1 * vertical.rates, 0.1 * vertical.rates, vertical.rates)
    xyz = np.column_stack([d["X"], d["Y"], d["Z"]])
    r = vv.tprogs_simulate(model, grid, xyz, d["CODE"], n=5, seed=0)
    cube = r.realizations.reshape((5, *grid.shape))
    assert cube.shape == (5, 30, 32, 32) and cube.dtype == np.int64
    cells, codes = grid.snap(xyz, d["CODE"])
    assert np.all(r.realizations[:, cells] == codes)
    assert np.abs(r.proportions.mean(axis=0) - p).max() <= 0.02
    np.testing.assert_allclose(r.probabilities.sum(axis=1), 1.0, atol=1e-12)
