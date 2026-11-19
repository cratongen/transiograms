import numpy as np
import pytest
import varvite as vv


@pytest.fixture(scope="module")
def pipeline():
    source = vv.datasets.layered_drillholes(seed=0)
    rng = np.random.default_rng(7)
    composites = {k: source[k] for k in ("HOLE_ID", "FROM", "TO", "X", "Y", "Z")}
    composites["GR"] = np.array([30.0, 75.0, 120.0])[source["CODE"]] + rng.normal(
        0.0, 12.0, len(source["CODE"])
    )
    hmm = vv.HiddenMarkovModel(3, n_init=5, seed=0, direction="down", ordering="mean")
    hmm.fit("GR", holes="HOLE_ID", data=composites)
    composites["FACIES"] = hmm.predict("GR", holes="HOLE_ID", data=composites)
    p = np.bincount(composites["FACIES"], minlength=3) / len(composites["FACIES"])
    e = vv.downhole_transiogram(
        "FACIES", holes="HOLE_ID", from_="FROM", to="TO", lags=np.arange(1.0, 9.0), data=composites
    )
    vertical = vv.TransiogramModel.fit_markov(e, proportions=p)
    model = vv.TransitionModel(p, 0.1 * vertical.rates, 0.1 * vertical.rates, vertical.rates)
    grid = vv.Grid.from_json(source["grid"])
    xyz = np.column_stack([composites["X"], composites["Y"], composites["Z"]])
    return source, composites, p, model, grid, xyz


def test_m7_hmm_segmentation_recovers_the_logged_facies(pipeline):
    source, composites, p, *_ = pipeline
    assert composites["FACIES"].dtype == np.int64
    assert np.mean(composites["FACIES"] == source["CODE"]) >= 0.97
    truth = np.bincount(source["CODE"], minlength=3) / len(source["CODE"])
    assert np.abs(p - truth).max() <= 0.02


@pytest.mark.parametrize("simulate", [vv.tprogs_simulate, vv.mcrf_simulate], ids=["tprogs", "mcrf"])
def test_m7_drillhole_arrays_in_categorical_grid_out(pipeline, simulate):
    _, composites, p, model, grid, xyz = pipeline
    n = 20
    r = simulate(model, grid, xyz, "FACIES", n=n, seed=0, data=composites)
    cube = r.realizations.reshape((n, *grid.shape))
    assert cube.shape == (n, 30, 32, 32) and cube.dtype == np.int64
    cells, codes = grid.snap(xyz, composites["FACIES"])
    assert len(cells) == len(composites["FACIES"])
    assert np.all(r.realizations[:, cells] == codes)
    np.testing.assert_allclose(r.probabilities.sum(axis=1), 1.0, atol=1e-12)
    if simulate is vv.tprogs_simulate:
        three_sigma = 3 * r.proportions.std(axis=0, ddof=1).max() / np.sqrt(n)
        assert three_sigma < 0.02
        assert np.abs(r.proportions.mean(axis=0) - p).max() <= 0.02
