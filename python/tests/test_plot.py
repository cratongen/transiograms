import matplotlib
import numpy as np
import pytest
import varvite as vv

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def test_every_plot_returns_a_figure():
    d = vv.datasets.varved_sequence()
    c = vv.MarkovChain().fit(d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
    cats = vv.Categories(d["names"])
    fig, ax = vv.plot.transition_matrix(c, categories=cats)
    assert fig is ax.figure
    s = vv.sojourn(d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
    vv.plot.sojourn(s, 0)
    at = d["HOLE_ID"] == "DH01"
    vv.plot.strip_log(d["CODE"][at], from_=d["FROM"][at], to=d["TO"][at], categories=cats)
    probs = np.eye(3)[d["CODE"][at]]
    fig, axes = vv.plot.strip_log(d["CODE"][at], from_=d["FROM"][at], to=d["TO"][at], posterior=probs)
    assert len(axes) == 2
    vv.plot.selection(
        {"states": np.arange(1, 4), "aic": [3.0, 1.0, 2.0], "bic": [3.0, 2.0, 2.5], "icl": [4, 3, 3.5]}
    )
    e = vv.downhole_transiogram(
        d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"], lags=np.arange(0.2, 3, 0.2)
    )
    fig, axes = vv.plot.transiogram(e, vv.TransiogramModel.markov(c), categories=cats)
    assert axes.shape == (3, 3)
    l = vv.datasets.layered_drillholes()
    grid = vv.Grid.from_json(l["grid"])
    vv.plot.transiogram(model=vv.TransitionModel.from_json(l["model"]), h=np.linspace(0, 20, 50))
    vv.plot.grid_slice(l["field"], grid, axis="y", index=5)
    vv.plot.grid_slice(np.linspace(0, 1, grid.n_cells), grid)
