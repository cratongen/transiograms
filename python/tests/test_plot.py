import matplotlib
import numpy as np
import pytest
import transiograms as tg

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def test_every_plot_returns_a_figure():
    d = tg.datasets.varved_sequence()
    c = tg.MarkovChain().fit(d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
    cats = tg.Categories(d["names"])
    fig, ax = tg.plot.transition_matrix(c, categories=cats)
    assert fig is ax.figure
    s = tg.sojourn(d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
    tg.plot.sojourn(s, 0)
    at = d["HOLE_ID"] == "DH01"
    tg.plot.strip_log(d["CODE"][at], from_=d["FROM"][at], to=d["TO"][at], categories=cats)
    probs = np.eye(3)[d["CODE"][at]]
    fig, axes = tg.plot.strip_log(d["CODE"][at], from_=d["FROM"][at], to=d["TO"][at], posterior=probs)
    assert len(axes) == 2
    tg.plot.selection(
        {"states": np.arange(1, 4), "aic": [3.0, 1.0, 2.0], "bic": [3.0, 2.0, 2.5], "icl": [4, 3, 3.5]}
    )
    e = tg.downhole_transiogram(
        d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"], lags=np.arange(0.2, 3, 0.2)
    )
    fig, axes = tg.plot.transiogram(e, tg.TransiogramModel.markov(c), categories=cats)
    assert axes.shape == (3, 3)
    l = tg.datasets.layered_drillholes()
    grid = tg.Grid.from_json(l["grid"])
    tg.plot.transiogram(model=tg.TransitionModel.from_json(l["model"]), h=np.linspace(0, 20, 50))
    tg.plot.grid_slice(l["field"], grid, axis="y", index=5)
    tg.plot.grid_slice(np.linspace(0, 1, grid.n_cells), grid)


def test_categories_of_the_wrong_length_name_the_mismatch():
    cats = tg.Categories(["a", "b"])
    with pytest.raises(tg.InvalidInput, match="categories has 2 names but the data needs 3"):
        tg.plot.transition_matrix(np.eye(3), categories=cats)
    with pytest.raises(tg.InvalidInput, match="needs at least 3"):
        tg.plot.strip_log(np.array([0, 1, 2]), categories=cats)
    tg.plot.strip_log(np.array([0, 0, 1]), categories=tg.Categories(["a", "b", "c"]))
