import pickle

import numpy as np
import pytest
import transiograms as tg
from transiograms import _transiograms


def test_import_smoke():
    assert isinstance(tg.__version__, str)
    assert {"Categories", "Grid", "TransiogramsError", "InvalidInput", "ConvergenceError"} <= set(tg.__all__)
    assert "plot" not in tg.__all__ and "datasets" not in tg.__all__ and "checks" not in tg.__all__
    assert not any(name.startswith("_") for name in tg.__all__)


def test_error_hierarchy():
    assert issubclass(tg.InvalidInput, tg.TransiogramsError)
    assert issubclass(tg.InvalidInput, ValueError)
    assert issubclass(tg.ConvergenceError, RuntimeError)
    with pytest.raises(tg.InvalidInput, match="appears twice"):
        tg.Categories(["a", "a"])
    with pytest.raises(ValueError):
        tg.Grid((0, 0, 0), (1, 0, 1), (1, 1, 1))


def test_categories_encode_decode_round_trip():
    c = tg.Categories(["sand", "silt", "clay"], colors=["#e6c35c", "#a0a0a0", "#6b4f2a"])
    assert len(c) == 3 and c.names == ["sand", "silt", "clay"]
    codes = c.encode(["clay", None, "sand", np.nan])
    assert codes.dtype == np.int64
    assert codes.tolist() == [2, -1, 0, -1]
    assert c.decode(codes).tolist() == ["clay", None, "sand", None]
    assert c.decode(np.array([1.0, np.nan])).tolist() == ["silt", None]
    with pytest.raises(tg.InvalidInput, match="gravel"):
        c.encode(["gravel"])
    with pytest.raises(tg.InvalidInput, match="codes must be integers"):
        c.decode([0.5])
    with pytest.raises(tg.InvalidInput):
        c.decode([3])


def test_categories_from_labels_sorts_numerically():
    assert tg.Categories.from_labels([10, 2, None, 2.0]).names == ["2", "10"]
    assert tg.Categories.from_labels(["b", "a"]).names == ["a", "b"]


def test_categories_json_and_pickle():
    c = tg.Categories(["a", "b"], colors=["red", "blue"])
    assert tg.Categories.from_json(c.to_json()) == c
    assert pickle.loads(pickle.dumps(c)) == c
    assert "Categories" in repr(c)
    with pytest.raises(tg.InvalidInput, match="extra"):
        tg.Categories.from_json('{"names": ["a"], "extra": 1}')
    with pytest.raises(tg.InvalidInput):
        tg.Categories.from_json('{"names": ["a", "a"]}')


def test_grid_geometry_and_snap():
    g = tg.Grid((0.0, 0.0, 0.0), (2.0, 1.0, 0.5), (3, 2, 4))
    assert g.n_cells == 24 and g.shape == (4, 2, 3) and g.count == (3, 2, 4)
    xyz = g.coords
    assert xyz.shape == (24, 3)
    np.testing.assert_allclose(xyz[0], [1.0, 0.5, 0.25])
    np.testing.assert_allclose(xyz[1], [3.0, 0.5, 0.25])
    np.testing.assert_allclose(xyz[3], [1.0, 1.5, 0.25])
    values = np.arange(24).reshape(g.shape)
    assert values[1, 0, 2] == 2 + 3 * (0 + 2 * 1)
    cells, codes = g.snap(np.array([[1.0, 0.5, 0.25], [1.4, 0.5, 0.25], [9.0, 9.0, 9.0]]), [0, 1, 2])
    assert cells.dtype == np.uint64 and codes.dtype == np.int64
    assert cells.tolist() == [0] and codes.tolist() == [0]
    cells, _ = g.snap(np.array([[1.0, 0.5]]), [1])
    assert cells.tolist() == [0]
    mask = np.ones(24, bool)
    mask[0] = False
    assert g.snap(np.array([[1.0, 0.5, 0.25]]), [1], mask=mask)[0].size == 0


def test_grid_json_pickle_eq():
    g = tg.Grid((1.0, 2.0, 3.0), (1.0, 1.0, 1.0), (2, 2, 1))
    assert tg.Grid.from_json(g.to_json()) == g
    assert pickle.loads(pickle.dumps(g)) == g
    assert g != tg.Grid((1.0, 2.0, 3.0), (1.0, 1.0, 1.0), (2, 2, 2))
    assert repr(g).startswith("Grid(")


def test_sequence_args_intervals_and_row_order():
    data = {
        "HOLE": np.array(["b", "a", "b", "a"]),
        "FROM": np.array([1.0, 0.0, 0.0, 1.0]),
        "TO": np.array([2.0, 1.0, 1.0, 3.0]),
        "CODE": np.array([1, 0, 0, -1]),
    }
    runs = _transiograms._sequence_runs(
        "CODE", holes="HOLE", from_="FROM", to="TO", data=data, direction="down"
    )
    assert runs == [(0, [2, 0], [0, 1]), (1, [1], [0])]
    up = _transiograms._sequence_runs("CODE", holes="HOLE", from_="FROM", to="TO", data=data)
    assert up == [(0, [0, 2], [1, 0]), (1, [1], [0])]
    with pytest.raises(tg.InvalidInput, match="not both"):
        _transiograms._sequence_runs([0, 1], from_=[0, 1], to=[1, 2], depth=[0, 1])
    with pytest.raises(tg.InvalidInput, match="codes must be integers"):
        _transiograms._sequence_runs([0, 1.5])
    with pytest.raises(tg.InvalidInput, match="data"):
        _transiograms._sequence_runs("CODE")


def test_sequence_args_samples_infer_step_and_split():
    depth = np.array([10.0, 10.5, 11.0, 13.0, 13.5])
    runs = _transiograms._sequence_runs(np.array([0.0, 1.0, np.nan, 2.0, 2.0]), depth=depth, direction="down")
    assert runs == [(0, [0, 1], [0, 1]), (0, [3, 4], [2, 2])]
    regular = _transiograms._sequence_runs([0, 0, 1, 1], holes=[7, 7, 8, 8], direction="down", categories=3)
    assert regular == [(0, [0, 1], [0, 0]), (1, [2, 3], [1, 1])]
    with pytest.raises(tg.InvalidInput):
        _transiograms._sequence_runs([0, 3], categories=3)


def test_hmm_sequence_input_keeps_rows():
    x = np.arange(12.0).reshape(6, 2)
    seqs, n = _transiograms._sequences(x, holes=["a", "b", "a", "a", "b", "b"], direction="down")
    assert n == 6
    assert [s[3] for s in seqs] == [[0, 2, 3], [1, 4, 5]]
    assert seqs[0][:2] == (3, 2) and seqs[0][2] == [0.0, 1.0, 4.0, 5.0, 6.0, 7.0]
    up, _ = _transiograms._sequences([np.array([1.0, 2.0, 3.0]), np.array([4.0])])
    assert up[0] == (3, 1, [3.0, 2.0, 1.0], [2, 1, 0]) and up[1][3] == [3]
    with pytest.raises(tg.InvalidInput, match="channels"):
        _transiograms._sequences([np.zeros((2, 2)), np.zeros((2, 3))])


def test_submodules_import():
    assert tg.plot.__name__ == "transiograms.plot"
    assert tg.datasets.__name__ == "transiograms.datasets"
    assert tg.checks.__name__ == "transiograms.checks"
