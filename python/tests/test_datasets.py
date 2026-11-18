import numpy as np
import pytest
import varvite as vv


@pytest.mark.parametrize(
    "name, keys",
    [
        ("varved_sequence", {"HOLE_ID", "FROM", "TO", "CODE", "generator"}),
        ("electrofacies_logs", {"HOLE_ID", "DEPTH", "GR", "RHOB", "STATE"}),
        ("layered_drillholes", {"X", "Y", "Z", "HOLE_ID", "CODE", "model", "grid"}),
    ],
)
def test_datasets_are_seeded_and_documented(name, keys):
    f = getattr(vv.datasets, name)
    a, b, c = f(seed=0), f(seed=0), f(seed=1)
    assert keys <= set(a)
    for k in keys:
        if isinstance(a[k], np.ndarray):
            np.testing.assert_array_equal(a[k], b[k])
    assert any(isinstance(a[k], np.ndarray) and not np.array_equal(a[k], c[k]) for k in keys)


def test_intervals_are_top_down_and_contiguous():
    d = vv.datasets.varved_sequence()
    for h in np.unique(d["HOLE_ID"]):
        at = d["HOLE_ID"] == h
        f, t = d["FROM"][at], d["TO"][at]
        assert f[0] == pytest.approx(0.0, abs=1e-9) and t[-1] == pytest.approx(30.0)
        np.testing.assert_allclose(f[1:], t[:-1])
    e = vv.datasets.electrofacies_logs()
    assert len(e["GR"]) == 6000 and set(np.unique(e["STATE"])) == {0, 1, 2}


def test_layered_holes_sample_the_field():
    d = vv.datasets.layered_drillholes()
    grid = vv.Grid.from_json(d["grid"])
    cells, codes = grid.snap(np.column_stack([d["X"], d["Y"], d["Z"]]), d["CODE"])
    np.testing.assert_array_equal(d["field"][cells], codes)
