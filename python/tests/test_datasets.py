import numpy as np
import pytest
import transiograms as tg


@pytest.mark.parametrize(
    "name, keys",
    [
        ("varved_sequence", {"HOLE_ID", "FROM", "TO", "CODE", "generator"}),
        ("electrofacies_logs", {"HOLE_ID", "DEPTH", "GR", "RHOB", "STATE"}),
        ("layered_drillholes", {"X", "Y", "Z", "HOLE_ID", "CODE", "model", "grid"}),
    ],
)
def test_datasets_are_seeded_and_documented(name, keys):
    f = getattr(tg.datasets, name)
    a, b, c = f(seed=0), f(seed=0), f(seed=1)
    assert keys <= set(a)
    for k in keys:
        if isinstance(a[k], np.ndarray):
            np.testing.assert_array_equal(a[k], b[k])
    assert any(isinstance(a[k], np.ndarray) and not np.array_equal(a[k], c[k]) for k in keys)


def test_intervals_are_top_down_and_contiguous():
    d = tg.datasets.varved_sequence()
    for h in np.unique(d["HOLE_ID"]):
        at = d["HOLE_ID"] == h
        f, t = d["FROM"][at], d["TO"][at]
        assert f[0] == pytest.approx(0.0, abs=1e-9) and t[-1] == pytest.approx(30.0)
        np.testing.assert_allclose(f[1:], t[:-1])
    e = tg.datasets.electrofacies_logs()
    assert len(e["GR"]) == 6000 and set(np.unique(e["STATE"])) == {0, 1, 2}


def test_layered_holes_sample_the_field():
    d = tg.datasets.layered_drillholes()
    grid = tg.Grid.from_json(d["grid"])
    cells, codes = grid.snap(np.column_stack([d["X"], d["Y"], d["Z"]]), d["CODE"])
    np.testing.assert_array_equal(d["field"][cells], codes)


def field_statistics(seed):
    d = tg.datasets.layered_drillholes(seed=seed)
    grid, model = tg.Grid.from_json(d["grid"]), tg.TransitionModel.from_json(d["model"])
    p = np.bincount(d["field"], minlength=3) / d["field"].size
    e = tg.grid_transiogram(d["field"], grid, offset=(0, 0, 1), n_lags=12)
    lengths = -1 / np.diag(tg.TransiogramModel.fit_markov(e, proportions=p).rates)
    gap = np.abs(e.probabilities - model.transition(np.outer(np.arange(1, 13), [0.0, 0.0, 1.0]))).max()
    return p, lengths, gap


def test_layered_drillholes_departs_from_its_model():
    """Pins the MCRF gap that the docstring and the transiograms example quantify."""
    p, lengths, gap = field_statistics(0)
    np.testing.assert_allclose(p, [0.525, 0.263, 0.211], atol=5e-4)
    np.testing.assert_allclose(lengths, [10.54, 6.79, 5.24], atol=5e-3)
    assert round(gap, 3) == 0.231
    assert round(field_statistics(2)[2], 3) == 0.423
    doc = tg.datasets.layered_drillholes.__doc__
    for quoted in ("0.525, 0.263 and 0.211", "10.54, 6.79 and 5.24 m", "up to 0.231", "0.423 at ``seed=2``"):
        assert quoted in doc
