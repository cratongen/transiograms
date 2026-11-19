import pickle

import numpy as np
import pytest
import transiograms as tg

P = np.array([0.5, 0.3, 0.2])
Q = np.array([[-0.25, 0.15, 0.10], [0.35, -0.5, 0.15], [0.1, 0.375, -0.475]])


def model3d(rotation=(0.0, 0.0, 0.0), lx=60.0, ly=40.0):
    lengths = [[lx, lx / 2, lx / 3], [ly, ly / 2, ly / 3], [6.0, 4.0, 3.0]]
    return tg.TransitionModel.from_geology(P, lengths, rotation=rotation)


def lag_grid(max_length):
    return np.r_[np.linspace(0.0, 20 * max_length, 200), 1e4 * max_length]


def check_valid(ts, p):
    assert np.all(ts >= 0.0)
    np.testing.assert_allclose(ts.sum(axis=2), 1.0, atol=1e-12)
    assert np.array_equal(ts[0], np.eye(len(p)))
    np.testing.assert_allclose(ts[-1], np.outer(np.ones(len(p)), p), atol=1e-10)


def test_m4_markov_model_is_valid_at_every_lag():
    m = tg.TransiogramModel.markov(Q)
    np.testing.assert_allclose(m.proportions, tg.MarkovChain.from_rates(Q).proportions, atol=1e-12)
    check_valid(m.transition(lag_grid(-1 / Q.diagonal().max())), m.proportions)


def test_m4_entrywise_model_is_valid_at_every_lag():
    ranges = np.full((3, 3), 6.0)
    m = tg.TransiogramModel.entrywise(P, "exponential", ranges, max_lag=200.0)
    check_valid(m.transition(lag_grid(6.0)), P)


def test_m4_three_d_model_is_valid_along_every_axis():
    m = model3d()
    for axis in np.eye(3):
        check_valid(m.transition(np.outer(lag_grid(60.0), axis)), P)


def test_reversal_and_determinant_identities():
    m = tg.TransiogramModel.markov(Q)
    p = m.proportions
    for h in (0.3, 2.0, 7.5):
        t, back = m.transition(h), m.transition(-h)
        np.testing.assert_allclose(p[:, None] * t, (p[:, None] * back).T, atol=1e-12)
        assert np.linalg.det(t) == pytest.approx(np.exp(h * np.trace(Q)), abs=1e-12)
    d = model3d()
    for h in ([0.0, 0.0, 2.0], [5.0, -3.0, 1.0]):
        t, back = d.transition(h), d.transition(-np.asarray(h))
        np.testing.assert_allclose(P[:, None] * t, (P[:, None] * back).T, atol=1e-12)


def test_principal_axis_interpolation_is_exact_and_lengths_lie_on_the_ellipsoid():
    m = model3d()
    np.testing.assert_array_equal(m.rates([0.0, 0.0, 3.0]), m.rates_z)
    np.testing.assert_array_equal(m.rates([2.0, 0.0, 0.0]), m.rates_x)
    axes = [m.mean_lengths(a) for a in np.eye(3)]
    u = np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5])
    got = m.mean_lengths(u)
    want = 1 / np.sqrt(sum((u[v] / axes[v]) ** 2 for v in range(3)))
    keep = np.arange(3) != m.background
    np.testing.assert_allclose(got[keep], want[keep], rtol=1e-12)


def test_rotation_by_ninety_degrees_swaps_the_horizontal_axes():
    a = model3d(lx=60.0, ly=20.0)
    b = model3d(rotation=(90.0, 0.0, 0.0), lx=20.0, ly=60.0)
    for lag in ([7.0, 0.0, 0.0], [0.0, 7.0, 0.0]):
        np.testing.assert_allclose(b.transition(lag), a.transition(lag), atol=1e-12)


def exact_overlaps(top, bottom, codes, h, k):
    n = np.zeros((k, k))
    for i in range(len(codes)):
        for j in range(len(codes)):
            lo, hi = max(top[i], top[j] - h), min(bottom[i], bottom[j] - h)
            n[codes[i], codes[j]] += max(hi - lo, 0.0)
    return n


def test_downhole_estimator_equals_interval_overlaps():
    top = np.array([0.0, 1.5, 2.0, 3.5, 4.0, 5.5])
    bottom = np.r_[top[1:], 6.0]
    codes = np.array([0, 1, 0, 1, 0, 1])
    lags = [0.25, 0.5, 1.0, 1.75]
    e = tg.downhole_transiogram(codes, lags=lags, direction="down", from_=top, to=bottom)
    for li, h in enumerate(lags):
        np.testing.assert_allclose(e.counts[li], exact_overlaps(top, bottom, codes, h, 2), atol=1e-12)
    np.testing.assert_allclose(e.counts.sum(axis=2), e.tail_totals, atol=1e-12)
    np.testing.assert_allclose(np.nansum(e.probabilities, axis=2), 1.0, atol=1e-12)


def test_point_estimator_equals_grid_estimator_on_vertical_holes():
    d = tg.datasets.layered_drillholes()
    grid = tg.Grid.from_json(d["grid"])
    xyz = np.column_stack([d["X"], d["Y"], d["Z"]])
    on_grid = tg.grid_transiogram(d["field"], grid, offset=(0, 0, 1), n_lags=6)
    holes_only = np.full(grid.n_cells, -1)
    cells, codes = grid.snap(xyz, d["CODE"])
    holes_only[cells] = codes
    g = tg.grid_transiogram(holes_only, grid, offset=(0, 0, 1), n_lags=6, categories=3)
    p = tg.experimental_transiogram(
        xyz,
        d["CODE"],
        holes=d["HOLE_ID"],
        direction=(0.0, -90.0),
        lag=1.0,
        n_lags=6,
        lag_tolerance=0.25,
        angle_tolerance=1.0,
        pairs="same_hole",
        categories=3,
    )
    np.testing.assert_array_equal(p.counts, g.counts)
    assert on_grid.counts.sum() > g.counts.sum()


def test_fit_markov_recovers_the_rates():
    chain = tg.MarkovChain.from_rates(Q, direction="down")
    beds = chain.simulate(200_000.0, seed=0)
    e = tg.downhole_transiogram(
        beds["code"], lags=np.arange(0.25, 6.0, 0.25), direction="down", from_=beds["from"], to=beds["to"]
    )
    m = tg.TransiogramModel.fit_markov(e)
    off = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(m.rates[off], Q[off], rtol=0.05)
    np.testing.assert_allclose(np.diag(m.rates), np.diag(Q), rtol=0.05)
    entry = tg.TransiogramModel.fit_entrywise(e, m.proportions)
    assert entry.kind == "entrywise"


def test_negative_entrywise_model_raises_naming_the_entry():
    nuggets = np.zeros((3, 3))
    nuggets[0, 2] = 0.9
    ranges = np.array([[50.0, 1.0, 1.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    with pytest.raises(tg.InvalidInput, match=r"\(0, 1"):
        tg.TransiogramModel.entrywise(
            [0.2, 0.4, 0.4], "spherical", ranges, nuggets=nuggets, complement=[1, 1, 2], max_lag=20.0
        )


def test_json_pickle_and_errors():
    m = model3d((30.0, 10.0, 0.0))
    assert tg.TransitionModel.from_json(m.to_json()) == m
    assert pickle.loads(pickle.dumps(m)) == m
    t = tg.TransiogramModel.markov(tg.MarkovChain.from_rates(Q))
    assert pickle.loads(pickle.dumps(t)) == t
    e = tg.downhole_transiogram([0, 1, 0, 1], lags=[1.0, 2.0])
    assert tg.ExperimentalTransiogram.from_json(e.to_json()) == e
    with pytest.raises(tg.InvalidInput):
        m.transition([np.inf, 0.0, 0.0])
    with pytest.raises(tg.InvalidInput):
        tg.TransitionModel(P, Q, Q, Q + 1.0)
    with pytest.raises(tg.InvalidInput):
        tg.experimental_transiogram(np.zeros((3, 3)), [0, 1, 0], lag=1.0, n_lags=2**40)


def test_closeness_equals_the_determinant_root_and_the_two_category_closed_form():
    m = model3d((30.0, 10.0, 0.0))
    lags = np.array([[0.0, 0.0, 0.0], [5.0, -3.0, 1.0], [0.0, 0.0, 2.0], [40.0, 10.0, -4.0]])
    c = m.closeness(lags)
    assert c.shape == (4,)
    assert c[0] == 1.0
    np.testing.assert_allclose(c, np.linalg.det(m.transition(lags)) ** 0.5, rtol=1e-10)
    assert m.closeness(lags[1]) == pytest.approx(c[1], rel=1e-14)
    two = tg.TransitionModel.from_geology([0.4, 0.6], [[20.0, 30.0], [10.0, 15.0], [2.0, 3.0]])
    h = np.outer([0.0, 0.5, 1.0, 3.0], [0.0, 0.0, 1.0])
    rate_sum = 1 / 2.0 + 1 / 3.0
    np.testing.assert_allclose(two.closeness(h), np.exp(-rate_sum * h[:, 2]), rtol=1e-10)


def test_batched_transition_raises_before_allocating_an_absurd_stack():
    k = 40
    p = np.full(k, 1 / k)
    big = tg.TransitionModel.from_geology(p, np.full((3, k), 5.0))
    with pytest.raises(tg.InvalidInput, match="transition matrices of 40 x 40"):
        big.transition(np.broadcast_to(np.ones(3), (10**7, 3)))
    markov = tg.TransiogramModel.markov(big.rates_z)
    with pytest.raises(tg.InvalidInput, match="transition matrices of 40 x 40"):
        markov.transition(np.broadcast_to(1.0, 10**7))
    assert big.closeness(np.ones((1000, 3))).shape == (1000,)
