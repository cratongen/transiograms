import itertools
import pickle

import numpy as np
import pytest
import transiograms as tg
from scipy import stats

Q = np.array([[-0.25, 0.15, 0.10], [0.35, -0.5, 0.15], [0.1, 0.375, -0.475]])
GOLDEN_N = np.array([[0, 30, 10, 5], [8, 0, 25, 12], [20, 6, 0, 9], [10, 14, 8, 0]], float)
GOLDEN_E = np.array(
    [
        [0, 20.6805682797, 15.4762764461, 8.8431552742],
        [17.022383202, 0, 17.8042536869, 10.1733631111],
        [11.68497564, 16.3315427453, 0, 6.9834816147],
        [9.292641158, 12.987888975, 9.719469867, 0],
    ]
)


def m1_chain():
    return tg.MarkovChain.from_matrix(tg.MarkovChain.from_rates(Q).transition_at(0.5), step=0.5)


def test_m1_recovery_within_binomial_and_kemeny_snell_tolerance():
    truth = m1_chain()
    x = truth.simulate(0.5 * 100_000, seed=0)[0]
    fit = tg.MarkovChain("fixed").fit(x[::-1], depth=0.25 + 0.5 * np.arange(x.size))
    assert fit.step == 0.5
    p, c = truth.transition_matrix, fit.counts
    sigma = np.sqrt(p * (1 - p) / c.sum(axis=1, keepdims=True))
    assert np.all(np.abs(fit.transition_matrix - p) <= 3 * sigma)
    sd = truth.proportion_std(100_000)
    np.testing.assert_allclose(sd, tg.checks.kemeny_snell_std(p, 100_000), rtol=1e-10)
    assert np.all(np.abs(np.bincount(x, minlength=3) / x.size - truth.proportions) <= 3 * sd)


@pytest.mark.parametrize("seed", range(5))
def test_every_fitted_chain_is_stochastic(seed):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 4, 300)
    holes = np.repeat(["a", "b", "c"], 100)
    for kind in ("fixed", "embedded"):
        c = tg.MarkovChain(kind, pseudocount=0.5).fit(codes, holes=holes)
        np.testing.assert_allclose(c.transition_matrix.sum(axis=1), 1.0, atol=1e-12)
        if kind == "embedded":
            assert np.all(np.diag(c.transition_matrix) == 0.0)
            q = c.generator
            np.testing.assert_allclose(q.sum(axis=1), 0.0, atol=1e-12 * np.abs(q).max())


def test_golden_quasi_independence_oracle():
    np.testing.assert_allclose(tg.checks.quasi_independence(GOLDEN_N), GOLDEN_E, atol=1e-8)
    e = tg.checks.quasi_independence(GOLDEN_N)
    x2 = np.sum((GOLDEN_N - e)[e > 0] ** 2 / e[e > 0])
    assert x2 == pytest.approx(29.298020503608846, rel=1e-9)


def test_independence_test_matches_numpy_and_scipy():
    d = tg.datasets.varved_sequence()
    c = tg.MarkovChain().fit("CODE", holes="HOLE_ID", from_="FROM", to="TO", data=d)
    e = tg.checks.quasi_independence(c.counts)
    for method in ("g", "pearson"):
        t = c.independence_test(method=method)
        np.testing.assert_allclose(t.expected, e, atol=1e-8)
        o, ee = c.counts[e > 0], e[e > 0]
        stat = (
            2 * np.sum(o[o > 0] * np.log(o[o > 0] / ee[o > 0]))
            if method == "g"
            else np.sum((o - ee) ** 2 / ee)
        )
        assert t.statistic == pytest.approx(stat, rel=1e-9)
        assert t.df == 3 * 3 - 3 * 3 + 1
        assert t.p_value == pytest.approx(stats.chi2.sf(stat, t.df), rel=1e-9)


def test_embedded_fit_recovers_the_generator_and_mean_lengths():
    d = tg.datasets.varved_sequence()
    truth = tg.MarkovChain.from_json(d["generator"])
    c = tg.MarkovChain().fit(d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
    np.testing.assert_allclose(c.mean_lengths, truth.mean_lengths, rtol=0.2)
    p = c.proportions
    nu, length = c.bed_frequencies, c.mean_lengths
    np.testing.assert_allclose(p, nu * length / np.sum(nu * length), atol=1e-12)


def test_never_tallies_across_holes_or_missing():
    c = tg.MarkovChain("fixed", on_empty_row="uniform").fit(
        [0, 0, 1, -1, 2, 2, 0, 1], holes=[1, 1, 1, 1, 1, 2, 2, 2]
    )
    expected = np.zeros((3, 3))
    for seq in ([1, 0, 0], [1, 0, 2]):
        for a, b in itertools.pairwise(seq):
            expected[a, b] += 1
    np.testing.assert_array_equal(c.counts, expected)


def test_direction_down_counts_in_row_order():
    c = tg.MarkovChain("fixed", direction="down", on_empty_row="uniform").fit([0, 0, 1, 2])
    n = np.zeros((3, 3))
    n[0, 0] = n[0, 1] = n[1, 2] = 1
    np.testing.assert_array_equal(c.counts, n)


def test_order_test_detects_second_order_and_has_the_right_df():
    rng = np.random.default_rng(1)
    x = [0, 1]
    for _ in range(20_000):
        x.append(x[-2] if rng.random() < 0.9 else rng.integers(0, 3))
    t = tg.order_test(np.array(x), order=2)
    assert t.df <= 3 * (3 - 1) ** 2
    assert t.p_value < 1e-6
    first = m1_chain().simulate(0.5 * 20_000, seed=2)[0][::-1]
    assert tg.order_test(first, order=2).p_value > 0.001


def test_select_order_shares_transitions():
    x = m1_chain().simulate(0.5 * 5_000, seed=3)[0][::-1]
    t = tg.select_order(x, max_order=2)
    assert list(t["order"]) == [0, 1, 2]
    assert t["n_transitions"][0] == x.size - 2
    assert np.argmin(t["bic"]) == 1


def test_homogeneity_of_one_chain_is_not_rejected():
    x = m1_chain().simulate(0.5 * 4_000, n=2, seed=4)[:, ::-1]
    t = tg.homogeneity_test(
        np.concatenate(x), np.repeat(["a", "b"], 4000), kind="fixed", holes=np.repeat(["a", "b"], 4000)
    )
    assert t.p_value > 0.001
    assert t.df == (2 - 1) * 3 * (3 - 1)


def test_symmetric_counts_give_zero_bowker_statistic():
    c = tg.MarkovChain("fixed", direction="down").fit([0, 1, 0, 2, 0, 1, 0, 2, 0])
    assert c.symmetry_test().statistic == pytest.approx(0.0, abs=1e-12)


def test_uniform_embedded_rows_have_entropy_log2_m_minus_1():
    pi = (np.ones((4, 4)) - np.eye(4)) / 3
    e = tg.MarkovChain.from_matrix(pi, kind="embedded").entropy()
    np.testing.assert_allclose(e.post, np.log2(3), atol=1e-12)


def test_difference_matrix_is_observed_minus_quasi_independent():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 4, 2000)
    c = tg.MarkovChain("embedded").fit(codes)
    e = tg.checks.quasi_independence(c.counts)
    want = c.transition_matrix - e / e.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(c.difference_matrix(), want, atol=1e-12)


def test_asymmetry_is_the_antisymmetric_part_of_the_flux():
    p = np.array([[0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.7, 0.1, 0.2]])
    c = tg.MarkovChain.from_matrix(p, step=1.0)
    rate = p - np.eye(3) if c.generator is None else c.generator
    w, v = np.linalg.eig(rate.T)
    nu = np.real(v[:, np.argmin(np.abs(w))])
    nu /= nu.sum()
    f = nu[:, None] * rate
    a = c.asymmetry()
    np.testing.assert_allclose(a, f - f.T, atol=1e-12)
    np.testing.assert_allclose(a, -a.T, atol=1e-15)
    np.testing.assert_allclose(a.sum(axis=1), 0.0, atol=1e-12)
    sym = tg.MarkovChain.from_matrix(np.full((3, 3), 1 / 3), step=1.0)
    np.testing.assert_allclose(sym.asymmetry(), 0.0, atol=1e-12)


def test_discrete_bridge_marginal_matches_closed_form():
    p = m1_chain().transition_matrix
    c = tg.MarkovChain.from_matrix(p, step=1.0)
    sims = c.simulate(7.0, n=20_000, seed=5, hard_from=[0.5, 6.5], hard_codes=[0, 2])
    assert np.all(sims[:, 0] == 0) and np.all(sims[:, 6] == 2)
    for t in (2, 4):
        want = tg.checks.markov_bridge_marginal(p, 0, 2, 6, t)
        got = np.bincount(sims[:, t], minlength=3) / len(sims)
        assert np.all(np.abs(got - want) <= 4 * np.sqrt(want * (1 - want) / len(sims)) + 1e-12)


def test_continuous_simulation_honours_hard_intervals():
    c = tg.MarkovChain.from_rates(Q)
    beds = c.simulate(40.0, n=3, seed=6, hard_from=[10.0], hard_to=[12.0], hard_codes=[1])
    for r in range(3):
        at = beds["realization"] == r
        f, t, code = beds["from"][at], beds["to"][at], beds["code"][at]
        assert f[0] == 0.0 and t[-1] == pytest.approx(40.0)
        inside = (f < 12.0) & (t > 10.0)
        assert np.all(code[inside] == 1)


def test_simulation_is_seeded():
    c = tg.MarkovChain.from_rates(Q)
    a, b = c.simulate(20.0, n=2, seed=9), c.simulate(20.0, n=2, seed=9)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_simulate_beds_has_no_repeats():
    pi = (np.ones((3, 3)) - np.eye(3)) / 2
    x = tg.MarkovChain.from_matrix(pi, kind="embedded").simulate_beds(50, n=4, seed=1)
    assert x.shape == (4, 50) and np.all(x[:, 1:] != x[:, :-1])


def test_sojourn_and_exponential_test():
    d = tg.datasets.varved_sequence()
    s = tg.sojourn(d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
    _t, surv = s.survival(0)
    assert np.all(np.diff(surv) <= 0) and 0 <= surv[-1] <= 1
    thick, open_ = s.thickness(0)
    assert len(thick) == s.n_beds[0] and open_.sum() == s.n_open[0]
    r = s.exponential_test(0, n_bootstrap=99, seed=0)
    assert r.name == "exponential" and 0 < r.p_value <= 1


def test_json_pickle_and_errors():
    c = tg.MarkovChain.from_rates(Q)
    assert tg.MarkovChain.from_json(c.to_json()) == c
    assert pickle.loads(pickle.dumps(c)) == c
    with pytest.raises(tg.InvalidInput):
        m1_chain().symmetry_test()
    with pytest.raises(tg.InvalidInput):
        tg.MarkovChain("sideways")
    with pytest.raises(ValueError, match="codes must be integers"):
        tg.MarkovChain().fit([0, 1.5, 2])
    with pytest.raises(tg.InvalidInput, match="not fitted"):
        _ = tg.MarkovChain().transition_matrix
    r = m1_chain().simulate(5.0)
    assert r.dtype == np.int64


def test_huge_requests_raise_before_allocating():
    with pytest.raises(tg.InvalidInput):
        m1_chain().simulate(0.5 * 2**40, n=2**20)
    with pytest.raises(tg.InvalidInput):
        tg.order_test(np.arange(40) % 4, order=33)
    with pytest.raises(tg.InvalidInput, match="histories"):
        tg.order_test(np.arange(2000) % 4, order=14)
    assert tg.order_test(np.arange(2000) % 4, order=5).expected.shape == (4**5, 4)
    wide = tg.MarkovChain.from_matrix(np.full((200, 200), 1 / 200), step=1.0)
    with pytest.raises(tg.InvalidInput, match="transition_at"):
        wide.transition_at(np.ones(10**6))


def test_fixed_hard_points_freeze_the_cell_that_contains_them():
    c = tg.MarkovChain.from_matrix(np.full((3, 3), 1 / 3), step=0.5)
    sims = c.simulate(5.0, n=50, seed=1, hard_from=[0.0, 2.5, 4.99], hard_codes=[2, 1, 0])
    assert np.all(sims[:, 0] == 2) and np.all(sims[:, 5] == 1) and np.all(sims[:, 9] == 0)
