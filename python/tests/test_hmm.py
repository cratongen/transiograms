import pickle

import numpy as np
import pytest
import varvite as vv
from scipy import stats

START = np.array([0.4, 0.3, 0.2, 0.1])
A = np.array(
    [
        [0.70, 0.10, 0.10, 0.10],
        [0.15, 0.60, 0.15, 0.10],
        [0.05, 0.25, 0.60, 0.10],
        [0.20, 0.20, 0.20, 0.40],
    ]
)


def gaussian_model(**kw):
    dists = [vv.Gaussian([m, -m], [[1.0, 0.3], [0.3, 0.8]]) for m in (0.0, 1.0, 2.0, 3.5)]
    return vv.HiddenMarkovModel(dists, start=START, transition=A, direction="down", **kw)


def categorical_model():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=4)
    return vv.HiddenMarkovModel(
        [vv.Categorical(p) for p in probs], start=START, transition=A, direction="down"
    )


def log_b(model, x):
    return np.column_stack([d.log_density(x) for d in model.distributions])


@pytest.mark.parametrize("make", [gaussian_model, categorical_model])
def test_m2_brute_force(make):
    m = make()
    _, obs = m.sample(8, seed=11)
    x = obs[0]
    ll, path, best, gamma, entropy = vv.checks.brute_force_hmm(m.start, m.transition, log_b(m, x))
    vpath, vlp = m.viterbi(x)
    np.testing.assert_array_equal(vpath, path)
    assert vlp[0] == pytest.approx(best, abs=1e-12)
    assert m.score(x) == pytest.approx(ll, abs=1e-12)
    np.testing.assert_allclose(m.predict_proba(x), gamma, atol=1e-12)
    h = (m.icl(x) - m.bic(x)) / 2
    assert h == pytest.approx(entropy, abs=1e-10)


def test_posteriors_sum_to_one_and_decoders_agree_in_shape():
    m = gaussian_model()
    _, obs = m.sample([50, 30], seed=1)
    g = m.predict_proba(obs)
    for p in g:
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    for algorithm in ("viterbi", "map", "posterior_viterbi"):
        paths = m.predict(obs, algorithm=algorithm)
        assert [len(p) for p in paths] == [50, 30]


def test_em_is_monotone_and_rows_are_stochastic():
    m = gaussian_model()
    _, obs = m.sample([200] * 5, seed=2)
    for implementation in ("log", "scaled"):
        f = vv.HiddenMarkovModel(
            4,
            covariance_type="full",
            n_init=2,
            seed=3,
            max_iter=60,
            direction="down",
            implementation=implementation,
        ).fit(obs)
        h = f.history
        assert np.all(np.diff(h) >= -1e-9 * np.abs(h[1:]))
        np.testing.assert_allclose(f.transition.sum(axis=1), 1.0, atol=1e-12)


def test_scaled_and_log_fits_agree():
    m = gaussian_model()
    _, obs = m.sample([150] * 3, seed=4)
    a = vv.HiddenMarkovModel(4, covariance_type="full", direction="down", max_iter=20).fit(obs)
    b = vv.HiddenMarkovModel(
        4, covariance_type="full", direction="down", max_iter=20, implementation="scaled"
    ).fit(obs)
    np.testing.assert_allclose(a.history, b.history, rtol=1e-8)


def test_recovery_from_the_truth_within_binomial_tolerance():
    dists = [vv.Gaussian([4.0 * j], 1.0, covariance_type="spherical") for j in range(4)]
    truth = vv.HiddenMarkovModel(dists, start=START, transition=A, direction="down")
    states, obs = truth.sample([10_000] * 10, seed=5)
    f = vv.HiddenMarkovModel(
        truth.distributions, start=truth.start, transition=truth.transition, params="t", direction="down"
    ).fit(obs)
    s = np.concatenate([x[:-1] for x in states])
    visits = np.bincount(s, minlength=4)[:, None]
    sigma = np.sqrt(A * (1 - A) / visits)
    assert np.all(np.abs(f.transition - A) <= 3 * sigma)


def test_fully_labelled_fit_is_the_counting_mle():
    truth = categorical_model()
    states, obs = truth.sample([300, 300], seed=6)
    f = vv.HiddenMarkovModel(
        [vv.Categorical(np.full(3, 1 / 3)) for _ in range(4)],
        start=np.full(4, 0.25),
        transition=np.full((4, 4), 0.25),
        direction="down",
        max_iter=1,
    ).fit(obs, labels=states)
    n = np.zeros((4, 4))
    for s in states:
        np.add.at(n, (s[:-1], s[1:]), 1)
    np.testing.assert_allclose(f.transition, n / n.sum(axis=1, keepdims=True), atol=1e-12)
    for j, d in enumerate(f.distributions):
        c = np.bincount(np.concatenate(obs)[np.concatenate(states) == j], minlength=3)
        np.testing.assert_allclose(d.probabilities, c / c.sum(), atol=1e-12)


def test_separate_holes_score_as_a_sum():
    m = gaussian_model()
    _, obs = m.sample([40, 60], seed=7)
    assert m.score(obs) == pytest.approx(m.score(obs[0]) + m.score(obs[1]), abs=1e-10)
    assert m.score(np.concatenate(obs)) != pytest.approx(m.score(obs), abs=1e-6)
    np.testing.assert_allclose(m.score_samples(obs), [m.score(obs[0]), m.score(obs[1])], atol=1e-10)


def test_state_permutation_leaves_the_score_unchanged():
    m = gaussian_model()
    _, obs = m.sample(100, seed=8)
    perm = [2, 0, 3, 1]
    p = vv.HiddenMarkovModel(
        [m.distributions[i] for i in perm],
        start=START[perm],
        transition=A[np.ix_(perm, perm)],
        direction="down",
    )
    assert p.score(obs) == pytest.approx(m.score(obs), abs=1e-10)


@pytest.mark.parametrize("direction", ["up", "down"])
def test_outputs_come_back_in_input_row_order(direction):
    m = vv.HiddenMarkovModel(
        [vv.Gaussian([0.0], 1.0), vv.Gaussian([4.0], 1.0)],
        start=[0.5, 0.5],
        transition=[[0.9, 0.1], [0.2, 0.8]],
        direction=direction,
    )
    _, obs = m.sample([30, 20], seed=9)
    listed = m.predict(obs)
    holes = np.repeat(["a", "b"], [30, 20])
    joined = m.predict(np.concatenate(obs), holes=holes)
    np.testing.assert_array_equal(np.concatenate(listed), joined)
    shuffled = np.r_[np.arange(30, 50), np.arange(30)]
    again = m.predict(np.concatenate(obs)[shuffled], holes=holes[shuffled])
    np.testing.assert_array_equal(again, joined[shuffled])


def test_densities_match_scipy():
    x = np.array([[0.1, -0.2], [1.5, 0.3], [np.nan, 0.4]])
    cov = [[1.0, 0.3], [0.3, 0.8]]
    g = vv.Gaussian([0.0, 0.1], cov)
    want = stats.multivariate_normal([0.0, 0.1], cov).logpdf(x[:2])
    np.testing.assert_allclose(g.log_density(x)[:2], want, rtol=1e-12)
    assert g.log_density(x)[2] == pytest.approx(stats.norm(0.1, np.sqrt(0.8)).logpdf(0.4), rel=1e-12)
    p = vv.Poisson([2.0, 0.5])
    k = np.array([[0, 1], [3, 0]])
    np.testing.assert_allclose(p.log_density(k), stats.poisson([2.0, 0.5]).logpmf(k).sum(axis=1), rtol=1e-12)
    c = vv.Categorical([0.2, 0.5, 0.3])
    np.testing.assert_allclose(c.log_density([0, 2, -1]), [np.log(0.2), np.log(0.3), 0.0], rtol=1e-12)
    mix = vv.GaussianMixture([0.3, 0.7], [[0.0], [3.0]], [1.0, 2.0], covariance_type="spherical")
    xs = np.array([[-1.0], [2.0]])
    want = np.log(0.3 * stats.norm(0, 1).pdf(xs[:, 0]) + 0.7 * stats.norm(3, np.sqrt(2)).pdf(xs[:, 0]))
    np.testing.assert_allclose(mix.log_density(xs), want, rtol=1e-12)


def test_distribution_samples_are_seeded_and_shaped():
    g = vv.Gaussian([0.0, 1.0], [1.0, 2.0], covariance_type="diag")
    assert g.sample(10, seed=1).shape == (10, 2)
    np.testing.assert_array_equal(g.sample(10, seed=1), g.sample(10, seed=1))
    assert vv.Categorical([0.5, 0.5]).sample(7).dtype == np.int64
    np.testing.assert_allclose(g.covariance, [1.0, 2.0])


def test_pseudocount_survives_restarts():
    m = categorical_model()
    _, obs = m.sample([100, 100], seed=10)
    f = vv.HiddenMarkovModel(3, emission="categorical", pseudocount=2.0, n_init=2, direction="down").fit(obs)
    assert all(d.pseudocount == 2.0 for d in f.distributions)


def test_sample_posterior_and_clamps():
    m = gaussian_model()
    _, obs = m.sample(40, seed=12)
    x = obs[0]
    labels = np.full(40, -1)
    labels[10:15] = 3
    draws = m.sample_posterior(x, n=50, seed=1, labels=labels)
    assert draws.shape == (50, 40) and np.all(draws[:, 10:15] == 3)
    allowed = np.ones((40, 4), bool)
    allowed[:, 0] = False
    assert np.all(m.predict(x, allowed=allowed) != 0)
    np.testing.assert_array_equal(m.sample_posterior(x, n=5, seed=2), m.sample_posterior(x, n=5, seed=2))


def test_json_pickle_and_errors():
    m = gaussian_model()
    assert vv.HiddenMarkovModel.from_json(m.to_json()) == m
    assert pickle.loads(pickle.dumps(m)) == m
    for d in m.distributions + [vv.Poisson([1.0]), vv.GaussianMixture([1.0], [[0.0]], [[[1.0]]])]:
        assert type(d).from_json(d.to_json()) == d
        assert pickle.loads(pickle.dumps(d)) == d
    with pytest.raises(vv.InvalidInput, match="not fitted"):
        _ = vv.HiddenMarkovModel(2).transition
    with pytest.raises(vv.InvalidInput):
        vv.HiddenMarkovModel(2, emission="laplace")
    with pytest.raises(vv.InvalidInput):
        vv.Categorical([0.5, 0.6])
    with pytest.raises(vv.InvalidInput, match="labels"):
        m.predict(np.zeros((5, 2)), labels=[0, 1, 9, 0, 0])
    with pytest.raises(vv.InvalidInput):
        m.sample_posterior(np.zeros((5, 2)), n=2**62)
