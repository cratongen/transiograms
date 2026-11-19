import pickle

import numpy as np
import pytest
import transiograms as tg


def hmm3():
    return tg.HiddenMarkovModel(
        [tg.Gaussian([m], 1.0) for m in (0.0, 2.0, 4.0)],
        start=[0.5, 0.3, 0.2],
        transition=[[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.25, 0.25, 0.5]],
        direction="down",
    )


def hsmm3():
    return tg.HiddenSemiMarkovModel(
        [tg.Gaussian([m], 1.0) for m in (0.0, 2.0, 4.0)],
        [tg.ShiftedPoisson(1.5), tg.NegativeBinomial(2.0, 0.5), tg.EmpiricalDuration([0.1, 0.5, 0.4])],
        start=[0.5, 0.3, 0.2],
        transition=[[0.0, 0.7, 0.3], [0.4, 0.0, 0.6], [0.5, 0.5, 0.0]],
        max_duration=[4, 5, 3],
        direction="down",
    )


def test_m3_geometric_hsmm_equals_the_hmm():
    h = hmm3()
    m = int(np.ceil(np.log(1e-14) / np.log(0.8))) + 1
    s = tg.HiddenSemiMarkovModel.from_hmm(h, max_duration=m)
    _, obs = h.sample([120, 80], seed=1)
    assert s.score(obs) == pytest.approx(h.score(obs), abs=1e-10)
    for a, b in zip(s.predict(obs), h.predict(obs)):
        np.testing.assert_array_equal(a, b)


def test_brute_force_k3_t8():
    m = hsmm3()
    _, obs = m.sample(8, seed=2)
    x = obs[0]
    log_b = np.column_stack([d.log_density(x) for d in m.distributions])
    pmfs = [d.pmf(int(md)) for d, md in zip(m.durations, m.max_duration)]
    log_d = [np.log(p) for p in pmfs]
    with np.errstate(divide="ignore"):
        log_s = [np.log(np.cumsum(p[::-1])[::-1]) for p in pmfs]
    ll, path, best, gamma, entropy = tg.checks.brute_force_hsmm(m.start, m.transition, log_d, log_s, log_b)
    assert m.score(x) == pytest.approx(ll, abs=1e-12)
    vpath, vlp = m.viterbi(x)
    np.testing.assert_array_equal(vpath, path)
    assert vlp[0] == pytest.approx(best, abs=1e-12)
    np.testing.assert_allclose(m.predict_proba(x), gamma, atol=1e-12)
    assert (m.icl(x) - m.bic(x)) / 2 == pytest.approx(entropy, abs=1e-10)


def test_duration_laws():
    for law, mean in [
        (tg.Geometric(0.25), 4.0),
        (tg.ShiftedPoisson(2.5), 3.5),
        (tg.NegativeBinomial(3.0, 0.6), 1 + 3 * 0.4 / 0.6),
    ]:
        p = law.pmf(400)
        assert p.sum() == pytest.approx(1.0, abs=1e-12) and np.all(p >= 0)
        assert law.mean(400) == pytest.approx(mean, rel=1e-9)
    np.testing.assert_allclose(tg.EmpiricalDuration([1.0, 3.0]).pmf(2), [0.25, 0.75], atol=1e-15)


def test_em_is_monotone_and_durations_are_recovered():
    truth = hsmm3()
    _, obs = truth.sample([400] * 5, seed=3)
    f = tg.HiddenSemiMarkovModel(
        truth.distributions,
        duration="poisson",
        max_duration=12,
        init_params="std",
        n_init=2,
        direction="down",
    ).fit(obs)
    h = f.history
    assert np.all(np.diff(h) >= -1e-9 * np.abs(h[1:]))
    np.testing.assert_allclose(np.diag(f.transition), 0.0)
    np.testing.assert_allclose(f.transition.sum(axis=1), 1.0, atol=1e-12)
    assert f.durations[0].mean(12) == pytest.approx(truth.durations[0].mean(4), rel=0.2)


def test_m3_bic_selects_three_states():
    d = tg.datasets.electrofacies_logs(seed=0)
    table = tg.select_states(
        np.column_stack([d["GR"], d["RHOB"]]),
        holes=d["HOLE_ID"],
        states=range(1, 7),
        n_init=5,
        seed=0,
        direction="down",
    )
    assert table["states"][np.argmin(table["bic"])] == 3
    assert np.all(np.isfinite(table["cv_per_step"]))


def test_selection_is_seeded():
    d = tg.datasets.electrofacies_logs(seed=1)
    x = np.column_stack([d["GR"], d["RHOB"]])[:1500]
    holes = d["HOLE_ID"][:1500]
    a = tg.select_states(x, holes=holes, states=[2, 3], model="hsmm", max_duration=40, cv_folds=2, n_init=1)
    b = tg.select_states(x, holes=holes, states=[2, 3], model="hsmm", max_duration=40, cv_folds=2, n_init=1)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_json_pickle_and_errors():
    m = hsmm3()
    assert tg.HiddenSemiMarkovModel.from_json(m.to_json()) == m
    assert pickle.loads(pickle.dumps(m)) == m
    for law in m.durations + [tg.Geometric(0.3)]:
        assert type(law).from_json(law.to_json()) == law
        assert pickle.loads(pickle.dumps(law)) == law
    with pytest.raises(tg.InvalidInput):
        tg.Geometric(1.5)
    with pytest.raises(tg.InvalidInput):
        tg.Geometric(0.5).pmf(2**60)
    with pytest.raises(tg.InvalidInput):
        hsmm3().predict(np.zeros(5), algorithm="posterior_viterbi")
    with pytest.raises(tg.InvalidInput, match="single entry"):
        tg.select_states(np.zeros(10), labels=np.zeros(10), states=[1, 2])
    with pytest.raises(tg.InvalidInput):
        tg.HiddenSemiMarkovModel(
            [tg.Gaussian([0.0], 1.0)],
            [tg.Geometric(0.5)],
            start=[1.0],
            transition=[[0.0]],
            max_duration=2**60,
        )


def test_pseudocounts_survive_restarts():
    _, obs = hsmm3().sample([200, 200], seed=4)
    f = tg.HiddenSemiMarkovModel(
        2, duration="empirical", max_duration=10, pseudocount=2.0, n_init=2, direction="down"
    ).fit(obs)
    assert all(d.pseudocount == 2.0 for d in f.durations)


def test_public_members_have_numpy_docstrings():
    cls = tg.HiddenSemiMarkovModel
    for name in (
        "score",
        "score_samples",
        "predict_proba",
        "viterbi",
        "sample",
        "sample_posterior",
        "aic",
        "bic",
        "icl",
    ):
        assert "Returns\n-------" in (getattr(cls, name).__doc__ or ""), name
    for name in ("durations", "max_duration", "history", "converged", "n_iter", "transition", "start"):
        assert getattr(cls, name).__doc__, name
