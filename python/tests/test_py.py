import pickle
import subprocess
import sys

import numpy as np
import pytest
import transiograms as tg


def instances():
    q = np.array([[-0.25, 0.15, 0.10], [0.35, -0.5, 0.15], [0.1, 0.375, -0.475]])
    chain = tg.MarkovChain.from_rates(q)
    hmm = tg.HiddenMarkovModel(
        [tg.Gaussian([0.0], 1.0), tg.Gaussian([3.0], 1.0)],
        start=[0.5, 0.5],
        transition=[[0.9, 0.1], [0.1, 0.9]],
    )
    geology = tg.TransitionModel.from_geology([0.5, 0.5], [[5, 5], [5, 5], [1, 1]])
    grid = tg.Grid((0, 0, 0), (1, 1, 1), (4, 3, 2))
    hard_xyz = np.array([[0.5, 0.5, 0.5], [3.5, 2.5, 1.5]])
    hard_codes = np.array([0, 1])
    mask = np.arange(grid.n_cells) != 5
    return [
        tg.Categories(["a", "b"], colors=["red", "blue"]),
        tg.Grid((0, 0, 0), (1, 1, 1), (2, 3, 4)),
        chain,
        tg.MarkovChain("fixed"),
        tg.MarkovChain.from_matrix([[0.0, 1.0], [1.0, 0.0]], kind="embedded"),
        hmm,
        tg.HiddenMarkovModel(3, emission="categorical"),
        tg.HiddenSemiMarkovModel.from_hmm(hmm, max_duration=30),
        tg.Categorical([0.5, 0.5]),
        tg.Gaussian([0.0, 1.0], [[1.0, 0.0], [0.0, 2.0]]),
        tg.Poisson([1.0]),
        tg.GaussianMixture([0.5, 0.5], [[0.0], [1.0]], [1.0, 2.0], covariance_type="spherical"),
        tg.Geometric(0.5),
        tg.ShiftedPoisson(2.0),
        tg.NegativeBinomial(2.0, 0.4),
        tg.EmpiricalDuration([0.2, 0.8], pseudocount=1.0),
        tg.TransiogramModel.markov(q),
        tg.TransitionModel.from_geology([0.5, 0.5], [[5, 5], [5, 5], [1, 1]]),
        tg.downhole_transiogram([0, 1, 1, 0], lags=[1.0]),
        tg.order_test(np.arange(30) % 3, order=1),
        chain.entropy(),
        tg.sojourn([0, 0, 1, 2, 2, 2, 0, 1, 1]),
        tg.mcp_predict(geology, grid, hard_xyz, hard_codes, mask=mask),
        tg.mcrf_simulate(geology, grid, hard_xyz, hard_codes, n=2, seed=3, mask=mask),
        tg.tprogs_simulate(geology, grid, hard_xyz, hard_codes, n=2, seed=3),
    ]


@pytest.mark.parametrize("obj", instances(), ids=lambda o: type(o).__name__)
def test_json_and_pickle_round_trip(obj):
    assert type(obj).from_json(obj.to_json()) == obj
    assert pickle.loads(pickle.dumps(obj)) == obj
    assert "\n" not in repr(obj)


def test_error_hierarchy():
    assert issubclass(tg.InvalidInput, tg.TransiogramsError) and issubclass(tg.InvalidInput, ValueError)
    assert issubclass(tg.ConvergenceError, tg.TransiogramsError) and issubclass(
        tg.ConvergenceError, RuntimeError
    )
    with pytest.raises(tg.InvalidInput) as e:
        tg.MarkovChain.from_matrix([[0.5, 0.6], [0.5, 0.5]])
    assert not str(e.value).endswith(".")
    with pytest.raises(tg.InvalidInput):
        tg.Categories.from_json('{"names": ["a"], "extra": 1}')
    alternating = tg.downhole_transiogram(np.arange(40) % 2, lags=[1.0, 2.0])
    with pytest.raises(tg.ConvergenceError) as e:
        tg.TransiogramModel.fit_markov(alternating)
    assert isinstance(e.value, RuntimeError) and "no matrix logarithm" in str(e.value)
    assert not str(e.value).endswith(".")


def test_from_json_rejects_inconsistent_shapes():
    with pytest.raises(tg.InvalidInput):
        tg.Prediction.from_json(
            '{"k":2,"probabilities":[1.0],"most_likely":[0],"entropy":[0.0],"corrected":0.0}'
        )
    with pytest.raises(tg.InvalidInput):
        tg.Realizations.from_json('{"n":1,"n_cells":2,"k":2,"codes":[0,5],"corrected":[0.0],"objective":[]}')
    with pytest.raises(tg.InvalidInput):
        tg.Sojourn.from_json(
            '{"n_beds":[1],"n_open":[0],"total":[1.0],"mean_length":[1.0],"thickness":[[1.0]],"open":[[]]}'
        )


def test_allocations_follow_the_process_data_limit():
    script = (
        "import resource, numpy as np, transiograms as tg\n"
        "resource.setrlimit(resource.RLIMIT_DATA, (2 * 1024**3, resource.RLIM_INFINITY))\n"
        "for f in (lambda: tg.Grid([0, 0, 0], [1, 1, 1], [10**5, 10**5, 10**4]).coords,\n"
        "          lambda: tg.Grid([0, 0, 0], [1, 1, 1], [2000, 2000, 100]).coords,\n"
        "          lambda: tg.Gaussian(np.zeros(3000), np.ones(3000), covariance_type='diag').sample(10**6)):\n"
        "    try:\n"
        "        f()\n"
        "        raise SystemExit('no error')\n"
        "    except tg.InvalidInput as e:\n"
        "        assert 'memory budget' in str(e), e\n"
        "assert tg.Grid([0, 0, 0], [1, 1, 1], [10, 10, 10]).coords.shape == (1000, 3)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr + done.stdout


def test_namespace_is_flat_and_submodules_are_not_star_exported():
    for name in (
        "MarkovChain",
        "HiddenMarkovModel",
        "HiddenSemiMarkovModel",
        "TransitionModel",
        "tprogs_simulate",
        "mcrf_simulate",
        "select_states",
        "experimental_transiogram",
    ):
        assert name in tg.__all__
    for name in ("plot", "datasets", "checks"):
        assert name not in tg.__all__
