import pickle
import subprocess
import sys

import numpy as np
import pytest
import varvite as vv


def instances():
    q = np.array([[-0.25, 0.15, 0.10], [0.35, -0.5, 0.15], [0.1, 0.375, -0.475]])
    chain = vv.MarkovChain.from_rates(q)
    hmm = vv.HiddenMarkovModel(
        [vv.Gaussian([0.0], 1.0), vv.Gaussian([3.0], 1.0)],
        start=[0.5, 0.5],
        transition=[[0.9, 0.1], [0.1, 0.9]],
    )
    geology = vv.TransitionModel.from_geology([0.5, 0.5], [[5, 5], [5, 5], [1, 1]])
    grid = vv.Grid((0, 0, 0), (1, 1, 1), (4, 3, 2))
    hard_xyz = np.array([[0.5, 0.5, 0.5], [3.5, 2.5, 1.5]])
    hard_codes = np.array([0, 1])
    mask = np.arange(grid.n_cells) != 5
    return [
        vv.Categories(["a", "b"], colors=["red", "blue"]),
        vv.Grid((0, 0, 0), (1, 1, 1), (2, 3, 4)),
        chain,
        vv.MarkovChain("fixed"),
        vv.MarkovChain.from_matrix([[0.0, 1.0], [1.0, 0.0]], kind="embedded"),
        hmm,
        vv.HiddenMarkovModel(3, emission="categorical"),
        vv.HiddenSemiMarkovModel.from_hmm(hmm, max_duration=30),
        vv.Categorical([0.5, 0.5]),
        vv.Gaussian([0.0, 1.0], [[1.0, 0.0], [0.0, 2.0]]),
        vv.Poisson([1.0]),
        vv.GaussianMixture([0.5, 0.5], [[0.0], [1.0]], [1.0, 2.0], covariance_type="spherical"),
        vv.Geometric(0.5),
        vv.ShiftedPoisson(2.0),
        vv.NegativeBinomial(2.0, 0.4),
        vv.EmpiricalDuration([0.2, 0.8], pseudocount=1.0),
        vv.TransiogramModel.markov(q),
        vv.TransitionModel.from_geology([0.5, 0.5], [[5, 5], [5, 5], [1, 1]]),
        vv.downhole_transiogram([0, 1, 1, 0], lags=[1.0]),
        vv.order_test(np.arange(30) % 3, order=1),
        chain.entropy(),
        vv.sojourn([0, 0, 1, 2, 2, 2, 0, 1, 1]),
        vv.mcp_predict(geology, grid, hard_xyz, hard_codes, mask=mask),
        vv.mcrf_simulate(geology, grid, hard_xyz, hard_codes, n=2, seed=3, mask=mask),
        vv.tprogs_simulate(geology, grid, hard_xyz, hard_codes, n=2, seed=3),
    ]


@pytest.mark.parametrize("obj", instances(), ids=lambda o: type(o).__name__)
def test_json_and_pickle_round_trip(obj):
    assert type(obj).from_json(obj.to_json()) == obj
    assert pickle.loads(pickle.dumps(obj)) == obj
    assert "\n" not in repr(obj)


def test_error_hierarchy():
    assert issubclass(vv.InvalidInput, vv.VarviteError) and issubclass(vv.InvalidInput, ValueError)
    assert issubclass(vv.ConvergenceError, vv.VarviteError) and issubclass(vv.ConvergenceError, RuntimeError)
    with pytest.raises(vv.InvalidInput) as e:
        vv.MarkovChain.from_matrix([[0.5, 0.6], [0.5, 0.5]])
    assert not str(e.value).endswith(".")
    with pytest.raises(vv.InvalidInput):
        vv.Categories.from_json('{"names": ["a"], "extra": 1}')
    alternating = vv.downhole_transiogram(np.arange(40) % 2, lags=[1.0, 2.0])
    with pytest.raises(vv.ConvergenceError) as e:
        vv.TransiogramModel.fit_markov(alternating)
    assert isinstance(e.value, RuntimeError) and "no matrix logarithm" in str(e.value)
    assert not str(e.value).endswith(".")


def test_from_json_rejects_inconsistent_shapes():
    with pytest.raises(vv.InvalidInput):
        vv.Prediction.from_json(
            '{"k":2,"probabilities":[1.0],"most_likely":[0],"entropy":[0.0],"corrected":0.0}'
        )
    with pytest.raises(vv.InvalidInput):
        vv.Realizations.from_json('{"n":1,"n_cells":2,"k":2,"codes":[0,5],"corrected":[0.0],"objective":[]}')
    with pytest.raises(vv.InvalidInput):
        vv.Sojourn.from_json(
            '{"n_beds":[1],"n_open":[0],"total":[1.0],"mean_length":[1.0],"thickness":[[1.0]],"open":[[]]}'
        )


def test_allocations_follow_the_process_data_limit():
    script = (
        "import resource, numpy as np, varvite as vv\n"
        "resource.setrlimit(resource.RLIMIT_DATA, (2 * 1024**3, resource.RLIM_INFINITY))\n"
        "for f in (lambda: vv.Grid([0, 0, 0], [1, 1, 1], [10**5, 10**5, 10**4]).coords,\n"
        "          lambda: vv.Grid([0, 0, 0], [1, 1, 1], [2000, 2000, 100]).coords,\n"
        "          lambda: vv.Gaussian(np.zeros(3000), np.ones(3000), covariance_type='diag').sample(10**6)):\n"
        "    try:\n"
        "        f()\n"
        "        raise SystemExit('no error')\n"
        "    except vv.InvalidInput as e:\n"
        "        assert 'memory budget' in str(e), e\n"
        "assert vv.Grid([0, 0, 0], [1, 1, 1], [10, 10, 10]).coords.shape == (1000, 3)\n"
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
        assert name in vv.__all__
    for name in ("plot", "datasets", "checks"):
        assert name not in vv.__all__
