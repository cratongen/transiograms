import pickle

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
