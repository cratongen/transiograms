"""Release audit: the public estimators and simulators give byte-identical results on 1 and 8
Rayon threads and on a repeated run with the same seed."""

import os
import subprocess
import sys

SCRIPT = r"""
import hashlib
import numpy as np
import transiograms as tg

h = hashlib.sha256()
def add(*arrays):
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes())

src = tg.datasets.layered_drillholes(seed=0)
rng = np.random.default_rng(7)
d = {k: src[k] for k in ("HOLE_ID", "FROM", "TO", "X", "Y", "Z", "CODE")}
d["GR"] = np.array([30.0, 75.0, 120.0])[src["CODE"]] + rng.normal(0.0, 12.0, len(src["CODE"]))

hmm = tg.HiddenMarkovModel(3, n_init=4, seed=3, direction="down", ordering="mean", max_iter=30)
hmm.fit("GR", holes="HOLE_ID", data=d)
add(hmm.transition, hmm.history, hmm.predict_proba("GR", holes="HOLE_ID", data=d))
add(hmm.sample_posterior("GR", holes="HOLE_ID", data=d, n=5, seed=2))

hsmm = tg.HiddenSemiMarkovModel(3, duration="poisson", n_init=3, seed=4, max_iter=15,
                                max_duration=30, direction="down")
hsmm.fit("GR", holes="HOLE_ID", data=d)
add(hsmm.transition, hsmm.history, hsmm.predict_proba("GR", holes="HOLE_ID", data=d))
add(hsmm.sample_posterior("GR", holes="HOLE_ID", data=d, n=5, seed=2))

chain = tg.MarkovChain("fixed", direction="down")
chain.fit("CODE", holes="HOLE_ID", from_="FROM", to="TO", step=1.0, data=d)
add(chain.transition_matrix)
add(chain.simulate(50, n=20, seed=5))

e = tg.downhole_transiogram("CODE", holes="HOLE_ID", from_="FROM", to="TO",
                            lags=np.arange(1.0, 9.0), data=d)
add(e.probabilities)
fit = tg.TransiogramModel.fit_markov(e)
add(fit.rates)
model = tg.TransitionModel(fit.proportions, 0.1 * fit.rates, 0.1 * fit.rates, fit.rates)
grid = tg.Grid.from_json(src["grid"])
xyz = np.column_stack([d["X"], d["Y"], d["Z"]])
for sim in (tg.tprogs_simulate, tg.mcrf_simulate):
    r = sim(model, grid, xyz, "CODE", n=3, seed=6, data=d)
    add(r.realizations, r.probabilities)
print(h.hexdigest())
"""


def _digest(threads):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    out = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip()


def test_results_are_identical_on_1_and_8_threads_and_repeated_runs():
    one = _digest(1)
    assert one == _digest(8)
    assert one == _digest(8)
