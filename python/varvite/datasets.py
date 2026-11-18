"""Seeded synthetic datasets (`vv.datasets`). Each function takes `seed=0` and returns a dict of NumPy
arrays and model JSON strings."""

import numpy as np


# >>> chain
def varved_sequence(seed=0):
    """Fining-upward cycles of sand, silt and clay in 12 vertical holes, 30 m each.

    Beds come from a continuous-lag Markov chain run upward (older to younger) whose embedded
    matrix favours sand to silt to clay to sand, with mean thicknesses 1.2, 0.8 and 0.5 m.

    Parameters
    ----------
    seed : int

    Returns
    -------
    dict
        ``HOLE_ID`` (str), ``FROM`` and ``TO`` (depth, top-down), ``CODE`` (0 sand, 1 silt,
        2 clay), ``names`` and ``generator`` (the true ``MarkovChain`` as JSON).
    """
    import varvite as vv

    embedded = np.array([[0.0, 0.85, 0.15], [0.1, 0.0, 0.9], [0.8, 0.2, 0.0]])
    chain = vv.MarkovChain.from_matrix(embedded, kind="embedded", mean_lengths=[1.2, 0.8, 0.5])
    length = 30.0
    beds = chain.simulate(length, n=12, seed=seed)
    holes, top, bottom, codes = [], [], [], []
    for r in range(12):
        at = beds["realization"] == r
        holes.append(np.full(at.sum(), f"DH{r + 1:02d}"))
        top.append((length - beds["to"][at])[::-1])
        bottom.append((length - beds["from"][at])[::-1])
        codes.append(beds["code"][at][::-1])
    return {
        "HOLE_ID": np.concatenate(holes),
        "FROM": np.concatenate(top),
        "TO": np.concatenate(bottom),
        "CODE": np.concatenate(codes),
        "names": ["sand", "silt", "clay"],
        "generator": chain.to_json(),
    }


# <<< chain


# >>> hmm
def electrofacies_logs(seed=0):
    """Gamma-ray and density logs of 20 holes from a 3-state Gaussian HMM.

    Each hole has 300 samples 0.1 m apart. The hidden states (shale, silty sand, clean sand) are
    sticky (0.95 on the diagonal) and well separated in both channels.

    Parameters
    ----------
    seed : int

    Returns
    -------
    dict
        ``HOLE_ID``, ``DEPTH``, ``GR`` (API), ``RHOB`` (g/cm3) and ``STATE`` per row, top-down,
        and ``model``: the true ``HiddenMarkovModel`` (on the scaled channels) as JSON.
    """
    import varvite as vv

    scale, shift = np.array([20.0, -0.08]), np.array([40.0, 2.65])
    means = [[0.0, 0.0], [3.0, 2.5], [6.0, 0.0]]
    truth = vv.HiddenMarkovModel(
        [vv.Gaussian(m, [0.5, 0.5], covariance_type="diag") for m in means],
        start=[0.4, 0.35, 0.25],
        transition=[[0.95, 0.03, 0.02], [0.02, 0.95, 0.03], [0.03, 0.02, 0.95]],
        direction="down",
    )
    states, obs = truth.sample([300] * 20, seed=seed)
    x = np.concatenate(obs) * scale + shift
    scaled = vv.HiddenMarkovModel(
        [vv.Gaussian(np.multiply(m, scale) + shift, 0.5 * scale**2, covariance_type="diag") for m in means],
        start=truth.start,
        transition=truth.transition,
        direction="down",
    )
    return {
        "HOLE_ID": np.repeat([f"W{h + 1:02d}" for h in range(20)], 300),
        "DEPTH": np.tile(0.05 + 0.1 * np.arange(300), 20),
        "GR": x[:, 0],
        "RHOB": x[:, 1],
        "STATE": np.concatenate(states),
        "model": scaled.to_json(),
    }


# <<< hmm

# >>> hsmm
# <<< hsmm


# >>> transiogram
def layered_drillholes(seed=0):
    """Vertical drillholes through a seeded realization of a known 3D transition model.

    Three facies (proportions 0.5, 0.3, 0.2; mean lengths 60, 40, 30 m along x and y, 6, 4, 3 m
    along z) are simulated by MCRF on a 32 x 32 x 30 grid of 10 x 10 x 1 m cells, then sampled
    by 25 vertical holes on a regular pattern at 1 m composites.

    Parameters
    ----------
    seed : int

    Returns
    -------
    dict
        ``X``, ``Y``, ``Z`` (composite midpoints), ``HOLE_ID``, ``CODE``; ``FROM`` and ``TO``
        (depth below the grid top); ``field`` (the realization, one code per cell, x fastest);
        ``model`` (``TransitionModel`` JSON) and ``grid`` (``Grid`` JSON).
    """
    import varvite as vv

    model = vv.TransitionModel.from_geology(
        [0.5, 0.3, 0.2], [[60.0, 40.0, 30.0], [60.0, 40.0, 30.0], [6.0, 4.0, 3.0]]
    )
    grid = vv.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (32, 32, 30))
    field = vv.mcrf_simulate(model, grid, np.empty((0, 3)), np.empty(0), seed=seed).realizations[0]
    cube = field.reshape(grid.shape)
    columns = [(i, j) for j in range(3, 32, 6) for i in range(3, 32, 6)]
    nz = grid.count[2]
    x, y, z, hole, code = [], [], [], [], []
    for h, (i, j) in enumerate(columns):
        x.append(np.full(nz, 5.0 + 10.0 * i))
        y.append(np.full(nz, 5.0 + 10.0 * j))
        z.append(0.5 + np.arange(nz)[::-1])
        hole.append(np.full(nz, f"H{h + 1:02d}"))
        code.append(cube[::-1, j, i])
    z = np.concatenate(z)
    return {
        "X": np.concatenate(x),
        "Y": np.concatenate(y),
        "Z": z,
        "FROM": nz - z - 0.5,
        "TO": nz - z + 0.5,
        "HOLE_ID": np.concatenate(hole),
        "CODE": np.concatenate(code),
        "field": field,
        "model": model.to_json(),
        "grid": grid.to_json(),
    }


# <<< transiogram

# >>> tpsim
# <<< tpsim

# >>> mcrf
# <<< mcrf
