"""Figures for varvite results (`vv.plot`). Every function takes `ax=None` and returns `(fig, ax)`."""


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "vv.plot needs matplotlib: pip install 'varvite[plot]' or conda install -c conda-forge matplotlib"
        ) from e
    return plt


def _axes(ax=None):
    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.figure, ax


# >>> chain
# <<< chain

# >>> hmm
# <<< hmm

# >>> hsmm
# <<< hsmm

# >>> transiogram
# <<< transiogram

# >>> tpsim
# <<< tpsim

# >>> mcrf
# <<< mcrf
