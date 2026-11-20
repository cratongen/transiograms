"""Figures for transiograms results (`tg.plot`). Every function takes `ax=None` and returns `(fig, ax)`."""

from transiograms.errors import InvalidInput


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "tg.plot needs matplotlib: pip install 'transiograms[plot]' or conda install -c conda-forge matplotlib"
        ) from e
    return plt


_PALETTE = ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7", "#999999"]


def _names_colors(categories, k, *, exact=False):
    if categories is not None and (len(categories) < k or (exact and len(categories) != k)):
        need = f"{k}" if exact else f"at least {k}"
        raise InvalidInput(f"categories has {len(categories)} names but the data needs {need}")
    names = getattr(categories, "names", None) or [str(i) for i in range(k)]
    colors = getattr(categories, "colors", None) or [_PALETTE[i % len(_PALETTE)] for i in range(k)]
    return list(names), list(colors)


def _axes(ax=None):
    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.figure, ax


# >>> chain
def transition_matrix(matrix, *, categories=None, annotate=True, ax=None, **kwargs):
    """Heat map of a transition matrix, rows "from" and columns "to".

    Parameters
    ----------
    matrix : array_like or MarkovChain
        ``(k, k)`` matrix, or a chain whose ``transition_matrix`` is drawn.
    categories : Categories, optional
        Names for the axes.
    annotate : bool
        Write each probability in its cell.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to ``imshow``.

    Returns
    -------
    (fig, ax)

    Raises
    ------
    ImportError
        When matplotlib is not installed.
    InvalidInput
        When ``categories`` has another number of names than the data needs.

    Examples
    --------
    >>> chain = tg.MarkovChain.from_matrix([[0.9, 0.1], [0.2, 0.8]])
    >>> fig, ax = tg.plot.transition_matrix(chain, categories=tg.Categories(["sand", "clay"]))
    >>> [t.get_text() for t in ax.get_xticklabels()]
    ['sand', 'clay']
    """
    import numpy as np

    m = np.asarray(getattr(matrix, "transition_matrix", matrix), dtype=float)
    fig, ax = _axes(ax)
    names, _ = _names_colors(categories, len(m), exact=True)
    kwargs.setdefault("cmap", "Blues")
    kwargs.setdefault("vmin", 0.0)
    kwargs.setdefault("vmax", 1.0)
    image = ax.imshow(m, **kwargs)
    ax.set_xticks(range(len(m)), names)
    ax.set_yticks(range(len(m)), names)
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    if annotate:
        for i in range(len(m)):
            for j in range(len(m)):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if m[i, j] > 0.6 else "black",
                )
    fig.colorbar(image, ax=ax, shrink=0.8)
    return fig, ax


def sojourn(sojourn, code, *, exponential=True, ax=None, **kwargs):
    """Kaplan-Meier survival of bed thickness for one category, open beds censored.

    Parameters
    ----------
    sojourn : Sojourn
    code : int
    exponential : bool
        Overlay ``exp(-t / mean_length)``, the survival a Markov chain implies.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to the survival ``step`` call.

    Returns
    -------
    (fig, ax)

    Raises
    ------
    ImportError
        When matplotlib is not installed.
    InvalidInput
        When ``code`` is not a category.

    Examples
    --------
    >>> d = tg.datasets.varved_sequence()
    >>> s = tg.sojourn(d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
    >>> fig, ax = tg.plot.sojourn(s, 0)
    >>> ax.get_ylabel()
    'P(thickness > t)'
    """
    import numpy as np

    fig, ax = _axes(ax)
    t, s = sojourn.survival(code)
    ax.step(np.r_[0.0, t], np.r_[1.0, s], where="post", label="Kaplan-Meier", **kwargs)
    if exponential and np.isfinite(sojourn.mean_length[code]):
        h = np.linspace(0.0, max(t.max() if t.size else 1.0, 1e-9), 200)
        ax.plot(h, np.exp(-h / sojourn.mean_length[code]), ls="--", label="exponential")
    ax.set_xlabel("thickness")
    ax.set_ylabel("P(thickness > t)")
    ax.legend()
    return fig, ax


# <<< chain


# >>> hmm
def strip_log(codes, *, depth=None, from_=None, to=None, posterior=None, categories=None, ax=None, **kwargs):
    """A categorical log as coloured intervals, depth increasing downward.

    Parameters
    ----------
    codes : array_like
        Category per row, -1 missing.
    depth : array_like, optional
        Sample depths; intervals span half the spacing either side.
    from_, to : array_like, optional
        Interval tops and bottoms, instead of ``depth``.
    posterior : array_like, optional
        ``(rows, K)`` probabilities, drawn as stacked bands in a second panel.
    categories : Categories, optional
    ax : matplotlib.axes.Axes or sequence of two, optional
    **kwargs
        Passed to ``barh``.

    Returns
    -------
    (fig, ax) or (fig, axes)
    Raises
    ------
    ImportError
        When matplotlib is not installed.
    InvalidInput
        When ``categories`` has fewer names than the codes need.

    Examples
    --------
    >>> fig, axes = tg.plot.strip_log([0, 0, 1, 2], depth=[0.5, 1.5, 2.5, 3.5], posterior=np.eye(3)[[0, 0, 1, 2]])
    >>> len(axes)
    2
    """
    import numpy as np

    plt = _pyplot()
    codes = np.asarray(codes)
    n = len(codes)
    if from_ is not None and to is not None:
        top, bottom = np.asarray(from_, float), np.asarray(to, float)
    else:
        z = np.arange(n, dtype=float) if depth is None else np.asarray(depth, float)
        half = np.diff(z) / 2 if n > 1 else np.array([0.5])
        top = z - np.r_[half[:1], half]
        bottom = z + np.r_[half, half[-1:]]
    k = int(max(codes.max() + 1, 0 if posterior is None else np.shape(posterior)[1]))
    names, colors = _names_colors(categories, k)
    if ax is None:
        width = (3.0, 1.0) if posterior is not None else (1.0,)
        fig, axes = plt.subplots(1, len(width), figsize=(1.6 * len(width), 6), sharey=True, squeeze=False)
        axes = list(axes[0])
    else:
        axes = list(ax) if np.iterable(ax) else [ax]
        fig = axes[0].figure
    first = axes[0]
    for c in range(k):
        at = codes == c
        first.barh(
            top[at], 1.0, height=(bottom - top)[at], align="edge", color=colors[c], label=names[c], **kwargs
        )
    first.set_xticks([])
    first.set_ylim(bottom.max(), top.min())
    first.set_ylabel("depth")
    first.legend(loc="upper left", bbox_to_anchor=(0, -0.02), fontsize="small", frameon=False)
    if posterior is not None:
        p = np.asarray(posterior, float)
        mid = (top + bottom) / 2
        left = np.zeros(len(p))
        for c in range(p.shape[1]):
            axes[1].fill_betweenx(mid, left, left + p[:, c], color=colors[c], step="mid", lw=0)
            left = left + p[:, c]
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("posterior")
        return fig, axes
    return fig, first


# <<< hmm


# >>> hsmm
def selection(table, *, criteria=("aic", "bic", "icl"), ax=None, **kwargs):
    """Information criteria against the number of states, from ``select_states``.

    Parameters
    ----------
    table : dict
        The ``select_states`` result.
    criteria : sequence of str
        Keys to draw; the minimum of each is marked.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to ``plot``.

    Returns
    -------
    (fig, ax)

    Raises
    ------
    ImportError
        When matplotlib is not installed.
    KeyError
        When a criterion is not a key of ``table``.

    Examples
    --------
    >>> table = {"states": [1, 2, 3], "aic": [30.0, 20.0, 25.0], "bic": [31.0, 24.0, 33.0]}
    >>> fig, ax = tg.plot.selection(table, criteria=("aic", "bic"))
    >>> [t.get_text() for t in ax.get_legend().get_texts()]
    ['AIC', 'BIC']
    """
    import numpy as np

    fig, ax = _axes(ax)
    k = np.asarray(table["states"])
    for name in criteria:
        v = np.asarray(table[name], float)
        (line,) = ax.plot(k, v, marker="o", label=name.upper(), **kwargs)
        best = int(np.nanargmin(v))
        ax.plot(k[best], v[best], marker="*", ms=14, color=line.get_color())
    ax.set_xlabel("number of states")
    ax.set_ylabel("criterion (lower is better)")
    ax.set_xticks(k)
    ax.legend()
    return fig, ax


# <<< hsmm


# >>> transiogram
def transiogram(experimental=None, model=None, *, h=None, categories=None, axes=None, **kwargs):
    """Grid of auto- (diagonal) and cross-transiograms.

    Parameters
    ----------
    experimental : ExperimentalTransiogram, optional
        Drawn as points at the mean lag of each class.
    model : TransiogramModel or TransitionModel, optional
        Drawn as curves; a ``TransitionModel`` along +z unless ``h`` is ``(n, 3)``.
    h : array_like, optional
        Lags of the curves; up to the largest experimental lag by default.
    categories : Categories, optional
    axes : array of matplotlib.axes.Axes, optional
        ``(k, k)``.
    **kwargs
        Passed to the model ``plot`` calls.

    Returns
    -------
    (fig, axes)
    Raises
    ------
    ImportError
        When matplotlib is not installed.
    InvalidInput
        When neither ``experimental`` nor ``model`` is given, or ``categories`` has another
        number of names than the data needs.

    Examples
    --------
    >>> m = tg.TransiogramModel.markov([[-0.5, 0.5], [1.0, -1.0]])
    >>> fig, axes = tg.plot.transiogram(model=m, h=np.linspace(0.0, 5.0, 20))
    >>> axes.shape
    (2, 2)
    """
    import numpy as np

    plt = _pyplot()
    if experimental is None and model is None:
        raise InvalidInput("give experimental, model or both")
    k = experimental.probabilities.shape[1] if experimental is not None else len(model.proportions)
    names, colors = _names_colors(categories, k, exact=True)
    if axes is None:
        fig, axes = plt.subplots(k, k, figsize=(2.4 * k, 2.0 * k), sharex=True, sharey=True, squeeze=False)
    else:
        axes = np.asarray(axes)
        fig = axes.flat[0].figure
    if h is None:
        top = experimental.lags.max() if experimental is not None else 10.0
        h = np.linspace(0.0, top, 200)
    h = np.asarray(h, float)
    curves = None
    if model is not None:
        if h.ndim == 2:
            curves, dist = model.transition(h), np.linalg.norm(h, axis=1)
        elif hasattr(model, "rates_z"):
            curves, dist = model.transition(np.outer(h, [0.0, 0.0, 1.0])), h
        else:
            curves, dist = model.transition(h), h
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if experimental is not None:
                x = np.where(np.isnan(experimental.mean_lags), experimental.lags, experimental.mean_lags)
                ax.plot(x, experimental.probabilities[:, i, j], "o", ms=3, color=colors[j])
            if curves is not None:
                ax.plot(dist, curves[:, i, j], color="black", lw=1.0, **kwargs)
            ax.set_ylim(-0.02, 1.02)
            if i == k - 1:
                ax.set_xlabel(f"lag, to {names[j]}")
            if j == 0:
                ax.set_ylabel(f"from {names[i]}")
    return fig, axes


# <<< transiogram


# >>> tpsim
def grid_slice(values, grid, *, axis="z", index=0, categories=None, ax=None, **kwargs):
    """One slice of a grid property.

    Parameters
    ----------
    values : array_like
        One value per cell, x fastest (a realization, ``most_likely``, an entropy).
    grid : Grid
    axis : {"x", "y", "z"}
        Axis normal to the slice.
    index : int
        Cell index along ``axis``.
    categories : Categories, optional
        Draw integer codes with category colours (-1 blank).
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to ``imshow``.

    Returns
    -------
    (fig, ax)

    Raises
    ------
    ImportError
        When matplotlib is not installed.
    InvalidInput
        When ``axis`` is not "x", "y" or "z", or ``categories`` has fewer names than the
        codes need.

    Examples
    --------
    >>> grid = tg.Grid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 3, 2))
    >>> fig, ax = tg.plot.grid_slice(np.arange(24) % 3, grid, axis="z", index=1)
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('x', 'y')
    """
    import numpy as np

    fig, ax = _axes(ax)
    cube = np.asarray(values).reshape(grid.shape)
    (ox, oy, oz), (sx, sy, sz), (nx, ny, nz) = grid.origin, grid.size, grid.count
    if axis == "z":
        image, extent, labels = cube[index], (ox, ox + nx * sx, oy, oy + ny * sy), ("x", "y")
    elif axis == "y":
        image, extent, labels = cube[:, index, :], (ox, ox + nx * sx, oz, oz + nz * sz), ("x", "z")
    elif axis == "x":
        image, extent, labels = cube[:, :, index], (oy, oy + ny * sy, oz, oz + nz * sz), ("y", "z")
    else:
        raise InvalidInput(f"axis must be 'x', 'y' or 'z', not {axis!r}")
    integer = np.issubdtype(image.dtype, np.integer)
    if categories is not None or integer:
        from matplotlib.colors import ListedColormap

        k = int(max(image.max() + 1, len(categories) if categories is not None else 0, 1))
        _names, colors = _names_colors(categories, k)
        image = np.ma.masked_less(image, 0)
        kwargs.setdefault("cmap", ListedColormap(colors[:k]))
        kwargs.setdefault("vmin", -0.5)
        kwargs.setdefault("vmax", k - 0.5)
        kwargs.setdefault("interpolation", "nearest")
    kwargs.setdefault("aspect", "auto")
    shown = ax.imshow(image, origin="lower", extent=extent, **kwargs)
    if categories is None and not integer:
        fig.colorbar(shown, ax=ax, shrink=0.8)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    return fig, ax


# <<< tpsim

# >>> mcrf
# <<< mcrf
