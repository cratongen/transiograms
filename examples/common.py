"""Figure style that render.py applies to every example; the scripts do not import it."""

import matplotlib as mpl

INK = "#000000"
GRAY = "#737373"
ACCENT = "#0072b2"
HIGHLIGHT = "#d55e00"
SERIES = [INK, GRAY, ACCENT, HIGHLIGHT, "#009e73", "#e69f00", "#cc79a7"]

mpl.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": GRAY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "legend.frameon": False,
        "image.cmap": "Greys",
    }
)
