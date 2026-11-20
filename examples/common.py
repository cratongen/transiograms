"""Figure style that render.py applies to every example; the scripts do not import it."""

import matplotlib as mpl

INK = "#1a1a1a"
GRAY = "#8c8c8c"
ACCENT = "#0072b2"
HIGHLIGHT = "#d55e00"
SERIES = [ACCENT, GRAY, HIGHLIGHT, "#009e73", "#e69f00", "#cc79a7", "#56b4e9"]

mpl.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": GRAY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "legend.frameon": False,
        "image.cmap": "cividis",
    }
)
