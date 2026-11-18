"""Figure style shared by the examples."""

import inspect
from pathlib import Path

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

SAVED: list[Path] = []


def save(fig, name: str) -> Path:
    """Saves `fig` as `name.png` next to the calling script and records it for render.py."""
    caller = Path(inspect.stack()[1].filename).resolve().parent
    path = caller / f"{name}.png"
    fig.savefig(path)
    SAVED.append(path)
    return path
