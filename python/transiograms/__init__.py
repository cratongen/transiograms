"""Markov models of geological categories: chains, hidden Markov models, transiograms, grid simulation."""

import copyreg as _copyreg

from transiograms import _transiograms, checks, datasets, plot  # noqa: F401
from transiograms._transiograms import *
from transiograms._transiograms import __version__  # noqa: F401
from transiograms.errors import ConvergenceError, InvalidInput, TransiogramsError  # noqa: F401

__all__ = [
    name
    for name in dir()
    if not name.startswith("_") and name not in ("checks", "datasets", "errors", "plot")
]


def _from_json(cls, text):
    return cls.from_json(text)


def _reduce(obj):
    return _from_json, (type(obj), obj.to_json())


for _cls in vars(_transiograms).values():
    if isinstance(_cls, type) and hasattr(_cls, "to_json") and hasattr(_cls, "from_json"):
        _copyreg.pickle(_cls, _reduce)
