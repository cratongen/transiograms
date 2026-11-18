"""Markov models for geostatistics: chains, hidden Markov models, transiograms and grid simulation."""

import copyreg as _copyreg

from varvite import _varvite, checks, datasets, plot  # noqa: F401
from varvite._varvite import *
from varvite._varvite import __version__  # noqa: F401
from varvite.errors import ConvergenceError, InvalidInput, VarviteError  # noqa: F401

__all__ = [
    name
    for name in dir()
    if not name.startswith("_") and name not in ("checks", "datasets", "errors", "plot")
]


def _from_json(cls, text):
    return cls.from_json(text)


def _reduce(obj):
    return _from_json, (type(obj), obj.to_json())


for _cls in vars(_varvite).values():
    if isinstance(_cls, type) and hasattr(_cls, "to_json") and hasattr(_cls, "from_json"):
        _copyreg.pickle(_cls, _reduce)
