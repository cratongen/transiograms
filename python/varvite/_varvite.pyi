from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: str

# >>> core
class Categories:
    def __init__(self, names: Sequence[str], *, colors: Sequence[str] | None = None) -> None: ...
    @staticmethod
    def from_labels(labels: ArrayLike) -> Categories: ...
    @property
    def names(self) -> list[str]: ...
    @property
    def colors(self) -> list[str] | None: ...
    def encode(self, labels: ArrayLike) -> NDArray[np.int64]: ...
    def decode(self, codes: ArrayLike) -> NDArray[np.object_]: ...
    def __len__(self) -> int: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Categories: ...

class Grid:
    def __init__(self, origin: Sequence[float], size: Sequence[float], count: Sequence[int]) -> None: ...
    @property
    def origin(self) -> tuple[float, float, float]: ...
    @property
    def size(self) -> tuple[float, float, float]: ...
    @property
    def count(self) -> tuple[int, int, int]: ...
    @property
    def n_cells(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...
    @property
    def coords(self) -> NDArray[np.float64]: ...
    def snap(
        self, coords: ArrayLike, codes: ArrayLike, *, mask: ArrayLike | None = None
    ) -> tuple[NDArray[np.uint64], NDArray[np.int64]]: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Grid: ...

def _sequence_runs(
    codes: ArrayLike | str,
    *,
    holes: ArrayLike | str | None = None,
    from_: ArrayLike | str | None = None,
    to: ArrayLike | str | None = None,
    depth: ArrayLike | str | None = None,
    step: float | None = None,
    gap: float = 1e-6,
    categories: Categories | int | None = None,
    data: Mapping[str, ArrayLike] | None = None,
    direction: str = "up",
) -> list[tuple[int, list[int], list[int]]]: ...
def _sequences(
    sequences: ArrayLike | list[ArrayLike] | str,
    *,
    holes: ArrayLike | str | None = None,
    data: Mapping[str, ArrayLike] | None = None,
    direction: str = "up",
) -> tuple[list[tuple[int, int, list[float], list[int]]], int]: ...

# <<< core

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
