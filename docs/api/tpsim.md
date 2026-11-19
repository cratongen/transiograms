# TP simulation

`Prediction`, `Realizations`, transition-probability cokriging and T-PROGS simulation. See the [guide](../guide/tp-simulation.md).

```python
class Prediction:
    @property
    def probabilities(self) -> NDArray[np.float64]: ...
    @property
    def most_likely(self) -> NDArray[np.int64]: ...
    @property
    def entropy(self) -> NDArray[np.float64]: ...
    @property
    def corrected(self) -> float: ...


class Realizations:
    @property
    def realizations(self) -> NDArray[np.int64]: ...
    @property
    def proportions(self) -> NDArray[np.float64]: ...
    @property
    def probabilities(self) -> NDArray[np.float64]: ...
    @property
    def most_likely(self) -> NDArray[np.int64]: ...
    @property
    def entropy(self) -> NDArray[np.float64]: ...
    @property
    def corrected(self) -> NDArray[np.float64]: ...
    @property
    def objective(self) -> list[NDArray[np.float64]]: ...
    def __len__(self) -> int: ...


def tp_cokriging(
    model: TransitionModel,
    grid: Grid,
    coords: ArrayLike | str,
    codes: ArrayLike | str,
    *,
    mask: ArrayLike | None = None,
    max_data: int = 12,
    closeness_limit: float = 0.01,
    max_per_octant: int = 0,
    svd_cutoff: float = 1e-8,
    memory: int | None = None,
    data: Mapping[str, ArrayLike] | None = None,
) -> Prediction: ...
def tprogs_simulate(
    model: TransitionModel,
    grid: Grid,
    coords: ArrayLike | str,
    codes: ArrayLike | str,
    *,
    n: int = 1,
    seed: int = 0,
    mask: ArrayLike | None = None,
    max_data: int = 12,
    closeness_limit: float = 0.01,
    max_per_octant: int = 0,
    svd_cutoff: float = 1e-8,
    quench: bool = True,
    quench_iterations: int = 3,
    quench_limit: float = 0.4,
    quench_tol: float = 1e-3,
    proportion_weight: float = 1.0,
    memory: int | None = None,
    data: Mapping[str, ArrayLike] | None = None,
) -> Realizations: ...
```
