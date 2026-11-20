# Transiograms

Experimental transiograms, the one-directional `TransiogramModel`, the 3D `TransitionModel`, and the `fit_quality` report that compares simulated realizations with their model. See the [guide](../guide/transiograms.md); the report is described in the [T-PROGS](../guide/tp-simulation.md) and [MCP and MCRF](../guide/mcrf-mcp.md) guides.

```python
class ExperimentalTransiogram:
    @property
    def lags(self) -> NDArray[np.float64]: ...
    @property
    def mean_lags(self) -> NDArray[np.float64]: ...
    @property
    def counts(self) -> NDArray[np.float64]: ...
    @property
    def probabilities(self) -> NDArray[np.float64]: ...
    @property
    def tail_totals(self) -> NDArray[np.float64]: ...
    @property
    def directed(self) -> bool: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> ExperimentalTransiogram: ...


def experimental_transiogram(
    coords: ArrayLike | str,
    codes: ArrayLike | str,
    *,
    holes: ArrayLike | str | None = None,
    direction: tuple[float, float] | None = None,
    lag: float,
    n_lags: int,
    lag_tolerance: float | None = None,
    angle_tolerance: float = 22.5,
    bandwidth: float = ...,
    vertical_bandwidth: float = ...,
    directed: bool = True,
    pairs: Literal["all", "same_hole", "different_hole"] = "all",
    categories: Categories | int | None = None,
    data: Mapping[str, ArrayLike] | None = None,
) -> ExperimentalTransiogram: ...
def downhole_transiogram(
    codes: ArrayLike | str,
    *,
    lags: ArrayLike,
    direction: Literal["up", "down"] = "up",
    holes: ArrayLike | str | None = None,
    from_: ArrayLike | str | None = None,
    to: ArrayLike | str | None = None,
    depth: ArrayLike | str | None = None,
    step: float | None = None,
    gap: float = 1e-6,
    categories: Categories | int | None = None,
    data: Mapping[str, ArrayLike] | None = None,
) -> ExperimentalTransiogram: ...
def grid_transiogram(
    codes: ArrayLike,
    grid: Grid,
    *,
    offset: tuple[int, int, int],
    n_lags: int,
    mask: ArrayLike | None = None,
    categories: Categories | int | None = None,
) -> ExperimentalTransiogram: ...


class TransiogramModel:
    @staticmethod
    def markov(rates: ArrayLike | MarkovChain) -> TransiogramModel: ...
    @staticmethod
    def entrywise(
        proportions: ArrayLike,
        structures: str | ArrayLike,
        ranges: ArrayLike,
        *,
        nuggets: ArrayLike | None = None,
        complement: ArrayLike | None = None,
        max_lag: float,
    ) -> TransiogramModel: ...
    @staticmethod
    def fit_markov(
        experimental: ExperimentalTransiogram,
        *,
        proportions: ArrayLike | None = None,
        background: int = -1,
        start: ArrayLike | None = None,
    ) -> TransiogramModel: ...
    @staticmethod
    def fit_entrywise(
        experimental: ExperimentalTransiogram,
        proportions: ArrayLike,
        *,
        structures: str | Sequence[str] | None = None,
        complement: ArrayLike | None = None,
    ) -> TransiogramModel: ...
    @property
    def kind(self) -> str: ...
    @property
    def proportions(self) -> NDArray[np.float64]: ...
    @property
    def rates(self) -> NDArray[np.float64] | None: ...
    def transition(self, h: float | ArrayLike) -> NDArray[np.float64]: ...
    def residuals(self, experimental: ExperimentalTransiogram) -> NDArray[np.float64]: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> TransiogramModel: ...


class TransitionModel:
    def __init__(
        self,
        proportions: ArrayLike,
        rates_x: ArrayLike,
        rates_y: ArrayLike,
        rates_z: ArrayLike,
        *,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        background: int = -1,
    ) -> None: ...
    @staticmethod
    def from_geology(
        proportions: ArrayLike,
        mean_lengths: ArrayLike,
        *,
        embedded_x: ArrayLike | None = None,
        embedded_y: ArrayLike | None = None,
        embedded_z: ArrayLike | None = None,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        background: int = -1,
    ) -> TransitionModel: ...
    @property
    def proportions(self) -> NDArray[np.float64]: ...
    @property
    def rates_x(self) -> NDArray[np.float64]: ...
    @property
    def rates_y(self) -> NDArray[np.float64]: ...
    @property
    def rates_z(self) -> NDArray[np.float64]: ...
    @property
    def rotation(self) -> tuple[float, float, float]: ...
    @property
    def background(self) -> int: ...
    @property
    def n_categories(self) -> int: ...
    def transition(self, lags: ArrayLike) -> NDArray[np.float64]: ...
    def rates(self, lag: ArrayLike) -> NDArray[np.float64]: ...
    def closeness(self, lags: ArrayLike) -> NDArray[np.float64] | float: ...
    def mean_lengths(self, lag: ArrayLike) -> NDArray[np.float64]: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> TransitionModel: ...


class FitQuality:
    @property
    def target_proportions(self) -> NDArray[np.float64]: ...
    @property
    def proportions(self) -> NDArray[np.float64]: ...
    @property
    def mean_proportions(self) -> NDArray[np.float64]: ...
    @property
    def offsets(self) -> NDArray[np.int64]: ...
    @property
    def model(self) -> NDArray[np.float64]: ...
    @property
    def mean(self) -> NDArray[np.float64]: ...
    @property
    def standard_error(self) -> NDArray[np.float64]: ...
    @property
    def proportion_error(self) -> float: ...
    @property
    def transiogram_error(self) -> float: ...
    @property
    def max_standard_errors(self) -> float: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> FitQuality: ...


def fit_quality(
    model: TransitionModel,
    grid: Grid,
    realizations: Realizations,
    *,
    directions: Sequence[tuple[int, int, int]] | None = None,
    n_lags: int = 6,
    mask: ArrayLike | None = None,
) -> FitQuality: ...
```
