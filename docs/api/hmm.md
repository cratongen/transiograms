# HMM

Emission distributions and `HiddenMarkovModel`. `Sequences` is a list of arrays (one per hole) or one array with `holes=`; `Rows` is a list of arrays or one array, matching the input. See the [guide](../guide/hidden-markov.md).

```python
Sequences: TypeAlias = ArrayLike | list[ArrayLike] | str
Rows: TypeAlias = NDArray | list[NDArray]


class Categorical:
    def __init__(
        self, probabilities: ArrayLike, *, pseudocount: float = 0.0, frozen: bool = False
    ) -> None: ...
    @property
    def probabilities(self) -> NDArray[np.float64]: ...
    @property
    def pseudocount(self) -> float: ...
    def log_density(self, x: ArrayLike) -> NDArray[np.float64]: ...
    def sample(self, n: int, *, seed: int = 0) -> NDArray: ...
    @property
    def frozen(self) -> bool: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Categorical: ...


class Gaussian:
    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike | float,
        *,
        covariance_type: Literal["full", "diag", "spherical"] = "full",
        frozen: bool = False,
    ) -> None: ...
    @property
    def mean(self) -> NDArray[np.float64]: ...
    @property
    def covariance(self) -> NDArray[np.float64] | float: ...
    @property
    def covariance_type(self) -> str: ...
    def log_density(self, x: ArrayLike) -> NDArray[np.float64]: ...
    def sample(self, n: int, *, seed: int = 0) -> NDArray: ...
    @property
    def frozen(self) -> bool: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Gaussian: ...


class Poisson:
    def __init__(self, rate: ArrayLike | float, *, frozen: bool = False) -> None: ...
    @property
    def rate(self) -> NDArray[np.float64]: ...
    def log_density(self, x: ArrayLike) -> NDArray[np.float64]: ...
    def sample(self, n: int, *, seed: int = 0) -> NDArray: ...
    @property
    def frozen(self) -> bool: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Poisson: ...


class GaussianMixture:
    def __init__(
        self,
        weights: ArrayLike,
        means: ArrayLike,
        covariances: ArrayLike,
        *,
        covariance_type: Literal["full", "diag", "spherical"] = "full",
        frozen: bool = False,
    ) -> None: ...
    @property
    def weights(self) -> NDArray[np.float64]: ...
    @property
    def means(self) -> NDArray[np.float64]: ...
    @property
    def covariances(self) -> NDArray[np.float64]: ...
    @property
    def covariance_type(self) -> str: ...
    def log_density(self, x: ArrayLike) -> NDArray[np.float64]: ...
    def sample(self, n: int, *, seed: int = 0) -> NDArray: ...
    @property
    def frozen(self) -> bool: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> GaussianMixture: ...


Distribution: TypeAlias = Categorical | Gaussian | Poisson | GaussianMixture


class HiddenMarkovModel:
    def __init__(
        self,
        distributions: int | Sequence[Distribution],
        *,
        start: ArrayLike | None = None,
        transition: ArrayLike | None = None,
        emission: Literal["gaussian", "categorical", "poisson", "mixture"] = "gaussian",
        covariance_type: Literal["full", "diag", "spherical"] = "diag",
        n_symbols: int | None = None,
        n_mix: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_init: int = 1,
        seed: int = 0,
        params: str = "ste",
        init_params: str | None = None,
        start_prior: float = 1.0,
        transition_prior: float = 1.0,
        covariance_prior: float = 1e-6,
        pseudocount: float = 0.0,
        direction: Literal["up", "down"] = "up",
        implementation: Literal["log", "scaled"] = "log",
        ordering: Literal["none", "mean", "occupancy"] = "none",
    ) -> None: ...
    @property
    def direction(self) -> str: ...
    def fit(
        self,
        sequences: Sequences,
        *,
        holes: ArrayLike | str | None = None,
        labels: ArrayLike | str | None = None,
        allowed: ArrayLike | None = None,
        data: Mapping[str, ArrayLike] | None = None,
    ) -> HiddenMarkovModel: ...
    def score(
        self,
        sequences: Sequences,
        *,
        holes: ArrayLike | str | None = None,
        data: Mapping[str, ArrayLike] | None = None,
    ) -> float: ...
    def score_samples(
        self,
        sequences: Sequences,
        *,
        holes: ArrayLike | str | None = None,
        data: Mapping[str, ArrayLike] | None = None,
    ) -> NDArray[np.float64]: ...
    def predict(
        self,
        sequences: Sequences,
        *,
        holes: ArrayLike | str | None = None,
        labels: ArrayLike | str | None = None,
        allowed: ArrayLike | None = None,
        data: Mapping[str, ArrayLike] | None = None,
        algorithm: Literal["viterbi", "map", "posterior_viterbi"] = "viterbi",
    ) -> Rows: ...
    def predict_proba(
        self,
        sequences: Sequences,
        *,
        holes: ArrayLike | str | None = None,
        labels: ArrayLike | str | None = None,
        allowed: ArrayLike | None = None,
        data: Mapping[str, ArrayLike] | None = None,
    ) -> Rows: ...
    def viterbi(
        self,
        sequences: Sequences,
        *,
        holes: ArrayLike | str | None = None,
        labels: ArrayLike | str | None = None,
        allowed: ArrayLike | None = None,
        data: Mapping[str, ArrayLike] | None = None,
    ) -> tuple[Rows, NDArray[np.float64]]: ...
    def sample(
        self, lengths: int | Sequence[int], *, seed: int = 0
    ) -> tuple[list[NDArray[np.int64]], list[NDArray]]: ...
    def sample_posterior(
        self,
        sequences: Sequences,
        *,
        n: int = 100,
        seed: int = 0,
        holes: ArrayLike | str | None = None,
        labels: ArrayLike | str | None = None,
        allowed: ArrayLike | None = None,
        data: Mapping[str, ArrayLike] | None = None,
    ) -> NDArray[np.int64]: ...
    def aic(
        self,
        sequences: Sequences,
        *,
        holes: ArrayLike | str | None = None,
        data: Mapping[str, ArrayLike] | None = None,
    ) -> float: ...
    def bic(
        self,
        sequences: Sequences,
        *,
        holes: ArrayLike | str | None = None,
        data: Mapping[str, ArrayLike] | None = None,
    ) -> float: ...
    def icl(
        self,
        sequences: Sequences,
        *,
        holes: ArrayLike | str | None = None,
        data: Mapping[str, ArrayLike] | None = None,
    ) -> float: ...
    @property
    def n_states(self) -> int: ...
    @property
    def start(self) -> NDArray[np.float64]: ...
    @property
    def transition(self) -> NDArray[np.float64]: ...
    @property
    def distributions(self) -> list[Distribution]: ...
    @property
    def history(self) -> NDArray[np.float64]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def n_iter(self) -> int: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> HiddenMarkovModel: ...
```
