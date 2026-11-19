# HSMM

Duration laws, `HiddenSemiMarkovModel` and `select_states`. The methods not repeated here match `HiddenMarkovModel`. See the [guide](../guide/semi-markov.md).

```python
class Geometric:
    def __init__(self, p: float, *, frozen: bool = False) -> None: ...
    @property
    def p(self) -> float: ...
    def pmf(self, max_duration: int) -> NDArray[np.float64]: ...
    def mean(self, max_duration: int) -> float: ...
    @property
    def frozen(self) -> bool: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Geometric: ...

class ShiftedPoisson:
    def __init__(self, rate: float, *, frozen: bool = False) -> None: ...
    @property
    def rate(self) -> float: ...
    def pmf(self, max_duration: int) -> NDArray[np.float64]: ...
    def mean(self, max_duration: int) -> float: ...
    @property
    def frozen(self) -> bool: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> ShiftedPoisson: ...

class NegativeBinomial:
    def __init__(self, r: float, p: float, *, frozen: bool = False) -> None: ...
    @property
    def r(self) -> float: ...
    @property
    def p(self) -> float: ...
    def pmf(self, max_duration: int) -> NDArray[np.float64]: ...
    def mean(self, max_duration: int) -> float: ...
    @property
    def frozen(self) -> bool: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> NegativeBinomial: ...

class EmpiricalDuration:
    def __init__(self, pmf: ArrayLike, *, pseudocount: float = 0.0, frozen: bool = False) -> None: ...
    @property
    def pseudocount(self) -> float: ...
    def pmf(self, max_duration: int) -> NDArray[np.float64]: ...
    def mean(self, max_duration: int) -> float: ...
    @property
    def frozen(self) -> bool: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> EmpiricalDuration: ...

DurationLaw: TypeAlias = Geometric | ShiftedPoisson | NegativeBinomial | EmpiricalDuration

class HiddenSemiMarkovModel:
    def __init__(
        self,
        distributions: int | Sequence[Distribution],
        durations: Sequence[DurationLaw] | None = None,
        *,
        start: ArrayLike | None = None,
        transition: ArrayLike | None = None,
        emission: Literal["gaussian", "categorical", "poisson", "mixture"] = "gaussian",
        duration: Literal["geometric", "poisson", "negative_binomial", "empirical"] = "geometric",
        covariance_type: Literal["full", "diag", "spherical"] = "diag",
        n_symbols: int | None = None,
        n_mix: int = 1,
        max_duration: int | Sequence[int] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_init: int = 1,
        seed: int = 0,
        params: str = "sted",
        init_params: str | None = None,
        start_prior: float = 1.0,
        transition_prior: float = 1.0,
        covariance_prior: float = 1e-6,
        pseudocount: float = 0.0,
        direction: Literal["up", "down"] = "up",
        ordering: Literal["none", "mean", "occupancy"] = "none",
    ) -> None: ...
    @staticmethod
    def from_hmm(hmm: HiddenMarkovModel, *, max_duration: int) -> HiddenSemiMarkovModel: ...
    @property
    def durations(self) -> list[DurationLaw]: ...
    @property
    def max_duration(self) -> NDArray[np.int64]: ...
    @property
    def truncated_mass(self) -> NDArray[np.float64]: ...
    def fit(
        self,
        sequences: Sequences,
        *,
        holes: ArrayLike | str | None = None,
        labels: ArrayLike | str | None = None,
        allowed: ArrayLike | None = None,
        data: Mapping[str, ArrayLike] | None = None,
    ) -> HiddenSemiMarkovModel: ...
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
        algorithm: Literal["viterbi", "map"] = "viterbi",
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
    def from_json(text: str) -> HiddenSemiMarkovModel: ...

def select_states(
    sequences: Sequences,
    *,
    holes: ArrayLike | str | None = None,
    data: Mapping[str, ArrayLike] | None = None,
    labels: ArrayLike | str | None = None,
    allowed: ArrayLike | None = None,
    states: Sequence[int] = ...,
    model: Literal["hmm", "hsmm"] = "hmm",
    emission: Literal["gaussian", "categorical", "poisson", "mixture"] = "gaussian",
    covariance_type: Literal["full", "diag", "spherical"] = "diag",
    n_symbols: int | None = None,
    n_mix: int = 1,
    duration: Literal["geometric", "poisson", "negative_binomial", "empirical"] = "geometric",
    max_duration: int | None = None,
    cv_folds: int | None = None,
    n_init: int = 5,
    seed: int = 0,
    max_iter: int = 100,
    tol: float = 1e-6,
    direction: Literal["up", "down"] = "up",
) -> dict[str, NDArray]: ...
```
