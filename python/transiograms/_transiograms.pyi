from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: str

# >>> core
class Categories:
    def __init__(self, names: Sequence[str] | object, *, colors: Sequence[str] | None = None) -> None: ...
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
    def __minetypes__(self) -> dict[str, object]: ...

class Grid:
    def __init__(
        self,
        origin: Sequence[float] | object,
        size: Sequence[float] | None = None,
        count: Sequence[int] | None = None,
    ) -> None: ...
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
        self,
        coords: ArrayLike | object,
        codes: ArrayLike | str | None = None,
        *,
        mask: ArrayLike | None = None,
    ) -> tuple[NDArray[np.uint64], NDArray[np.int64]]: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Grid: ...
    def __minetypes__(self) -> dict[str, object]: ...

def _sequence_runs(
    codes: ArrayLike | str | object,
    *,
    holes: ArrayLike | str | None = None,
    from_: ArrayLike | str | None = None,
    to: ArrayLike | str | None = None,
    depth: ArrayLike | str | None = None,
    step: float | None = None,
    gap: float = 1e-6,
    categories: Categories | int | object | None = None,
    data: Mapping[str, ArrayLike] | object | None = None,
    direction: str = "up",
) -> list[tuple[int, list[int], list[int]]]: ...
def _sequences(
    sequences: ArrayLike | list[ArrayLike] | str | object,
    *,
    holes: ArrayLike | str | None = None,
    data: Mapping[str, ArrayLike] | object | None = None,
    direction: str = "up",
) -> tuple[list[tuple[int, int, list[float], list[int]]], int]: ...

# <<< core

# >>> chain
class TestResult:
    @property
    def name(self) -> str: ...
    @property
    def method(self) -> str: ...
    @property
    def statistic(self) -> float: ...
    @property
    def df(self) -> int: ...
    @property
    def p_value(self) -> float: ...
    @property
    def expected(self) -> NDArray[np.float64] | None: ...
    @property
    def small_expected(self) -> float: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> TestResult: ...

class Entropy:
    @property
    def post(self) -> NDArray[np.float64]: ...
    @property
    def pre(self) -> NDArray[np.float64]: ...
    @property
    def post_normalized(self) -> NDArray[np.float64]: ...
    @property
    def pre_normalized(self) -> NDArray[np.float64]: ...
    @property
    def chain(self) -> float: ...
    @property
    def max(self) -> float: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Entropy: ...

class MarkovChain:
    def __init__(
        self,
        kind: Literal["embedded", "fixed"] = "embedded",
        *,
        direction: Literal["up", "down"] = "up",
        step: float | None = None,
        self_transitions: bool = False,
        pseudocount: float = 0.0,
        on_empty_row: Literal["raise", "uniform"] = "raise",
    ) -> None: ...
    def fit(
        self,
        codes: ArrayLike | str | object,
        *,
        holes: ArrayLike | str | None = None,
        from_: ArrayLike | str | None = None,
        to: ArrayLike | str | None = None,
        depth: ArrayLike | str | None = None,
        step: float | None = None,
        gap: float = 1e-6,
        categories: Categories | int | object | None = None,
        data: Mapping[str, ArrayLike] | object | None = None,
    ) -> MarkovChain: ...
    @staticmethod
    def from_matrix(
        matrix: ArrayLike,
        *,
        kind: Literal["embedded", "fixed"] = "fixed",
        step: float = 1.0,
        mean_lengths: ArrayLike | None = None,
        direction: Literal["up", "down"] = "up",
    ) -> MarkovChain: ...
    @staticmethod
    def from_rates(rates: ArrayLike, *, direction: Literal["up", "down"] = "up") -> MarkovChain: ...
    @staticmethod
    def from_geology(
        proportions: ArrayLike,
        mean_lengths: ArrayLike,
        embedded: ArrayLike | None = None,
        *,
        background: int = -1,
        symmetric: bool = False,
        direction: Literal["up", "down"] = "up",
    ) -> MarkovChain: ...
    @property
    def kind(self) -> str: ...
    @property
    def direction(self) -> str: ...
    @property
    def step(self) -> float | None: ...
    @property
    def n_categories(self) -> int: ...
    @property
    def counts(self) -> NDArray[np.float64] | None: ...
    @property
    def transition_matrix(self) -> NDArray[np.float64]: ...
    @property
    def generator(self) -> NDArray[np.float64] | None: ...
    @property
    def proportions(self) -> NDArray[np.float64]: ...
    @property
    def mean_lengths(self) -> NDArray[np.float64] | None: ...
    @property
    def bed_frequencies(self) -> NDArray[np.float64]: ...
    @property
    def projection_distance(self) -> float | None: ...
    def transition_at(self, h: float | ArrayLike) -> NDArray[np.float64]: ...
    def independence_test(self, *, method: Literal["g", "pearson"] = "g") -> TestResult: ...
    def symmetry_test(self) -> TestResult: ...
    def entropy(self) -> Entropy: ...
    def difference_matrix(self) -> NDArray[np.float64]: ...
    def asymmetry(self) -> NDArray[np.float64]: ...
    def proportion_std(self, n_steps: int) -> NDArray[np.float64]: ...
    def simulate(
        self,
        length: float,
        *,
        n: int = 1,
        seed: int = 0,
        hard_from: ArrayLike | None = None,
        hard_to: ArrayLike | None = None,
        hard_codes: ArrayLike | None = None,
        start: int | Literal["stationary"] | None = None,
    ) -> NDArray[np.int64] | dict[str, NDArray]: ...
    def simulate_beds(
        self,
        n_beds: int,
        *,
        n: int = 1,
        seed: int = 0,
        hard_from: ArrayLike | None = None,
        hard_to: ArrayLike | None = None,
        hard_codes: ArrayLike | None = None,
        start: int | Literal["stationary"] | None = None,
    ) -> NDArray[np.int64]: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> MarkovChain: ...

class Sojourn:
    @property
    def n_beds(self) -> NDArray[np.int64]: ...
    @property
    def n_open(self) -> NDArray[np.int64]: ...
    @property
    def total(self) -> NDArray[np.float64]: ...
    @property
    def mean_length(self) -> NDArray[np.float64]: ...
    def thickness(self, code: int) -> tuple[NDArray[np.float64], NDArray[np.bool_]]: ...
    def survival(self, code: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def exponential_test(self, code: int, *, n_bootstrap: int = 999, seed: int = 0) -> TestResult: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Sojourn: ...

def order_test(
    codes: ArrayLike | str | object,
    *,
    order: int = 2,
    kind: Literal["embedded", "fixed"] = "fixed",
    direction: Literal["up", "down"] = "up",
    method: Literal["g", "pearson"] = "g",
    holes: ArrayLike | str | None = None,
    from_: ArrayLike | str | None = None,
    to: ArrayLike | str | None = None,
    depth: ArrayLike | str | None = None,
    step: float | None = None,
    gap: float = 1e-6,
    categories: Categories | int | object | None = None,
    data: Mapping[str, ArrayLike] | object | None = None,
) -> TestResult: ...
def select_order(
    codes: ArrayLike | str | object,
    *,
    max_order: int = 3,
    direction: Literal["up", "down"] = "up",
    holes: ArrayLike | str | None = None,
    from_: ArrayLike | str | None = None,
    to: ArrayLike | str | None = None,
    depth: ArrayLike | str | None = None,
    step: float | None = None,
    gap: float = 1e-6,
    categories: Categories | int | object | None = None,
    data: Mapping[str, ArrayLike] | object | None = None,
) -> dict[str, NDArray]: ...
def homogeneity_test(
    codes: ArrayLike | str | object,
    groups: ArrayLike | str,
    *,
    kind: Literal["embedded", "fixed"] = "embedded",
    direction: Literal["up", "down"] = "up",
    method: Literal["g", "pearson"] = "g",
    holes: ArrayLike | str | None = None,
    from_: ArrayLike | str | None = None,
    to: ArrayLike | str | None = None,
    depth: ArrayLike | str | None = None,
    step: float | None = None,
    gap: float = 1e-6,
    categories: Categories | int | object | None = None,
    data: Mapping[str, ArrayLike] | object | None = None,
) -> TestResult: ...
def sojourn(
    codes: ArrayLike | str | object,
    *,
    direction: Literal["up", "down"] = "up",
    holes: ArrayLike | str | None = None,
    from_: ArrayLike | str | None = None,
    to: ArrayLike | str | None = None,
    depth: ArrayLike | str | None = None,
    step: float | None = None,
    gap: float = 1e-6,
    categories: Categories | int | object | None = None,
    data: Mapping[str, ArrayLike] | object | None = None,
) -> Sojourn: ...

# <<< chain

# >>> hmm
Sequences: TypeAlias = ArrayLike | list[ArrayLike] | str | object
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

# <<< hmm

# >>> hsmm
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
    states: Sequence[int] | None = None,
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

# <<< hsmm

# >>> transiogram
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
    coords: ArrayLike | str | object,
    codes: ArrayLike | str | None = None,
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
    categories: Categories | int | object | None = None,
    data: Mapping[str, ArrayLike] | object | None = None,
) -> ExperimentalTransiogram: ...
def downhole_transiogram(
    codes: ArrayLike | str | object,
    *,
    lags: ArrayLike,
    direction: Literal["up", "down"] = "up",
    holes: ArrayLike | str | None = None,
    from_: ArrayLike | str | None = None,
    to: ArrayLike | str | None = None,
    depth: ArrayLike | str | None = None,
    step: float | None = None,
    gap: float = 1e-6,
    categories: Categories | int | object | None = None,
    data: Mapping[str, ArrayLike] | object | None = None,
) -> ExperimentalTransiogram: ...
def grid_transiogram(
    codes: ArrayLike | str | object,
    grid: Grid | object | None = None,
    *,
    offset: tuple[int, int, int],
    n_lags: int,
    mask: ArrayLike | None = None,
    categories: Categories | int | object | None = None,
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
    grid: Grid | object,
    realizations: Realizations,
    *,
    directions: Sequence[tuple[int, int, int]] | None = None,
    n_lags: int = 6,
    mask: ArrayLike | None = None,
) -> FitQuality: ...

# <<< transiogram

# >>> tpsim
class Prediction:
    @property
    def probabilities(self) -> NDArray[np.float64]: ...
    @property
    def most_likely(self) -> NDArray[np.int64]: ...
    @property
    def entropy(self) -> NDArray[np.float64]: ...
    @property
    def corrected(self) -> float: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Prediction: ...

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
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Realizations: ...
    def __minetypes__(self) -> dict[str, object]: ...

def tp_cokriging(
    model: TransitionModel,
    grid: Grid | object,
    coords: ArrayLike | str | object,
    codes: ArrayLike | str | None = None,
    *,
    mask: ArrayLike | None = None,
    max_data: int = 12,
    closeness_limit: float = 0.01,
    max_per_octant: int = 0,
    svd_cutoff: float = 1e-8,
    memory: int | None = None,
    data: Mapping[str, ArrayLike] | object | None = None,
) -> Prediction: ...
def tprogs_simulate(
    model: TransitionModel,
    grid: Grid | object,
    coords: ArrayLike | str | object,
    codes: ArrayLike | str | None = None,
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
    data: Mapping[str, ArrayLike] | object | None = None,
) -> Realizations: ...

# <<< tpsim

# >>> mcrf
def mcp_predict(
    model: TransitionModel,
    grid: Grid | object,
    coords: ArrayLike | str | object,
    codes: ArrayLike | str | None = None,
    *,
    mask: ArrayLike | None = None,
    neighbours: Literal["sectors"] | int | None = None,
    closeness_limit: float = 0.01,
    memory: int | None = None,
    data: Mapping[str, ArrayLike] | object | None = None,
) -> Prediction: ...
def mcrf_simulate(
    model: TransitionModel,
    grid: Grid | object,
    coords: ArrayLike | str | object,
    codes: ArrayLike | str | None = None,
    *,
    n: int = 1,
    seed: int = 0,
    mask: ArrayLike | None = None,
    neighbours: Literal["sectors"] | int | None = None,
    closeness_limit: float = 0.01,
    memory: int | None = None,
    data: Mapping[str, ArrayLike] | object | None = None,
) -> Realizations: ...

# <<< mcrf
