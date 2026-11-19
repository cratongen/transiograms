# Chain

Markov chains, statistical tests, sojourns and 1D simulation. Docstrings in Python (`help(tg.MarkovChain.fit)`) hold the details; the [guide](../guide/markov-chains.md) explains the methods.

```python
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
        start: int | Literal["stationary"] = "stationary",
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
        start: int | Literal["stationary"] = "stationary",
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


def order_test(
    codes: ArrayLike | str,
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
    categories: Categories | int | None = None,
    data: Mapping[str, ArrayLike] | None = None,
) -> TestResult: ...
def select_order(
    codes: ArrayLike | str,
    *,
    max_order: int = 3,
    direction: Literal["up", "down"] = "up",
    holes: ArrayLike | str | None = None,
    from_: ArrayLike | str | None = None,
    to: ArrayLike | str | None = None,
    depth: ArrayLike | str | None = None,
    step: float | None = None,
    gap: float = 1e-6,
    categories: Categories | int | None = None,
    data: Mapping[str, ArrayLike] | None = None,
) -> dict[str, NDArray]: ...
def homogeneity_test(
    codes: ArrayLike | str,
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
    categories: Categories | int | None = None,
    data: Mapping[str, ArrayLike] | None = None,
) -> TestResult: ...
def sojourn(
    codes: ArrayLike | str,
    *,
    direction: Literal["up", "down"] = "up",
    holes: ArrayLike | str | None = None,
    from_: ArrayLike | str | None = None,
    to: ArrayLike | str | None = None,
    depth: ArrayLike | str | None = None,
    step: float | None = None,
    gap: float = 1e-6,
    categories: Categories | int | None = None,
    data: Mapping[str, ArrayLike] | None = None,
) -> Sojourn: ...
```
