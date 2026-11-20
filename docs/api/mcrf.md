# MCRF and MCP

Markov-type categorical prediction and Markov chain random field simulation; both return the classes of the [TP simulation](tpsim.md) page. See the [guide](../guide/mcrf-mcp.md).

```python
def mcp_predict(
    model: TransitionModel,
    grid: Grid,
    coords: ArrayLike | str,
    codes: ArrayLike | str,
    *,
    mask: ArrayLike | None = None,
    neighbours: Literal["sectors"] | int | None = None,
    closeness_limit: float = 0.01,
    memory: int | None = None,
    data: Mapping[str, ArrayLike] | None = None,
) -> Prediction: ...
def mcrf_simulate(
    model: TransitionModel,
    grid: Grid,
    coords: ArrayLike | str,
    codes: ArrayLike | str,
    *,
    n: int = 1,
    seed: int = 0,
    mask: ArrayLike | None = None,
    neighbours: Literal["sectors"] | int | None = None,
    closeness_limit: float = 0.01,
    memory: int | None = None,
    data: Mapping[str, ArrayLike] | None = None,
) -> Realizations: ...
```
