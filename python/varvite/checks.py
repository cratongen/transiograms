"""Pure NumPy reference implementations (`vv.checks`) used by tests and examples as oracles."""

import itertools

import numpy as np

_MAX_PATHS = 500_000


def _logsumexp(x):
    x = np.asarray(x, dtype=float)
    m = np.max(x) if x.size else -np.inf
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(x - m)))


def _paths(k, t):
    if k**t > _MAX_PATHS:
        raise ValueError(f"{k}^{t} paths exceed the {_MAX_PATHS} a brute-force check enumerates")
    return np.array(list(itertools.product(range(k), repeat=t)), dtype=np.int64).reshape(-1, t)


def _path_summary(paths, logp, k):
    ll = _logsumexp(logp)
    best = int(np.argmax(logp))
    t = paths.shape[1]
    with np.errstate(invalid="ignore"):
        q = np.exp(logp - ll) if np.isfinite(ll) else np.zeros_like(logp)
    gamma = np.zeros((t, k))
    for j in range(k):
        gamma[:, j] = ((paths == j) * q[:, None]).sum(axis=0)
    positive = q > 0
    entropy = float(-np.sum(q[positive] * np.log(q[positive])))
    return ll, paths[best].copy(), float(logp[best]), gamma, entropy


# >>> chain
def quasi_independence(counts, *, tol=1e-12, max_iter=10_000):
    """Expected counts of an embedded table under quasi-independence (zero diagonal).

    Iterative proportional fitting of ``E_ij = a_i b_j`` (``i != j``) to the row and column totals.

    Parameters
    ----------
    counts : array_like
        ``(m, m)`` transition counts with a zero diagonal.

    Returns
    -------
    numpy.ndarray
        ``(m, m)`` expected counts, zero diagonal.
    """
    n = np.asarray(counts, dtype=float)
    rows, cols = n.sum(axis=1), n.sum(axis=0)
    e = 1.0 - np.eye(len(n))
    for _ in range(max_iter):
        r = e.sum(axis=1)
        e *= np.divide(rows, r, out=np.zeros_like(rows), where=r > 0)[:, None]
        c = e.sum(axis=0)
        e *= np.divide(cols, c, out=np.zeros_like(cols), where=c > 0)[None, :]
        if np.max(np.abs(e.sum(axis=1) - rows)) <= tol * max(rows.sum(), 1.0):
            break
    return e


def kemeny_snell_std(P, n):
    """Kemeny and Snell standard deviation of the state proportions over ``n`` steps.

    ``sqrt(pi_k (2 Z_kk - 1 - pi_k) / n)`` with ``Z = (I - P + 1 pi^T)^-1``.
    """
    P = np.asarray(P, dtype=float)
    m = len(P)
    a = np.vstack([(P.T - np.eye(m)), np.ones(m)])
    pi = np.linalg.lstsq(a, np.r_[np.zeros(m), 1.0], rcond=None)[0]
    z = np.linalg.inv(np.eye(m) - P + np.outer(np.ones(m), pi))
    return np.sqrt(np.maximum(pi * (2 * np.diag(z) - 1 - pi) / n, 0.0))


def markov_bridge_marginal(P, a, b, n, t):
    """``P(x_t = j | x_0 = a, x_n = b) = (P^t)_aj (P^(n-t))_jb / (P^n)_ab`` for every ``j``."""
    P = np.asarray(P, dtype=float)
    mp = np.linalg.matrix_power
    return mp(P, t)[a] * mp(P, n - t)[:, b] / mp(P, n)[a, b]


# <<< chain


# >>> hmm
def brute_force_hmm(start, transition, log_b):
    """Exhaustive HMM inference over all ``K^T`` state paths.

    Parameters
    ----------
    start : array_like
        ``(K,)`` start probabilities.
    transition : array_like
        ``(K, K)`` transition matrix.
    log_b : array_like
        ``(T, K)`` emission log densities.

    Returns
    -------
    tuple
        ``(log_likelihood, best_path, best_log_probability, gamma, path_entropy)``; ties of the
        best path go to the lexicographically smallest path.
    """
    log_b = np.asarray(log_b, dtype=float)
    t, k = log_b.shape
    with np.errstate(divide="ignore"):
        ls, la = np.log(np.asarray(start, float)), np.log(np.asarray(transition, float))
    paths = _paths(k, t)
    logp = ls[paths[:, 0]] + log_b[np.arange(t), paths].sum(axis=1)
    if t > 1:
        logp = logp + la[paths[:, :-1], paths[:, 1:]].sum(axis=1)
    return _path_summary(paths, logp, k)


# <<< hmm


# >>> hsmm
def brute_force_hsmm(start, transition, log_d, log_survivor, log_b):
    """Exhaustive HSMM inference with a right-censored last segment.

    Every state path is cut into runs; runs longer than their state's ``len(log_d[j])`` are
    impossible. A path weighs ``pi * prod p (between runs) * prod d (non-final runs) * D (final
    run) * prod b``.

    Parameters
    ----------
    start, transition : array_like
        ``(K,)`` and ``(K, K)`` with a zero diagonal.
    log_d, log_survivor : sequence of array_like
        Per state, the duration log pmf and log survivor ``log P(U >= u)``, index ``u - 1``.
    log_b : array_like
        ``(T, K)`` emission log densities.

    Returns
    -------
    tuple
        As ``brute_force_hmm``.
    """
    log_b = np.asarray(log_b, dtype=float)
    t, k = log_b.shape
    with np.errstate(divide="ignore"):
        ls, la = np.log(np.asarray(start, float)), np.log(np.asarray(transition, float))
    paths = _paths(k, t)
    logp = np.empty(len(paths))
    for n, path in enumerate(paths):
        runs = [(s, len(list(g))) for s, g in itertools.groupby(path)]
        total = ls[runs[0][0]] + log_b[np.arange(t), path].sum()
        for i, (s, length) in enumerate(runs):
            table = log_survivor[s] if i == len(runs) - 1 else log_d[s]
            if length > len(table):
                total = -np.inf
                break
            total += table[length - 1]
            if i > 0:
                total += la[runs[i - 1][0], s]
        logp[n] = total
    return _path_summary(paths, logp, k)


# <<< hsmm

# >>> transiogram
# <<< transiogram

# >>> tpsim
# <<< tpsim


# >>> mcrf
def mcp_brute_force(p, T_list, codes, *, tol=1e-13, max_iter=10_000):
    """MCP conditional by maximum entropy on the full joint table.

    Iterative proportional fitting of the ``K^(G+1)`` table of ``(x_0, x_1, ..., x_G)`` to the
    pairwise margins ``P(x_0 = k, x_g = c) = p_k T_g[k, c]``, then the distribution of ``x_0``
    given ``x_g = codes[g]``.

    Parameters
    ----------
    p : array_like
        ``(K,)`` proportions.
    T_list : sequence of array_like
        One ``(K, K)`` transition matrix ``T(x_g - x_0)`` per neighbour.
    codes : sequence of int
        Category of each neighbour.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` conditional probabilities of ``x_0``.
    """
    p = np.asarray(p, dtype=float)
    k, g = len(p), len(T_list)
    if k ** (g + 1) > _MAX_PATHS:
        raise ValueError(f"{k}^{g + 1} cells exceed the {_MAX_PATHS} a brute-force check enumerates")
    table = np.full((k,) * (g + 1), 1.0 / k ** (g + 1))
    targets = [p[:, None] * np.asarray(t, dtype=float) for t in T_list]
    for _ in range(max_iter):
        worst = 0.0
        for i, target in enumerate(targets):
            other = tuple(a for a in range(1, g + 1) if a != i + 1)
            margin = table.sum(axis=other)
            ratio = np.divide(target, margin, out=np.zeros_like(target), where=margin > 0)
            shape = [1] * (g + 1)
            shape[0], shape[i + 1] = k, k
            table *= ratio.reshape(shape)
            worst = max(worst, float(np.max(np.abs(margin - target))))
        if worst < tol:
            break
    cond = table[(slice(None), *codes)]
    return cond / cond.sum()


# <<< mcrf
