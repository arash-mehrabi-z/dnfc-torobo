"""
MCMC convergence diagnostics: split-Rhat, effective sample size (ESS), and autocorrelation.

Conventions (match Stan / ArviZ):
  * a parameter's draws are an array ``(C, N)`` = C chains x N draws;
  * ``split_rhat`` and ``ess`` split each chain in half (-> 2C half-chains) before computing,
    so within-chain non-stationarity is penalised;
  * ESS uses the combined multi-chain autocorrelation with Geyer's initial positive + monotone
    sequence truncation; ``tau_int = N_total / ESS`` is the integrated autocorrelation time.

Self-test:  python -m bayes.diagnostics   (checks iid -> ESS~N, Rhat~1; AR(1) -> ESS ~ N(1-rho)/(1+rho))
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _autocov_1d(y: np.ndarray) -> np.ndarray:
    """Biased autocovariance of a 1-D series at lags 0..N-1 (FFT). ``acov[0]`` = population var."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    y = y - y.mean()
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(y, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n].real
    return acov / n


def _split(chains: np.ndarray) -> np.ndarray:
    """(C, N) -> (2C, N//2): split each chain into its two halves."""
    chains = np.asarray(chains, dtype=np.float64)
    c, n = chains.shape
    h = n // 2
    return np.concatenate([chains[:, :h], chains[:, h:2 * h]], axis=0)


def split_rhat(chains: np.ndarray) -> float:
    """Split-Rhat (Gelman-Rubin) for one scalar parameter, draws shaped ``(C, N)``."""
    x = _split(chains)
    m, n = x.shape
    if n < 2:
        return float("nan")
    means = x.mean(axis=1)
    vars = x.var(axis=1, ddof=1)
    W = vars.mean()
    if W <= 0 or not np.isfinite(W):
        return float("nan")
    B = n * means.var(ddof=1)
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def ess(chains: np.ndarray) -> float:
    """Effective sample size for one scalar parameter, draws shaped ``(C, N)`` (split-chain)."""
    x = _split(chains)
    m, n = x.shape
    if n < 8:
        return float("nan")
    acov = np.stack([_autocov_1d(x[i]) for i in range(m)])  # (m, n)
    chain_mean = x.mean(axis=1)
    mean_var = acov[:, 0].mean() * n / (n - 1)
    var_plus = mean_var * (n - 1) / n
    if m > 1:
        var_plus += chain_mean.var(ddof=1)
    if var_plus <= 0 or not np.isfinite(var_plus):
        return float("nan")

    mean_acov = acov.mean(axis=0)             # (n,)
    rho = 1.0 - (mean_var - mean_acov) / var_plus
    rho[0] = 1.0

    # Geyer initial positive sequence on paired autocorrelations Gamma_k = rho[2k+1] + rho[2k+2].
    gamma = []
    k = 0
    while 2 * k + 2 <= n - 1:
        gamma.append(rho[2 * k + 1] + rho[2 * k + 2])
        k += 1
    gamma = np.asarray(gamma)
    nonpos = np.where(gamma <= 0)[0]
    K = int(nonpos[0]) if nonpos.size else gamma.size
    g = gamma[:K].copy()
    # Geyer initial monotone sequence: make the kept pairs non-increasing.
    for i in range(1, g.size):
        if g[i] > g[i - 1]:
            g[i] = g[i - 1]

    tau_int = 1.0 + 2.0 * g.sum()             # integrated autocorrelation time
    tau_int = max(tau_int, 1.0 / np.log10(m * n))  # floor (as in Stan), avoids ESS > N blowups
    return float(m * n / tau_int)


def autocorr(chain: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation rho[0..max_lag] of a single 1-D chain."""
    acov = _autocov_1d(chain)
    return (acov[: max_lag + 1] / acov[0]) if acov[0] > 0 else np.full(max_lag + 1, np.nan)


def summarize(draws: np.ndarray, names: Sequence[str] | None = None) -> Dict[str, np.ndarray]:
    """Per-parameter Rhat / ESS for draws shaped ``(C, N, P)`` (or ``(C, N)`` for a single param).

    Returns dict with arrays ``rhat`` (P,), ``ess`` (P,), and (if given) ``names``.
    """
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim == 2:
        draws = draws[:, :, None]
    c, n, p = draws.shape
    rhat = np.array([split_rhat(draws[:, :, j]) for j in range(p)])
    ess_ = np.array([ess(draws[:, :, j]) for j in range(p)])
    out = {"rhat": rhat, "ess": ess_}
    if names is not None:
        out["names"] = np.asarray(names, dtype=object)
    return out


def _self_test() -> None:
    rng = np.random.default_rng(0)
    C, N = 4, 4000

    # (1) iid normal: Rhat ~ 1, ESS ~ C*N.
    x = rng.standard_normal((C, N))
    print(f"iid N(0,1):     Rhat={split_rhat(x):.3f}  ESS={ess(x):8.0f}  (target ~{C*N})")

    # (2) AR(1): ESS ~ C*N * (1-rho)/(1+rho).
    for rho in (0.5, 0.9):
        y = np.zeros((C, N))
        y[:, 0] = rng.standard_normal(C)
        noise = rng.standard_normal((C, N)) * np.sqrt(1 - rho ** 2)
        for t in range(1, N):
            y[:, t] = rho * y[:, t - 1] + noise[:, t]
        theory = C * N * (1 - rho) / (1 + rho)
        print(f"AR(1) rho={rho}:   Rhat={split_rhat(y):.3f}  ESS={ess(y):8.0f}  (theory ~{theory:.0f})")

    # (3) non-converged: chains with different means -> Rhat >> 1.
    z = rng.standard_normal((C, N)) + np.arange(C)[:, None] * 3.0
    print(f"shifted chains: Rhat={split_rhat(z):.3f}  ESS={ess(z):8.0f}  (Rhat should be >> 1)")


if __name__ == "__main__":
    _self_test()
