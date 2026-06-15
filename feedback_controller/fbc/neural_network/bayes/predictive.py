"""
Week-4 predictive-uncertainty machinery (see main.tex, section 5).

Posterior predictive over actions by Monte-Carlo averaging a set of last-layer samples Theta (M, D):

    mean(s,g)   = (1/M) sum_m  f_theta^(m)(s,g)                      -> (N, 7)
    std(s,g)    = sqrt( (1/M) sum_m || f_theta^(m)(s,g) - mean ||^2 ) -> (N,)   (total spread over the 7 dims)

with f_theta = tanh(W phi(s,g) + b) and phi from the frozen trunk (shared across samples).

Sample sources compared:
  * point  -- the trained theta_0 (deterministic; std == 0),
  * analytic -- the closed-form Gaussian posterior (the near-exact reference; see ``sample_analytic``),
  * RWMH / SGLD -- MCMC draws.

Three input regimes: in-distribution (interp split), extrapolation (extrap split), and target-perturbed
(interp states with Gaussian noise added to the 9 target cartesian coords).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from bayes.dnfc_bll import (load_triples, ACTION_DIM, COORDS_DIM,
                            _S0, STATE_DIM, _T0, TARGET_DIM, _A0)


# --------------------------------------------------------------------------------------
# regimes
# --------------------------------------------------------------------------------------
def perturb_targets(target_repr: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Add N(0, sigma^2) noise to the 9 cartesian target coords (cols 0:9); keep the one-hot."""
    rng = np.random.default_rng(seed)
    out = np.array(target_repr, copy=True)
    out[:, :COORDS_DIM] = out[:, :COORDS_DIM] + rng.normal(0.0, sigma, size=(out.shape[0], COORDS_DIM))
    return out


# --------------------------------------------------------------------------------------
# analytic (linear-Gaussian) posterior over the last layer -- the reference
# --------------------------------------------------------------------------------------
def analytic_posterior(post) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form Gaussian posterior over (W, b) under the linearised (tanh~=id) likelihood.

    Each of the 7 output dims is an independent Bayesian ridge regression on the SAME augmented
    features [phi, 1] (so one shared covariance). Returns ``mu`` (d+1, 7) and shared ``Sigma`` (d+1, d+1)
    in float64.
    """
    Phi = post.Phi.detach().double()                          # (N', d)
    A = post.A.detach().double()                              # (N', 7)
    N, d = Phi.shape
    Phit = torch.cat([Phi, torch.ones(N, 1, dtype=torch.float64, device=Phi.device)], dim=1)  # (N', d+1)
    prec = Phit.t() @ Phit / post.sigma_a ** 2 + torch.eye(d + 1, dtype=torch.float64,
                                                           device=Phi.device) / post.alpha ** 2
    Sigma = torch.linalg.inv(prec)
    mu = Sigma @ (Phit.t() @ A) / post.sigma_a ** 2           # (d+1, 7)
    return mu, Sigma


def sample_analytic(post, M: int, seed: int = 0) -> torch.Tensor:
    """Draw ``M`` i.i.d. last-layer samples ``Theta`` (M, D) from the analytic Gaussian posterior."""
    mu, Sigma = analytic_posterior(post)                      # (d+1,7), (d+1,d+1)
    d1 = mu.shape[0]
    d = d1 - 1
    L = torch.linalg.cholesky(Sigma)                          # (d+1, d+1)
    g = torch.Generator(device=Sigma.device).manual_seed(seed)
    z = torch.randn(ACTION_DIM, d1, M, generator=g, dtype=torch.float64, device=Sigma.device)
    samp = mu.t()[:, :, None] + torch.einsum("ab,jbm->jam", L, z)   # (7, d+1, M)
    W = samp[:, :d, :].permute(2, 0, 1)                       # (M, 7, d)
    b = samp[:, d, :].t()                                     # (M, 7)
    theta = torch.cat([W.reshape(M, -1), b], dim=1)
    return theta.to(post.dtype).to(post.device)


# --------------------------------------------------------------------------------------
# posterior predictive mean / std
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_mean_std(wrapper, Theta: torch.Tensor, state: np.ndarray, target_repr: np.ndarray,
                     chunk: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    """Posterior-predictive ``mean`` (N,7) and ``std`` (N,) over action samples ``Theta`` (M, D)."""
    phi = wrapper.features(state, target_repr)                # (N, d)
    M, d = Theta.shape[0], wrapper.d
    W = Theta[:, : ACTION_DIM * d].reshape(M, ACTION_DIM, d)
    b = Theta[:, ACTION_DIM * d:].reshape(M, ACTION_DIM)
    means, stds = [], []
    for i in range(0, phi.shape[0], chunk):
        ph = phi[i:i + chunk]                                 # (n, d)
        pred = torch.tanh(torch.einsum("nd,mod->mno", ph, W) + b[:, None, :])  # (M, n, 7)
        mean = pred.mean(dim=0)                               # (n, 7)
        std = (pred - mean[None]).pow(2).sum(dim=2).mean(dim=0).sqrt()         # (n,)
        means.append(mean)
        stds.append(std)
    return torch.cat(means), torch.cat(stds)


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with tie handling -- like scipy.stats.rankdata(method='average')."""
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=np.intp)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def auroc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    """AUROC = P(score(pos) > score(neg)) via the Mann-Whitney U statistic (tie-aware)."""
    neg, pos = np.asarray(neg_scores, float), np.asarray(pos_scores, float)
    n_neg, n_pos = len(neg), len(pos)
    if n_neg == 0 or n_pos == 0:
        return float("nan")
    ranks = _rankdata_avg(np.concatenate([neg, pos]))
    r_pos = ranks[n_neg:].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def action_rmse(mean_pred: torch.Tensor, expert: np.ndarray) -> np.ndarray:
    """Per-state action RMSE (over the 7 dims) between predictive mean and the expert action."""
    expert = torch.as_tensor(expert, device=mean_pred.device, dtype=mean_pred.dtype)
    return (mean_pred - expert).pow(2).mean(dim=1).sqrt().cpu().numpy()


def calibration_curve(std: np.ndarray, err: np.ndarray, n_bins: int = 10):
    """Bin states by predicted std (quantile bins); return (mean_std, mean_err) per bin."""
    std, err = np.asarray(std), np.asarray(err)
    edges = np.quantile(std, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-12
    idx = np.clip(np.digitize(std, edges[1:-1]), 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 3:
            xs.append(std[m].mean())
            ys.append(err[m].mean())
    return np.array(xs), np.array(ys)
