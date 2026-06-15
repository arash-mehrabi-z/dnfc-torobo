"""
Random-walk Metropolis-Hastings (RWMH) for the Bayesian last layer (week 2 of main.tex).

Isotropic Gaussian proposals  theta' = theta + tau * xi,  xi ~ N(0, I)  (the gradient-free baseline).
The C chains are run as a batch (one ``log_posterior_batch`` matmul per iteration). During burn-in the
per-chain step size ``tau`` is adapted by Robbins-Monro on log-tau toward the high-dimensional
benchmark acceptance rate (~0.234); after burn-in ``tau`` is frozen and draws are recorded.

    log tau <- log tau + gamma_t (alpha_t - target),   gamma_t = (t+1)^(-0.6),
    alpha_t = min(1, exp(logpost' - logpost))  (the per-chain acceptance probability; low variance).

``run_rwmh`` returns a ``RWMHResult`` with the thinned post-burn-in samples plus full per-iteration
traces (log-posterior, tau, acceptance, a few selected coordinates) for diagnostics and plots.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from bayes.dnfc_bll import BLLPosterior


@dataclass
class RWMHResult:
    samples: np.ndarray            # (C, M, D) thinned post-burn-in draws of theta
    logp: np.ndarray               # (C, n_iter) unnormalised log-posterior trace
    tau: np.ndarray                # (C, n_iter) step-size trace
    accept: np.ndarray            # (C, n_iter) acceptance indicator (0/1)
    coord_trace: np.ndarray        # (C, n_iter, K) selected-coordinate traces
    coord_idx: np.ndarray          # (K,) indices of the recorded coordinates
    accept_rate: np.ndarray        # (C,) post-burn-in acceptance rate per chain
    tau_final: np.ndarray          # (C,) frozen step size per chain
    burn_in: int
    thin: int
    seconds: float                 # wall-clock of the sampling loop (for ESS/sec)
    meta: dict = field(default_factory=dict)


def run_rwmh(post: BLLPosterior,
             theta_init: torch.Tensor,
             n_iter: int = 12000,
             burn_in: int = 4000,
             thin: int = 8,
             tau0: float = 1e-2,
             target_accept: float = 0.234,
             adapt_exponent: float = 0.6,
             record_coords: Optional[Sequence[int]] = None,
             seed: int = 0,
             progress_every: int = 0) -> RWMHResult:
    """Run ``C = theta_init.shape[0]`` RWMH chains in parallel against ``post.log_posterior``."""
    device, dtype = post.device, post.dtype
    theta = torch.as_tensor(theta_init, device=device, dtype=dtype).clone()
    C, D = theta.shape
    gen = torch.Generator(device=device).manual_seed(seed)

    logp = post.log_posterior_batch(theta)                          # (C,)
    log_tau = torch.full((C,), float(np.log(tau0)), device=device, dtype=dtype)

    if record_coords is None:
        # last 7 entries are the output biases; sprinkle a few weights across the matrix.
        record_coords = [0, D // 4, D // 2, D - 8, D - 4, D - 1]
    coord_idx = torch.as_tensor(list(record_coords), device=device, dtype=torch.long)

    logp_tr = np.empty((C, n_iter), dtype=np.float64)
    tau_tr = np.empty((C, n_iter), dtype=np.float64)
    acc_tr = np.empty((C, n_iter), dtype=np.float64)
    coord_tr = np.empty((C, n_iter, coord_idx.numel()), dtype=np.float64)

    n_keep = len(range(burn_in, n_iter, thin))
    samples = np.empty((C, n_keep, D), dtype=np.float32)
    keep_i = 0
    post_burn_accept = torch.zeros(C, device=device, dtype=dtype)

    t0 = time.time()
    for it in range(n_iter):
        xi = torch.randn(C, D, generator=gen, device=device, dtype=dtype)
        tau = torch.exp(log_tau)
        prop = theta + tau[:, None] * xi
        logp_prop = post.log_posterior_batch(prop)
        dlogp = logp_prop - logp
        log_u = torch.log(torch.rand(C, generator=gen, device=device, dtype=dtype))
        accept = log_u < dlogp                                       # (C,) bool
        theta = torch.where(accept[:, None], prop, theta)
        logp = torch.where(accept, logp_prop, logp)

        if it < burn_in:                                            # adapt log-tau (Robbins-Monro)
            alpha = torch.exp(torch.clamp(dlogp, max=0.0))          # acceptance probability
            gamma = (it + 1) ** (-adapt_exponent)
            log_tau = log_tau + gamma * (alpha - target_accept)
        else:
            post_burn_accept += accept.to(dtype)
            if (it - burn_in) % thin == 0:
                samples[:, keep_i, :] = theta.detach().cpu().numpy()
                keep_i += 1

        logp_tr[:, it] = logp.detach().cpu().numpy()
        tau_tr[:, it] = tau.detach().cpu().numpy()
        acc_tr[:, it] = accept.detach().cpu().numpy()
        coord_tr[:, it, :] = theta[:, coord_idx].detach().cpu().numpy()

        if progress_every and (it + 1) % progress_every == 0:
            ar = acc_tr[:, max(0, it - progress_every):it + 1].mean()
            print(f"    it {it+1:6d}/{n_iter}  accept~{ar:.3f}  "
                  f"tau~{tau.mean().item():.2e}  logp~{logp.mean().item():.4e}")
    seconds = time.time() - t0

    n_post = n_iter - burn_in
    return RWMHResult(
        samples=samples[:, :keep_i, :],
        logp=logp_tr, tau=tau_tr, accept=acc_tr,
        coord_trace=coord_tr, coord_idx=coord_idx.cpu().numpy(),
        accept_rate=(post_burn_accept / max(n_post, 1)).cpu().numpy(),
        tau_final=torch.exp(log_tau).detach().cpu().numpy(),
        burn_in=burn_in, thin=thin, seconds=seconds,
        meta=dict(n_iter=n_iter, tau0=tau0, target_accept=target_accept, seed=seed,
                  C=C, D=D, temperature=post.temperature),
    )


def time_rwmh_update(post: BLLPosterior, theta: torch.Tensor, tau: float = 1e-4,
                     n: int = 3000) -> float:
    """Seconds per pure RWMH update (propose + full-N' eval + accept), no recording.

    Used for a fair ESS/second comparison: RWMH's intrinsic cost is one *full* log-posterior
    evaluation per step (the exact MH ratio cannot be subsampled).
    """
    device, dtype = post.device, post.dtype
    theta = theta.clone()
    C, D = theta.shape
    gen = torch.Generator(device=device).manual_seed(123)
    logp = post.log_posterior_batch(theta)
    tau_t = torch.full((C,), tau, device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        prop = theta + tau_t[:, None] * torch.randn(C, D, generator=gen, device=device, dtype=dtype)
        logp_prop = post.log_posterior_batch(prop)
        accept = torch.log(torch.rand(C, generator=gen, device=device, dtype=dtype)) < (logp_prop - logp)
        theta = torch.where(accept[:, None], prop, theta)
        logp = torch.where(accept, logp_prop, logp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n


def dispersed_init(post: BLLPosterior, n_chains: int = 4, sigma: float = 2e-3,
                   seed: int = 0, around_theta0: bool = True) -> torch.Tensor:
    """Over-dispersed chain starts: ``theta_0 + sigma * N(0, I)`` (or prior draws if not around_theta0).

    Dispersing around the trained point estimate keeps the sharp-likelihood directions recoverable
    while still giving the between-chain spread split-Rhat needs.
    """
    g = torch.Generator(device=post.device).manual_seed(seed)
    base = post.theta0 if around_theta0 else torch.zeros(post.theta_dim, device=post.device, dtype=post.dtype)
    noise = torch.randn(n_chains, post.theta_dim, generator=g, device=post.device, dtype=post.dtype)
    scale = sigma if around_theta0 else post.alpha
    return base[None, :] + scale * noise
