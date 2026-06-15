"""
Stochastic gradient Langevin dynamics (SGLD) for the Bayesian last layer (week 3 of main.tex).

The primary, gradient-aware sampler. Per-chain minibatches of B triples drive the Welling-Teh update

    theta_{k+1} = theta_k + (tau_k / 2) * g_hat(theta_k) + sqrt(tau_k) * xi_k,   xi_k ~ N(0, I),

where  g_hat = (1/T)( (N'/B) sum_{i in B_k} grad log p(a_i|theta) + grad log p(theta) )  is the
unbiased minibatch estimate of the tempered-log-posterior gradient (``grad_log_posterior_batch``),
and the step size follows the decreasing schedule  tau_k = max(tau_min, tau_0 (1 + k/k0)^(-0.55)).
SGLD has no accept/reject step; the floor ``tau_min`` gives a stationary tail for clean diagnostics
(the schedule is plotted as the step-size trajectory diagnostic).

C chains run as a batch; each draws its own independent minibatch every step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from bayes.dnfc_bll import BLLPosterior


@dataclass
class SGLDResult:
    samples: np.ndarray            # (C, M, D) thinned post-burn-in draws of theta
    logp: np.ndarray               # (C, M) full log-posterior at the thinned draws (computed post-hoc)
    tau_sched: np.ndarray          # (n_iter,) step-size schedule
    grad_norm: np.ndarray          # (C, n_iter) minibatch-gradient norm trace
    coord_trace: np.ndarray        # (C, n_iter, K) selected-coordinate traces
    coord_idx: np.ndarray          # (K,)
    keep_iters: np.ndarray         # (M,) iteration index of each kept sample (for logp trace x-axis)
    burn_in: int
    thin: int
    seconds: float                 # wall-clock of the sampling loop
    sec_per_update: float          # pure update cost (no recording) for fair ESS/sec
    meta: dict = field(default_factory=dict)


def sgld_schedule(n_iter: int, tau0: float, tau_min: float, k0: float,
                  exponent: float = 0.55) -> np.ndarray:
    """Decreasing step size tau_k = max(tau_min, tau0 (1 + k/k0)^(-exponent))."""
    k = np.arange(n_iter)
    return np.maximum(tau_min, tau0 * (1.0 + k / k0) ** (-exponent))


def run_sgld(post: BLLPosterior,
             theta_init: torch.Tensor,
             n_iter: int = 100000,
             burn_in: int = 50000,
             thin: int = 25,
             tau0: float = 2e-9,
             tau_min: float = 5e-10,
             k0: float = 10000.0,
             exponent: float = 0.55,
             record_coords: Optional[Sequence[int]] = None,
             seed: int = 0,
             progress_every: int = 0) -> SGLDResult:
    """Run ``C = theta_init.shape[0]`` SGLD chains against ``post`` (minibatch B = ``post.batch_size``)."""
    device, dtype = post.device, post.dtype
    theta = torch.as_tensor(theta_init, device=device, dtype=dtype).clone()
    C, D = theta.shape
    B = post.batch_size
    gen = torch.Generator(device=device).manual_seed(seed)

    tau = sgld_schedule(n_iter, tau0, tau_min, k0, exponent)
    tau_t = torch.as_tensor(tau, device=device, dtype=dtype)

    if record_coords is None:
        record_coords = [0, D // 4, D // 2, D - 8, D - 4, D - 1]
    coord_idx = torch.as_tensor(list(record_coords), device=device, dtype=torch.long)

    grad_norm_tr = np.empty((C, n_iter), dtype=np.float64)
    coord_tr = np.empty((C, n_iter, coord_idx.numel()), dtype=np.float64)
    keep_iters = list(range(burn_in, n_iter, thin))
    samples = np.empty((C, len(keep_iters), D), dtype=np.float32)
    keep_i = 0

    sync = device.type == "cuda"
    t_update = 0.0
    t0_wall = time.time()
    for it in range(n_iter):
        if sync:
            torch.cuda.synchronize()
        t0 = time.time()
        idx = torch.randint(0, post.n_prime, (C, B), generator=gen, device=device)
        grad = post.grad_log_posterior_batch_fast(theta, idx)                    # (C, D)
        noise = torch.randn(C, D, generator=gen, device=device, dtype=dtype)
        tk = tau_t[it]
        theta = theta + 0.5 * tk * grad + torch.sqrt(tk) * noise
        if sync:
            torch.cuda.synchronize()
        t_update += time.time() - t0

        grad_norm_tr[:, it] = grad.norm(dim=1).detach().cpu().numpy()
        coord_tr[:, it, :] = theta[:, coord_idx].detach().cpu().numpy()
        if it >= burn_in and (it - burn_in) % thin == 0:
            samples[:, keep_i, :] = theta.detach().cpu().numpy()
            keep_i += 1

        if progress_every and (it + 1) % progress_every == 0:
            print(f"    it {it+1:7d}/{n_iter}  tau={float(tk):.2e}  "
                  f"||grad||~{grad.norm(dim=1).mean().item():.2e}")
    seconds = time.time() - t0_wall

    samples = samples[:, :keep_i, :]
    # Full log-posterior at the kept draws (post-hoc, untimed) for trace/ESS.
    with torch.no_grad():
        logp = np.stack([
            post.log_posterior_batch(torch.as_tensor(samples[:, m, :], device=device, dtype=dtype)
                                     ).cpu().numpy()
            for m in range(samples.shape[1])
        ], axis=1) if samples.shape[1] else np.zeros((C, 0))

    return SGLDResult(
        samples=samples, logp=logp, tau_sched=tau,
        grad_norm=grad_norm_tr, coord_trace=coord_tr, coord_idx=coord_idx.cpu().numpy(),
        keep_iters=np.array(keep_iters[:keep_i]), burn_in=burn_in, thin=thin,
        seconds=seconds, sec_per_update=t_update / n_iter,
        meta=dict(n_iter=n_iter, tau0=tau0, tau_min=tau_min, k0=k0, exponent=exponent,
                  B=B, C=C, D=D, seed=seed, temperature=post.temperature),
    )


def time_sgld_update(post: BLLPosterior, theta: torch.Tensor, tau: float = 1e-9,
                     n: int = 3000) -> float:
    """Seconds per pure SGLD update (minibatch gradient + Langevin step), no recording."""
    device, dtype = post.device, post.dtype
    theta = theta.clone()
    C, D = theta.shape
    B = post.batch_size
    gen = torch.Generator(device=device).manual_seed(123)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        idx = torch.randint(0, post.n_prime, (C, B), generator=gen, device=device)
        grad = post.grad_log_posterior_batch_fast(theta, idx)
        theta = theta + 0.5 * tau * grad + np.sqrt(tau) * torch.randn(
            C, D, generator=gen, device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n
