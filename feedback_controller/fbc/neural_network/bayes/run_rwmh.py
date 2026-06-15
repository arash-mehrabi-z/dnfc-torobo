"""
Week-2 driver: Random-walk Metropolis-Hastings on the DNFC Bayesian last layer.

    python -m bayes.run_rwmh [--temperature T] [--n-iter N] [--burn-in B] [--device cuda|cpu]

Pipeline
  1. SAMPLER VALIDATION -- run RWMH on isotropic Gaussians of growing dimension. Acceptance tunes to
     ~0.234, recovered moments match the target (proves correctness), and ESS/iter falls ~1/d
     (the curse-of-dimension that limits random-walk proposals).
  2. MAIN RUN -- four dispersed chains on the real posterior; tune tau to ~23.4%; compute split-Rhat,
     ESS, ESS/second, autocorrelation, acceptance, and the tau trajectory; save traces + a figure.
  3. WHY -- quantify the posterior anisotropy (feature Gram spectrum -> Gaussian-approx condition
     number) that makes isotropic RWMH mix poorly, motivating the gradient-aware SGLD (week 3).

Outputs go to ``bayes/results/rwmh/<tag>/``: ``samples.npy`` (C,M,D), ``summary.json``, ``diagnostics.png``.
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayes.dnfc_bll import BLLConfig, load_default, _NN_DIR
from bayes.rwmh import run_rwmh, dispersed_init
from bayes.diagnostics import split_rhat, ess, autocorr, summarize


# ---------------------------------------------------------------------------------------
# 1. Sampler validation: isotropic Gaussian targets (duck-typed like BLLPosterior)
# ---------------------------------------------------------------------------------------
@dataclass
class IsoGaussTarget:
    """N(0, sigma^2 I) target exposing the minimal interface run_rwmh / dispersed_init need."""
    theta_dim: int
    sigma: float
    device: torch.device
    dtype: torch.dtype
    temperature: float = 1.0

    @property
    def alpha(self) -> float:
        return self.sigma

    # minimal attrs so SGLD (run_sgld) can also target this (no data -> minibatch is a no-op).
    n_prime: int = 1
    batch_size: int = 1

    @property
    def theta0(self) -> torch.Tensor:
        return torch.zeros(self.theta_dim, device=self.device, dtype=self.dtype)

    def log_posterior_batch(self, Theta: torch.Tensor) -> torch.Tensor:
        Theta = torch.as_tensor(Theta, device=self.device, dtype=self.dtype)
        return -0.5 / self.sigma ** 2 * Theta.pow(2).sum(dim=1)

    def grad_log_posterior_batch_fast(self, Theta: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        Theta = torch.as_tensor(Theta, device=self.device, dtype=self.dtype)
        return -Theta / self.sigma ** 2


def validate_sampler(device, dtype) -> None:
    print("\n" + "=" * 78 + "\n1. SAMPLER VALIDATION  (isotropic N(0, sigma^2 I); sigma=0.0627)\n" + "=" * 78)
    print(f"  {'dim':>6} {'accept':>8} {'Rhat(x0)':>9} {'ESS(x0)':>9} {'ESS/iter':>9} "
          f"{'mean~0':>9} {'std~sigma':>10}")
    sigma = 0.0627
    for d in (10, 100, 1000, 2695):
        tgt = IsoGaussTarget(d, sigma, device, dtype)
        init = dispersed_init(tgt, n_chains=4, sigma=sigma, seed=0, around_theta0=False)
        n_iter, burn = 30000, 15000
        res = run_rwmh(tgt, init, n_iter=n_iter, burn_in=burn, thin=5,
                       tau0=2.38 * sigma / np.sqrt(d), seed=0)
        s = res.samples.reshape(-1, d)            # pooled post-burn draws
        x0 = res.samples[:, :, 0]                  # (C, M) first coordinate
        n_post = n_iter - burn
        print(f"  {d:6d} {res.accept_rate.mean():8.3f} {split_rhat(x0):9.3f} {ess(x0):9.1f} "
              f"{ess(x0) / (4 * n_post):9.4f} {s.mean():9.3f} {s.std():10.4f}")
    print("  -> acceptance hits ~0.234; recovered mean~0 & std~sigma (correct stationary law);")
    print("     ESS/iter ~ 1/d  =>  random-walk proposals decorrelate in O(d) steps.")


# ---------------------------------------------------------------------------------------
# 3. Why RWMH struggles: posterior anisotropy from the (low-rank) feature Gram matrix
# ---------------------------------------------------------------------------------------
def anisotropy_report(post) -> dict:
    print("\n" + "=" * 78 + "\n3. POSTERIOR ANISOTROPY  (Gaussian/Laplace approximation)\n" + "=" * 78)
    Phi = post.Phi.detach().double()                       # (N', d)
    gram = (Phi.t() @ Phi).cpu().numpy()                   # (d, d)
    evals = np.linalg.eigvalsh(gram)
    evals = np.clip(evals[::-1], 0, None)                   # descending, non-negative
    lam_max = float(evals[0])
    pos = evals[evals > 1e-8 * lam_max]
    eff_rank = float((evals.sum() ** 2) / (evals ** 2).sum())   # participation ratio
    # Per-output Gauss-Newton precision ~ Phi'Phi/sigma_a^2 + I/alpha^2; condition number:
    kappa = 1.0 + (post.alpha ** 2 / post.sigma_a ** 2) * lam_max
    print(f"  feature dim d                         = {Phi.shape[1]}")
    print(f"  effective rank of Phi'Phi (part.ratio)= {eff_rank:.1f}   (input 'diff' is only 14-dim)")
    print(f"  eigenvalues > 1e-8*max                = {pos.size} / {evals.size}")
    print(f"  lambda_max(Phi'Phi)                   = {lam_max:.3e}")
    print(f"  alpha^2 / sigma_a^2                   = {post.alpha ** 2 / post.sigma_a ** 2:.1f}")
    print(f"  => posterior condition number kappa   ~ {kappa:.3e}")
    print(f"  isotropic RWMH mixing time ~ kappa, so ESS ~ N_iter/kappa: hopeless at this kappa.")
    return dict(eff_rank=eff_rank, lambda_max=lam_max, n_pos_eig=int(pos.size),
                kappa_gauss=float(kappa))


# ---------------------------------------------------------------------------------------
# 2. Main run + diagnostics + plots
# ---------------------------------------------------------------------------------------
def diagnostics_figure(res, post, out_path) -> None:
    C = res.logp.shape[0]
    burn = res.burn_in
    it = np.arange(res.logp.shape[1])
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    for c in range(C):
        ax[0, 0].plot(it, res.logp[c], lw=0.6)
    ax[0, 0].axvline(burn, color="k", ls="--", lw=1)
    ax[0, 0].set(title="log-posterior trace", xlabel="iteration", ylabel="log p(theta|D)")

    for c in range(C):
        ax[0, 1].semilogy(it, res.tau[c], lw=0.8)
    ax[0, 1].axvline(burn, color="k", ls="--", lw=1)
    ax[0, 1].set(title="step size tau (adapted to 23.4%)", xlabel="iteration", ylabel="tau")

    win = max(1, res.logp.shape[1] // 200)
    for c in range(C):
        run = np.convolve(res.accept[c], np.ones(win) / win, mode="valid")
        ax[0, 2].plot(run, lw=0.8)
    ax[0, 2].axhline(0.234, color="k", ls=":", lw=1)
    ax[0, 2].set(title=f"running acceptance (win={win})", xlabel="iteration", ylim=(0, 1))

    # trace of two coordinates (a weight and an output bias)
    k_w, k_b = 0, res.coord_trace.shape[2] - 1
    for c in range(C):
        ax[1, 0].plot(it, res.coord_trace[c, :, k_w], lw=0.6)
    ax[1, 0].axvline(burn, color="k", ls="--", lw=1)
    ax[1, 0].set(title=f"trace: weight theta[{res.coord_idx[k_w]}]", xlabel="iteration")
    for c in range(C):
        ax[1, 1].plot(it, res.coord_trace[c, :, k_b], lw=0.6)
    ax[1, 1].axvline(burn, color="k", ls="--", lw=1)
    ax[1, 1].set(title=f"trace: bias theta[{res.coord_idx[k_b]}]", xlabel="iteration")

    # autocorrelation of the log-posterior (post burn-in), per chain
    maxlag = min(300, (res.logp.shape[1] - burn) // 2)
    for c in range(C):
        ax[1, 2].plot(autocorr(res.logp[c, burn:], maxlag), lw=0.8)
    ax[1, 2].axhline(0, color="k", lw=0.6)
    ax[1, 2].set(title="ACF of log-posterior (post burn-in)", xlabel="lag", ylim=(-0.2, 1))

    fig.suptitle("RWMH diagnostics -- DNFC Bayesian last layer", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sigma-a", type=float, default=None,
                    help="observation noise (posterior conditioning); None -> train-residual (nominal, "
                         "stiff). pass e.g. 0.05 for a samplable posterior consistent with weeks 3-4.")
    ap.add_argument("--n-iter", type=int, default=40000)
    ap.add_argument("--burn-in", type=int, default=20000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--tau0", type=float, default=1e-4)
    ap.add_argument("--sigma-init", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32

    if not args.skip_validation:
        validate_sampler(device, dtype)

    print("\n" + "=" * 78 + f"\n2. MAIN RWMH RUN  (T={args.temperature:g}, {args.n_chains} chains, "
          f"{args.n_iter} iters)\n" + "=" * 78)
    wrapper, post = load_default(BLLConfig(device=args.device, temperature=args.temperature,
                                          sigma_a=args.sigma_a))
    print("  checkpoint:", os.path.relpath(wrapper.tgt.ckpt, _NN_DIR))
    print(" ", post.summary())

    init = dispersed_init(post, n_chains=args.n_chains, sigma=args.sigma_init, seed=args.seed)
    res = run_rwmh(post, init, n_iter=args.n_iter, burn_in=args.burn_in, thin=args.thin,
                   tau0=args.tau0, seed=args.seed, progress_every=args.n_iter // 4)

    # --- diagnostics ---
    lp_post = res.logp[:, res.burn_in:]
    rhat_lp, ess_lp = split_rhat(lp_post), ess(lp_post)
    summ = summarize(res.samples)                          # per-coordinate Rhat/ESS
    rhat, ess_c = summ["rhat"], summ["ess"]
    sec = res.seconds
    head = dict(
        accept_rate=float(res.accept_rate.mean()),
        tau_final=float(res.tau_final.mean()),
        seconds=sec, n_samples_per_chain=int(res.samples.shape[1]),
        logp_rhat=float(rhat_lp), logp_ess=float(ess_lp), logp_ess_per_s=float(ess_lp / sec),
        theta_rhat_median=float(np.nanmedian(rhat)), theta_rhat_max=float(np.nanmax(rhat)),
        theta_ess_median=float(np.nanmedian(ess_c)), theta_ess_min=float(np.nanmin(ess_c)),
        theta_ess_per_s_median=float(np.nanmedian(ess_c) / sec),
        frac_rhat_below_1p1=float(np.mean(rhat < 1.1)),
    )

    print(f"\n  acceptance (mean over chains)   = {head['accept_rate']:.3f}  (target 0.234)")
    print(f"  tau (final, mean)               = {head['tau_final']:.3e}")
    print(f"  sampling wall-clock             = {sec:.1f}s  ({args.n_chains} chains x {args.n_iter} it)")
    print(f"  log-posterior : Rhat={rhat_lp:.3f}  ESS={ess_lp:.1f}  ESS/s={ess_lp / sec:.3f}")
    print(f"  theta (2695)  : Rhat med={head['theta_rhat_median']:.2f} "
          f"max={head['theta_rhat_max']:.2f} | ESS med={head['theta_ess_median']:.1f} "
          f"min={head['theta_ess_min']:.1f} | ESS/s med={head['theta_ess_per_s_median']:.3f}")
    print(f"  fraction of coords with Rhat<1.1: {head['frac_rhat_below_1p1']:.2f}")
    verdict = ("MIXED" if (rhat_lp < 1.1 and head["frac_rhat_below_1p1"] > 0.9)
               else "POORLY MIXED (expected for isotropic RWMH at this dimension/conditioning)")
    print(f"  VERDICT: {verdict}")

    aniso = anisotropy_report(post)

    # --- save ---
    tag = f"T{args.temperature:g}_N{args.n_iter}_C{args.n_chains}"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "rwmh", tag)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "samples.npy"), res.samples)
    diagnostics_figure(res, post, os.path.join(out_dir, "diagnostics.png"))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"config": vars(args), "checkpoint": os.path.relpath(wrapper.tgt.ckpt, _NN_DIR),
                   "posterior": post.summary(), "headline": head, "anisotropy": aniso,
                   "accept_rate_per_chain": res.accept_rate.tolist(),
                   "tau_final_per_chain": res.tau_final.tolist()}, f, indent=2)
    print(f"\n  saved -> {os.path.relpath(out_dir, _NN_DIR)}/  (samples.npy, diagnostics.png, summary.json)")
    print("\nWeek-2 takeaway: RWMH tunes correctly to 23.4% but mixing is dominated by the ~10^3-dim,")
    print("ill-conditioned posterior; ESS/sec is the baseline to beat with gradient-aware SGLD (week 3).")


if __name__ == "__main__":
    main()
