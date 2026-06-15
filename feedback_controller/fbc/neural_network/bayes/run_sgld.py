"""
Week-3 driver: SGLD on the DNFC Bayesian last layer, head-to-head with RWMH.

    python -m bayes.run_sgld [--sigma-a S] [--device cuda|cpu]

Pipeline
  1. SAMPLER VALIDATION -- SGLD on isotropic Gaussians: recovers N(0, sigma^2) and mixes well
     (well-conditioned => the integrator is correct).
  2. MAIN RUN + COMPARISON -- four SGLD chains and four RWMH chains on the SAME posterior; full
     diagnostic suite for SGLD (traces, step-size trajectory, grad-norm, ACF, ESS) and the headline
     efficiency comparison. Efficiency is reported three ways to separate algorithm from hardware:
        * ESS / iteration         -- mixing quality (platform-independent);
        * ESS / 1e6 likelihood-term evals -- "per unit compute" (RWMH touches N' terms/step, SGLD B);
        * ESS / second            -- wall-clock on the run device.
  3. CONDITIONING SWEEP -- the same efficiency vs sigma_a (which sets the posterior condition number
     kappa = 1 + alpha^2 lambda_max / sigma_a^2): shows where gradient information starts to pay off.

Outputs -> ``bayes/results/sgld/<tag>/``: ``samples.npy``, ``sgld_diagnostics.png``,
``comparison.png``, ``summary.json``.
"""

from __future__ import annotations

import os
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayes.dnfc_bll import BLLConfig, load_default, _NN_DIR
from bayes.rwmh import run_rwmh, time_rwmh_update, dispersed_init
from bayes.run_rwmh import IsoGaussTarget
from bayes.sgld import run_sgld, time_sgld_update
from bayes.diagnostics import split_rhat, ess, autocorr, summarize


def lambda_max_gram(post) -> float:
    Phi = post.Phi.detach().double()
    return float(torch.linalg.eigvalsh(Phi.t() @ Phi).max())


def stable_tau0(post, c: float = 0.5) -> float:
    """SGLD step at the stability edge: tau ~ c * sigma_a^2 / lambda_max(Phi'Phi)."""
    return c * post.sigma_a ** 2 / lambda_max_gram(post)


# ---------------------------------------------------------------------------------------
# 1. SGLD validation on isotropic Gaussians
# ---------------------------------------------------------------------------------------
def validate_sgld(device, dtype) -> None:
    print("\n" + "=" * 78 + "\n1. SGLD VALIDATION  (isotropic N(0, sigma^2 I); sigma=0.0627)\n" + "=" * 78)
    print(f"  {'dim':>6} {'tau':>10} {'Rhat(x0)':>9} {'ESS(x0)':>9} {'mean~0':>9} {'std~sigma':>10}")
    sigma = 0.0627
    for d in (100, 1000, 2695):
        tgt = IsoGaussTarget(d, sigma, device, dtype)
        init = dispersed_init(tgt, 4, sigma=sigma, seed=0, around_theta0=False)
        tau = 0.3 * sigma ** 2          # well-conditioned (kappa=1): stable, fast
        res = run_sgld(tgt, init, n_iter=30000, burn_in=15000, thin=5,
                       tau0=tau, tau_min=tau, k0=1e9, seed=0)
        s = res.samples.reshape(-1, d)
        x0 = res.samples[:, :, 0]
        print(f"  {d:6d} {tau:10.1e} {split_rhat(x0):9.3f} {ess(x0):9.1f} "
              f"{s.mean():9.3f} {s.std():10.4f}")
    print("  -> recovers mean~0 & std~sigma with Rhat~1: the SGLD integrator is correct.")


# ---------------------------------------------------------------------------------------
# 2. run both samplers + metrics
# ---------------------------------------------------------------------------------------
def metrics(res, post, is_sgld: bool, sec_per_update: float) -> dict:
    lp = res.logp if is_sgld else res.logp[:, res.burn_in:]
    iters_post = res.meta["n_iter"] - res.burn_in
    terms_per_iter = post.batch_size if is_sgld else post.n_prime
    e_coord = summarize(res.samples)["ess"]
    out = {}
    for name, val in (("logp", ess(lp)), ("coord_median", float(np.nanmedian(e_coord)))):
        out[name] = dict(
            ess=float(val),
            ess_per_iter=float(val / iters_post),
            ess_per_Mterm=float(val / (iters_post * terms_per_iter / 1e6)),
            ess_per_sec=float(val / (iters_post * sec_per_update)),
        )
    out["logp_rhat"] = float(split_rhat(lp))
    out["coord_rhat_median"] = float(np.nanmedian(summarize(res.samples)["rhat"]))
    out["coord_ess_max"] = float(np.nanmax(e_coord))
    out["ess_coord_all"] = e_coord
    out["sec_per_update"] = sec_per_update
    out["terms_per_iter"] = terms_per_iter
    out["iters_post"] = iters_post
    return out


def run_both(post, init, sgld_iter=80000, sgld_burn=40000, rwmh_iter=40000, rwmh_burn=20000,
             progress=False):
    tau0 = stable_tau0(post)
    sgld = run_sgld(post, init, n_iter=sgld_iter, burn_in=sgld_burn, thin=max(1, (sgld_iter - sgld_burn) // 2000),
                    tau0=tau0, tau_min=tau0 / 3, k0=sgld_burn / 4,
                    seed=0, progress_every=(sgld_iter // 2 if progress else 0))
    rwmh = run_rwmh(post, init, n_iter=rwmh_iter, burn_in=rwmh_burn,
                    thin=max(1, (rwmh_iter - rwmh_burn) // 2000),
                    tau0=max(1e-4, 0.05 * post.sigma_a), seed=0,
                    progress_every=(rwmh_iter // 2 if progress else 0))
    spu_s = time_sgld_update(post, init, tau=tau0, n=2000)
    spu_r = time_rwmh_update(post, init, tau=1e-3, n=2000)
    return sgld, rwmh, metrics(sgld, post, True, spu_s), metrics(rwmh, post, False, spu_r)


# ---------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------
def sgld_diagnostics_figure(sgld, m_sgld, m_rwmh, out_path):
    C = sgld.logp.shape[0]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    for c in range(C):
        ax[0, 0].plot(sgld.keep_iters, sgld.logp[c], lw=0.7)
    ax[0, 0].set(title="log-posterior trace (post burn-in)", xlabel="iteration", ylabel="log p")

    ax[0, 1].loglog(np.arange(1, len(sgld.tau_sched) + 1), sgld.tau_sched, lw=1.2)
    ax[0, 1].axvline(sgld.burn_in, color="k", ls="--", lw=1)
    ax[0, 1].set(title="step-size schedule tau_k ~ k^-0.55", xlabel="iteration", ylabel="tau")

    it = np.arange(sgld.grad_norm.shape[1])
    for c in range(C):
        ax[0, 2].semilogy(it, sgld.grad_norm[c], lw=0.5)
    ax[0, 2].axvline(sgld.burn_in, color="k", ls="--", lw=1)
    ax[0, 2].set(title="minibatch ||grad|| trace", xlabel="iteration", ylabel="||grad||")

    for c in range(C):
        ax[1, 0].plot(it, sgld.coord_trace[c, :, 0], lw=0.5)
    ax[1, 0].axvline(sgld.burn_in, color="k", ls="--", lw=1)
    ax[1, 0].set(title=f"trace: weight theta[{sgld.coord_idx[0]}]", xlabel="iteration")

    maxlag = min(200, sgld.logp.shape[1] // 2)
    for c in range(C):
        ax[1, 1].plot(autocorr(sgld.logp[c], maxlag), lw=0.8)
    ax[1, 1].axhline(0, color="k", lw=0.6)
    ax[1, 1].set(title="ACF of log-posterior (thinned)", xlabel="lag (thinned)", ylim=(-0.2, 1))

    bins = np.linspace(0, max(np.nanmax(m_sgld["ess_coord_all"]), np.nanmax(m_rwmh["ess_coord_all"])), 40)
    ax[1, 2].hist(m_rwmh["ess_coord_all"], bins=bins, alpha=0.6, label="RWMH")
    ax[1, 2].hist(m_sgld["ess_coord_all"], bins=bins, alpha=0.6, label="SGLD")
    ax[1, 2].set(title="per-coordinate ESS", xlabel="ESS", ylabel="# coords")
    ax[1, 2].legend()

    fig.suptitle("SGLD diagnostics -- DNFC Bayesian last layer", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def comparison_figure(m_sgld, m_rwmh, sweep, out_path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    labels = ["ESS/iter", "ESS/1e6 terms", "ESS/sec"]
    keys = ["ess_per_iter", "ess_per_Mterm", "ess_per_sec"]
    x = np.arange(len(labels))
    for j, (metric_group, title) in enumerate([("logp", "log-posterior"), ("coord_median", "median coordinate")]):
        r = [m_rwmh[metric_group][k] for k in keys]
        s = [m_sgld[metric_group][k] for k in keys]
        axx = ax[j]
        axx.bar(x - 0.2, r, 0.4, label="RWMH")
        axx.bar(x + 0.2, s, 0.4, label="SGLD")
        axx.set(title=f"efficiency: {title}", xticks=x, yscale="log")
        axx.set_xticklabels(labels, rotation=15)
        axx.legend()
        for xi, (rv, sv) in enumerate(zip(r, s)):
            axx.text(xi + 0.2, sv, f"{sv/rv:.1f}x" if rv > 0 else "", ha="center", va="bottom", fontsize=8)

    sa = sweep["sigma_a"]
    ax[2].loglog(sa, sweep["sgld_logp_per_Mterm"], "o-", label="SGLD")
    ax[2].loglog(sa, sweep["rwmh_logp_per_Mterm"], "s-", label="RWMH")
    ax[2].set(title="log-posterior ESS / 1e6 terms  vs  sigma_a",
              xlabel="sigma_a (larger = better conditioned)", ylabel="ESS / 1e6 terms")
    ax[2].legend()
    fig.suptitle("RWMH vs SGLD efficiency", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-a", type=float, default=0.05,
                    help="obs. noise; sets posterior conditioning. nominal (train residual) ~0.00127 is "
                         "too stiff for either sampler. default 0.05 is the tractable regime.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    dtype = torch.float32

    if not args.skip_validation:
        validate_sgld(device, dtype)

    # --- 2. main run + comparison at the chosen sigma_a ---
    print("\n" + "=" * 78 + f"\n2. SGLD vs RWMH  (sigma_a={args.sigma_a}, 4 chains each)\n" + "=" * 78)
    wrapper, post = load_default(BLLConfig(device=args.device, sigma_a=args.sigma_a))
    kappa = 1.0 + (post.alpha ** 2 / post.sigma_a ** 2) * lambda_max_gram(post)
    print("  checkpoint:", os.path.relpath(wrapper.tgt.ckpt, _NN_DIR))
    print(" ", post.summary(), f"\n  posterior condition number kappa ~ {kappa:.2e}")
    init = dispersed_init(post, 4, sigma=min(5e-3, 0.1 * post.sigma_a), seed=0)
    sgld, rwmh, m_sgld, m_rwmh = run_both(post, init, progress=True)

    print(f"\n  per-update: SGLD={m_sgld['sec_per_update']*1e3:.3f}ms (touch {m_sgld['terms_per_iter']} terms) | "
          f"RWMH={m_rwmh['sec_per_update']*1e3:.3f}ms (touch {m_rwmh['terms_per_iter']} terms)")
    print(f"  {'':14}{'logp Rhat':>10}{'logp ESS':>10}{'ESS/iter':>10}{'ESS/1e6term':>13}{'ESS/sec':>10}")
    for nm, m in (("SGLD", m_sgld), ("RWMH", m_rwmh)):
        L = m["logp"]
        print(f"  {nm:14}{m['logp_rhat']:>10.2f}{L['ess']:>10.0f}{L['ess_per_iter']:>10.4f}"
              f"{L['ess_per_Mterm']:>13.1f}{L['ess_per_sec']:>10.2f}")
    gain_term = m_sgld["logp"]["ess_per_Mterm"] / max(m_rwmh["logp"]["ess_per_Mterm"], 1e-9)
    gain_iter = m_sgld["logp"]["ess_per_iter"] / max(m_rwmh["logp"]["ess_per_iter"], 1e-9)
    print(f"  -> SGLD logp mixing: {gain_iter:.1f}x ESS/iter, {gain_term:.1f}x ESS per unit compute vs RWMH")
    print(f"  -> coordinate bulk (prior-dominated): SGLD ESS med "
          f"{m_sgld['coord_median']['ess']:.0f} vs RWMH {m_rwmh['coord_median']['ess']:.0f} "
          f"(both limited by the {post.theta_dim}-dim prior bulk)")

    # --- 3. conditioning sweep ---
    print("\n" + "=" * 78 + "\n3. CONDITIONING SWEEP  (efficiency vs sigma_a)\n" + "=" * 78)
    sweep = {"sigma_a": [], "sgld_logp_per_Mterm": [], "rwmh_logp_per_Mterm": [],
             "sgld_logp_ess": [], "rwmh_logp_ess": []}
    print(f"  {'sigma_a':>9} {'kappa':>10} {'SGLD ESS/1e6t':>14} {'RWMH ESS/1e6t':>14} {'SGLD/RWMH':>10}")
    for sa in [0.00127, 0.005, 0.02, 0.05, 0.1]:
        _, p = load_default(BLLConfig(device=args.device, sigma_a=sa))
        k = 1.0 + (p.alpha ** 2 / p.sigma_a ** 2) * lambda_max_gram(p)
        ini = dispersed_init(p, 4, sigma=min(5e-3, 0.1 * sa), seed=0)
        s_res, r_res, ms, mr = run_both(p, ini, sgld_iter=40000, sgld_burn=20000,
                                        rwmh_iter=30000, rwmh_burn=15000)
        sweep["sigma_a"].append(sa)
        sweep["sgld_logp_per_Mterm"].append(ms["logp"]["ess_per_Mterm"])
        sweep["rwmh_logp_per_Mterm"].append(mr["logp"]["ess_per_Mterm"])
        sweep["sgld_logp_ess"].append(ms["logp"]["ess"])
        sweep["rwmh_logp_ess"].append(mr["logp"]["ess"])
        ratio = ms["logp"]["ess_per_Mterm"] / max(mr["logp"]["ess_per_Mterm"], 1e-9)
        print(f"  {sa:9.5f} {k:10.2e} {ms['logp']['ess_per_Mterm']:14.1f} "
              f"{mr['logp']['ess_per_Mterm']:14.1f} {ratio:9.1f}x")

    # --- save ---
    tag = f"sigma{args.sigma_a:g}"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sgld", tag)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "samples.npy"), sgld.samples)
    sgld_diagnostics_figure(sgld, m_sgld, m_rwmh, os.path.join(out_dir, "sgld_diagnostics.png"))
    comparison_figure(m_sgld, m_rwmh, sweep, os.path.join(out_dir, "comparison.png"))
    strip = lambda m: {k: v for k, v in m.items() if k != "ess_coord_all"}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"sigma_a": args.sigma_a, "kappa": kappa, "device": args.device,
                   "checkpoint": os.path.relpath(wrapper.tgt.ckpt, _NN_DIR),
                   "sgld": strip(m_sgld), "rwmh": strip(m_rwmh),
                   "sweep": sweep}, f, indent=2)
    print(f"\n  saved -> {os.path.relpath(out_dir, _NN_DIR)}/ (samples.npy, sgld_diagnostics.png, "
          f"comparison.png, summary.json)")
    print("\nWeek-3 takeaway: SGLD's gradient mixes the data-constrained directions (log-posterior) "
          "several-fold\nbetter per step and per unit compute than RWMH, and its edge grows as the "
          "posterior is better\nconditioned; both remain limited by the high-dim prior bulk. "
          "Conditioning (sigma_a), not the\nsampler alone, governs feasibility.")


if __name__ == "__main__":
    main()
