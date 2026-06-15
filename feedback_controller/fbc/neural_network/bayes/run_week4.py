"""
Week-4 driver: predictive uncertainty of the Bayesian last layer (see main.tex, section 5).

    python -m bayes.run_week4 [--sigma-a 0.05] [--device cuda|cpu]

Question: does the last-layer posterior's predictive spread flag out-of-distribution states?
Compared methods: point estimate (theta_0), the analytic Gaussian posterior (exact reference),
RWMH samples, SGLD samples. Regimes: in-distribution (interp test), extrapolation (extrap test),
target-perturbed (interp states + noise on the 9 target coords).

Reports: action MSE (interp/extrap), predictive-std distributions, extrapolation-detection AUROC
(interp vs extrap), std-vs-error calibration, and std vs perturbation level.
Outputs -> ``bayes/results/week4/sigma<S>/``: ``std_distributions.png``, ``uncertainty_metrics.png``,
``summary.json``.
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

from bayes.dnfc_bll import BLLConfig, load_default, load_triples, _NN_DIR
from bayes.rwmh import run_rwmh, dispersed_init
from bayes.sgld import run_sgld
from bayes.run_sgld import stable_tau0
from bayes.predictive import (sample_analytic, predict_mean_std, perturb_targets,
                              auroc, action_rmse, calibration_curve)

METHOD_COLORS = {"point": "gray", "analytic": "black", "RWMH": "tab:orange", "SGLD": "tab:blue"}
PERTURB_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]


def pool_tail(samples: np.ndarray, m_total: int) -> torch.Tensor:
    """Pool the last ``m_total`` thinned draws across chains: (C, m, D) -> (<=m_total, D)."""
    C = samples.shape[0]
    per = max(1, m_total // C)
    return torch.as_tensor(samples[:, -per:, :].reshape(-1, samples.shape[-1]))


def get_samples(post, init, m_mcmc=200, m_analytic=1000):
    """Return {method: Theta (M, D)} for point / analytic / RWMH / SGLD."""
    theta = {}
    theta["point"] = post.theta0[None, :]                                    # M=1 -> std 0
    theta["analytic"] = sample_analytic(post, m_analytic, seed=0)
    tau0 = stable_tau0(post)
    sgld = run_sgld(post, init, n_iter=80000, burn_in=40000, thin=20,
                    tau0=tau0, tau_min=tau0 / 3, k0=10000.0, seed=0)
    rwmh = run_rwmh(post, init, n_iter=40000, burn_in=20000, thin=10,
                    tau0=max(1e-4, 0.05 * post.sigma_a), seed=0)
    theta["SGLD"] = pool_tail(sgld.samples, m_mcmc).to(post.device, post.dtype)
    theta["RWMH"] = pool_tail(rwmh.samples, m_mcmc).to(post.device, post.dtype)
    return theta, {"SGLD": sgld.samples.shape, "RWMH": rwmh.samples.shape}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-a", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    wrapper, post = load_default(BLLConfig(device=args.device, sigma_a=args.sigma_a))
    print(f"Week-4 predictive uncertainty | {post.summary()}")
    print("  checkpoint:", os.path.relpath(wrapper.tgt.ckpt, _NN_DIR))

    # --- regimes ---
    ds = wrapper.tgt.dataset_dir
    interp_file = wrapper.tgt.train_file.replace("train_", "test_")          # test_interp_0.85.npy
    extrap_file = interp_file.replace("interp", "extrap")                    # test_extrap_0.85.npy
    s_i, g_i, a_i = load_triples(ds, interp_file)
    s_e, g_e, a_e = load_triples(ds, extrap_file)
    print(f"  interp test: {s_i.shape[0]} states ({interp_file}) | extrap test: {s_e.shape[0]} ({extrap_file})")

    # --- samples ---
    init = dispersed_init(post, 4, sigma=min(5e-3, 0.1 * post.sigma_a), seed=0)
    print("  drawing posterior samples (analytic + RWMH + SGLD) ...")
    theta, shapes = get_samples(post, init)
    for k, v in theta.items():
        print(f"    {k:9s} M={v.shape[0]}")

    # --- predictive mean/std per method per regime ---
    pred = {}   # pred[method][regime] = (mean (N,7) tensor, std (N,) np)
    for name, Th in theta.items():
        pred[name] = {}
        for rname, (s, g) in (("interp", (s_i, g_i)), ("extrap", (s_e, g_e))):
            mean, std = predict_mean_std(wrapper, Th, s, g)
            pred[name][rname] = (mean, std.cpu().numpy())

    # ============================ metrics ============================
    print("\n" + "=" * 78 + "\nACTION ERROR  (MSE vs expert action)\n" + "=" * 78)
    print(f"  {'method':10}{'interp MSE':>14}{'extrap MSE':>14}{'extrap/interp':>15}")
    mse = {}
    for name in theta:
        mi = float(((pred[name]["interp"][0] - torch.as_tensor(a_i, device=post.device, dtype=post.dtype)).pow(2)).mean())
        me = float(((pred[name]["extrap"][0] - torch.as_tensor(a_e, device=post.device, dtype=post.dtype)).pow(2)).mean())
        mse[name] = dict(interp=mi, extrap=me, ratio=me / mi)
        print(f"  {name:10}{mi:>14.3e}{me:>14.3e}{me/mi:>15.2f}")

    print("\n" + "=" * 78 + "\nEXTRAPOLATION DETECTION  (predictive std as OOD score: interp vs extrap)\n" + "=" * 78)
    print(f"  {'method':10}{'median std (ID)':>17}{'median std (OOD)':>18}{'AUROC':>8}")
    auroc_res = {}
    for name in theta:
        std_i, std_e = pred[name]["interp"][1], pred[name]["extrap"][1]
        au = auroc(std_i, std_e)                          # higher std on extrap -> AUROC > 0.5
        auroc_res[name] = au
        print(f"  {name:10}{np.median(std_i):>17.3e}{np.median(std_e):>18.3e}{au:>8.3f}"
              + ("   (no spread)" if name == "point" else ""))

    # --- target-perturbation sweep (interp states, noise on target coords) ---
    print("\n" + "=" * 78 + "\nTARGET PERTURBATION  (mean predictive std vs noise on target coords)\n" + "=" * 78)
    perturb = {name: [] for name in theta}
    header = "  " + "level".ljust(10) + "".join(f"{n:>12}" for n in theta)
    print(header)
    for lvl in PERTURB_LEVELS:
        g_p = perturb_targets(g_i, lvl, seed=0)
        row = f"  {lvl:<10.3f}"
        for name, Th in theta.items():
            _, std = predict_mean_std(wrapper, Th, s_i, g_p)
            v = float(std.cpu().numpy().mean())
            perturb[name].append(v)
            row += f"{v:>12.3e}"
        print(row)

    # --- calibration: pool interp+extrap, bin by std, realised error per bin ---
    calib = {}
    for name in theta:
        if name == "point":
            continue
        std = np.concatenate([pred[name]["interp"][1], pred[name]["extrap"][1]])
        err = np.concatenate([action_rmse(pred[name]["interp"][0], a_i),
                              action_rmse(pred[name]["extrap"][0], a_e)])
        calib[name] = calibration_curve(std, err, n_bins=10)

    # ============================ figures ============================
    tag = f"sigma{args.sigma_a:g}"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "week4", tag)
    os.makedirs(out_dir, exist_ok=True)

    # Fig 1: predictive-std distributions, interp vs extrap, per (spread-bearing) method
    spread_methods = [m for m in ("analytic", "RWMH", "SGLD")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for j, name in enumerate(spread_methods):
        si, se = pred[name]["interp"][1], pred[name]["extrap"][1]
        lo = max(min(si.min(), se.min()), 1e-9)
        bins = np.logspace(np.log10(lo), np.log10(max(si.max(), se.max()) + 1e-12), 40)
        ax[j].hist(si, bins=bins, alpha=0.6, label="interp (ID)", color="tab:green", density=True)
        ax[j].hist(se, bins=bins, alpha=0.6, label="extrap (OOD)", color="tab:red", density=True)
        ax[j].set(title=f"{name}: predictive std  (AUROC={auroc_res[name]:.3f})",
                  xscale="log", xlabel="predictive action std", ylabel="density")
        ax[j].legend()
    fig.suptitle("Predictive-std distributions: in-distribution vs extrapolation", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "std_distributions.png"), dpi=110)
    plt.close(fig)

    # Fig 2: AUROC bars | calibration | perturbation
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    names_s = spread_methods
    ax[0].bar(names_s, [auroc_res[n] for n in names_s],
              color=[METHOD_COLORS[n] for n in names_s])
    ax[0].axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax[0].set(title="extrapolation-detection AUROC", ylim=(0, 1), ylabel="AUROC")
    ax[0].legend()

    for name in names_s:
        xs, ys = calib[name]
        ax[1].plot(xs, ys, "o-", color=METHOD_COLORS[name], label=name)
    ax[1].set(title="calibration: predicted std vs realised error", xscale="log",
              xlabel="predicted action std (bin mean)", ylabel="realised action RMSE")
    ax[1].legend()

    for name in theta:
        ax[2].plot(PERTURB_LEVELS, perturb[name], "o-", color=METHOD_COLORS[name], label=name)
    ax[2].set(title="predictive std vs target perturbation", xlabel="target-coord noise std (m)",
              ylabel="mean predictive std")
    ax[2].legend()
    fig.suptitle("Bayesian last-layer uncertainty quality", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "uncertainty_metrics.png"), dpi=110)
    plt.close(fig)

    # ============================ save + verdict ============================
    summary = dict(sigma_a=args.sigma_a, kappa_note="see week3", checkpoint=os.path.relpath(wrapper.tgt.ckpt, _NN_DIR),
                   n_interp=int(s_i.shape[0]), n_extrap=int(s_e.shape[0]),
                   action_mse=mse, auroc=auroc_res,
                   perturbation={"levels": PERTURB_LEVELS, **{k: v for k, v in perturb.items()}},
                   median_std={n: dict(interp=float(np.median(pred[n]["interp"][1])),
                                       extrap=float(np.median(pred[n]["extrap"][1]))) for n in theta})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78 + "\nVERDICT\n" + "=" * 78)
    print(f"  action error rises on extrapolation (point estimate): "
          f"{mse['point']['extrap']/mse['point']['interp']:.2f}x")
    best = max(("analytic", "RWMH", "SGLD"), key=lambda n: auroc_res[n])
    print(f"  best OOD-detection AUROC: {best} = {auroc_res[best]:.3f}  "
          f"(analytic {auroc_res['analytic']:.3f} | SGLD {auroc_res['SGLD']:.3f} | RWMH {auroc_res['RWMH']:.3f})")
    mono = all(perturb["analytic"][i] <= perturb["analytic"][i + 1] + 1e-9 for i in range(len(PERTURB_LEVELS) - 1))
    print(f"  analytic predictive std rises monotonically with target perturbation: {mono}")
    print(f"\n  saved -> {os.path.relpath(out_dir, _NN_DIR)}/ (std_distributions.png, uncertainty_metrics.png, summary.json)")


if __name__ == "__main__":
    main()
