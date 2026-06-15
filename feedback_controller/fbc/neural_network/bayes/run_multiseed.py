"""
Multi-seed robustness: repeat the RWMH-vs-SGLD efficiency comparison and the predictive-uncertainty
experiments across several independently-trained DNFC controllers (different init seeds), and report
mean +/- std. Uses the non-AMC medium model (`2+2l_lat:sub-nvel`, 10 seeds available); architecturally
identical to the +AMC variant used elsewhere.

    python -m bayes.run_multiseed [--seeds 0 1 2 3 4] [--sigma-a 0.05] [--device cuda|cpu]

Turns the single-model headline numbers into error bars: the point is to show the SGLD>RWMH ordering
and the qualitative findings hold across models, not just one. Outputs -> ``bayes/results/multiseed/``.
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
from bayes.rwmh import dispersed_init
from bayes.run_sgld import run_both, lambda_max_gram
from bayes.predictive import sample_analytic, predict_mean_std, perturb_targets, auroc
from bayes.run_week4 import pool_tail, PERTURB_LEVELS

V_NAME = "2+2l_lat:sub-nvel"            # non-AMC model with 10 seeds
UNC_METHODS = ["analytic", "SGLD", "RWMH"]
COLORS = {"analytic": "black", "SGLD": "tab:blue", "RWMH": "tab:orange"}


def run_one_seed(train_no: int, sigma_a: float, device: str) -> dict:
    wrapper, post = load_default(BLLConfig(device=device, sigma_a=sigma_a, v_name=V_NAME, train_no=train_no))
    init = dispersed_init(post, 4, sigma=min(5e-3, 0.1 * sigma_a), seed=0)

    # --- efficiency (week-3 head-to-head) ---
    sgld, rwmh, m_sgld, m_rwmh = run_both(post, init)

    # --- predictive uncertainty (week-4) ---
    ds = wrapper.tgt.dataset_dir
    interp_file = wrapper.tgt.train_file.replace("train_", "test_")
    extrap_file = interp_file.replace("interp", "extrap")
    s_i, g_i, a_i = load_triples(ds, interp_file)
    s_e, g_e, a_e = load_triples(ds, extrap_file)
    theta = {"point": post.theta0[None, :],
             "analytic": sample_analytic(post, 1000, seed=0),
             "SGLD": pool_tail(sgld.samples, 200).to(post.device, post.dtype),
             "RWMH": pool_tail(rwmh.samples, 200).to(post.device, post.dtype)}

    au, mse_ratio, pert = {}, {}, {n: [] for n in theta}
    for nm, Th in theta.items():
        mi, sti = predict_mean_std(wrapper, Th, s_i, g_i)
        me, ste = predict_mean_std(wrapper, Th, s_e, g_e)
        au[nm] = auroc(sti.cpu().numpy(), ste.cpu().numpy())
        ei = float((mi - torch.as_tensor(a_i, device=post.device, dtype=post.dtype)).pow(2).mean())
        ee = float((me - torch.as_tensor(a_e, device=post.device, dtype=post.dtype)).pow(2).mean())
        mse_ratio[nm] = ee / ei
    for lvl in PERTURB_LEVELS:
        g_p = perturb_targets(g_i, lvl, seed=0)
        for nm, Th in theta.items():
            _, std = predict_mean_std(wrapper, Th, s_i, g_p)
            pert[nm].append(float(std.cpu().numpy().mean()))

    return dict(
        alpha=post.alpha,
        kappa=1.0 + (post.alpha ** 2 / post.sigma_a ** 2) * lambda_max_gram(post),
        sgld_logp={k: m_sgld["logp"][k] for k in ("ess_per_iter", "ess_per_Mterm", "ess_per_sec")},
        rwmh_logp={k: m_rwmh["logp"][k] for k in ("ess_per_iter", "ess_per_Mterm", "ess_per_sec")},
        sgld_logp_rhat=m_sgld["logp_rhat"], rwmh_logp_rhat=m_rwmh["logp_rhat"],
        auroc=au, mse_ratio=mse_ratio, perturb=pert,
    )


def _ms(vals):  # mean, std
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sigma-a", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Multi-seed robustness | model={V_NAME} | seeds={args.seeds} | sigma_a={args.sigma_a}")
    per_seed = []
    for k in args.seeds:
        print(f"  === seed {k} ===", flush=True)
        r = run_one_seed(k, args.sigma_a, args.device)
        print(f"    kappa={r['kappa']:.2e} | logp ESS/1e6term  SGLD={r['sgld_logp']['ess_per_Mterm']:.2f} "
              f"RWMH={r['rwmh_logp']['ess_per_Mterm']:.3f} "
              f"(ratio {r['sgld_logp']['ess_per_Mterm']/max(r['rwmh_logp']['ess_per_Mterm'],1e-9):.1f}x) | "
              f"AUROC analytic={r['auroc']['analytic']:.3f}", flush=True)
        per_seed.append(r)

    S = len(per_seed)
    # ---- aggregate ----
    agg = {}
    for samp in ("sgld", "rwmh"):
        for k in ("ess_per_iter", "ess_per_Mterm", "ess_per_sec"):
            agg[f"{samp}_logp_{k}"] = _ms([r[f"{samp}_logp"][k] for r in per_seed])
    ratio_Mterm = [r["sgld_logp"]["ess_per_Mterm"] / max(r["rwmh_logp"]["ess_per_Mterm"], 1e-9) for r in per_seed]
    agg["sgld_over_rwmh_ess_per_Mterm"] = _ms(ratio_Mterm)
    for nm in UNC_METHODS:
        agg[f"auroc_{nm}"] = _ms([r["auroc"][nm] for r in per_seed])
    agg["mse_ratio_point"] = _ms([r["mse_ratio"]["point"] for r in per_seed])
    pert_arr = {nm: np.array([r["perturb"][nm] for r in per_seed]) for nm in theta_keys(per_seed)}

    # ============================ report table ============================
    lines = []
    lines.append(f"# Multi-seed results ({S} seeds, non-AMC model, sigma_a={args.sigma_a})\n")
    lines.append("All values mean +/- std across independently-trained seeds.\n")
    lines.append("## Sampler efficiency (log-posterior)\n")
    lines.append("| metric | RWMH | SGLD | SGLD/RWMH |")
    lines.append("|---|---|---|---|")
    for k, lab in (("ess_per_iter", "ESS / iteration"), ("ess_per_Mterm", "ESS / 1e6 term-evals"),
                   ("ess_per_sec", "ESS / sec (GPU)")):
        rm, rs = agg[f"rwmh_logp_{k}"]
        sm, ss = agg[f"sgld_logp_{k}"]
        rr = "—"
        if k == "ess_per_Mterm":
            rr = f"**{agg['sgld_over_rwmh_ess_per_Mterm'][0]:.1f}x ± {agg['sgld_over_rwmh_ess_per_Mterm'][1]:.1f}**"
        lines.append(f"| {lab} | {rm:.3g} ± {rs:.2g} | {sm:.3g} ± {ss:.2g} | {rr} |")
    lines.append("\n## Predictive uncertainty\n")
    lines.append("| method | extrap-detection AUROC | (chance = 0.5) |")
    lines.append("|---|---|---|")
    for nm in UNC_METHODS:
        m, s = agg[f"auroc_{nm}"]
        lines.append(f"| {nm} | {m:.3f} ± {s:.3f} | {'null (not OOD)' if abs(m-0.5)<0.06 else ''} |")
    mr_m, mr_s = agg["mse_ratio_point"]
    lines.append(f"\nAction MSE extrap/interp (point estimate): **{mr_m:.2f} ± {mr_s:.2f}** "
                 f"(≈1 ⇒ extrap not harder than interp).\n")
    lines.append("## Target-perturbation: mean predictive std vs noise level\n")
    lines.append("| level (m) | " + " | ".join(UNC_METHODS) + " |")
    lines.append("|---|" + "---|" * len(UNC_METHODS))
    for j, lvl in enumerate(PERTURB_LEVELS):
        row = f"| {lvl:.2f} | " + " | ".join(f"{pert_arr[nm][:, j].mean():.4f} ± {pert_arr[nm][:, j].std():.4f}"
                                              for nm in UNC_METHODS) + " |"
        lines.append(row)
    table = "\n".join(lines)
    print("\n" + table)

    # ============================ figures ============================
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "multiseed")
    os.makedirs(out_dir, exist_ok=True)

    # efficiency: per-seed scatter + mean for ESS/1e6-terms; and the per-seed ratio
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    rwmh_pts = [r["rwmh_logp"]["ess_per_Mterm"] for r in per_seed]
    sgld_pts = [r["sgld_logp"]["ess_per_Mterm"] for r in per_seed]
    ax[0].scatter(np.zeros(S) + 0.05 * np.random.randn(S), rwmh_pts, color=COLORS["RWMH"], label="RWMH", zorder=3)
    ax[0].scatter(np.ones(S) + 0.05 * np.random.randn(S), sgld_pts, color=COLORS["SGLD"], label="SGLD", zorder=3)
    ax[0].errorbar([0, 1], [np.mean(rwmh_pts), np.mean(sgld_pts)],
                   yerr=[np.std(rwmh_pts), np.std(sgld_pts)], fmt="_", ms=40, color="k", lw=2, zorder=2)
    ax[0].set(title="log-posterior ESS / 1e6 term-evals (per seed)", xticks=[0, 1],
              yscale="log", ylabel="ESS / 1e6 terms")
    ax[0].set_xticklabels(["RWMH", "SGLD"])
    ax[1].scatter(args.seeds, ratio_Mterm, color="tab:purple", zorder=3)
    ax[1].axhline(np.mean(ratio_Mterm), color="k", ls="--",
                  label=f"mean {np.mean(ratio_Mterm):.1f}x ± {np.std(ratio_Mterm):.1f}")
    ax[1].axhline(1.0, color="gray", ls=":", label="parity")
    ax[1].set(title="SGLD / RWMH compute efficiency (per seed)", xlabel="seed",
              ylabel="ESS-per-compute ratio")
    ax[1].legend()
    for a in ax:
        a.legend()
    fig.suptitle(f"RWMH vs SGLD efficiency across {S} trained seeds", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "multiseed_efficiency.png"), dpi=110)
    plt.close(fig)

    # uncertainty: perturbation bands + AUROC per-seed
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for nm in UNC_METHODS:
        m = pert_arr[nm].mean(0)
        s = pert_arr[nm].std(0)
        ax[0].plot(PERTURB_LEVELS, m, "o-", color=COLORS[nm], label=nm)
        ax[0].fill_between(PERTURB_LEVELS, m - s, m + s, color=COLORS[nm], alpha=0.2)
    ax[0].set(title="predictive std vs target perturbation (mean ± std over seeds)",
              xlabel="target-coord noise std (m)", ylabel="mean predictive std")
    ax[0].legend()
    for i, nm in enumerate(UNC_METHODS):
        pts = [r["auroc"][nm] for r in per_seed]
        ax[1].scatter(np.zeros(S) + i + 0.06 * np.random.randn(S), pts, color=COLORS[nm], zorder=3)
        ax[1].errorbar([i], [np.mean(pts)], yerr=[np.std(pts)], fmt="_", ms=30, color="k", lw=2)
    ax[1].axhline(0.5, color="gray", ls=":", label="chance")
    ax[1].set(title="extrapolation-detection AUROC (per seed)", xticks=range(len(UNC_METHODS)),
              ylim=(0.3, 0.7), ylabel="AUROC")
    ax[1].set_xticklabels(UNC_METHODS)
    ax[1].legend()
    fig.suptitle(f"Predictive uncertainty across {S} trained seeds", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "multiseed_uncertainty.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(out_dir, "TABLE.md"), "w") as f:
        f.write(table)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "sigma_a": args.sigma_a, "model": V_NAME,
                   "aggregate": {k: list(v) for k, v in agg.items()},
                   "per_seed": [{k: (v if not isinstance(v, dict) else v) for k, v in r.items()
                                 if k not in ("perturb",)} for r in per_seed]}, f, indent=2, default=float)
    print(f"\nsaved -> {os.path.relpath(out_dir, _NN_DIR)}/ (multiseed_efficiency.png, "
          f"multiseed_uncertainty.png, TABLE.md, summary.json)")


def theta_keys(per_seed):
    return list(per_seed[0]["perturb"].keys())


if __name__ == "__main__":
    main()
