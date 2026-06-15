"""
Week-1 verification for the Bayesian last-layer DNFC (see ``main.tex``, Week 1):

    "Wrap the existing trained DNFC with a frozen encoder and frozen controller trunk.
     Implement the tempered, subsampled Gaussian log-posterior and its gradient over the
     Bayesian last layer. Verify against the trained point estimate."

Run:  python -m bayes.verify_week1

Checks
  A. Wrapper equivalence -- the decomposed (frozen trunk + Bayesian head) path reproduces the
     ORIGINAL monolithic GeneralModel.forward at theta_0, and theta<->(W,b) round-trips exactly.
  B. Point-estimate fit -- action R^2 / RMSE at theta_0 on the N' subset; reports the resolved
     posterior hyper-parameters (theta_dim, N', B, T, alpha, sigma_a).
  C. Gradient correctness -- autograd grad_log_posterior vs finite differences (float64), plus the
     analytic prior gradient. The autograd graph is identical for any (alpha, sigma_a, T), so a clean
     moderate-noise check validates the stiff real posterior too.
  D. SGLD subsample scaling -- the minibatch gradient estimator (with the N'/B rescaling) is an
     unbiased estimate of the exact full-N' gradient: its average converges to the full gradient.
  E. online_tester integration -- wrapper.as_controller(theta) returns a drop-in GeneralModel with the
     exact ``model(target, state) -> (action, x_des, diff)`` signature online_test() expects.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from bayes.dnfc_bll import (BLLConfig, DNFCLastLayer, BLLPosterior,
                            load_triples, load_default, _NN_DIR)


def _hdr(s: str) -> None:
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _tensors(wrapper, n):
    state, target_repr, action = load_triples(wrapper.tgt.dataset_dir, wrapper.tgt.train_file)
    n = min(n, state.shape[0])
    to = lambda a: torch.as_tensor(a[:n], device=wrapper.device, dtype=wrapper.dtype)
    return to(state), to(target_repr), to(action)


def check_A_equivalence(wrapper: DNFCLastLayer) -> bool:
    _hdr("A. Wrapper equivalence  (decomposed path == original GeneralModel.forward)")
    s, g, _ = _tensors(wrapper, 2048)
    a_decomposed = wrapper.predict(wrapper.theta0, s, g)   # trunk + Bayesian head + tanh
    a_original = wrapper.predict_original(s, g)            # monolithic forward
    max_diff = float((a_decomposed - a_original).abs().max())

    W, b = wrapper.unflatten(wrapper.theta0)
    rt = float((wrapper.flatten(W, b) - wrapper.theta0).abs().max())
    match_W = float((W - wrapper._head.weight).abs().max())
    match_b = float((b - wrapper._head.bias).abs().max())

    ok = (max_diff < 1e-5) and (rt == 0.0) and (match_W == 0.0) and (match_b == 0.0)
    print(f"  max |a_decomposed - a_original|     = {max_diff:.3e}   (tol 1e-5)")
    print(f"  theta->(W,b)->theta round-trip err  = {rt:.3e}")
    print(f"  unflatten(theta0) vs head (W,b) err = {match_W:.3e}, {match_b:.3e}")
    print(f"  -> {_status(ok)}")
    return ok


def check_B_fit(post: BLLPosterior) -> bool:
    _hdr("B. Trained point-estimate fit  +  resolved hyper-parameters")
    th0 = post.theta0
    with torch.no_grad():
        resid = post.A - post.w.action_from_features(th0, post.Phi)
        rmse = float(resid.pow(2).mean().sqrt())
        mae = float(resid.abs().mean())
        ss_res = float(resid.pow(2).sum())
        ss_tot = float((post.A - post.A.mean(0, keepdim=True)).pow(2).sum())
        rmse_zero = float(post.A.pow(2).mean().sqrt())  # naive predict-zero baseline
        r2 = 1.0 - ss_res / ss_tot
    print("  " + post.summary())
    print(f"  action RMSE @ theta_0 (N' subset)   = {rmse:.3e}")
    print(f"  action MAE  @ theta_0 (N' subset)   = {mae:.3e}")
    print(f"  RMSE of predict-zero baseline       = {rmse_zero:.3e}   ({rmse_zero / rmse:.1f}x worse than theta_0)")
    print(f"  R^2 (variance explained)            = {r2:.4f}")
    print(f"  sigma_a (obs. noise)                = {post.sigma_a:.3e}   (== residual std by construction)")
    print(f"  alpha   (prior std)                 = {post.alpha:.3e}   (trained theta_0 std)")
    print(f"  log p(theta_0 | D)  [tempered, T={post.temperature:g}] = {post.log_posterior_value(th0):.4e}")
    ok = (r2 > 0.9) and (rmse < 0.5 * rmse_zero) and np.isfinite(post.log_posterior_value(th0))
    print(f"  -> {_status(ok)}")
    return ok


def _fd_directional(post: BLLPosterior, theta: torch.Tensor, v: torch.Tensor, h: float) -> float:
    with torch.no_grad():
        fp = post.log_posterior(theta + h * v)
        fm = post.log_posterior(theta - h * v)
    return float((fp - fm) / (2 * h))


def check_C_gradient() -> bool:
    _hdr("C. Gradient correctness  (autograd vs finite differences, float64)")
    # Moderate-noise float64 posterior: same autograd graph as the real one, well-conditioned for FD.
    _, post = load_default(BLLConfig(dtype=torch.float64, sigma_a=0.1, alpha=0.1, n_prime=1024))
    rng = torch.Generator().manual_seed(0)
    worst = 0.0
    for tag, theta in [("theta_0", post.theta0),
                       ("prior draw", 0.1 * torch.randn(post.theta_dim, dtype=torch.float64, generator=rng))]:
        grad = post.grad_log_posterior(theta)
        rel_errs = []
        for _ in range(8):
            v = torch.randn(post.theta_dim, dtype=torch.float64, generator=rng)
            v = v / v.norm()
            analytic = float(grad @ v)
            fd = _fd_directional(post, theta, v, h=1e-6)
            rel_errs.append(abs(analytic - fd) / (abs(analytic) + 1e-12))
        m = max(rel_errs)
        worst = max(worst, m)
        print(f"  [{tag:11s}] ||grad||={float(grad.norm()):.4e}   max rel err (8 dirs) = {m:.3e}")

    theta = post.theta0
    theta_req = theta.detach().clone().requires_grad_(True)
    (gp,) = torch.autograd.grad(post.beta * post.log_prior(theta_req), theta_req)
    gp_analytic = -post.beta * theta / post.alpha ** 2
    prior_err = float((gp - gp_analytic).abs().max())
    print(f"  prior grad: autograd vs -beta*theta/alpha^2  max err = {prior_err:.3e}")

    ok = (worst < 1e-6) and (prior_err < 1e-9)
    print(f"  -> {_status(ok)}")
    return ok


def check_D_sgld_unbiased(post: BLLPosterior) -> bool:
    _hdr("D. SGLD subsample scaling  (minibatch gradient -> exact full-N' gradient)")
    theta = post.theta0
    g_full = post.grad_log_posterior(theta, idx=None)
    gen = torch.Generator(device=post.device).manual_seed(0)

    def mean_minibatch_grad(num_batches: int) -> torch.Tensor:
        acc = torch.zeros_like(g_full)
        for _ in range(num_batches):
            acc += post.grad_log_posterior(theta, idx=post.sample_minibatch(generator=gen))
        return acc / num_batches

    def report(num_batches: int) -> float:
        gm = mean_minibatch_grad(num_batches)
        cos = float(torch.nn.functional.cosine_similarity(gm, g_full, dim=0))
        rel = float((gm - g_full).norm() / g_full.norm())
        print(f"  avg of {num_batches:4d} minibatch grads:  cos(full)={cos:.5f}   rel L2 err={rel:.3e}")
        return rel

    rel_1, rel_50, rel_800 = report(1), report(50), report(800)
    ok = (rel_800 < rel_50 < rel_1) and (rel_800 < 0.05)
    print(f"  error decreases with more minibatches: {rel_1:.3e} -> {rel_50:.3e} -> {rel_800:.3e}")
    print(f"  -> {_status(ok)}")
    return ok


def check_E_controller_adapter(wrapper: DNFCLastLayer) -> bool:
    _hdr("E. online_tester integration  (as_controller -> drop-in GeneralModel)")
    s, g, _ = _tensors(wrapper, 256)
    ref = wrapper.predict_original(s, g)  # action at theta_0 from the monolithic model

    # (1) as_controller(theta_0) returns a GeneralModel with the exact online_test signature/values.
    model = wrapper.as_controller(wrapper.theta0)
    out = model(g, s)  # GeneralModel.forward(target_repr, state)
    sig_ok = isinstance(out, tuple) and len(out) == 3
    acts, x_des, diff = out
    shape_ok = tuple(acts.shape) == (s.shape[0], 7)
    err0 = float((acts - ref).abs().max())

    # (2) a different theta actually changes the action, and restoring theta_0 brings it back exactly.
    theta_rand = wrapper.theta0 + 0.5 * torch.randn_like(wrapper.theta0)
    acts_rand = wrapper.as_controller(theta_rand)(g, s)[0]
    changed = float((acts_rand - ref).abs().mean()) > 1e-3
    acts_restore = wrapper.as_controller(wrapper.theta0)(g, s)[0]  # restore theta_0
    restored = float((acts_restore - ref).abs().max()) < 1e-6

    ok = sig_ok and shape_ok and (err0 < 1e-5) and changed and restored
    print(f"  forward returns (action, x_des, diff) tuple : {sig_ok}")
    print(f"  action shape == (N, 7)                      : {shape_ok}  {tuple(acts.shape)}")
    print(f"  as_controller(theta_0) == original forward  : err {err0:.3e}")
    print(f"  swapping theta changes the action           : {changed}")
    print(f"  restoring theta_0 recovers it exactly       : {restored}")
    print(f"  -> {_status(ok)}")
    return ok


def main() -> None:
    torch.manual_seed(0)
    print("Building frozen DNFC wrapper + tempered subsampled posterior ...")
    wrapper, post = load_default()
    print("  derived checkpoint :", os.path.relpath(wrapper.tgt.ckpt, _NN_DIR))
    print(f"  model_name         : {wrapper.tgt.model_name}")
    print(f"  dataset / ds_ratio : {wrapper.tgt.dataset_name} / {wrapper.tgt.ds_ratio}")
    print(f"  N' subset drawn from train split: {post.idx_subset.shape[0]} triples"
          f" (pool {'train_indices' if wrapper.tgt.split_file else 'ALL rows'})")

    results = {
        "A. wrapper equivalence": check_A_equivalence(wrapper),
        "B. point-estimate fit": check_B_fit(post),
        "C. gradient correctness": check_C_gradient(),
        "D. SGLD subsample scaling": check_D_sgld_unbiased(post),
        "E. online_tester integration": check_E_controller_adapter(wrapper),
    }

    _hdr("SUMMARY")
    for name, ok in results.items():
        print(f"  [{_status(ok)}]  {name}")
    allok = all(results.values())
    print("\n" + ("ALL CHECKS PASSED -- Week-1 foundation is ready for RWMH (week 2)."
                  if allok else "SOME CHECKS FAILED -- see above."))
    raise SystemExit(0 if allok else 1)


if __name__ == "__main__":
    main()
