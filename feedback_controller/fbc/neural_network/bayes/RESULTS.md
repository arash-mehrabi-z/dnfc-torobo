# Bayesian Last-Layer Uncertainty in the DNFC Controller — Results

Comparing random-walk Metropolis–Hastings (RWMH) and stochastic-gradient Langevin dynamics (SGLD)
for last-layer Bayesian inference on the DNFC neural feedback controller. This document consolidates
the weeks-1–4 results (see `main.tex` for the proposal). All numbers are reproducible with the
drivers in this package — see `README.md`.

## Setup

We freeze a trained DNFC (`GeneralModel`, medium / `2+2l_lat:sub-nvel+AMC`, 12.053K params,
`interp_0.85` split, `train_no_0`, epoch 4000) and place a Gaussian prior on **only its final
controller linear layer**:

- feature `φ(s,g) = relu(W₀(enc1(g) − s) + b₀) ∈ ℝ³⁸⁴` from the frozen trunk (precomputed once);
- action `f_θ(s,g) = tanh(W φ + b)`, Bayesian parameter `θ = (W, b) ∈ ℝ²⁶⁹⁵` (7×384 + 7);
- prior `N(0, α²I)`, likelihood `N(a_i | f_θ, σ_a² I₇)`, tempered posterior `p(θ|D)^{1/T}`.

**Hyperparameters** (reported up front): `α = 0.0627` (= std of the trained last layer), `T = 1`,
`N' = 4096` training triples (drawn from the genuine train split), SGLD minibatch `B = 128`.
The observation noise **`σ_a`** is the key modelling knob — it sets the posterior **condition
number** `κ = 1 + (α²/σ_a²)·λ_max(Φ'Φ)`, which dominates everything below:

| σ_a | meaning | κ (cond. number) | samplable? |
|---|---|---|---|
| 0.00127 | train-residual (overconfident) | 6.5×10⁶ | no — needle-sharp posterior |
| **0.05** | **deliberate "actions ≈ exact" (used for wks 3–4)** | **4.2×10³** | yes |

The feature Gram `Φ'Φ` has **effective rank ≈ 3.6** (the controller's input `diff` is only 14-D),
so the likelihood constrains a tiny subspace tightly and leaves the ~2690-D bulk to the prior —
the source of the ill-conditioning.

**Robustness (5 seeds).** The weeks-3–4 comparison numbers below are reported as **mean ± std across
5 independently-trained controllers** (non-AMC `2+2l_lat:sub-nvel` model, `train_no_0…4`; the `+AMC`
suffix is a training label only — same architecture and θ ∈ ℝ²⁶⁹⁵). Per-seed `α` ranges 0.060–0.068
and `κ ≈ 3.7–4.3×10³`. Reproduce with `python -m bayes.run_multiseed`; figures
`results/multiseed/{multiseed_efficiency.png, multiseed_uncertainty.png}`. Weeks 1–2 below report the
single-model machinery/diagnosis (architecture-level, seed-independent).

## Week 1 — Foundation (verified)

`bayes/verify_week1.py` → all 5 checks pass:

- the decomposed (frozen-trunk + Bayesian-head) path reproduces the original `GeneralModel.forward`
  **exactly** (max diff 0);
- the trained point estimate explains the actions (R² = 0.99 on the N' subset);
- the analytic log-posterior gradient matches finite differences to ~1e-7;
- the SGLD minibatch gradient estimator (N'/B scaling) is unbiased;
- `as_controller(θ)` is a drop-in `GeneralModel` for `online_tester.py`.

## Week 2 — RWMH: correct, well-tuned, but defeated by dimension

`bayes/run_rwmh.py` (4 chains, nominal σ_a = 0.00127, κ = 6.5×10⁶). Step size adapts cleanly to the
23.4% benchmark, but the chains do not mix.

| metric | value |
|---|---|
| acceptance (tuned) | **0.234** ✓ |
| log-posterior split-R̂ / ESS | **2.11 / 5.2** (from 8000 draws) |
| θ split-R̂ (median / max) | 6.2 / 20.4 |
| coordinates with R̂ < 1.1 | **0 %** |

This is **not an implementation bug**: on isotropic Gaussians the sampler recovers the target
exactly, and its ESS/iteration falls as **1/d** — the random-walk curse of dimension:

| isotropic dim d | 10 | 100 | 1000 | 2695 |
|---|---|---|---|---|
| RWMH ESS (x₀) | 1989 | 240 | 13 | **11** |

So RWMH is a faithful but fundamentally dimension-limited baseline at θ ∈ ℝ²⁶⁹⁵.
Figure: `results/rwmh/T1_N40000_C4/diagnostics.png` (traces, τ trajectory, acceptance, ACF).

## Week 3 — SGLD: gradient information pays off where it matters

`bayes/run_sgld.py` (σ_a = 0.05, κ = 4.2×10³). The SGLD integrator is validated on isotropic
Gaussians, where — unlike RWMH — it mixes **dimension-independently**:

| isotropic dim d=2695 | RWMH | SGLD |
|---|---|---|
| ESS (x₀) | 11 | **4554** (~400×) |

On the **actual** (ill-conditioned) posterior, both are limited by κ, so per-iteration mixing is
**comparable**; SGLD's advantage is **compute** (it touches B = 128 vs RWMH's N' = 4096 likelihood
terms/step). Three complementary efficiency metrics (log-posterior, σ_a = 0.05):

| metric | RWMH | SGLD | SGLD advantage |
|---|---|---|---|
| ESS / iteration | (3.2 ± 1.5)×10⁻⁴ | (2.7 ± 0.8)×10⁻⁴ | ~comparable |
| **ESS / 10⁶ term-evals** | 0.078 ± 0.037 | 2.11 ± 0.63 | **32.9× ± 15.4** |
| ESS / sec (GPU) | 2.26 ± 1.1 | 1.75 ± 0.52 | RWMH (overhead-bound)¹ |

(mean ± std over 5 seeds; the SGLD/RWMH per-compute ratio is > 1 on **every** seed: 56, 28, 41, 30, 9.3×.)

¹ On this small last layer the GPU is kernel-overhead-bound, so RWMH's full-batch step is as cheap
as SGLD's minibatch. On **CPU** (FLOP-bound) SGLD is **2.3× cheaper per step** (0.30 vs 0.70 ms) —
the minibatch advantage is real; it just doesn't surface in GPU wall-clock at this scale.

**Conditioning sweep** (single model) — SGLD's per-compute advantage over RWMH (ESS per 10⁶ terms) holds across σ_a:

| σ_a | 0.00127 | 0.005 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|
| SGLD / RWMH | 27× | 31× | **95×** | 36× | 31× |

Both still stall on the high-dimensional prior bulk (per-coordinate ESS ≈ 4–8). Figures:
`results/sgld/sigma0.05/{sgld_diagnostics.png, comparison.png}`, multi-seed
`results/multiseed/multiseed_efficiency.png`.

**Takeaway:** SGLD dominates on the two properties that generalize — **scaling to high dimension**
and **ESS per unit compute** — while being honestly comparable per-iteration on this small,
ill-conditioned problem.

## Week 4 — Predictive uncertainty

`bayes/run_week4.py` (σ_a = 0.05). Four methods — point estimate, the closed-form **analytic
Gaussian posterior** (exact reference), RWMH, SGLD — over three input regimes. The analytic
reference disentangles "does the posterior encode the signal" from "did the sampler capture it."

**(a) The uncertainty machinery works for genuine input shift.** Perturbing the target coordinates
raises predictive std monotonically, and SGLD tracks the exact posterior far better than RWMH:

| target-coord noise (m) | 0.0 | 0.02 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|
| analytic (exact) | 0.0186 ± .0004 | 0.0207 ± .0005 | 0.0285 ± .0012 | 0.0462 ± .0030 | **0.0927 ± .0089** |
| SGLD | 0.0167 ± .0005 | 0.0185 ± .0005 | 0.0254 ± .0011 | 0.0411 ± .0030 | **0.0834 ± .0104** |
| RWMH | 0.0147 ± .0003 | 0.0157 ± .0003 | 0.0200 ± .0006 | 0.0304 ± .0017 | **0.0584 ± .0056** |

(mean ± std over 5 seeds; `results/multiseed/multiseed_uncertainty.png`.) This is the project's
positive result: **a usable distribution-shift signal, recovered more faithfully by SGLD** — at
every level and every seed the ordering is analytic ≳ SGLD > RWMH, i.e. RWMH systematically
under-estimates the spread (poor mixing; median ID std RWMH 0.0147 vs SGLD 0.0167 vs exact 0.0186).

**(b) The dataset's `extrap_0.85` split is not a true input shift** — so extrapolation detection is a
(correct) null, even for the exact posterior:

| method | extrap-detection AUROC (predictive std as OOD score) |
|---|---|
| analytic (exact) | 0.460 ± 0.010 |
| SGLD | 0.457 ± 0.046 |
| RWMH | 0.475 ± 0.034 |

All ≈ 0.5 (chance) — with tight bars on the **exact** analytic posterior, so the null is robust
across seeds, not a sampler artifact. Action MSE extrap/interp (point estimate) = **0.84 ± 0.04**
(≈ 1 ⇒ extrap is no harder than interp). Evidence that the split is on-manifold: **0.0 %** of extrap target coords fall outside the training
range (interp itself is 18.2 % outside), and feature z-scores vs training are identical
(interp 0.60 ≈ extrap 0.61). The frozen encoder maps extrap inputs to familiar regions, the
controller generalizes, and the uncertainty *correctly* reports "not more uncertain." The paper
therefore uses the **target-perturbation** experiment as its distribution-shift demonstration and
reports the extrap-split null honestly. Figures:
`results/week4/sigma0.05/{std_distributions.png, uncertainty_metrics.png}`.

## Conclusions

1. **Gradient information is necessary** for last-layer Bayesian inference at θ ∈ ℝ²⁶⁹⁵: RWMH is
   correct and well-tuned but dimension-limited; SGLD mixes dimension-independently and is
   **33× ± 15 more efficient per unit compute** (mean ± std over 5 seeds; > 1 on every seed).
2. **For predictive uncertainty, SGLD ≻ RWMH** — it recovers the exact posterior's spread and
   shift-response far more faithfully; RWMH under-estimates uncertainty.
3. **Conditioning (σ_a), not the sampler alone, governs feasibility.** The train-residual σ_a gives
   an unsamplable, overconfident posterior; a deliberate σ_a = 0.05 restores both samplability and a
   meaningful uncertainty signal.
4. **Bayesian last-layer MCMC does produce a usable extrapolation signal** under genuine input shift;
   the nominal extrap split simply isn't one.

All four conclusions hold across **5 independently-trained seeds** (mean ± std reported above;
`python -m bayes.run_multiseed`), so they reflect the method/architecture, not one lucky model.

## Caveats / open items

- σ_a = 0.05 is a reported modelling choice, not learned; sensitivity is partially mapped by the
  week-3 sweep but not propagated through week 4.
- MCMC ESS is small in absolute terms (tens), so RWMH/SGLD predictive stds carry estimator noise;
  the analytic reference is the reliable ground truth.
- A genuine out-of-hull extrapolation split would let the AUROC experiment show real OOD detection;
  deferred in favour of the perturbation demonstration.
