# `bayes/` — Bayesian last-layer uncertainty for the DNFC controller

Last-layer Bayesian inference on the trained DNFC feedback controller, comparing **RWMH** (gradient-free)
and **SGLD** (gradient-aware) MCMC. Proposal: `../main.tex`. Consolidated findings: [`RESULTS.md`](RESULTS.md).

## Environment

Use the **`dnfc2`** conda env (the repo's bundled `.venv` is macOS-built and does not run here):

```bash
PY=/home/arash/anaconda3/envs/dnfc2/bin/python   # Python 3.9, torch 2.5.1 (CUDA)
```

Run everything as a module **from the `neural_network/` directory** (one level up):

```bash
cd /home/arash/catkin_ws/src/feedback_controller/fbc/neural_network
```

## What it wraps

The model/checkpoint is **not hardcoded** — `dnfc_bll.resolve_target()` rebuilds the weight path from
the repo's `config.py` using the same `get_model_name` convention as `testers.py`, so editing
`config.py` (dataset / `ds_ratio` / `v_name` / complexity) moves this study with it. Current target:
medium `2+2l_lat:sub-nvel+AMC` (12.053K), `interp_0.85`, `train_no_0`, epoch 4000 → θ ∈ ℝ²⁶⁹⁵.

A posterior sample can be dropped straight into `online_tester.py`:

```python
from bayes.dnfc_bll import load_default
wrapper, post = load_default()
tester.model = wrapper.as_controller(theta_sample)   # exact GeneralModel (action, x_des, diff) signature
```

## Modules

| file | role |
|---|---|
| `dnfc_bll.py` | frozen DNFC wrapper + tempered/subsampled Gaussian log-posterior, gradient (autograd + analytic), batched forms |
| `diagnostics.py` | split-R̂, ESS (Geyer), autocorrelation — self-tested vs iid / AR(1) |
| `rwmh.py` | batched 4-chain RWMH, Robbins–Monro step-size adaptation to 23.4% |
| `sgld.py` | batched 4-chain SGLD, `τ_k ∝ k^−0.55` schedule, analytic minibatch gradient |
| `predictive.py` | analytic (exact) posterior, posterior-predictive mean/std, AUROC, calibration, target perturbation |
| `verify_week1.py` | week-1 verification suite (5 checks) |
| `run_rwmh.py` | week-2 driver: RWMH + sampler validation + anisotropy diagnosis |
| `run_sgld.py` | week-3 driver: SGLD validation + RWMH-vs-SGLD efficiency + σ_a sweep |
| `run_week4.py` | week-4 driver: predictive uncertainty across the three regimes |
| `run_multiseed.py` | multi-seed robustness: wks 3–4 across N trained seeds → mean ± std + error-bar figures |

## Reproduce

```bash
$PY -m bayes.diagnostics                 # self-test the diagnostics (iid / AR(1))
$PY -m bayes.verify_week1                # week 1: foundation, 5 checks -> all PASS

$PY -m bayes.run_rwmh                     # week 2: RWMH (nominal sigma_a; the stiff/worst case)
$PY -m bayes.run_sgld   --sigma-a 0.05    # week 3: SGLD vs RWMH efficiency + conditioning sweep
$PY -m bayes.run_week4  --sigma-a 0.05    # week 4: predictive uncertainty (point/analytic/RWMH/SGLD)
$PY -m bayes.run_multiseed --seeds 0 1 2 3 4   # robustness: wks 3-4 across 5 seeds (non-AMC) -> error bars
```

Common flags: `--sigma-a` (observation noise = posterior conditioning; default 0.05 for wks 3–4),
`--device cuda|cpu`. `run_rwmh.py` also takes `--temperature`. Outputs land in
`bayes/results/<sampler>/<tag>/` (`samples.npy`, diagnostic `*.png`, `summary.json`).

## Key knobs (`BLLConfig` in `dnfc_bll.py`)

- `sigma_a` — observation noise; sets posterior width **and** condition number `κ = 1 + α²λ_max/σ_a²`.
  Default `None` → train-residual (0.00127, too stiff to sample); pass `0.05` for the samplable regime.
- `alpha` — prior std; `None` → std of the trained last layer (0.0627).
- `temperature` — `T` in `p^{1/T}`; `T ≥ 1` flattens. (Note: `main.tex` prose says "small T flattens",
  which conflicts with the stated `p^{1/T}`; we follow the formula. Tempering rescales width but **not**
  conditioning, so it does not fix mixing — see `RESULTS.md`.)
- `n_prime` / `batch_size` — N' likelihood subset (4096) / SGLD minibatch (128).
- `model_complexity` / `v_name` / `train_no` / `epoch` — which trained DNFC to wrap (rest from
  `config.py`). `v_name='2+2l_lat:sub-nvel'` selects the non-AMC 10-seed model used for the multi-seed
  robustness; `train_no` picks the seed.
