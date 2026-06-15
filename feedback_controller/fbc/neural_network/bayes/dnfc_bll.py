"""
Bayesian last-layer wrapper around the trained DNFC controller.

Week-1 deliverable for the RWMH-vs-SGLD study (see ``main.tex``):

  * wrap the trained DNFC with a FROZEN encoder + FROZEN controller trunk, exposing
    only the final controller linear layer as the Bayesian parameter vector
    ``theta in R^{7(d+1)}`` (d = controller hidden width; medium model -> d=384 -> 2695 params);
  * implement the *tempered, subsampled* Gaussian log-posterior over ``theta`` and
    its gradient -- the single target that both RWMH (week 2) and SGLD (week 3) sample;
  * expose everything needed to verify the wrapper against the trained point estimate, and a
    drop-in adapter so a posterior sample runs through ``online_tester.py`` unchanged.

This module is deliberately glued to the repo's own files (``nn_models.GeneralModel``,
``config.Config``) so it tracks YOUR pipeline:

  * the architecture is the exact ``GeneralModel`` built by ``train_w_datasets.py``;
  * the checkpoint path is rebuilt from ``Config`` via the SAME ``get_model_name`` convention
    that ``testers.py`` / ``online_tester.py`` use, so changing ``config.py`` moves this too;
  * the N' likelihood subset is drawn from the genuine TRAIN split (``split_indices_*.pt``),
    i.e. the triples the controller was actually fit on.

Model recap (``GeneralModel.forward(target_repr, state)``)::

    x_des = enc1(target_repr)            # MLP encoder: target(13) -> latent(14, state-shaped)
    diff  = x_des - state                # (14,)
    phi   = relu(W0 @ diff + b0)         # controller trunk: first linear + ReLU  -> (d,)
    a     = tanh(W @ phi + b)            # FINAL linear layer + tanh   <-- BAYESIAN here  -> (7,)

Frozen at trained values: ``enc1`` and ``(W0, b0)``.  Bayesian: ``theta = (W, b)``.
Because the trunk is frozen, each triple's feature ``phi = phi(s, g)`` is constant, so we
precompute ``Phi`` (N' x d) and the expert actions ``A`` (N' x 7) ONCE; the posterior over
``theta`` then only touches the cheap last-layer map ``a = tanh(Phi @ W^T + b)``.

NB (deployment): in ``online_tester.online_test`` the model output is a joint VELOCITY and the
rollout integrates ``state[:7] += 5 * a`` (a fixed x5 gain). That gain is a deployment detail, not
part of the model or likelihood -- the posterior is over the action predictor ``a`` exactly as
trained. Use ``DNFCLastLayer.as_controller(theta)`` to obtain a ``GeneralModel`` for that loop.

Prior / likelihood / posterior (Gaussian; fixed obs. noise ``sigma_a``; prior scale ``alpha``)::

    p(theta)        = N(theta | 0, alpha^2 I)
    p(a_i | theta)  = N(a_i | f_theta(s_i, g_i), sigma_a^2 I_7),   f_theta = tanh(W phi_i + b)
    p(theta | D)   propto  p(D | theta) p(theta)

Tempering follows the proposal's formula ``p(theta | D)^{1/T}`` -> ``log_post = (1/T)(loglik + logprior)``.
Under this formula **T >= 1 flattens** (T = 1 untempered). NOTE: ``main.tex`` prose says "small T flattens",
which is inconsistent with ``p^{1/T}``; we follow the written formula and keep T identical across samplers.
"""

from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

# The trained model + dim/path conventions live in the (flat) neural_network/ package one level up.
_NN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _NN_DIR not in sys.path:
    sys.path.insert(0, _NN_DIR)
from nn_models import GeneralModel  # noqa: E402
from config import Config  # noqa: E402


# --------------------------------------------------------------------------------------
# Feature-vector layout of the DNFC trajectory arrays, shape (n_traj, n_steps, 35):
#   [ step(1) | state(14) | target(9) | onehot(4) | action(7) ]
# (matches TrajectoryDataset.__getitem__ in train_w_datasets.py; target_repr = target+onehot = 13)
# --------------------------------------------------------------------------------------
STEP_DIM, STATE_DIM, COORDS_DIM, ONEHOT_DIM, ACTION_DIM = 1, 14, 9, 4, 7
TARGET_DIM = COORDS_DIM + ONEHOT_DIM  # 13: encoder input (target_repr)

_S0 = STEP_DIM                # state slice start            -> 1
_T0 = _S0 + STATE_DIM         # target_repr slice start      -> 15
_A0 = _T0 + TARGET_DIM        # action slice start           -> 28
assert _A0 + ACTION_DIM == 35, "unexpected feature layout"


@dataclass
class BLLConfig:
    """All choices needed to reproduce the Bayesian last-layer posterior.

    The model/dataset selection is intentionally thin: by default everything is read from the
    repo's ``Config`` (``config.py``) + the repo's weight-path convention, so this study wraps
    exactly the DNFC your ``train_w_datasets.py`` trains and ``online_tester.py`` deploys. Only
    the things ``Config`` does NOT pin -- which complexity / seed / epoch -- are set here.
    """

    # --- which trained DNFC (rest derived from Config, like testers.py) ---
    model_complexity: str = "medium"            # train_w_datasets.py / online_tester.py default
    train_no: int = 0
    epoch: int = 4000
    use_custom_loss: Optional[bool] = None      # None -> Config().use_custom_loss
    v_name: Optional[str] = None                # None -> Config().v_name; override to pick a model variant
                                                #   (e.g. "2+2l_lat:sub-nvel" = the non-AMC 10-seed model)
    ckpt: Optional[str] = None                  # None -> derive from Config + complexity/train_no/epoch
    dataset_dir: Optional[str] = None           # None -> data/torobo/{Config().dataset_name}
    train_file: Optional[str] = None            # None -> Config().ds_file_name (e.g. train_interp_0.85.npy)
    use_train_split: bool = True                # draw N' from the genuine train split (split_indices_*.pt)

    # --- posterior hyper-parameters (reported up front) ---
    alpha: Optional[float] = None               # prior std; None -> std of trained last-layer theta
    sigma_a: Optional[float] = None             # obs. noise std; None -> residual std at theta_0
    temperature: float = 1.0                    # T in p(theta|D)^{1/T}; T=1 untempered (see module docstring)
    n_prime: int = 4096                         # N': size of the fixed training subset (D)
    batch_size: int = 128                       # B: SGLD minibatch size
    subset_seed: int = 0                        # RNG seed for choosing the N' subset

    # --- runtime ---
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


@dataclass
class _Resolved:
    """Concrete paths/dims for a BLLConfig, rebuilt from the repo's Config."""
    ckpt: str
    dataset_dir: str
    train_file: str
    split_file: Optional[str]
    enc_hid: int
    cont_hid: int
    use_custom_loss: bool
    dataset_name: str
    ds_ratio: str
    model_name: str


def resolve_target(cfg: BLLConfig) -> _Resolved:
    """Rebuild the checkpoint path + dataset paths from the repo's ``Config``.

    Mirrors ``testers.Tester.load_model`` exactly:
        weights/{dataset_name}|{ds_ratio}|{model_name}|{num_params}K_params/train_no_{i}/fbc_{epoch}.pth
    so the directory string matches byte-for-byte what ``train_w_datasets.py`` wrote.
    """
    rc = Config()
    ucl = rc.use_custom_loss if cfg.use_custom_loss is None else cfg.use_custom_loss
    if cfg.v_name is not None:                  # local Config instance -> safe to override (config.py untouched)
        rc.v_name = cfg.v_name
    enc_hid, cont_hid, _, _ = rc.get_model_dims(cfg.model_complexity)
    # Count params on a throwaway model the same way training/testers do (numel is device-independent).
    tmp = GeneralModel(STATE_DIM, TARGET_DIM, ACTION_DIM, enc_hid, cont_hid, use_image=False)
    num_params = rc.get_params_num(tmp)                       # == sum(numel)/1e3
    model_name = rc.get_model_name(False, ucl, False) + f"|{num_params}K_params"

    ckpt = cfg.ckpt or os.path.join(
        _NN_DIR, "weights", f"{rc.dataset_name}|{rc.ds_ratio}|{model_name}",
        f"train_no_{cfg.train_no}", f"fbc_{cfg.epoch}.pth")
    dataset_dir = cfg.dataset_dir or os.path.join(_NN_DIR, "data", "torobo", rc.dataset_name)
    train_file = cfg.train_file or rc.ds_file_name
    split_file = (os.path.join(dataset_dir, rc.train_val_file)
                  if (cfg.use_train_split and getattr(rc, "train_val_file", None)) else None)
    return _Resolved(ckpt, dataset_dir, train_file, split_file, enc_hid, cont_hid,
                     ucl, rc.dataset_name, rc.ds_ratio, model_name)


# ======================================================================================
# 1. Frozen DNFC wrapper: trunk -> features phi(s, g); Bayesian last layer theta -> action
# ======================================================================================
class DNFCLastLayer:
    """Loads a trained ``GeneralModel``, freezes it, and splits it at the final linear layer."""

    def __init__(self, cfg: BLLConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype
        self.tgt = resolve_target(cfg)
        if not os.path.isfile(self.tgt.ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found:\n  {self.tgt.ckpt}\n"
                f"Derived from config.py (dataset={self.tgt.dataset_name}, ds_ratio={self.tgt.ds_ratio}, "
                f"model_name={self.tgt.model_name}). Check Config.v_name / complexity / train_no / epoch.")

        self.d = self.tgt.cont_hid  # controller hidden width (feature dim)
        model = GeneralModel(
            encoded_space_dim=STATE_DIM, target_dim=TARGET_DIM, action_dim=ACTION_DIM,
            enc_hid=self.tgt.enc_hid, cont_hid=self.tgt.cont_hid, use_image=False,
        )
        model.load_state_dict(torch.load(self.tgt.ckpt, map_location="cpu", weights_only=True))
        model.eval().to(self.device, self.dtype)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        # Trunk submodules (all frozen).
        self._enc1 = model.enc1                            # target_repr(13) -> x_des(14)
        self._ctrl_in = model.mlp_controller.linear[0]     # Linear(14 -> d)
        self._ctrl_relu = model.mlp_controller.linear[1]   # ReLU
        self._head = model.mlp_controller.linear[2]        # Linear(d -> 7) == Bayesian last layer

        # Trained point estimate theta_0 = flatten(W0, b0).
        self.theta0 = self.flatten(self._head.weight.detach(), self._head.bias.detach())
        assert self.theta0.numel() == ACTION_DIM * (self.d + 1)

    # ---- theta <-> (W, b) ---------------------------------------------------------------
    @property
    def theta_dim(self) -> int:
        return ACTION_DIM * (self.d + 1)

    def flatten(self, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(W (7,d), b (7,)) -> theta (7*(d+1),). Row-major W then b."""
        return torch.cat([W.reshape(-1), b.reshape(-1)]).to(self.device, self.dtype)

    def unflatten(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """theta (7*(d+1),) -> (W (7,d), b (7,)). Inverse of ``flatten``."""
        W = theta[: ACTION_DIM * self.d].reshape(ACTION_DIM, self.d)
        b = theta[ACTION_DIM * self.d:].reshape(ACTION_DIM)
        return W, b

    # ---- frozen trunk: features ---------------------------------------------------------
    @torch.no_grad()
    def features(self, state: torch.Tensor, target_repr: torch.Tensor) -> torch.Tensor:
        """phi(s, g) = relu(W0 (enc1(g) - s) + b0)  ->  (N, d). Trunk is frozen (no grad)."""
        state = torch.as_tensor(state, device=self.device, dtype=self.dtype)
        target_repr = torch.as_tensor(target_repr, device=self.device, dtype=self.dtype)
        x_des = self._enc1(target_repr)
        diff = x_des - state
        return self._ctrl_relu(self._ctrl_in(diff))

    # ---- Bayesian last layer: features -> action ----------------------------------------
    def action_from_features(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """f_theta(phi) = tanh(phi @ W^T + b)  ->  (N, 7). Differentiable in ``theta``."""
        W, b = self.unflatten(theta)
        return torch.tanh(phi @ W.t() + b)

    def predict(self, theta: torch.Tensor, state: torch.Tensor,
                target_repr: torch.Tensor) -> torch.Tensor:
        """Full forward through the decomposed (trunk + Bayesian head) path. ``theta=None`` -> theta_0."""
        if theta is None:
            theta = self.theta0
        return self.action_from_features(theta, self.features(state, target_repr))

    @torch.no_grad()
    def predict_original(self, state: torch.Tensor, target_repr: torch.Tensor) -> torch.Tensor:
        """Action from the ORIGINAL monolithic ``GeneralModel.forward`` (for equivalence checks)."""
        state = torch.as_tensor(state, device=self.device, dtype=self.dtype)
        target_repr = torch.as_tensor(target_repr, device=self.device, dtype=self.dtype)
        return self.model(target_repr, state)[0]

    # ---- integration: drop-in GeneralModel for online_tester.py -------------------------
    @torch.no_grad()
    def as_controller(self, theta: Optional[torch.Tensor] = None) -> GeneralModel:
        """Write ``theta`` into the last layer and return the underlying ``GeneralModel``.

        The returned object has the exact ``model(target_repr, state) -> (action, x_des, diff)``
        signature ``online_tester.online_test`` expects, e.g.::

            wrapper, post = load_default()
            tester.model = wrapper.as_controller(theta_sample)   # then run online_test(...)

        Pass ``theta=None`` to restore the trained point estimate (theta_0).
        """
        W, b = self.unflatten(self.theta0 if theta is None else
                              torch.as_tensor(theta, device=self.device, dtype=self.dtype))
        self._head.weight.copy_(W)
        self._head.bias.copy_(b)
        return self.model


# ======================================================================================
# 2. Tempered, subsampled Gaussian log-posterior over the Bayesian last layer + gradient
# ======================================================================================
def load_triples(dataset_dir: str, file_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a (n_traj, n_steps, 35) DNFC array into (state, target_repr, action) triples."""
    arr = np.load(os.path.join(dataset_dir, file_name))
    flat = arr.reshape(-1, arr.shape[-1])
    return (flat[:, _S0:_S0 + STATE_DIM],
            flat[:, _T0:_T0 + TARGET_DIM],
            flat[:, _A0:_A0 + ACTION_DIM])


def load_train_split_indices(split_file: Optional[str]) -> Optional[np.ndarray]:
    """Return the flattened TRAIN indices from a ``split_indices_*.pt`` file (as in train_w_datasets.py)."""
    if not split_file or not os.path.isfile(split_file):
        return None
    split = torch.load(split_file, weights_only=False)
    return np.asarray(split["train_indices"]).astype(np.int64).reshape(-1)


class BLLPosterior:
    """Tempered, subsampled Gaussian log-posterior over ``theta`` and its gradient.

    Holds precomputed frozen features ``Phi`` (N' x d) and expert actions ``A`` (N' x 7) for a
    fixed N' subset drawn (once, with ``subset_seed``) from ``candidate_idx`` -- the train-split
    rows by default -- so evaluations are O(N' * d) matmuls in ``theta`` only.
    """

    def __init__(self, wrapper: DNFCLastLayer, state: np.ndarray, target_repr: np.ndarray,
                 action: np.ndarray, cfg: Optional[BLLConfig] = None,
                 candidate_idx: Optional[np.ndarray] = None):
        self.w = wrapper
        self.cfg = cfg or wrapper.cfg
        self.device = wrapper.device
        self.dtype = wrapper.dtype
        self.d = wrapper.d
        self.theta_dim = wrapper.theta_dim

        # --- fix the N' subset once, drawn from the candidate pool (train split by default) ---
        pool = np.arange(state.shape[0]) if candidate_idx is None else np.asarray(candidate_idx)
        self.n_prime = min(self.cfg.n_prime, len(pool))
        rng = np.random.default_rng(self.cfg.subset_seed)
        self.idx_subset = np.sort(rng.choice(pool, size=self.n_prime, replace=False))
        s = torch.as_tensor(state[self.idx_subset], device=self.device, dtype=self.dtype)
        g = torch.as_tensor(target_repr[self.idx_subset], device=self.device, dtype=self.dtype)
        self.A = torch.as_tensor(action[self.idx_subset], device=self.device, dtype=self.dtype)

        # --- precompute frozen features Phi (N' x d) ONCE ---
        self.Phi = self.w.features(s, g)  # no grad; constant w.r.t. theta

        # --- resolve + freeze prior scale alpha and obs. noise sigma_a, then report ---
        theta0 = self.w.theta0.to(self.device, self.dtype)
        self.alpha = float(self.cfg.alpha) if self.cfg.alpha is not None else float(theta0.std())
        if self.cfg.sigma_a is not None:
            self.sigma_a = float(self.cfg.sigma_a)
        else:
            with torch.no_grad():
                resid = self.A - self.w.action_from_features(theta0, self.Phi)
            self.sigma_a = float(resid.std())

        self.temperature = float(self.cfg.temperature)
        self.beta = 1.0 / self.temperature          # exponent in p^{1/T}
        self.batch_size = int(self.cfg.batch_size)

    # ---- pieces -------------------------------------------------------------------------
    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """log N(theta | 0, alpha^2 I), including the normalising constant."""
        const = -0.5 * self.theta_dim * math.log(2 * math.pi * self.alpha ** 2)
        return const - 0.5 / self.alpha ** 2 * theta.pow(2).sum()

    def log_likelihood(self, theta: torch.Tensor, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sum_i log N(a_i | f_theta(phi_i), sigma_a^2 I_7) over the N' subset (or rows ``idx``)."""
        Phi = self.Phi if idx is None else self.Phi[idx]
        A = self.A if idx is None else self.A[idx]
        pred = self.w.action_from_features(theta, Phi)
        sq = (A - pred).pow(2).sum()
        const = -0.5 * A.numel() * math.log(2 * math.pi * self.sigma_a ** 2)
        return const - 0.5 / self.sigma_a ** 2 * sq

    # ---- log-posterior (RWMH target) ----------------------------------------------------
    def log_posterior(self, theta: torch.Tensor) -> torch.Tensor:
        """Tempered, FULL-N' unnormalised log-posterior ``(1/T)(loglik + logprior)``.

        The density RWMH targets and the quantity to trace-plot. Scalar tensor; differentiable
        in ``theta`` when ``theta.requires_grad``.
        """
        theta = torch.as_tensor(theta, device=self.device, dtype=self.dtype)
        return self.beta * (self.log_likelihood(theta) + self.log_prior(theta))

    def log_posterior_value(self, theta: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self.log_posterior(theta))

    @torch.no_grad()
    def log_posterior_batch(self, Theta: torch.Tensor) -> torch.Tensor:
        """Tempered full-N' log-posterior for a BATCH of parameter vectors.

        ``Theta`` is ``(C, theta_dim)`` (one row per chain) -> ``(C,)``. Same density as
        ``log_posterior`` but vectorised over chains in a single matmul, for multi-chain RWMH.
        """
        Theta = torch.as_tensor(Theta, device=self.device, dtype=self.dtype)
        C = Theta.shape[0]
        W = Theta[:, : ACTION_DIM * self.d].reshape(C, ACTION_DIM, self.d)  # (C,7,d)
        b = Theta[:, ACTION_DIM * self.d:].reshape(C, ACTION_DIM)            # (C,7)
        pred = torch.tanh(torch.einsum("nd,cod->cno", self.Phi, W) + b[:, None, :])  # (C,N',7)
        sq = (self.A[None] - pred).pow(2).sum(dim=(1, 2))                    # (C,)
        ll_const = -0.5 * self.A.numel() * math.log(2 * math.pi * self.sigma_a ** 2)
        lp_const = -0.5 * self.theta_dim * math.log(2 * math.pi * self.alpha ** 2)
        ll = ll_const - 0.5 / self.sigma_a ** 2 * sq
        lp = lp_const - 0.5 / self.alpha ** 2 * Theta.pow(2).sum(dim=1)
        return self.beta * (ll + lp)

    # ---- gradient (SGLD drift; idx=None gives the exact full-data gradient) --------------
    def grad_log_posterior(self, theta: torch.Tensor,
                           idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Gradient of the tempered log-posterior.

        ``idx is None`` -> exact full-N' gradient ``(1/T)(grad loglik_full + grad logprior)``.
        ``idx`` a minibatch -> the SGLD unbiased estimator with the N'/B rescaling:
            ``(1/T)( (N'/B) sum_{i in idx} grad loglik_i  +  grad logprior )``.
        """
        theta = torch.as_tensor(theta, device=self.device, dtype=self.dtype).detach().requires_grad_(True)
        scale = 1.0 if idx is None else self.n_prime / float(len(idx))
        loss = self.beta * (scale * self.log_likelihood(theta, idx) + self.log_prior(theta))
        (grad,) = torch.autograd.grad(loss, theta)
        return grad

    def sample_minibatch(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Random row indices into the N' subset, size B (for SGLD)."""
        return torch.randint(0, self.n_prime, (self.batch_size,),
                             device=self.device, generator=generator)

    def grad_log_posterior_batch(self, Theta: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Per-chain SGLD drift: gradient of the tempered log-posterior with a per-chain minibatch.

        ``Theta`` is ``(C, theta_dim)`` and ``idx`` is ``(C, B)`` long row-indices into the N' subset
        (each chain its own independent minibatch). Returns ``(C, theta_dim)`` where row c is
        ``(1/T)( (N'/B) sum_{i in idx[c]} grad log p(a_i|theta_c) + grad log p(theta_c) )`` -- the
        unbiased minibatch estimator of the exact full-N' gradient, vectorised over chains.
        """
        Theta = torch.as_tensor(Theta, device=self.device, dtype=self.dtype).detach().requires_grad_(True)
        C, B = idx.shape
        W = Theta[:, : ACTION_DIM * self.d].reshape(C, ACTION_DIM, self.d)   # (C,7,d)
        b = Theta[:, ACTION_DIM * self.d:].reshape(C, ACTION_DIM)             # (C,7)
        Phi_mb = self.Phi[idx]                                               # (C,B,d)
        A_mb = self.A[idx]                                                   # (C,B,7)
        pred = torch.tanh(torch.einsum("cbd,cod->cbo", Phi_mb, W) + b[:, None, :])  # (C,B,7)
        sq = (A_mb - pred).pow(2).sum(dim=(1, 2))                            # (C,)
        ll = -0.5 / self.sigma_a ** 2 * sq                                   # (C,) (const dropped: no grad)
        lp = -0.5 / self.alpha ** 2 * Theta.pow(2).sum(dim=1)                # (C,)
        loss = (self.beta * ((self.n_prime / float(B)) * ll + lp)).sum()     # per-chain terms are independent
        (grad,) = torch.autograd.grad(loss, Theta)
        return grad

    @torch.no_grad()
    def grad_log_posterior_batch_fast(self, Theta: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Analytic version of ``grad_log_posterior_batch`` (no autograd) -- the SGLD drift.

        Closed-form gradient of the Gaussian-likelihood + Gaussian-prior tempered log-posterior through
        the ``tanh`` last layer, vectorised over chains. Same result as the autograd method but ~free
        of graph overhead, so the SGLD minibatch step is genuinely cheap. ``Theta`` (C, theta_dim),
        ``idx`` (C, B) -> (C, theta_dim).
        """
        Theta = torch.as_tensor(Theta, device=self.device, dtype=self.dtype)
        C, B = idx.shape
        W = Theta[:, : ACTION_DIM * self.d].reshape(C, ACTION_DIM, self.d)   # (C,7,d)
        b = Theta[:, ACTION_DIM * self.d:].reshape(C, ACTION_DIM)             # (C,7)
        Phi_mb = self.Phi[idx]                                               # (C,B,d)
        A_mb = self.A[idx]                                                   # (C,B,7)
        pred = torch.tanh(torch.einsum("cbd,cod->cbo", Phi_mb, W) + b[:, None, :])  # (C,B,7)
        # d/dz [-1/(2 sig^2) (a - tanh z)^2] = (1/sig^2)(a - tanh z)(1 - tanh^2 z)
        g_z = (1.0 / self.sigma_a ** 2) * (A_mb - pred) * (1.0 - pred ** 2)  # (C,B,7)
        dW = torch.einsum("cbo,cbd->cod", g_z, Phi_mb)                       # (C,7,d)
        db = g_z.sum(dim=1)                                                  # (C,7)
        grad_ll = torch.cat([dW.reshape(C, -1), db], dim=1)                  # (C, theta_dim)
        grad_prior = -Theta / self.alpha ** 2
        return self.beta * ((self.n_prime / float(B)) * grad_ll + grad_prior)

    # ---- convenience --------------------------------------------------------------------
    @property
    def theta0(self) -> torch.Tensor:
        return self.w.theta0.to(self.device, self.dtype)

    def summary(self) -> str:
        return (f"BLLPosterior(theta_dim={self.theta_dim}, d={self.d}, N'={self.n_prime}, "
                f"B={self.batch_size}, T={self.temperature:g}, alpha={self.alpha:.4g}, "
                f"sigma_a={self.sigma_a:.4g}, device={self.device}, dtype={self.dtype})")


# ======================================================================================
# Convenience builder
# ======================================================================================
def load_default(cfg: Optional[BLLConfig] = None) -> Tuple[DNFCLastLayer, BLLPosterior]:
    """Build the frozen wrapper + posterior from a ``BLLConfig`` (defaults track ``config.py``).

    Returns ``(wrapper, posterior)`` ready for RWMH/SGLD: ``posterior.log_posterior(theta)`` and
    ``posterior.grad_log_posterior(theta, idx)`` are the sampler hooks; ``posterior.theta0`` is the
    trained point estimate; ``wrapper.as_controller(theta)`` plugs a sample into online_tester.
    """
    cfg = cfg or BLLConfig()
    wrapper = DNFCLastLayer(cfg)
    state, target_repr, action = load_triples(wrapper.tgt.dataset_dir, wrapper.tgt.train_file)
    candidate_idx = load_train_split_indices(wrapper.tgt.split_file)
    posterior = BLLPosterior(wrapper, state, target_repr, action, cfg, candidate_idx=candidate_idx)
    return wrapper, posterior


if __name__ == "__main__":
    wrapper, post = load_default()
    print("checkpoint:", os.path.relpath(wrapper.tgt.ckpt, _NN_DIR))
    print(post.summary())
    th0 = post.theta0
    print("log p(theta_0 | D) =", post.log_posterior_value(th0))
    g = post.grad_log_posterior(th0)
    print("grad shape:", tuple(g.shape), "| ||grad|| =", float(g.norm()))
