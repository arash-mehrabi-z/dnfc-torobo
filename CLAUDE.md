# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Robotics research project implementing **Desired Neural Feedback Control (DNFC)** for the Torobo humanoid robot. Trains neural network controllers that map from desired task coordinates + robot state → joint velocities for 7-DOF arm manipulation.

**Task**: Block stacking with 3 objects (A, B, C) in 4 phases (one-hot encoded):
1. Grasp B → Place B on A → Grasp C → Place C on A

All code lives in [feedback_controller/fbc/neural_network/](feedback_controller/fbc/neural_network/).

## Architecture

### Two Controller Models

| Model | Architecture | Key Difference |
|-------|-------------|----------------|
| **DNFC** (`GeneralModel`) | Two-stream encoder + controller MLP | Uses EE pose history (N consecutive poses) as additional input stream |
| **MLP Baseline** (`MLPBaseline`) | Two-stream encoder + controller MLP | Uses only joint state (no EE pose history) |

Both models share the same latent-space-difference control paradigm:
1. **Target encoder**: target representation (9D coords + 4D one-hot = 13D) → `x_des` (14D latent)
2. **State encoder**: robot state → `x_curr` (14D latent). For DNFC, input is EE poses + joint state; for baseline, just joint state.
3. **Controller**: `x_des - x_curr` → `tanh(MLP) * action_scale` → 7D joint velocity

**Custom loss** (DNFC only): `MSE(actions) + C × MSE(x_des, x_curr)` where C=1e-5.

### Key Files

| File | Purpose |
|------|---------|
| [config.py](feedback_controller/fbc/neural_network/config.py) | All hyperparameters: dataset, model dims, loss settings |
| [nn_models.py](feedback_controller/fbc/neural_network/nn_models.py) | `GeneralModel`, `MLPBaseline`, `CustomLoss`, `KLLoss`, MLP building blocks |
| [train_w_datasets.py](feedback_controller/fbc/neural_network/train_w_datasets.py) | Training script with `TrajectoryDataset`, noise injection, validation |
| [testers.py](feedback_controller/fbc/neural_network/testers.py) | `Tester` class for emulated (offline) evaluation |
| [emulated_test.py](feedback_controller/fbc/neural_network/emulated_test.py) | Script that runs `Tester` and generates result plots/CSVs |
| [online_tester.py](feedback_controller/fbc/neural_network/online_tester.py) | Real robot testing with ROS interface |
| [global_defines.py](feedback_controller/fbc/neural_network/global_defines.py) | `TOR` class with robot constants and joint limits |
| [torkin.py](feedback_controller/fbc/neural_network/torkin.py) | Forward/inverse kinematics (used by Tester to compute EE poses) |

## Common Commands

```bash
cd feedback_controller/fbc/neural_network
source .venv/bin/activate

# Train (iterates model complexities, runs num_trains=2 seeds each)
python train_w_datasets.py

# Emulated testing (no robot)
python emulated_test.py

# Online testing (requires ROS + Torobo robot)
python online_tester.py
```

**Configuration**: Edit [config.py](feedback_controller/fbc/neural_network/config.py) before training:
- `dataset_name`: e.g., `"trajs:72_blocks:3_triangle_v_scarce"`
- `ds_ratio`: `"interp_0.85"` or `"extrap_0.85"` (interpolation vs extrapolation split)
- `use_custom_loss` / `C`: toggle DNFC custom loss and its weight
- `num_consecutive_poses`: number of stacked EE poses (default 4)
- `noise_scale`: training noise magnitude as fraction of per-dimension std (default 0.10)
- `action_scale`: multiplier for action outputs (default 50)

Model complexity is set in the training loop (edit `train_w_datasets.py`):
```python
for model_complexity in ['high']:  # Options: 'low', 'medium', 'high', 'xhigh', 'XXhigh'
```

## Dataset Structure

**Location**: `feedback_controller/fbc/neural_network/data/torobo/{dataset_name}/`
- Training file: `train_{ds_ratio}_eef_pos_R_noNorm.npy`
- Test file: `test_{ds_ratio}.npy`
- Split indices: `split_indices_{ds_ratio}.pt`

**Trajectory shape**: `(N_episodes, 299, 47)`

| Index | Content | Dimension |
|-------|---------|-----------|
| `[0]` | Step number | 1 |
| `[1:15]` | State (7 joint pos + 7 joint vel) | 14 |
| `[15:24]` | Target coords (A, B, C positions) | 9 |
| `[24:28]` | One-hot task phase | 4 |
| `[28:35]` | Action (joint velocities) | 7 |
| `[35:47]` | EE pose (3D position + 9D rotation matrix) | 12 |

## Code Patterns

### Forward Pass

```python
# DNFC (GeneralModel): 3 inputs, 4 outputs
acts_pred, x_des, x_curr, diff = model(target_repr, ee_repr, joint_state)

# MLP Baseline: 2 inputs, 4 outputs (no ee_repr)
acts_pred, x_des, x_curr, diff = baseline(target_repr, joint_state)
```

### Training Noise Injection

Training adds per-dimension Gaussian noise scaled by each feature's std:
```python
batch_noise = torch.randn(...) * ee_repr_std * noise_scale  # For EE repr
batch_state_noise = torch.randn(...) * joint_state_std * noise_scale  # For joint state
```

### Checkpoint Paths

```
weights/{dataset_name}|{ds_ratio}|{model_name}/train_no_{i}/fbc_{epoch}.pth
```
Model name encodes: loss type, target type, architecture variant, param count. Example:
`cus_los_1e-05|tar_cart|robo_enc_eef_N=4_plus_joints_noise=0.1|25.301K_params`

### Emulated Testing

The `Tester` computes EE poses at inference time using forward kinematics (`torkin.py`), maintaining a sliding window of `num_consecutive_poses` joint configurations. Success = number of 4 milestones reached (end-effector within 2cm of target position).

## Model Complexity Levels

| Level | enc_hid | cont_hid | Approx Params |
|-------|---------|----------|---------------|
| low | 64 | 192 | ~6K |
| medium | 128 | 384 | ~24K |
| high | 256 | 768 | ~96K |
| xhigh | 512 | 1536 | ~384K |
| XXhigh | 1024 | 3072 | ~1.5M |

## Training Details

- **Optimizer**: Adam, lr=3e-4
- **Batch size**: 256
- **Epochs**: 14,000
- **Validation interval**: every 100 epochs
- **Seeds**: 2 training runs per complexity (configurable via `num_trains`)
- **Checkpoints saved**: every 100 epochs
- **Loss logged to**: `weights/{model}/train_no_{i}/loss.csv`
- Training exits with error if weight directory already exists (prevents accidental overwrites)

## Robot Constraints

- **7-DOF right arm** with 2 fixed torso joints (prepended for FK)
- **299 steps** per episode (fixed)
- **State**: 14D (7 joint positions + 7 joint velocities)
- **Milestone threshold**: 2cm Euclidean distance in Cartesian space
