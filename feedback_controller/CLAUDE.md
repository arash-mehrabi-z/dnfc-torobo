# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a robotics research project implementing **Desired Neural Feedback Control (DNFC)** for the Torobo humanoid robot. The codebase trains neural network controllers that map from desired task coordinates and current robot state to joint velocities, using a custom loss function that incorporates both action prediction error and latent state representation error.

## High-Level Architecture

### Core Concept
- **Input**: Target Cartesian coordinates (x,y,z for 3 objects A,B,C) + one-hot encoded task phase + current robot state (joint positions & velocities for 7 joints)
- **Models**:
  1. **DNFC (Desired Neural Feedback Control)**: Two-stage architecture
     - Encoder MLP: Maps target representation to latent space (matching state dimensionality: 14D for 7 joints × 2)
     - Controller MLP: Maps latent error (desired - current state) to joint velocity commands
  2. **MLP Baseline**: Direct MLP mapping from concatenated inputs to actions
  3. **Diffusion Policy Baseline** (NEW): Conditional denoising diffusion model
     - 1D U-Net noise prediction network with FiLM conditioning
     - Predicts action sequences (16 steps) using receding horizon control
     - Conditions on observation history (2 timesteps)
     - Uses iterative denoising (100 steps) during inference
- **Training**:
  - DNFC: Custom loss combining MSE on actions with MSE on latent representations (weighted by constant C)
  - Baselines: MSE on actions only
  - Diffusion: MSE on noise prediction with DDPM scheduler

### Key Components

**Configuration** ([config.py](fbc/neural_network/config.py)):
- Centralized configuration for dataset names, model variants, loss parameters
- Model complexity levels: low/medium/high/xhigh (controls hidden layer sizes)
- Dataset paths follow pattern: `trajs:{num}_blocks:3_triangle_v_scarce` or `_random`

**Neural Models** ([nn_models.py](fbc/neural_network/nn_models.py), [diffusion_models.py](fbc/neural_network/diffusion_models.py)):
- `GeneralModel`: DNFC architecture with encoder + controller
- `MLPBaseline`: Direct state-to-action mapping for comparison
- `DiffusionPolicyModel` (NEW): Diffusion-based sequence prediction model
  - `ConditionalUnet1D`: 1D U-Net noise prediction network
  - `DDPMScheduler`: Denoising diffusion probabilistic model scheduler
- `CustomLoss`: Combines torque prediction MSE with latent state MSE (scaled by C parameter)
- Encoder supports both image input (CNN or AlexNet) and Cartesian coordinate input (MLP)

**Training** ([train_w_datasets.py](fbc/neural_network/train_w_datasets.py), [train_diffusion.py](fbc/neural_network/train_diffusion.py)):
- DNFC/MLP Baseline: Trains multiple model complexities (low/medium/high/xhigh) with multiple random seeds (10 runs each)
- Diffusion Policy (NEW): Dedicated training script with EMA, cosine LR schedule, AdamW optimizer
- Adds Gaussian noise (std=0.004) to states during training for robustness (DNFC/MLP only)
- Saves checkpoints every 100 epochs with loss plots and model weights
- Validation split is pre-computed and stored in `split_indices_{ds_ratio}.pt` (DNFC/MLP) or random split (Diffusion)

**Testing** ([testers.py](fbc/neural_network/testers.py), [online_tester.py](fbc/neural_network/online_tester.py)):
- `Tester` class: Loads models (DNFC, MLP Baseline, Diffusion Policy), runs emulated/offline tests, computes metrics
- Diffusion inference: Receding horizon control with observation history deque and action buffer management
- Online testing: ROS integration for real robot deployment (supports all model types)
- Metrics: DTW distance to ground truth trajectories, success rate (milestone completion)
- Task phases: Grasp object B → Place on A → Grasp object C → Place on A

**Robot Interface**:
- [global_defines.py](fbc/neural_network/global_defines.py): TOR class with robot constants, joint limits, predefined poses
- [torkin.py](fbc/neural_network/torkin.py): Forward/inverse kinematics using generated code
- [udp_comm.py](fbc/neural_network/udp_comm.py): UDP communication for robot commands (implied, not read but referenced)

## Common Commands

### Training

**DNFC and MLP Baseline:**
```bash
# Train models (runs all complexity levels with 10 seeds each)
cd fbc/neural_network
python train_w_datasets.py
```

**Diffusion Policy:**
```bash
# Train diffusion policy baseline
cd fbc/neural_network
python train_diffusion.py
```

Training configuration is set in [config.py](fbc/neural_network/config.py):
- `episodes_num_ds`: Number of trajectories in dataset
- `dataset_name`: Dataset folder name (e.g., "trajs:72_blocks:3_triangle_v_scarce")
- `ds_ratio`: Train/test split identifier (e.g., "extrap_0.85", "interp_0.95")
- `use_custom_loss`: Enable custom loss (True) vs. MSE-only (False) [DNFC only]
- `C`: Weight for latent state loss component (default: 1e-5) [DNFC only]
- `obs_horizon`, `pred_horizon`, `action_horizon`: Temporal parameters for diffusion policy

### Testing

**Emulated Testing:**
```bash
# Offline emulation testing for DNFC/MLP
cd fbc/neural_network
python emulated_test.py  # Edit file to configure test parameters

# Test diffusion policy
python test_diffusion_emulated.py
```

**Online Robot Testing (requires ROS):**
```bash
python online_tester.py  # Supports DNFC, MLP Baseline, and Diffusion Policy
```

Testing loads trained weights from:
```
fbc/neural_network/weights/{dataset_name}|{ds_ratio}|{model_name}/train_no_{i}/fbc_{epoch}.pth
```

Example paths:
- DNFC: `weights/trajs:72_blocks:3_triangle_v_scarce|extrap_0.85|cus_los_1e-05|tar_cart|2+2l_lat:sub-nvel|24.085K_params/train_no_0/fbc_4000.pth`
- Diffusion: `weights/trajs:72_blocks:3_triangle_v_scarce|extrap_0.85|diffusion_pol|oh:2|ph:16|ah:8|65322.0K_params/train_no_0/fbc_4000.pth`

### Data Exploration
Jupyter notebooks are in [fbc/neural_network/](fbc/neural_network/):
- Data preprocessing and visualization notebooks
- Results analysis notebooks

## Dataset Structure

Datasets are stored in [fbc/neural_network/data/torobo/](fbc/neural_network/data/torobo/):
```
trajs:{num}_blocks:3_{variant}/
├── train_{ratio}.npy        # Training episodes (N × 299 × features)
├── test_{ratio}.npy         # Test episodes
└── split_indices_{ratio}.pt # Train/val split indices
```

Each trajectory step contains:
- `[0]`: Step number
- `[1:15]`: State (7 joint positions + 7 velocities)
- `[15:24]`: Target coordinates (3×3 for objects A, B, C)
- `[24:28]`: One-hot task phase (4 phases)
- `[28:35]`: Action (7 joint velocities)

## Code Patterns

### State Representation
State is always 14D: `[q1, q2, ..., q7, dq1, dq2, ..., dq7]` (positions + velocities)

### Model Forward Pass
```python
# DNFC model
action_pred, x_des, diff = model(target_repr, state)
# x_des: encoder output (desired latent state)
# diff: x_des - state (used as controller input)

# Baseline model
action_pred = baseline(torch.cat((target_repr, state), dim=1))

# Diffusion Policy model
obs_seq = torch.stack(list(obs_deque))  # (obs_horizon, obs_dim)
obs_seq = obs_seq.unsqueeze(0)  # (1, obs_horizon, obs_dim)
action_seq = diffusion_model.get_action(obs_seq)  # (1, pred_horizon, action_dim)
# Execute first action_horizon actions, then replan
```

### Checkpoint Naming Convention
Model names encode architecture: `{loss_type}|{target_type}|{architecture}|{num_params}K_params`
- **DNFC/MLP:**
  - Loss: `cus_los_{C}` or `mse_los`
  - Target: `tar_cart` or `tar_img`
  - Architecture: `base|{v_name_base}` or just `{v_name}`
- **Diffusion Policy:**
  - Format: `diffusion_pol|oh:{obs_horizon}|ph:{pred_horizon}|ah:{action_horizon}|{num_params}K_params`
  - Example: `diffusion_pol|oh:2|ph:16|ah:8|65322.0K_params`

### Virtual Environment
A virtual environment exists at [fbc/neural_network/.venv/](fbc/neural_network/.venv/) but no requirements.txt is tracked. Dependencies include:
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- ROS libraries (rospy, roslibpy)
- DTW library
- PIL
- **Diffusion Policy additional dependencies:**
  - `diffusers` - HuggingFace library for diffusion models (DDPMScheduler, EMAModel)
  - `transformers` - Required by diffusers
  - `scikit-image` - Image processing utilities

## Important Constraints

- **7-DOF arm**: All models assume 7 joints for the right arm
- **Fixed trajectory length**: 299 steps per episode
- **ROS dependency**: Online testing requires ROS environment with Torobo robot interface
- **Data locality**: Datasets are not version-controlled (stored in data/ directories)
- **Git status**: There's an untracked Jupyter notebook for diffusion policy experiments

## Results and Evaluation

Results are saved to:
```
fbc/neural_network/results/{dataset_name}_{params}K_{ds_ratio}/ep:{epoch}/on_{model_variant}/
├── perf.csv              # Performance metrics per episode
├── plt_{eps}_{train}.png # 3D trajectory visualizations
├── latent_reps_{eps}_{train}.png
└── all_states_{model}.pickle
```

Metrics tracked:
- Success rate (percentage of task milestones reached)
- DTW distance (normalized and unnormalized) between predicted and ground truth trajectories
- Per-joint MAE during training/validation


## Diffusion Policy Integration (NEW)

A Diffusion Policy baseline has been integrated into the project for comparison with DNFC and MLP baselines. See dedicated documentation:

- **[DIFFUSION_INTEGRATION.md](fbc/neural_network/DIFFUSION_INTEGRATION.md)** - Comprehensive technical documentation
- **[QUICK_START_DIFFUSION.md](fbc/neural_network/QUICK_START_DIFFUSION.md)** - Quick start guide with examples

### Key Files

**Implementation:**
- [diffusion_models.py](fbc/neural_network/diffusion_models.py) - U-Net architecture and diffusion model wrapper
- [diffusion_dataset.py](fbc/neural_network/diffusion_dataset.py) - Sequence-based dataset for temporal modeling
- [train_diffusion.py](fbc/neural_network/train_diffusion.py) - Training script with EMA and cosine LR schedule
- [test_diffusion_emulated.py](fbc/neural_network/test_diffusion_emulated.py) - Example test script

**Modified Files:**
- [config.py](fbc/neural_network/config.py) - Added diffusion hyperparameters (obs_horizon, pred_horizon, action_horizon)
- [testers.py](fbc/neural_network/testers.py) - Added `load_diffusion_model()` and `get_emulated_diffusion()`
- [online_tester.py](fbc/neural_network/online_tester.py) - Added `online_test_diffusion()` for real robot testing

### Quick Start

**Train:**
```bash
cd fbc/neural_network
python train_diffusion.py  # Trains medium complexity by default
```

**Test:**
```python
from testers import Tester

tester = Tester()
tester.load_diffusion_model(train_no=0, epoch_no=4000, model_complexity="medium")
joints, success = tester.get_emulated_diffusion(5, return_path_point=True)
print(f"Reached {success}/4 milestones")
```

### Key Differences

| Aspect | DNFC/MLP Baseline | Diffusion Policy |
|--------|-------------------|------------------|
| **Prediction** | Single action (7D) | Action sequence (16×7D) |
| **Temporal** | Current state only | History (2 timesteps) |
| **Inference** | 1 forward pass | 100 denoising iterations |
| **Replanning** | Every step | Every 8 steps |
| **Data normalization** | None | [-1, 1] range |

For detailed implementation notes, troubleshooting, and hyperparameter tuning, see [DIFFUSION_INTEGRATION.md](fbc/neural_network/DIFFUSION_INTEGRATION.md).

