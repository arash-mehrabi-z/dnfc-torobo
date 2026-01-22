# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Robotics research project implementing **Desired Neural Feedback Control (DNFC)** for the Torobo humanoid robot. Trains neural network controllers that map from desired task coordinates + robot state → joint velocities.

## Architecture

### Models

| Model | Architecture | Prediction | Use Case |
|-------|--------------|------------|----------|
| **DNFC** | Encoder MLP → Controller MLP | Single action (7D) | Fast reactive control |
| **MLP Baseline** | Direct MLP | Single action (7D) | Simple baseline |
| **Diffusion Policy (U-Net)** | 1D U-Net + DDPM | Action sequence (16×7D) | Smooth multi-step plans |
| **Diffusion Policy (Transformer)** | Transformer Encoder-Decoder + DDPM | Action sequence (16×7D) | Attention-based multi-step plans |

**DNFC (Desired Neural Feedback Control)**:
- Encoder: target representation → latent space (14D: 7 joint pos + 7 vel)
- Controller: latent error (desired - current) → joint velocity
- Custom loss: MSE(actions) + C × MSE(latent representations)

**Diffusion Policy (U-Net)**:
- Conditions on observation history (obs_horizon=2)
- Predicts pred_horizon=16 future actions
- Executes action_horizon=8 before replanning
- 100 denoising iterations during inference
- Uses 1D convolutional U-Net with FiLM conditioning

**Diffusion Policy (Transformer)**:
- Same temporal structure as U-Net version
- Uses encoder-decoder transformer architecture
- Cross-attention conditioning on observations
- Causal attention mask for autoregressive structure
- Better at capturing long-range dependencies

### Key Files

| File | Purpose |
|------|---------|
| [config.py](fbc/neural_network/config.py) | All hyperparameters and model settings |
| [nn_models.py](fbc/neural_network/nn_models.py) | DNFC, MLP Baseline, CustomLoss |
| [diffusion_models.py](fbc/neural_network/diffusion_models.py) | ConditionalUnet1D, TransformerForDiffusion, DiffusionPolicyModel, DiffusionTransformerPolicyModel |
| [train_w_datasets.py](fbc/neural_network/train_w_datasets.py) | Training for DNFC/MLP |
| [train_diffusion.py](fbc/neural_network/train_diffusion.py) | Training for U-Net Diffusion Policy |
| [train_diffusion_transformer.py](fbc/neural_network/train_diffusion_transformer.py) | Training for Transformer Diffusion Policy |
| [testers.py](fbc/neural_network/testers.py) | Tester class for emulated testing |
| [online_tester.py](fbc/neural_network/online_tester.py) | Real robot testing (ROS) |
| [global_defines.py](fbc/neural_network/global_defines.py) | TOR class: robot constants, joint limits |
| [torkin.py](fbc/neural_network/torkin.py) | Forward/inverse kinematics |

## Common Commands

### Environment Setup
```bash
cd fbc/neural_network
source .venv/bin/activate  # Activate virtual environment
pip install torch torchvision numpy matplotlib scikit-learn dtw-python pillow
pip install diffusers transformers scikit-image  # For diffusion policy
```

### Training
```bash
cd fbc/neural_network

# DNFC + MLP Baseline (trains all complexities with 10 seeds each)
python train_w_datasets.py

# Diffusion Policy (U-Net)
python train_diffusion.py

# Diffusion Policy (Transformer)
python train_diffusion_transformer.py
```

Configuration in [config.py](fbc/neural_network/config.py):
- `dataset_name`: e.g., `"trajs:72_blocks:3_triangle_v_scarce"`
- `ds_ratio`: `"extrap_0.85"` or `"interp_0.95"`
- `use_custom_loss`, `C`: DNFC loss settings
- `obs_horizon`, `pred_horizon`, `action_horizon`: Diffusion temporal params

### Testing
```bash
cd fbc/neural_network

# Emulated testing
python emulated_test.py                      # DNFC/MLP
python test_diffusion_emulated.py            # U-Net Diffusion
python test_diffusion_transformer_emulated.py # Transformer Diffusion

# Online robot testing (requires ROS)
python online_tester.py
```

### Using the Tester Class
```python
from testers import Tester

tester = Tester()

# Load DNFC + Baseline
tester.load_model(train_no=0, epoch_no=4000, use_custom_loss=True, model_complexity='medium')
joints, success = tester.get_emulated(use_baseline=False, num=5, return_path_point=True)

# Load U-Net Diffusion
tester.load_diffusion_model(train_no=0, epoch_no=4000, model_complexity='medium')
joints, success = tester.get_emulated_diffusion(5, return_path_point=True)

# Load Transformer Diffusion
tester.load_diffusion_transformer_model(train_no=0, epoch_no=4000, model_complexity='medium')
joints, success = tester.get_emulated_diffusion_transformer(5, return_path_point=True)
```

## Dataset Structure

Location: `fbc/neural_network/data/torobo/trajs:{num}_blocks:3_{variant}/`

**Trajectory shape**: `(N_episodes, 299, 35)`

| Index | Content | Dim |
|-------|---------|-----|
| `[0]` | Step number | 1 |
| `[1:15]` | State (joint pos + vel) | 14 |
| `[15:24]` | Target coords (A, B, C) | 9 |
| `[24:28]` | One-hot task phase | 4 |
| `[28:35]` | Action (joint velocities) | 7 |

**Task phases**: Grasp B → Place on A → Grasp C → Place on A

## Code Patterns

### Model Forward Pass
```python
# DNFC: returns (action, desired_latent, latent_error)
action, x_des, diff = model(target_repr, state)

# MLP Baseline
action = baseline(torch.cat((target_repr, state), dim=1))

# U-Net Diffusion: flatten obs for FiLM conditioning
obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
action_seq = diffusion_model.get_action(obs_seq)  # (1, pred_horizon, 7)

# Transformer Diffusion: pass obs sequence for cross-attention
# obs_seq shape: (B, obs_horizon, obs_dim)
action_seq = transformer_diffusion_model.get_action(obs_seq)  # (1, pred_horizon, 7)
```

### Checkpoint Paths
```
weights/{dataset}|{ds_ratio}|{model_name}/train_no_{i}/fbc_{epoch}.pth
```

Model name formats:
- DNFC: `cus_los_1e-05|tar_cart|2+2l_lat:sub-nvel|24.085K_params`
- MLP: `mse_los|tar_cart|base|3l_base|24.091K_params`
- U-Net Diffusion: `diffusion_pol|oh:2|ph:16|ah:8|65322.0K_params`
- Transformer Diffusion: `diffusion_transformer|oh:2|ph:16|ah:8|170.631K_params`

## Constraints

- **7-DOF arm**: All models use 7 joints (right arm)
- **Fixed trajectory**: 299 steps per episode
- **State dim**: Always 14D (7 positions + 7 velocities)
- **ROS required**: Online testing needs Torobo robot interface
- **Datasets**: Stored locally in `data/`, not version-controlled

## Results Structure

```
results/{dataset}_{params}K_{ds_ratio}/ep:{epoch}/on_{model}/
├── perf.csv              # DTW distance, success rate per episode
├── plt_{eps}_{train}.png # 3D trajectory visualization
└── all_states_{model}.pickle
```

## Model Complexity Levels

| Level | DNFC/MLP Hidden | Diffusion U-Net Channels | Transformer (layer/head/emb) | Params |
|-------|-----------------|--------------------------|------------------------------|--------|
| minimal | - | [16, 32] | 2/2/64 | ~25K |
| low | 64/192 | [128, 256, 512] | 4/4/128 | ~6K / ~6M / ~200K |
| medium | 128/384 | [256, 512, 1024] | 8/4/256 | ~24K / ~65M / ~1.5M |
| high | 256/768 | [512, 1024, 2048] | 12/8/512 | ~96K / ~260M / ~12M |
| xhigh | 512/1536 | [512, 1024, 2048, 4096] | 12/12/768 | ~384K / ~520M / ~45M |

## Additional Documentation

For diffusion policy implementation details, hyperparameter tuning, and troubleshooting:
- [DIFFUSION_INTEGRATION.md](fbc/neural_network/DIFFUSION_INTEGRATION.md)
- [QUICK_START_DIFFUSION.md](fbc/neural_network/QUICK_START_DIFFUSION.md)
