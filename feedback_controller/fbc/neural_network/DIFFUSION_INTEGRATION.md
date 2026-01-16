# Diffusion Policy Integration

This document describes the integration of Diffusion Policy as a baseline in the feedback controller project.

## Overview

Diffusion Policy is a conditional generative model that uses denoising diffusion to predict action sequences. Unlike the existing DNFC and MLP baseline models which predict single-step actions, Diffusion Policy:

1. Predicts a **sequence of future actions** (pred_horizon=16)
2. Uses **receding horizon control** (executes action_horizon=8 actions before replanning)
3. Conditions on **observation history** (obs_horizon=2 timesteps)
4. Uses **iterative denoising** (100 diffusion steps) during inference

## Files Created

### Core Implementation

1. **[diffusion_models.py](diffusion_models.py)** - Neural network architectures
   - `SinusoidalPosEmb`: Positional encoding for diffusion timesteps
   - `Conv1dBlock`, `ConditionalResidualBlock1D`: Building blocks for U-Net
   - `ConditionalUnet1D`: 1D U-Net noise prediction network with FiLM conditioning
   - `DiffusionPolicyModel`: Wrapper class integrating U-Net with DDPM scheduler

2. **[diffusion_dataset.py](diffusion_dataset.py)** - Data loading and preprocessing
   - `DiffusionTrajectoryDataset`: Sequence-based dataset for diffusion training
   - Creates sequences with temporal structure (obs_horizon, pred_horizon, action_horizon)
   - Handles episode boundary padding
   - Normalizes data to [-1, 1] range
   - Helper functions: `create_sample_indices`, `sample_sequence`, `normalize_data`, `unnormalize_data`

3. **[train_diffusion.py](train_diffusion.py)** - Training script
   - Dedicated training script for diffusion policy
   - Uses EMA (Exponential Moving Average) for training stability
   - Cosine learning rate schedule with warmup
   - AdamW optimizer with weight decay
   - Supports multiple model complexities (low, medium, high, xhigh)

### Modified Files

4. **[config.py](config.py)** - Configuration parameters
   - Added diffusion-specific parameters:
     - `obs_horizon = 2`: Past observations to condition on
     - `pred_horizon = 16`: Future actions to predict
     - `action_horizon = 8`: Actions to execute before replanning
     - `num_diffusion_iters_train = 100`: Diffusion steps during training
     - `num_diffusion_iters_inference = 100`: Diffusion steps during inference
   - Added `get_diffusion_dims(model_complexity)`: Returns U-Net channel dimensions
   - Updated `get_model_name()`: Handles diffusion model naming

5. **[testers.py](testers.py)** - Testing infrastructure
   - Added `load_diffusion_model()`: Load trained diffusion policy
   - Added `get_emulated_diffusion()`: Emulated testing with receding horizon control
   - Uses observation deque for temporal conditioning
   - Handles action buffer and replanning logic

6. **[online_tester.py](online_tester.py)** - Real robot testing
   - Added `online_test_diffusion()`: Online testing with real robot
   - Integrates with ROS communication
   - Implements receding horizon control for real-time execution

### Test Scripts

7. **[test_diffusion_emulated.py](test_diffusion_emulated.py)** - Example test script
   - Compares Diffusion Policy vs Baseline performance
   - Runs emulated tests on all episodes
   - Saves results to CSV

## Usage

### 1. Training

Train a diffusion policy model:

```bash
cd fbc/neural_network
python train_diffusion.py
```

**Configuration**: Edit [config.py](config.py) to adjust:
- `dataset_name`: Which dataset to use
- `ds_ratio`: Train/test split (e.g., "extrap_0.85")
- `obs_horizon`, `pred_horizon`, `action_horizon`: Temporal parameters
- `num_diffusion_iters_train`: Number of diffusion steps (100 recommended)

**Model Complexity**: The script trains models with different complexities:
- `low`: [128, 256, 512] channels → ~6M parameters
- `medium`: [256, 512, 1024] channels → ~65M parameters
- `high`: [512, 1024, 2048] channels → ~260M parameters
- `xhigh`: [512, 1024, 2048, 4096] channels → ~520M parameters

**Training Details**:
- Batch size: 256
- Learning rate: 1e-4 (AdamW with weight decay 1e-6)
- LR schedule: Cosine with 500 warmup steps
- EMA power: 0.75
- Checkpoints saved every 100 epochs

### 2. Emulated Testing

Test a trained model in emulated environment:

```bash
python test_diffusion_emulated.py
```

Or use the Tester class directly:

```python
from testers import Tester

tester = Tester()
tester.load_diffusion_model(
    train_no=0,
    epoch_no=4000,
    model_complexity='medium'
)

# Test on episode 5
joints, path_point = tester.get_emulated_diffusion(5, return_path_point=True)
print(f"Reached {path_point}/4 milestones")
```

### 3. Online Testing (Real Robot)

For testing with the real Torobo robot:

```python
from online_tester import online_test_diffusion
from testers import Tester

tester = Tester()
tester.load_diffusion_model(0, 4000, 'medium')

# Test on episode 0
all_joints, traj_point, _, all_states = online_test_diffusion(tester, 0)
```

**Note**: Requires ROS environment with Torobo robot interface.

## Architecture Details

### Observation Space
- **Dimension**: 27D
- **Components**:
  - Joint positions (7D)
  - Joint velocities (7D)
  - Target coordinates for 3 objects: A, B, C (9D)
  - One-hot task phase (4D): [grasp B, place B, grasp C, place C]

### Action Space
- **Dimension**: 7D (joint velocities for 7 joints)

### Model Architecture

**ConditionalUnet1D**:
- Input: Noisy action sequence (B, pred_horizon=16, action_dim=7)
- Conditioning: Flattened observation history (B, obs_horizon × obs_dim = 2 × 27 = 54)
- Output: Predicted noise (B, 16, 7)

**Structure**:
```
Input (B, 7, 16)
  ↓ Conv1d projection
Down path: [Down1, Down2, Down3] with skip connections
  ↓ Downsample after each level
Bottleneck: [Mid1, Mid2] residual blocks
  ↓
Up path: [Up1, Up2, Up3] with skip connections from down path
  ↓ Upsample after each level
Final Conv1d → Output (B, 7, 16)
```

Each residual block uses **FiLM conditioning**:
- Observation embedding → [scale, bias] per channel
- Apply: `out = scale * features + bias`

### Training Process

1. **Data Loading**:
   - Load trajectories (N_eps, 299, 35)
   - Create sequences with padding at episode boundaries
   - Normalize observations and actions to [-1, 1]

2. **Forward Diffusion** (training):
   - Sample noise: `ε ~ N(0, I)`
   - Sample timestep: `t ~ Uniform(0, T)`
   - Add noise to action: `a_t = √(α_t) * a_0 + √(1 - α_t) * ε`

3. **Noise Prediction**:
   - Predict noise: `ε_θ = UNet(a_t, t, obs)`
   - Loss: `MSE(ε_θ, ε)`

4. **Reverse Diffusion** (inference):
   ```python
   a_T ~ N(0, I)  # Start from noise
   for t in [T, T-1, ..., 1]:
       ε_θ = UNet(a_t, t, obs)
       a_{t-1} = denoise(a_t, ε_θ, t)  # DDPM step
   return a_0  # Denoised action sequence
   ```

### Receding Horizon Control

During execution:
1. Observe last `obs_horizon=2` states
2. Predict `pred_horizon=16` future actions
3. Execute only `action_horizon=8` actions
4. Repeat (replanning with updated observations)

**Execution Pattern**:
```
Time:     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Predict:  |-------- 16 actions predicted ---------|
Observe:  |o|o|                                      <- 2 obs used
Execute:     |x|x|x|x|x|x|x|x|                       <- 8 actions executed
Replan:                       |o|o|                  <- replan with new obs
                              |-------- 16 actions predicted ---------|
```

## Key Differences from DNFC/Baseline

| Aspect | DNFC/Baseline | Diffusion Policy |
|--------|---------------|------------------|
| Prediction | Single action | Action sequence (16 steps) |
| Temporal context | Current state only | History (2 timesteps) |
| Inference | Single forward pass | Iterative denoising (100 steps) |
| Training objective | MSE on actions (+latent) | MSE on noise prediction |
| Data normalization | None | [-1, 1] normalization |
| Replanning | Every step | Every 8 steps |

## Model Checkpoints

Trained models are saved to:
```
fbc/neural_network/weights/
  {dataset_name}|{ds_ratio}|{model_name}/
    train_no_{i}/
      fbc_{epoch}.pth
```

Example path:
```
weights/trajs:72_blocks:3_triangle_v_scarce|extrap_0.85|diffusion_pol|oh:2|ph:16|ah:8|65322.0K_params/train_no_0/fbc_4000.pth
```

## Performance Considerations

### Inference Speed
- **Bottleneck**: 100 denoising iterations per prediction
- **Mitigation**:
  - Execute 8 actions before replanning (amortizes cost)
  - Consider reducing `num_diffusion_iters_inference` (e.g., 50 steps)
  - Use smaller model complexity for faster inference

### Memory Usage
- Medium complexity: ~65M parameters
- High complexity: ~260M parameters
- Batch size during training: 256 samples

### Training Time
- ~4 hours for 5000 epochs on medium complexity (NVIDIA GPU)
- EMA updates add ~10% overhead
- Cosine LR schedule improves convergence

## Hyperparameter Tuning

### Critical Parameters

1. **obs_horizon** (default: 2)
   - More history → better temporal reasoning
   - More history → larger U-Net input dimension
   - Recommended: 2-3

2. **pred_horizon** (default: 16)
   - Longer horizon → smoother trajectories
   - Longer horizon → harder to predict
   - Should be ≥ 2 × action_horizon

3. **action_horizon** (default: 8)
   - Larger → fewer replannings, faster execution
   - Smaller → more reactive to environment changes
   - Recommended: pred_horizon / 2

4. **num_diffusion_iters** (default: 100)
   - More iterations → better denoising quality
   - More iterations → slower inference
   - Training: 100 recommended
   - Inference: can reduce to 50 for speed

### Ablation Studies

To understand impact of each parameter:

1. **Vary obs_horizon**: 1, 2, 3, 4
2. **Vary pred_horizon**: 8, 16, 32
3. **Vary action_horizon**: 4, 8, 16
4. **Vary num_diffusion_iters**: 25, 50, 100

Compare metrics:
- Success rate (milestones reached)
- DTW distance to ground truth
- Inference time per action

## Troubleshooting

### Common Issues

1. **NaN losses during training**
   - Check data normalization (should be [-1, 1])
   - Reduce learning rate
   - Increase batch size
   - Check for corrupted data

2. **Poor inference performance**
   - Ensure using EMA weights (not raw weights)
   - Check normalization/unnormalization is symmetric
   - Verify action buffer logic (should replan at task changes)

3. **Slow inference**
   - Reduce `num_diffusion_iters_inference`
   - Use smaller model complexity
   - Consider caching action sequences longer (increase action_horizon)

4. **Task switching not detected**
   - Check one-hot updates in observation deque
   - Verify milestone distance thresholds in `close_enough()`
   - Force replanning by clearing action buffer at task switches

## References

- **Diffusion Policy Paper**: [Chi et al., 2023](https://diffusion-policy.cs.columbia.edu/)
- **Original Implementation**: Push-T environment demo (included in `diffusion_policy_state_pusht_demo.ipynb`)
- **DDPM Scheduler**: HuggingFace Diffusers library

## Future Improvements

1. **Multi-modal distributions**: Diffusion can represent multiple solutions
2. **Vision conditioning**: Replace Cartesian coordinates with image observations
3. **Hierarchical planning**: Use longer pred_horizon with hierarchical action abstraction
4. **Faster sampling**: DDIM or DPM-Solver for fewer diffusion steps
5. **Conditional generation**: Specify desired behavior through language or demonstrations
