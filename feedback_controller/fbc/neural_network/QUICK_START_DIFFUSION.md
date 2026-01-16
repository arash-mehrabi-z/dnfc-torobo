# Diffusion Policy Quick Start Guide

## Installation

Ensure you have the required dependencies:
```bash
pip install torch torchvision diffusers scikit-image
```

## Training in 3 Steps

### 1. Configure Training
Edit [config.py](config.py):
```python
# Set dataset
self.dataset_name = "trajs:72_blocks:3_triangle_v_scarce"
self.ds_ratio = "extrap_0.85"

# Diffusion parameters (defaults are good)
self.obs_horizon = 2
self.pred_horizon = 16
self.action_horizon = 8
```

### 2. Run Training
```bash
cd fbc/neural_network
python train_diffusion.py
```

Training will:
- Train on 'medium' complexity by default
- Run 10 training instances with different seeds
- Save checkpoints every 100 epochs
- Generate loss plots

### 3. Monitor Progress
Watch the terminal output:
```
Epoch: 100:
train_loss: 0.0234
val_loss: 0.0267
```

Loss plots saved to:
```
weights/{dataset}|{ratio}|{model_name}/train_no_{i}/loss_{epoch//100}.png
```

## Testing in 2 Steps

### 1. Emulated Test
```bash
python test_diffusion_emulated.py
```

Or in Python:
```python
from testers import Tester

tester = Tester()
tester.load_diffusion_model(train_no=0, epoch_no=4000, model_complexity='medium')

# Test on episode 5
joints, path_point = tester.get_emulated_diffusion(5, return_path_point=True)
print(f"Success: {path_point}/4 milestones reached")
```

### 2. Real Robot Test (if available)
```python
from online_tester import online_test_diffusion
from testers import Tester

tester = Tester()
tester.load_diffusion_model(0, 4000, 'medium')

all_joints, success, _, states = online_test_diffusion(tester, 0)
print(f"Success: {success}/4 milestones")
```

## Quick Comparison: DNFC vs Baseline vs Diffusion

```python
from testers import Tester

tester = Tester()

# Load all models
tester.load_model(0, 4000, use_custom_loss=True, model_complexity='medium')  # DNFC + Baseline
tester.load_diffusion_model(0, 4000, 'medium')  # Diffusion

eps_num = 5

# Test DNFC
_, dnfc_success = tester.get_emulated(False, eps_num, return_path_point=True)

# Test Baseline
_, base_success = tester.get_emulated(True, eps_num, return_path_point=True)

# Test Diffusion
_, diff_success = tester.get_emulated_diffusion(eps_num, return_path_point=True)

print(f"DNFC:      {dnfc_success}/4")
print(f"Baseline:  {base_success}/4")
print(f"Diffusion: {diff_success}/4")
```

## Model Complexity Levels

Choose based on available compute:

| Complexity | U-Net Channels | Parameters | Training Time (est.) |
|------------|----------------|------------|---------------------|
| low        | [128, 256, 512] | ~6M | 2 hours |
| medium     | [256, 512, 1024] | ~65M | 4 hours |
| high       | [512, 1024, 2048] | ~260M | 8 hours |
| xhigh      | [512, 1024, 2048, 4096] | ~520M | 12 hours |

To train different complexity:
```python
# In train_diffusion.py, line 108:
for model_complexity in ['low']:  # Change to 'medium', 'high', or 'xhigh'
```

## Tuning Performance

### Faster Inference (at cost of quality)
In [config.py](config.py):
```python
self.num_diffusion_iters_inference = 50  # Default: 100
```

### More Reactive Planning
```python
self.action_horizon = 4  # Default: 8 (smaller = replan more often)
```

### Better Temporal Modeling
```python
self.obs_horizon = 3  # Default: 2 (more history)
```

### Longer Planning Horizon
```python
self.pred_horizon = 32  # Default: 16
self.action_horizon = 16  # Keep ratio ~2:1
```

## Troubleshooting

**Training loss not decreasing?**
- Check data loaded correctly: `trajectories.shape` should be printed
- Verify normalization: data should be in [-1, 1]
- Try smaller learning rate: change `learning_rate = 5e-5` in train_diffusion.py

**Inference too slow?**
- Reduce diffusion iterations: set `num_diffusion_iters_inference = 50`
- Use smaller model: try 'low' complexity
- Increase action_horizon to replan less often

**Poor test performance?**
- Train longer (check if loss still decreasing)
- Ensure using correct checkpoint (epoch_no)
- Verify EMA weights are being used (they are by default)

**Memory errors during training?**
- Reduce batch_size: change to 128 or 64 in train_diffusion.py
- Use smaller model complexity
- Enable gradient checkpointing (advanced)

## File Locations

After training, find your models here:
```
fbc/neural_network/weights/
  trajs:72_blocks:3_triangle_v_scarce|extrap_0.85|
    diffusion_pol|oh:2|ph:16|ah:8|65322.0K_params/
      train_no_0/
        fbc_0.pth      # Initial checkpoint
        fbc_100.pth    # After 100 epochs
        fbc_4000.pth   # After 4000 epochs
        loss.csv       # Training log
        loss_40.png    # Loss plot at epoch 4000
```

## Next Steps

1. **Train your first model**: Run `python train_diffusion.py`
2. **Test it**: Run `python test_diffusion_emulated.py`
3. **Compare methods**: Modify test script to compare DNFC, Baseline, Diffusion
4. **Tune parameters**: Experiment with obs_horizon, pred_horizon, action_horizon
5. **Read full docs**: See [DIFFUSION_INTEGRATION.md](DIFFUSION_INTEGRATION.md)

## Questions?

- **What's the difference from DNFC?** Diffusion predicts action sequences, DNFC predicts single actions
- **Why is inference slow?** 100 denoising iterations required; can reduce to 50
- **When to use diffusion?** When you need smooth multi-step plans
- **When to use DNFC?** When you need fast single-step reactions

For detailed architecture and implementation notes, see [DIFFUSION_INTEGRATION.md](DIFFUSION_INTEGRATION.md).
