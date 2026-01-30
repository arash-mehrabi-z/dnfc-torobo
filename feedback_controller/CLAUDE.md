# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural network-based feedback controller for robotic manipulation on the Torobo robot (7-DOF arm). The system learns control policies from trajectory demonstrations, using latent space representations of target objects to generate real-time motor commands.

## Running Commands

All Python scripts are in `fbc/neural_network/` and should be run from that directory with the virtual environment activated:

```bash
cd fbc/neural_network
source .venv/bin/activate
```

**Training:**
```bash
python train_w_datasets.py
```
Training iterates through model complexities (low, medium, high, xhigh) and runs multiple training sessions. Configure via `config.py` before running.

**Offline Testing:**
```bash
python -c "from testers import Tester; t = Tester(); t.load_model(train_no=0, epoch_no=4000, use_custom_loss=True, model_complexity='high')"
```

**Online Robot Testing (requires ROS and Torobo):**
```bash
python online_tester.py
```

## Architecture

### Core Files

- [config.py](fbc/neural_network/config.py) - Central configuration: dataset paths, model hyperparameters, loss settings (C=1e-5 for custom loss)
- [nn_models.py](fbc/neural_network/nn_models.py) - Neural network architectures:
  - `GeneralModel` - Main controller: encoder MLP → latent space → controller MLP → action
  - `MLPBaseline` - Direct state+target → action baseline for comparison
  - `CustomLoss` - MSE on actions + scaled MSE penalty on latent space difference
- [train_w_datasets.py](fbc/neural_network/train_w_datasets.py) - Training pipeline with `TrajectoryDataset` loader
- [testers.py](fbc/neural_network/testers.py) - `Tester` class for offline model evaluation
- [online_tester.py](fbc/neural_network/online_tester.py) - Real-time robot control with ROS/UDP communication

### Control Flow

1. Target representation (9D coords + 4D one-hot for task phase) → Encoder → Latent state `x_des`
2. Current state (7 joint positions + 7 velocities) = `x_t`
3. Difference `x_des - x_t` → Controller MLP → tanh → 7D joint velocity commands

### Data Layout

- `data/torobo/{dataset_name}/` - Training datasets (e.g., `trajs:72_blocks:3_triangle_v_scarce`)
  - `train_extrap_0.85.npy`, `test_extrap_0.85.npy` - Train/test splits
  - `split_indices_extrap_0.85.pt` - Index splits for reproducibility
- `weights/{dataset}|{split}|{model_name}/train_no_{n}/` - Saved model checkpoints

### Key Dependencies

PyTorch, torchvision, NumPy, Matplotlib, ROS (rospy for online testing), dtw-python

## Configuration

Edit `config.py` to change:
- `use_custom_loss` / `C` - Toggle and weight for latent space penalty
- `dataset_name` / `ds_ratio` - Which dataset and train/test split
- `v_name` - Model architecture variant identifier
- Model dimensions via `get_model_dims(complexity)` where complexity ∈ {low, medium, high, xhigh}