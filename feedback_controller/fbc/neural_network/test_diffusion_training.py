"""Quick test to verify diffusion training works without NaN issues"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusion_models import DiffusionPolicyModel
from diffusion_dataset import DiffusionTrajectoryDataset
from config import Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = Config()

# Dataset
cur_dir = os.path.dirname(__file__)
ds_root_dir = os.path.join(cur_dir, f'data/torobo/{config.dataset_name}')

print("Loading dataset...")
dataset = DiffusionTrajectoryDataset(
    ds_root_dir, config.ds_file_name,
    config.joints_num,
    pred_horizon=config.pred_horizon,
    obs_horizon=config.obs_horizon,
    action_horizon=config.action_horizon
)

# Small subset for testing
train_size = 100
train_set = torch.utils.data.Subset(dataset, range(train_size))
dataloader = DataLoader(train_set, batch_size=16, shuffle=True)

# Model
obs_dim = config.state_dim + config.coords_dim + config.onehot_dim
model = DiffusionPolicyModel(
    obs_dim=obs_dim,
    action_dim=config.joints_num,
    obs_horizon=config.obs_horizon,
    pred_horizon=config.pred_horizon,
    action_horizon=config.action_horizon,
    num_diffusion_iters=100,
    down_dims=[256, 512, 1024]  # medium complexity
).to(device)

print(f"\nModel on device: {device}")
print(f"Training on {len(train_set)} samples\n")

# Test a few training iterations
for epoch in range(5):
    epoch_loss = 0
    for batch in dataloader:
        nobs = batch['obs'].to(device).float()
        naction = batch['action'].to(device).float()
        B = nobs.shape[0]

        obs_cond = nobs.flatten(start_dim=1)
        noise = torch.randn(naction.shape, device=device)
        timesteps = torch.randint(0, 100, (B,), device=device).long()

        noisy_actions = model.noise_scheduler.add_noise(naction, noise, timesteps)
        noise_pred = model.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

        loss = nn.functional.mse_loss(noise_pred, noise)
        epoch_loss += loss.item()

        # Check for NaN
        if torch.isnan(loss):
            print(f"ERROR: NaN loss at epoch {epoch}")
            print(f"  obs range: [{nobs.min()}, {nobs.max()}]")
            print(f"  action range: [{naction.min()}, {naction.max()}]")
            print(f"  noise_pred range: [{noise_pred.min()}, {noise_pred.max()}]")
            break

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch}: loss = {avg_loss:.6f}")

    if torch.isnan(torch.tensor(avg_loss)):
        print("\nTraining failed: NaN loss detected")
        break
else:
    print("\n✓ Training test successful! No NaN issues.")
