from diffusion_dataset import DiffusionTrajectoryDataset
from config import Config
import os
import numpy as np

config = Config()
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

print('\n=== Observation Stats ===')
print('Observation dimension:', len(dataset.stats['obs']['min']))
obs_range = dataset.stats['obs']['max'] - dataset.stats['obs']['min']
for i in range(len(obs_range)):
    if obs_range[i] == 0:
        print(f"Dim {i:2d}: CONSTANT = {dataset.stats['obs']['min'][i]:.6f}")
    else:
        print(f"Dim {i:2d}: [{dataset.stats['obs']['min'][i]:.6f}, {dataset.stats['obs']['max'][i]:.6f}] (range: {obs_range[i]:.6f})")

print('\n=== Action Stats ===')
print('Action dimension:', len(dataset.stats['action']['min']))
action_range = dataset.stats['action']['max'] - dataset.stats['action']['min']
for i in range(len(action_range)):
    if action_range[i] == 0:
        print(f"Dim {i:2d}: CONSTANT = {dataset.stats['action']['min'][i]:.6f}")
    else:
        print(f"Dim {i:2d}: [{dataset.stats['action']['min'][i]:.6f}, {dataset.stats['action']['max'][i]:.6f}] (range: {action_range[i]:.6f})")

print('\n=== Checking for NaN in normalized data ===')
sample = dataset[0]
print(f"Sample obs shape: {sample['obs'].shape}")
print(f"Sample action shape: {sample['action'].shape}")
print(f"NaN in obs: {np.isnan(sample['obs']).any()}")
print(f"NaN in action: {np.isnan(sample['action']).any()}")
