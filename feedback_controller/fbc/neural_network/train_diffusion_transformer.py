"""
Training script for transformer-based diffusion policy.
Similar to train_diffusion.py but uses TransformerForDiffusion instead of ConditionalUnet1D.
"""
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion_models import DiffusionTransformerPolicyModel
from diffusion_dataset import DiffusionTrajectoryDataset, unnormalize_data
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from config import Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = Config()

# General settings
current_dir_path = os.path.dirname(__file__)
current_dir_path = Path(current_dir_path)
fbc_root_dir_path = current_dir_path.parent.absolute()
dataset_name = config.dataset_name
ds_root_dir = os.path.join(fbc_root_dir_path,
                            f'neural_network/data/torobo/{dataset_name}')
ds_file_name = config.ds_file_name

# Model settings
joints_num = config.joints_num
action_dim = joints_num
obs_dim = config.state_dim + config.coords_dim + config.onehot_dim

# Training settings
num_epochs = 15000 + 1
batch_size = 256
learning_rate = 1e-4
weight_decay = 1e-3  # Higher weight decay for transformers
validation_interval = 100
num_trains = 2
lr_warmup_steps = 1000  # Warmup is critical for transformers


def log_loss(n, train_loss, val_loss, weights_storage_root_dir):
    """Log training and validation losses to CSV file"""
    loss_file_path = os.path.join(weights_storage_root_dir, "loss.csv")
    with open(loss_file_path, 'a') as f:
        writer = csv.writer(f)
        row = [n, train_loss, val_loss]
        writer.writerow(row)

    print("===")
    print(f"Epoch: {n}:")
    print(f"train_loss: {train_loss}")
    print(f"val_loss: {val_loss}")


def train_epoch(model, dataloader, optimizer, lr_scheduler, ema, device):
    """Training loop for one epoch of transformer diffusion policy"""
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        # batch['obs']: (B, obs_horizon, obs_dim)
        # batch['action']: (B, pred_horizon, action_dim)
        nobs = batch['obs'].to(device).float()
        naction = batch['action'].to(device).float()
        B = nobs.shape[0]

        # Sample noise
        noise = torch.randn(naction.shape, device=device)

        # Sample diffusion timestep
        timesteps = torch.randint(
            0, model.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # Add noise to actions (forward diffusion)
        noisy_actions = model.noise_scheduler.add_noise(
            naction, noise, timesteps
        )

        # Predict noise with observation conditioning
        # Note: transformer expects (B, obs_horizon, obs_dim) for conditioning
        noise_pred = model.noise_pred_net(
            sample=noisy_actions,
            timestep=timesteps,
            cond=nobs  # Pass full observation sequence for cross-attention
        )

        # Compute loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # Optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for transformer stability
        torch.nn.utils.clip_grad_norm_(model.noise_pred_net.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()

        # Update EMA
        if ema is not None:
            ema.step(model.noise_pred_net.parameters())

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, device):
    """Validation for transformer diffusion policy"""
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            nobs = batch['obs'].to(device).float()
            naction = batch['action'].to(device).float()
            B = nobs.shape[0]

            noise = torch.randn(naction.shape, device=device)
            timesteps = torch.randint(
                0, model.noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            noisy_actions = model.noise_scheduler.add_noise(
                naction, noise, timesteps
            )

            # Predict noise with observation conditioning
            noise_pred = model.noise_pred_net(
                sample=noisy_actions,
                timestep=timesteps,
                cond=nobs
            )

            loss = nn.functional.mse_loss(noise_pred, noise)
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


def visualize_losses(train_losses, val_losses, n, file_name, title, fig,
                     weights_storage_root_dir, validation_interval):
    """Visualize training and validation losses"""
    plt.figure(fig)
    plt.clf()

    # Plot validation loss
    plt.plot([-1] + list(range(0, n+1, validation_interval)), np.log(val_losses),
             label='Validation Loss')

    # Plot training loss
    plt.plot(list(range(len(train_losses))), np.log(train_losses),
             label='Training Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.title(title)

    # Update the plot
    plt.draw()
    plt.pause(0.001)

    # Save the plot
    loss_fig_path = os.path.join(weights_storage_root_dir, file_name)
    plt.savefig(loss_fig_path)


def create_csv_files(weights_storage_root_dir):
    """Create CSV file for logging losses"""
    loss_file_path = os.path.join(weights_storage_root_dir, "loss.csv")
    with open(loss_file_path, 'w') as f:
        writer = csv.writer(f)
        row = ["n", "train_loss", "val_loss"]
        writer.writerow(row)


# Main training loop
for model_complexity in ['low']:  # ['minimal', 'low', 'medium', 'high', 'xhigh']
    n_layer, n_head, n_emb, p_drop_attn, n_cond_layers = config.get_transformer_diffusion_dims(model_complexity)

    for i_train in range(num_trains):
        fig = plt.figure(figsize=(12.8, 9.6))

        model_name = config.get_model_name(False, False, False, use_diffusion=True, use_transformer=True)

        # Create dataset
        dataset = DiffusionTrajectoryDataset(
            ds_root_dir, ds_file_name,
            joints_num,
            pred_horizon=config.pred_horizon,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon
        )

        # Split into train and validation
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        print(f"train_set: {len(train_set)}")
        print(f"val_set: {len(val_set)}")

        # Create transformer-based diffusion model
        model = DiffusionTransformerPolicyModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=config.obs_horizon,
            pred_horizon=config.pred_horizon,
            action_horizon=config.action_horizon,
            num_diffusion_iters=config.num_diffusion_iters_train,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=0.0,
            p_drop_attn=p_drop_attn,
            causal_attn=True,
            n_cond_layers=n_cond_layers
        )
        model = model.to(device)

        num_params = sum(p.numel() for p in model.parameters()) / 1e3
        model_name += f"|{num_params}K_params"

        # Create dataloaders
        train_dataloader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True, num_workers=1,
                                      pin_memory=True, persistent_workers=True)
        val_dataloader = DataLoader(val_set, batch_size=batch_size,
                                    shuffle=False, num_workers=1,
                                    pin_memory=True, persistent_workers=True)

        # Setup directory
        weights_storage_root_dir = os.path.join(
            current_dir_path,
            f"weights/{dataset_name}|{config.ds_ratio}|{model_name}/train_no_{i_train}"
        )

        print(f"=== Train No: {i_train} ===")
        print(f"train num_samples: {len(train_dataloader.dataset)}")
        print(f"val num_samples: {len(val_dataloader.dataset)}")
        print(f"device: {device}")
        print(f"dataset_name: {dataset_name}")
        print(f"model_name: {model_name}")
        print(f"{num_params} K parameters")
        print(f"Transformer config: n_layer={n_layer}, n_head={n_head}, n_emb={n_emb}")
        print(f"weights_storage_root_dir: {weights_storage_root_dir}")

        if os.path.exists(weights_storage_root_dir):
            print("The weight directory exists... Are you training one more time?")
            sys.exit(1)
        else:
            os.makedirs(weights_storage_root_dir)

        create_csv_files(weights_storage_root_dir)

        # Create EMA model for stable training
        ema = EMAModel(
            parameters=model.noise_pred_net.parameters(),
            power=0.75
        )

        # Use AdamW optimizer with weight decay groups
        # Transformers benefit from separating parameters by weight decay
        optim_groups = model.noise_pred_net.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95)  # Standard betas for transformers
        )

        # Cosine LR scheduler with warmup (critical for transformers)
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=len(train_dataloader) * num_epochs
        )

        # Initialize loss tracking
        train_loss = 1e-10
        train_losses = [train_loss]
        val_losses = []

        # Training loop
        for n in range(num_epochs):
            if n == 0:
                # Initial validation using EMA weights
                ema.store(model.noise_pred_net.parameters())
                ema.copy_to(model.noise_pred_net.parameters())
                val_loss = validate_epoch(model, val_dataloader, device)
                val_losses.append(val_loss**0.5)
                log_loss(n, train_loss, val_loss, weights_storage_root_dir)
                ema.restore(model.noise_pred_net.parameters())

            start_time = time.time()
            train_loss = train_epoch(model, train_dataloader, optimizer,
                                    lr_scheduler, ema, device)
            train_losses.append(train_loss**0.5)
            end_time = time.time()

            if n % validation_interval == 0:
                epoch_time = end_time - start_time
                print(f"Last epoch taken time: {epoch_time:.3f} seconds")

                # Store original weights, copy EMA for validation/saving
                ema.store(model.noise_pred_net.parameters())
                ema.copy_to(model.noise_pred_net.parameters())

                # Run validation with EMA weights
                val_loss = validate_epoch(model, val_dataloader, device)
                val_losses.append(val_loss**0.5)

                log_loss(n, train_loss, val_loss, weights_storage_root_dir)

                # Save checkpoint (saves EMA weights)
                weight_file_path = os.path.join(weights_storage_root_dir,
                                               f"fbc_{n}.pth")
                torch.save(model.state_dict(), weight_file_path)

                # Restore original weights for continued training
                ema.restore(model.noise_pred_net.parameters())

                # Visualize losses
                visualize_losses(train_losses, val_losses, n,
                               f"loss_{n//100}.png", "Transformer Diffusion Training Loss",
                               fig, weights_storage_root_dir, validation_interval)

        plt.close('all')
