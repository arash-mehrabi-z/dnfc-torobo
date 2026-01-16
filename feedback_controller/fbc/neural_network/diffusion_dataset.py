import os
import numpy as np
import torch
from torch.utils.data import Dataset


def create_sample_indices(
        episode_ends: np.ndarray, sequence_length: int,
        pad_before: int = 0, pad_after: int = 0):
    """
    Create indices for all possible sequences in the dataset.
    Handles padding at episode boundaries.

    Args:
        episode_ends: Array of one-past-last index for each episode
        sequence_length: Length of sequences to extract
        pad_before: Number of timesteps to pad before each episode
        pad_after: Number of timesteps to pad after each episode

    Returns:
        indices: Array of shape (N, 4) where each row is:
            [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
    """
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    """
    Extract a sequence from the dataset with proper padding.

    Args:
        train_data: Dictionary with 'obs' and 'action' arrays
        sequence_length: Target length of sequence
        buffer_start_idx: Start index in the full dataset
        buffer_end_idx: End index in the full dataset
        sample_start_idx: Start index in the output sequence
        sample_end_idx: End index in the output sequence

    Returns:
        result: Dictionary with padded 'obs' and 'action' sequences
    """
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


def get_data_stats(data):
    """
    Compute min and max for each dimension in the data.

    Args:
        data: Array of shape (N, ..., dim)

    Returns:
        stats: Dictionary with 'min' and 'max' arrays of shape (dim,)
    """
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    """
    Normalize data to [-1, 1] range.

    Args:
        data: Array to normalize
        stats: Dictionary with 'min' and 'max'

    Returns:
        ndata: Normalized array in [-1, 1]
    """
    # Compute range, avoiding division by zero
    data_range = stats['max'] - stats['min']
    # For constant dimensions (range = 0), set range to 1 to avoid division by zero
    data_range = np.where(data_range == 0, 1.0, data_range)

    # normalize to [0,1]
    ndata = (data - stats['min']) / data_range
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    """
    Unnormalize data from [-1, 1] to original range.

    Args:
        ndata: Normalized array in [-1, 1]
        stats: Dictionary with 'min' and 'max'

    Returns:
        data: Unnormalized array
    """
    # Compute range, matching normalization logic
    data_range = stats['max'] - stats['min']
    data_range = np.where(data_range == 0, 1.0, data_range)

    ndata = (ndata + 1) / 2
    data = ndata * data_range + stats['min']
    return data


class DiffusionTrajectoryDataset(Dataset):
    """
    Dataset for diffusion policy training.
    Loads trajectory data and creates sequences with temporal structure.
    """
    def __init__(self, ds_root_dir, file_name, joint_dim,
                 pred_horizon, obs_horizon, action_horizon):
        """
        Args:
            ds_root_dir: Root directory containing the dataset
            file_name: Name of the .npy file containing trajectories
            joint_dim: Number of joints (7 for this robot)
            pred_horizon: Number of future actions to predict (16)
            obs_horizon: Number of past observations to condition on (2)
            action_horizon: Number of actions to execute before replanning (8)
        """
        self.joint_dim = joint_dim
        self.state_dim = 2 * joint_dim  # positions + velocities
        self.coords_dim = 3 * 3  # 3 objects × (x, y, z)
        self.onehot_dim = 4  # 4 task phases
        self.obs_dim = self.state_dim + self.coords_dim + self.onehot_dim  # 14 + 9 + 4 = 27
        self.action_dim = joint_dim

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # Load trajectory data
        data_file_path = os.path.join(ds_root_dir, file_name)
        trajectories = np.load(data_file_path)  # (N_eps, 299, 35)
        print(f"Loaded data from {data_file_path}")
        print(f"Trajectories shape: {trajectories.shape}")

        num_episodes = trajectories.shape[0]
        num_steps = trajectories.shape[1]

        # Extract observations and actions from trajectories
        # Observation: state (14) + target coords (9) + one-hot (4) = 27
        # Action: joint velocities (7)
        all_obs = []
        all_actions = []
        episode_ends = []

        for ep_idx in range(num_episodes):
            ep_traj = trajectories[ep_idx]  # (299, 35)

            # Extract state (positions + velocities)
            state = ep_traj[:, 1:1+self.state_dim]  # (299, 14)

            # Extract target coordinates
            target_start_idx = 1 + self.state_dim
            target_end_idx = target_start_idx + self.coords_dim
            target_coords = ep_traj[:, target_start_idx:target_end_idx]  # (299, 9)

            # Extract one-hot task phase
            onehot_start_idx = target_end_idx
            onehot_end_idx = onehot_start_idx + self.onehot_dim
            one_hot = ep_traj[:, onehot_start_idx:onehot_end_idx]  # (299, 4)

            # Concatenate to form observation
            obs = np.concatenate([state, target_coords, one_hot], axis=1)  # (299, 27)

            # Extract actions (joint velocities)
            action = ep_traj[:, -self.joint_dim:]  # (299, 7)

            all_obs.append(obs)
            all_actions.append(action)
            episode_ends.append((ep_idx + 1) * num_steps)

        # Concatenate all episodes
        all_obs = np.concatenate(all_obs, axis=0)  # (N_eps * 299, 27)
        all_actions = np.concatenate(all_actions, axis=0)  # (N_eps * 299, 7)
        episode_ends = np.array(episode_ends)

        # Store raw data
        train_data = {
            'obs': all_obs,
            'action': all_actions
        }

        # Compute sample indices with padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1
        )

        # Compute statistics and normalize data to [-1, 1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)

            # Check for constant dimensions
            data_range = stats[key]['max'] - stats[key]['min']
            constant_dims = np.where(data_range == 0)[0]
            if len(constant_dims) > 0:
                print(f"Warning: {key} has constant values in dimensions: {constant_dims}")
                print(f"  Values: {stats[key]['min'][constant_dims]}")

            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data

        print(f"Dataset created with {len(self.indices)} sequences")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the start/end indices for this sequence
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # Get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # Discard unused observations (only keep first obs_horizon)
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]

        return nsample
