"""
Plot end-effector 3D trajectories from saved all_states file.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torkin import TorKin

# Results directory
RESULTS_DIR = os.path.expanduser(
    "~/catkin_ws/src/feedback_controller/fbc/neural_network/results/"
    "trajs:360_blocks:3_imgs_2cams_909.691K_interp_0.85/ep:4400/"
    "image_model_img_task_enc_rob_enc"
)


def load_states(results_dir, filename="all_states_image_model"):
    """Load states from pickle file."""
    file_path = os.path.join(results_dir, filename)
    with open(file_path, 'rb') as f:
        all_states = pickle.load(f)
    print(f"Loaded {len(all_states)} episodes from {file_path}")
    return all_states


def compute_ee_trajectory(states_episode, kin):
    """Compute end-effector positions for an episode.

    Args:
        states_episode: List of states, each state is [j1..j7, v1..v7]
        kin: TorKin instance

    Returns:
        x, y, z: Lists of end-effector positions
    """
    x, y, z = [], [], []
    for state in states_episode:
        # First 7 values are joint positions
        joints = state[:7]
        # Prepend torso values [0, 0] for FK
        q_full = np.array([0, 0] + list(joints))
        pos, _ = kin.forwardkin(1, q_full)
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])
    return x, y, z


def plot_ee_trajectory_3d(x, y, z, eps_num, save_path):
    """Plot and save 3D end-effector trajectory."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(x, y, z, 'b-', linewidth=1.5, label='EE Trajectory')
    ax.scatter(x, y, z, c=np.arange(len(x)), cmap='viridis', s=5)

    # Mark start and end points
    ax.scatter([x[0]], [y[0]], [z[0]], c='g', s=100, marker='o', label='Start')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c='r', s=100, marker='x', label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Episode {eps_num}: End-Effector Trajectory')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main(results_dir, filename="all_states_image_model"):
    # Initialize forward kinematics
    kin = TorKin()

    # Load states
    all_states = load_states(results_dir, filename)

    # Create output directory for plots
    plots_dir = os.path.join(results_dir, "ee_trajectory_plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Process each episode
    for eps_num, states_episode in enumerate(all_states):
        if len(states_episode) == 0:
            print(f"Episode {eps_num}: No data, skipping")
            continue

        # Compute EE trajectory
        x, y, z = compute_ee_trajectory(states_episode, kin)

        # Plot and save
        save_path = os.path.join(plots_dir, f"ee_trajectory_eps_{eps_num}.png")
        plot_ee_trajectory_3d(x, y, z, eps_num, save_path)

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main(RESULTS_DIR)