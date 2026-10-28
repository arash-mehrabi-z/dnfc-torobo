"""
Plot end-effector 3D trajectories from saved all_states file.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torkin import TorKin

# Results directory (latent_dim = 14: 909.691K params, suffix-less v_name,
# ep:5000, DT=0.1 -- the run used in the score/DTW analysis).
RESULTS_DIR = os.path.expanduser(
    "~/catkin_ws/src/feedback_controller/fbc/neural_network/results/"
    "trajs:360_blocks:3_imgs_2cams_909.691K_interp_0.85/ep:5000/"
    "image_model_img_task_enc_rob_enc_raw_actionDT=0.1"
)

# Test data path for cube positions
TEST_DATA_PATH = os.path.expanduser(
    "~/catkin_ws/src/feedback_controller/fbc/neural_network/data/torobo/"
    "trajs:360_blocks:3_imgs_2cams/test_interp_0.85.npy"
)
NORM_PARAMS_PATH = os.path.expanduser(
    "~/catkin_ws/src/feedback_controller/fbc/neural_network/data/torobo/"
    "trajs:360_blocks:3_imgs_2cams/normalization_params.npz"
)

# Data structure indices
STEP_SIZE = 1
STATE_SIZE = 14
TARGET_SIZE = 6  # x,y for 3 cubes
CUBE_Z = 0.865  # Fixed z position for cubes on table
CUBE_SIZE = 0.01  # Cube footprint (x, y) in meters
CUBE_HEIGHT = 0.05  # Cube height (z) in meters -- taller than the footprint


def draw_3d_cube(ax, center, size, color, alpha=0.8, label=None, height=None):
    """Draw a 3D box on the given axes.

    Args:
        ax: Matplotlib 3D axes
        center: (x, y, z) center position of the box
        size: Footprint size in x and y
        color: Face color of the box
        alpha: Transparency (0-1)
        label: Label for legend
        height: Size in z. Defaults to ``size`` (a true cube) when None.
    """
    cx, cy, cz = center
    s = size / 2  # Half footprint (x, y)
    h = (size if height is None else height) / 2  # Half height (z)

    # Define the 8 vertices of the box
    vertices = [
        [cx - s, cy - s, cz - h],
        [cx + s, cy - s, cz - h],
        [cx + s, cy + s, cz - h],
        [cx - s, cy + s, cz - h],
        [cx - s, cy - s, cz + h],
        [cx + s, cy - s, cz + h],
        [cx + s, cy + s, cz + h],
        [cx - s, cy + s, cz + h],
    ]

    # Define the 6 faces using vertex indices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
    ]

    # Create the 3D polygon collection
    cube = Poly3DCollection(faces, alpha=alpha, linewidths=1, edgecolors='black')
    cube.set_facecolor(color)
    ax.add_collection3d(cube)

    # Add a dummy scatter point for the legend
    if label:
        ax.scatter([], [], [], c=color, s=100, marker='s', label=label)


def load_states(results_dir, filename="all_states_image_model"):
    """Load states from pickle file."""
    file_path = os.path.join(results_dir, filename)
    with open(file_path, 'rb') as f:
        all_states = pickle.load(f)
    print(f"Loaded {len(all_states)} episodes from {file_path}")
    return all_states


def load_cube_positions(test_data_path, norm_params_path):
    """Load and denormalize cube positions from test data.

    Returns:
        List of cube positions for each episode, where each element is:
        [(x_A, y_A, z_A), (x_B, y_B, z_B), (x_C, y_C, z_C)]
    """
    # Load test data and normalization params
    test_data = np.load(test_data_path, allow_pickle=True)
    norm_params = np.load(norm_params_path)
    xy_mean = norm_params['xy_mean'].flatten()
    xy_std = norm_params['xy_std'].flatten()

    all_cube_positions = []
    for episode in test_data:
        # Extract normalized goal from first timestep
        # Goal is at indices [STEP_SIZE+STATE_SIZE : STEP_SIZE+STATE_SIZE+TARGET_SIZE]
        goal_normalized = episode[0][STEP_SIZE + STATE_SIZE : STEP_SIZE + STATE_SIZE + TARGET_SIZE]

        # Denormalize: real = normalized * std + mean
        goal_real = goal_normalized * xy_std + xy_mean

        # Extract cube positions (x, y) and add fixed z
        cube_A = (goal_real[0], goal_real[1], CUBE_Z)
        cube_B = (goal_real[2], goal_real[3], CUBE_Z)
        cube_C = (goal_real[4], goal_real[5], CUBE_Z)

        all_cube_positions.append([cube_A, cube_B, cube_C])

    print(f"Loaded cube positions for {len(all_cube_positions)} episodes")
    return all_cube_positions


def load_gt_states(test_data_path, norm_params_path):
    """Load ground-truth joint-state sequences (one per test episode).

    Each test-data row is laid out as [step, state(14), coords(6), onehot(4),
    ee_pose(7), ...]; the state at indices [STEP_SIZE : STEP_SIZE+STATE_SIZE]
    holds [j1..j7, v1..v7]. The stored state is *normalized* (state_mean/
    state_std), so we denormalize it back to real joint angles -- otherwise FK
    on normalized values produces a wildly wrong trajectory. The rollout states
    in all_states_image_model are already in real units, so after this the two
    are directly comparable. Returns a list of (num_steps, 14) arrays.
    """
    test_data = np.load(test_data_path, allow_pickle=True)
    norm_params = np.load(norm_params_path)
    state_mean = norm_params['state_mean']  # (14,)
    state_std = norm_params['state_std']    # (14,)

    gt_states = []
    for episode in test_data:
        state_norm = np.asarray(episode)[:, STEP_SIZE:STEP_SIZE + STATE_SIZE]
        state_real = state_norm * state_std + state_mean  # denormalize
        gt_states.append(state_real)
    print(f"Loaded ground-truth states for {len(gt_states)} episodes")
    return gt_states


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


def plot_ee_trajectory_3d(x, y, z, eps_num, save_path, cube_positions=None,
                          gt_xyz=None):
    """Plot and save 3D end-effector trajectory with cube positions.

    Args:
        x, y, z: End-effector trajectory coordinates
        eps_num: Episode number
        save_path: Path to save the plot
        cube_positions: List of 3 cube positions [(x,y,z), (x,y,z), (x,y,z)]
        gt_xyz: Optional (gx, gy, gz) ground-truth EE trajectory to overlay
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(x, y, z, 'b-', linewidth=1.5, label=r'LFEL$_0$ Trajectory')
    ax.scatter(x, y, z, c=np.arange(len(x)), cmap='viridis', s=5)

    # Overlay the ground-truth trajectory (dashed) for comparison
    if gt_xyz is not None:
        gx, gy, gz = gt_xyz
        ax.plot(gx, gy, gz, 'k--', linewidth=1.5, label='Ground Truth')

    # Mark start and end points
    ax.scatter([x[0]], [y[0]], [z[0]], c='g', s=100, marker='o', label='Start')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c='r', s=100, marker='x', label='End')

    # Plot 3D cubes at their positions
    if cube_positions is not None:
        cube_colors = ['red', 'green', 'blue']  # A=red, B=green, C=blue
        cube_labels = ['Cube A', 'Cube B', 'Cube C']
        for i, (cx, cy, cz) in enumerate(cube_positions):
            draw_3d_cube(ax, (cx, cy, cz), CUBE_SIZE, cube_colors[i],
                         alpha=0.8, label=cube_labels[i], height=CUBE_HEIGHT)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    # ax.set_title(f'Episode {eps_num}: End-Effector Trajectory')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main(results_dir, filename="all_states_image_model",
         test_data_path=None, norm_params_path=None):
    # Initialize forward kinematics
    kin = TorKin()

    # Load states
    all_states = load_states(results_dir, filename)

    # Load cube positions if paths provided
    all_cube_positions = None
    gt_states = None
    if test_data_path and norm_params_path:
        all_cube_positions = load_cube_positions(test_data_path, norm_params_path)
    if test_data_path and norm_params_path:
        # Ground-truth EE trajectories, computed once per test episode below.
        gt_states = load_gt_states(test_data_path, norm_params_path)
    gt_traj_cache = {}

    # all_states holds one rollout per perf.csv row: each test episode is repeated
    # once per trained model (e.g. 18 test episodes x 3 trains = 54 rollouts,
    # ordered 0,0,0,1,1,1,...). The cube positions only have one entry per test
    # episode, so map rollout index -> test episode by the number of trains.
    num_trains = 1
    if all_cube_positions:
        num_trains = max(1, len(all_states) // len(all_cube_positions))

    # Create output directory for plots
    plots_dir = os.path.join(results_dir, "ee_trajectory_plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Process each rollout
    for rollout_num, states_episode in enumerate(all_states):
        if len(states_episode) == 0:
            print(f"Rollout {rollout_num}: No data, skipping")
            continue

        # The test episode (and thus the cube layout) this rollout belongs to.
        eps_num = rollout_num // num_trains
        train_num = rollout_num % num_trains

        # Compute EE trajectory
        x, y, z = compute_ee_trajectory(states_episode, kin)

        # Get cube positions for this episode (if available)
        cube_positions = None
        if all_cube_positions and eps_num < len(all_cube_positions):
            cube_positions = all_cube_positions[eps_num]

        # Ground-truth EE trajectory for this episode (shared across trains;
        # compute once per episode and cache).
        gt_xyz = None
        if gt_states is not None and eps_num < len(gt_states):
            if eps_num not in gt_traj_cache:
                gt_traj_cache[eps_num] = compute_ee_trajectory(
                    gt_states[eps_num], kin)
            gt_xyz = gt_traj_cache[eps_num]

        # Plot and save
        save_path = os.path.join(
            plots_dir, f"ee_trajectory_eps_{eps_num}_train_{train_num}.png")
        plot_ee_trajectory_3d(x, y, z, eps_num, save_path, cube_positions,
                              gt_xyz=gt_xyz)
        # break

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main(RESULTS_DIR, test_data_path=TEST_DATA_PATH, norm_params_path=NORM_PARAMS_PATH)