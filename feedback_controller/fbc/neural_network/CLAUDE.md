# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The `neural_network` package of a `feedback_controller` ROS (catkin) workspace. It trains and deploys neural feedback controllers (DNFC — "Deep Neural Feedback Controller") that map a task goal + the robot's current state to joint torques/actions, for a Torobo 7-DOF arm (with legacy Baxter data). Training is plain PyTorch and offline; deployment is a ROS node that closes the loop on the real robot.

## Environment

- **Always use the `dnfc2` conda env**: `/home/arash/anaconda3/envs/dnfc2/bin/python`. The repo's `.venv` is a broken macOS env — do not use it. (torch 2.5.1, CUDA available.)
- ROS/`rospy`, `torobo_msgs`, `cv_bridge` are only importable inside the catkin workspace with ROS sourced. Anything importing `rospy`, `udp_comm`, `torcomm`, or `global_defines` (i.e. `online_tester.py`, `torkin.py`) only runs on the robot/ROS machine. Training and emulated testing do **not** need ROS.

## Common commands

Run everything from this directory (`fbc/neural_network`) with the dnfc2 python.

- **Train**: `python train_w_datasets.py` — config is edited *in the file*, not via CLI. Key knobs near line 466: `use_baseline`, `use_image`, `use_two_stream`, `num_epochs`, `batch_size`, `model_complexity` loop (line 493), `num_trains` (independent reruns). Writes checkpoints + loss CSVs + plots to `weights/<dataset>|<ds_ratio>|<model_name>/train_no_<i>/`. **Refuses to overwrite an existing weights dir** (`sys.exit(1)`), so bump/clear the dir to retrain.
- **Emulated (offline) test**: `python emulated_test.py` — loads a checkpoint and rolls out the controller against recorded trajectories without the robot. Edit `epoch_num`, `train_no`, `model_complexity` at the top. Writes to `results/...`.
- **Online (robot) test**: `python online_tester.py` — ROS node, requires the live robot + cameras. Edit `epoch_no`, `train_num`, camera topics near line 1085. **Refuses to overwrite an existing results dir.**
- **Find image crop region**: `python find_crop_coords.py` — interactive OpenCV tool to pick the `crop_params` (top/left/height/width) used by the image datasets/testers. These crop values are duplicated as literals in `train_w_datasets.py` (~line 478) and the testers — keep them in sync.

There is no test suite, linter, or build step; the data-prep / analysis work lives in the notebooks (`preprocess_data_v2.ipynb`, `data.ipynb`, `analyze.ipynb`, `get_results.ipynb`).

## Architecture

### Config is the single source of truth (`config.py`)

`Config` centralizes dataset name, ratios, dims, and — critically — **model naming and dimensioning**:
- `get_model_name(use_baseline, use_custom_loss, use_image, use_two_stream)` builds the canonical `model_name` string (e.g. `mse_los|tar_img_static|img_task_enc_rob_enc`). This string plus dataset name + param count forms the **weights/results directory path**, so the variant names (`v_name`, `v_name_base`, `v_name_two_stream`) directly determine where checkpoints are saved/loaded. Changing a variant name orphans old checkpoints.
- `get_model_dims` / `get_image_model_dims` / `get_two_stream_dims` map a `model_complexity` string (`low|medium|high|xhigh`) to layer widths. Pick the complexity at train time and pass the *same* one at test time or the architecture won't match the checkpoint.

### Models (`nn_models.py`)

Three controller architectures, all selected by the `use_baseline`/`use_two_stream`/`use_image` flags in the training script:

- **`GeneralModel`** (the main DNFC). Encodes a desired latent `x_des` from the task goal, encodes the robot's current state `x_encoded` from consecutive EE poses, and feeds the **difference** `x_des - x_encoded` to an MLP controller. This subtraction (a learned feedback error) is the core idea.
  - `use_image=True`: goal = a single front-camera image at t=0 → CNN (`SingleImageEncoder`) → concat one-hot touch history → `x_des`; current-state stream = two consecutive EE poses (pos+quat, 14 dims) → `MLP_3L` → `x_encoded`.
  - `use_image=False`: goal = object coordinates + one-hot → MLP → `x_des`.
- **`TwoStreamBaseline`**: same inputs as `GeneralModel` but **concatenates** the two streams instead of subtracting — the ablation that shows the subtraction matters.
- **`MLPBaseline`**: plain MLP over `[target_repr, state]`, tanh output.
- Custom losses (`CustomLoss`, `KLLoss`) add a latent-consistency term weighted by `config.C`; only used with `GeneralModel` (`use_custom_loss`). Baselines force MSE.

### Data (`train_w_datasets.py` datasets)

Trajectories are dense `.npy` arrays of shape `(num_trajs, num_steps, feature_dim)`, loaded from `data/torobo/<dataset_name>/`. Each step row is laid out as `[step, state(2*7), coords(6), onehot(4), ee_pose(7), ..., action(7)]`; the dataset classes slice this by fixed offsets, so **the column layout is a hard contract** between the preprocessing notebooks and `TrajectoryDataset.__getitem__`.
- `TrajectoryDataset` serves both `use_image` modes (image loaded from `triangle_images_fixed_cam/traj_<NNN>/step_<NNN>.jpg`); `TwoStreamDataset` stacks multiple history images from front+side cameras.
- A separate `traj_mapping_<ds_ratio>.npz` maps dataset row index → original trajectory folder number; image loading depends on it. `ds_ratio` (e.g. `interp_0.85`, `extrap_0.85`) selects the train/test split files and encodes the interpolation/extrapolation generalization regime.
- Training adds Gaussian noise (`noise_std`) to EE poses for robustness.
- Normalization stats live in `data/torobo/<dataset_name>/normalization_params.npz` and are applied at **deploy/test time** by the testers (xy mean/std, action_std, state mean/std, ee_pose pos mean/std), not baked into the dataset.

### Deployment (`testers.py`, `online_tester.py`, `udp_comm.py`, `torkin.py`)

- `Tester` (`testers.py`) is the shared inference harness: `load_model` reconstructs the exact architecture from `Config` + complexity and loads the `.pth`; `get_emulated`/`calculate_cartesian_perform` roll the controller forward; it owns normalization and forward kinematics (`TorKin` from `torkin.py`, generated symbolically in `kinematics/`).
- `online_tester.py` wraps `Tester` in a ROS node: subscribes to camera topics, buffers images, computes EE pose from live joint states via `TorKin`, runs the model, and sends torque commands. `udp_comm.py` reads Torobo joint state and pushes commands over UDP. `global_defines.py` holds all ROS topic names / joint-name constants for the Torobo robot.

## Gotchas

- Both `train_w_datasets.py` and `online_tester.py` intentionally **abort rather than overwrite** existing weights/results directories.
- `use_baseline` or `use_two_stream` silently force `use_custom_loss = False`.
- The `weights/`, `weights_old/`, `weights_baxter/`, `results/`, `results_old/` trees and `results.zip` (~1.5 GB) are large artifacts, not source — avoid reading them wholesale.
- `kinematics/` mixes MATLAB source (`*.m`) with auto-generated Python (`generated_pycode/`); the Python is what runs.
