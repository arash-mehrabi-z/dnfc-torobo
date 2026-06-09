# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for a paper on a **Deep Neural Feedback Controller (DNFC)** for robot manipulation. A neural network drives a robot toward task goals by repeatedly predicting the next action from the current state, in closed loop. The headline experiment compares DNFC against a plain MLP **baseline** on a **Torobo** humanoid's right arm (7-DOF) doing a 3-point pick-and-place task. A separate, self-contained experiment in `test_w_nn.py` runs the same idea on a 2-DOF MuJoCo `Reacher-v4` arm.

The repo lives under a `catkin_ws`, but there is **no ROS package here** (no `package.xml`/`CMakeLists.txt`) — it is a plain Python tree. Everything of interest is in `fbc/neural_network/`, and **all scripts must be run from inside that directory** (they use bare imports like `from nn_models import ...` and relative paths like `data/torobo/...`).

## Running things

There is no build system, no test runner, no linter, and no `requirements.txt` (the `readme.odt` references a nonexistent `requirements.py`). "Tests" are rollout/evaluation scripts. Two separate Python runtimes are involved:

- **Offline (training, emulation, analysis):** the bundled `fbc/neural_network/.venv` (Python 3.9; torch, torchvision, sklearn, `dtw`, gymnasium). This venv was created on macOS — several scripts contain hardcoded absolute paths from other machines (`/Users/denizakkoc/...`, `/home/deniz/...`, `/home/erhan/...`) that must be edited for the data/IK-table files to load.
- **Online (real/simulated robot):** the **system ROS** Python — `online_tester.py` and `udp_comm.py` do `import rospy` and `from torobo_msgs.msg import ...`, and need a running ROS master plus a Torobo control server reachable over UDP at `localhost:50000`.

```bash
cd fbc/neural_network            # required working dir for everything below

python3 train_w_datasets.py      # train Torobo DNFC/baseline (offline venv)
python3 emulated_test.py         # offline rollout via FK only, cartesian success + plots (no ROS)
python3 online_tester.py         # online rollout on Torobo, DTW vs ground truth (needs ROS + UDP server)
python3 test_w_nn.py             # standalone 2-DOF Reacher-v4 experiment (gymnasium/MuJoCo)
python3 torkin.py                # smoke-test the kinematics (standalone IK demo)
```

Selecting *which* model/dataset to train or test is done by **editing `config.py`** and the script-top flags (e.g. `use_baseline`, `use_custom_loss`, `epoch_no`), not via CLI args. `train_w_datasets.py` refuses to start if the target weights directory already exists (delete it to retrain). Data `.npy` files and `.pt` train/val splits are produced by the notebooks (`preprocess_data_v2.ipynb`, `data.ipynb`) from raw trajectories; analysis/figures live in `get_results.ipynb`, `trajectory_visualization.ipynb`, `inspection.ipynb`.

## Architecture

**The two models** (`nn_models.py`):
- `GeneralModel` (**DNFC**): an encoder maps the *target representation* to a latent **desired state** `x_des`; the controller MLP consumes `diff = x_des - state` and outputs a `tanh`-bounded action (joint-velocity delta). The structural prior is that control is driven by the *difference* between a desired and current state in a learned space.
- `MLPBaseline`: concatenates `[target, state]` and regresses the action directly — no latent / no difference structure. This is the comparison point.
- `CustomLoss`: `MSE(action) + C * MSE(x_des[:7], state[:7])`. The second term pulls the latent desired-state onto the actual joint-position space; toggling it (`use_custom_loss` / `C` in `config.py`) is the main ablation.

**The data layout** is the single most important convention — every script slices a flat per-step vector by the dims in `config.py`:

```
[ step(1) | state(14 = 7 joint pos + 7 joint vel) | target(9 = 3 points × xyz) | onehot(4) | action(7 joint deltas) ]
```

Datasets are `.npy` arrays of shape `(num_trajs, num_steps≈299, 35)`. The task is pick-and-place over points A/B/C; the `onehot` selects which of 4 sub-goals (grepB → putB → grepC → putC) is active. A rollout integrates `state[:7] += velocity; state[7:] = velocity` and advances the one-hot when the end-effector is `close_enough` (forward-kinematics distance ≤ 1.5 cm) to the next milestone (`testers.py`, `online_tester.py`).

**Config-driven paths (critical gotcha):** weights and results live in deeply nested directories whose names are *reconstructed from `config.py`* at both train and test time, e.g.
`weights/{dataset_name}|{ds_ratio}|{model_name}/train_no_{i}/fbc_{epoch}.pth`
where `model_name` encodes loss type, target type, model variant, and parameter count (e.g. `cus_los_1e-05|tar_cart|2+2l_lat:sub-nvel|6.037K_params`). `Tester.load_model` rebuilds these strings, so **`config.py` must match between a training run and the test run that loads its weights**, or nothing is found. Training sweeps model size over `['low','medium','high','xhigh']` (`Config.get_model_dims`) × `num_trains` seeds.

**Kinematics** (`torkin.py` + `kinematics/`): `TorKin` wraps auto-generated forward-kinematics and Jacobian functions in `kinematics/generated_pycode/` (`FK_*`, `Jpos/Jzaxis/Jyaxis_*`), which were themselves generated from the MATLAB symbolic scripts (`sym_*kin.m`, `gen_*code_py.m`). `forwardkin(bodypartix, q)` returns `(p, R)`; `q` is padded to 10 = torso(2) + arm(7) + gripper(1). Bodypart indices come from `global_defines.TOR` (`_TORSO=0, _RARM=1, _LARM=3, _HEAD=5`); the arm experiments use `_RARM=1`. Iterative IK (`_IKwZAXIS` batch, `_IIKwZAXIS` teleop) solves position + Z/Y-axis orientation and optionally seeds from KD-tree IK tables — those table files are hardcoded under `/home/erhan/...` and, if absent, IK silently falls back to a default seed. Note `torkin.py` does `import roslibpy as rospy` (the WebSocket client `roslibpy`, **not** ROS `rospy`) and only uses it for `rospy.sleep`.

**Communication** (`udp_comm.py`): `Comm` subscribes to the Torobo joint-state ROS topic and sends `setq`/`move_point` text commands to the control server over UDP. Used only by `online_tester.py`.

## Conventions worth matching

- Scripts are **run-as-`__main__` config blocks**, not reusable functions: hyperparameters, flags, and dataset selection are top-level module variables, and run state is shared through module-level globals (`global` declarations are pervasive, e.g. in `train_w_datasets.py` and `test_w_nn.py`). Match this style when extending an existing script rather than refactoring it into functions.
- `Config` is the one place dims/names are defined; read new fields from it instead of hardcoding the `35`-vector offsets.
- "DNFC" in code/plots = `GeneralModel`; "Basel"/baseline = `MLPBaseline`; "DTW" = the Cartesian end-effector path distance vs. ground truth (`online_tester.py`).
