import torch
import rospy
from nn_models import GeneralModel, MLPBaseline, TwoStreamBaseline
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROSImage
from config import Config
import socket
import time
import numpy as np
from torkin import TorKin
from global_defines import TOR
from threading import Lock
import torch.nn as nn
import matplotlib.pyplot as plt
from testers import Tester
from udp_comm import Comm
import os
import random
import csv
import pickle
from PIL import Image
from torchvision import transforms
from cv_bridge import CvBridge
# from tslearn.metrics import dtw_path
from dtw import *

plt.switch_backend('agg')

comm = Comm()
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
cur_file_dir_path = os.path.dirname(__file__)
config = Config()
bridge = CvBridge()

# Load normalization parameters
norm_params_path = os.path.join(
    cur_file_dir_path,
    f'data/torobo/{config.dataset_name}/normalization_params.npz'
)
norm_params = np.load(norm_params_path)
xy_mean = norm_params['xy_mean'].flatten()  # (6,) - x,y for 3 objects
xy_std = norm_params['xy_std'].flatten()    # (6,)
action_std = torch.tensor(norm_params['action_std'].flatten()).to(device).float()  # (7,)
state_mean = torch.tensor(norm_params['state_mean'].flatten()).to(device).float()  # (14,)
state_std = torch.tensor(norm_params['state_std'].flatten()).to(device).float()    # (14,)


class ImageBuffer:
    """Buffer to store recent images for TwoStreamBaseline input."""
    def __init__(self, num_history_images=3, image_size=(128, 128),
                 image_topic='/camera/color/image_raw'):
        self.num_history_images = num_history_images
        self.image_size = image_size
        self.images = []
        self.lock = Lock()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        # Subscribe to camera topic
        self.image_sub = rospy.Subscriber(
            image_topic, ROSImage, self.image_callback, queue_size=1)
        print(f"Subscribed to image topic: {image_topic}")

    def image_callback(self, msg):
        """Callback for incoming images."""
        try:
            # Convert ROS Image to PIL Image without cv_bridge
            # This avoids the cv_bridge Python version mismatch issue
            img_data = np.frombuffer(msg.data, dtype=np.uint8)

            if msg.encoding == 'rgb8':
                img_array = img_data.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'bgr8':
                img_array = img_data.reshape((msg.height, msg.width, 3))
                img_array = img_array[:, :, ::-1]  # BGR to RGB
            elif msg.encoding == 'rgba8':
                img_array = img_data.reshape((msg.height, msg.width, 4))[:, :, :3]
            elif msg.encoding == 'bgra8':
                img_array = img_data.reshape((msg.height, msg.width, 4))[:, :, :3]
                img_array = img_array[:, :, ::-1]  # BGR to RGB
            else:
                rospy.logwarn(f"Unsupported image encoding: {msg.encoding}")
                return

            pil_image = Image.fromarray(img_array)

            with self.lock:
                self.images.append(pil_image)
                # Keep only the most recent images
                if len(self.images) > self.num_history_images:
                    self.images.pop(0)
        except Exception as e:
            rospy.logwarn(f"Error processing image: {e}")

    def get_image_stack(self):
        """Get stacked images for model input."""
        with self.lock:
            if len(self.images) < self.num_history_images:
                # Pad with copies of the first image if not enough history
                if len(self.images) == 0:
                    return None
                images_to_use = [self.images[0]] * (self.num_history_images - len(self.images)) + self.images
            else:
                images_to_use = self.images[-self.num_history_images:]

        # Transform and stack images
        transformed = [self.transform(img) for img in images_to_use]
        image_stack = torch.cat(transformed, dim=0)  # (num_images * 3, H, W)
        return image_stack.unsqueeze(0).to(device).float()  # (1, num_images * 3, H, W)


# Global image buffers (initialized when needed)
image_buffer_front = None
image_buffer_side = None

# Crop parameters for image preprocessing (must match training)
CROP_PARAMS = {
    'top': 210,
    'left': 220,
    'height': 150,
    'width': 200,
}


class TargetImageCapture:
    """Capture a single image at t=0 as the target representation."""
    def __init__(self, image_size=(128, 128), crop_params=None,
                 image_topic='/camera/color/image_raw'):
        self.image_size = image_size
        self.crop_params = crop_params
        self.captured_image = None
        self.lock = Lock()

        # Build transform with optional cropping
        transform_list = []
        if crop_params is not None:
            transform_list.append(
                transforms.Lambda(lambda img: transforms.functional.crop(
                    img,
                    top=crop_params['top'],
                    left=crop_params['left'],
                    height=crop_params['height'],
                    width=crop_params['width']
                ))
            )
        transform_list.extend([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.transform = transforms.Compose(transform_list)

        # Subscribe to camera topic
        self.image_sub = rospy.Subscriber(
            image_topic, ROSImage, self.image_callback, queue_size=1)
        print(f"TargetImageCapture subscribed to: {image_topic}")

    def image_callback(self, msg):
        """Callback for incoming images - only captures first image."""
        if self.captured_image is not None:
            return  # Already captured

        try:
            img_data = np.frombuffer(msg.data, dtype=np.uint8)

            if msg.encoding == 'rgb8':
                img_array = img_data.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'bgr8':
                img_array = img_data.reshape((msg.height, msg.width, 3))
                img_array = img_array[:, :, ::-1]  # BGR to RGB
            elif msg.encoding == 'rgba8':
                img_array = img_data.reshape((msg.height, msg.width, 4))[:, :, :3]
            elif msg.encoding == 'bgra8':
                img_array = img_data.reshape((msg.height, msg.width, 4))[:, :, :3]
                img_array = img_array[:, :, ::-1]
            else:
                rospy.logwarn(f"Unsupported image encoding: {msg.encoding}")
                return

            pil_image = Image.fromarray(img_array.copy())

            with self.lock:
                self.captured_image = pil_image
                print("Target image captured at t=0")
        except Exception as e:
            rospy.logwarn(f"Error capturing target image: {e}")

    def get_target_image(self):
        """Get the captured target image as a tensor."""
        with self.lock:
            if self.captured_image is None:
                return None
            image_tensor = self.transform(self.captured_image)
            return image_tensor.unsqueeze(0).to(device).float()  # (1, 3, H, W)

    def reset(self):
        """Reset to capture a new target image."""
        with self.lock:
            self.captured_image = None

    def save_target_image(self, path):
        """Save the captured target image for debugging."""
        with self.lock:
            if self.captured_image is not None:
                self.captured_image.save(path)
                print(f"Target image saved to: {path}")


def online_test(tester: Tester, eps_num, use_baseline=False, model=None,
                use_image=False, target_image_capture=None,
                results_dir=None, save_every_n_steps=10):
    """Online test for robot control.

    Args:
        tester: Tester instance
        eps_num: Episode number
        use_baseline: If True, use MLPBaseline interface (only when model=None)
        model: Optional model to use. If None, uses tester.model or tester.baseline.
        use_image: If True, use image at t=0 as target representation
        target_image_capture: TargetImageCapture instance (required if use_image=True)
        results_dir: Directory to save images (optional, only used if use_image=True)
        save_every_n_steps: Save image every N steps (default: 10)
    """
    step_size = tester.step_size
    state_size = tester.state_size
    target_size = tester.target_size
    onehot_size = tester.onehot_size

    # Parameters that differ between image and coordinate modes
    if use_image:
        traj_step_size = 180
        velocity_scale = 5
        sleep_time = 0.2
        init_sleep = 7
    else:
        traj_step_size = 900
        velocity_scale = 1
        sleep_time = 0.05
        init_sleep = 5

    elem = tester.dataset[eps_num]
    # Initial state from dataset is NORMALIZED
    state_normalized = torch.tensor(
        elem[0][step_size:step_size + state_size].tolist()
    ).to(device).float()

    # Denormalize to get real state for robot control
    state_real = state_normalized * state_std + state_mean

    one_hot = elem[0][step_size + state_size + target_size:
                      step_size + state_size + target_size + onehot_size].tolist()
    # goal contains NORMALIZED coordinates from the preprocessed dataset
    goal_normalized = elem[0][step_size + state_size:
                              step_size + state_size + target_size].tolist()

    # Denormalize coordinates for physical object positioning
    goal_real = (np.array(goal_normalized) * xy_std) + xy_mean

    # 6D coordinates: x,y for 3 objects (z is fixed at 0.865)
    z_fixed = 0.865
    obstA = torch.tensor([goal_real[0], goal_real[1], z_fixed])
    obstB = torch.tensor([goal_real[2], goal_real[3], z_fixed])
    obstC = torch.tensor([goal_real[4], goal_real[5], z_fixed])
    grepB = [obstB[0], obstB[1], 0.87]
    putB = [obstA[0], obstA[1], 0.9]
    grepC = [obstC[0], obstC[1], 0.87]
    putC = [obstA[0], obstA[1], 0.93]

    print("Points")
    print(obstA)
    print(obstB)
    print(obstC)

    comm.move('point1', obstA)
    comm.move('point2', obstB)
    comm.move('point3', obstC)
    comm.create_and_pub_msg(state_real[:7])
    rospy.sleep(init_sleep)

    # Handle target image capture if using image mode
    target_image = None
    if use_image:
        if target_image_capture is None:
            raise ValueError("target_image_capture required when use_image=True")
        target_image_capture.reset()
        print("Waiting for target image capture...")
        while target_image_capture.get_target_image() is None:
            rospy.sleep(0.1)
        print("Target image captured, starting test...")

        # Save target image for debugging
        if results_dir is not None:
            images_dir = os.path.join(results_dir, f"images_eps_{eps_num}")
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            target_image_capture.save_target_image(
                os.path.join(images_dir, "target_image_t0_raw.jpg"))
            target_image_tensor = target_image_capture.get_target_image()
            if target_image_tensor is not None:
                img_for_viz = target_image_tensor.squeeze(0).cpu()
                img_for_viz = img_for_viz.permute(1, 2, 0).numpy()
                img_for_viz = (img_for_viz * 255).astype(np.uint8)
                processed_img = Image.fromarray(img_for_viz)
                processed_img.save(os.path.join(images_dir, "target_image_t0_processed.jpg"))
                print(f"Processed image saved (shape: {target_image_tensor.shape})")

    traj_point = 0
    all_joints = []
    latent_reps = []
    all_states = []

    # Determine which model to use
    if model is not None:
        active_model = model
    elif use_baseline:
        active_model = tester.baseline
    else:
        active_model = tester.model

    active_model.eval()

    for i in range(traj_step_size):
        comm.which(f'\n step {i}\n')

        state_nn = torch.unsqueeze(state_normalized, 0).float()
        touch_history = torch.tensor([one_hot]).to(device).float()

        with torch.no_grad():
            if use_baseline and model is None:
                goal_tensor = torch.tensor(goal_normalized).to(device).float()
                goal_nn = torch.unsqueeze(goal_tensor, 0)
                basel_input = torch.cat((goal_nn, touch_history, state_nn), dim=1)
                velocities_tensor = active_model(basel_input)
            elif use_image:
                target_image = target_image_capture.get_target_image()
                if target_image is None:
                    rospy.logwarn("Target image not available, skipping step")
                    continue
                velocities_tensor, x_des, _ = active_model(
                    target_image, state_nn, touch_history)
                latent_reps.append(torch.squeeze(x_des, 0).tolist())
            else:
                goal_tensor = torch.tensor(goal_normalized).to(device).float()
                goal_nn = torch.unsqueeze(goal_tensor, 0)
                velocities_tensor, x_des, _ = active_model(goal_nn, state_nn, touch_history)
                latent_reps.append(torch.squeeze(x_des, 0).tolist())

        velocities_tensor = torch.squeeze(velocities_tensor, 0)
        velocities_real = velocities_tensor * action_std
        state_real[:7] += (velocity_scale * velocities_real)

        comm.create_and_pub_msg(state_real[:7])
        rospy.sleep(sleep_time)
        comm.jsLock.acquire()
        js_real = torch.tensor((list(comm.joint_state))).to(device).float()
        comm.jsLock.release()

        state_real = torch.cat((js_real, velocities_real), dim=0).to(device)
        state_normalized = (state_real - state_mean) / state_std

        all_joints.append(state_real[:7])
        all_states.append(state_real.tolist())

        pos = tester.get_end_eff(js_real.tolist())
        if one_hot[0] == 1:
            pos[2] -= 0.02
            comm.move('point2', pos)
        elif one_hot[2] == 1:
            pos[2] -= 0.02
            comm.move('point3', pos)

        # Waypoint checking
        if traj_point == 0 and tester.close_enough(state_real, grepB):
            traj_point = 1
            one_hot = [1, 0, 0, 0]
            print(f'here1 at step {i}')
        elif traj_point == 1 and tester.close_enough(state_real, putB):
            traj_point = 2
            one_hot = [0, 1, 0, 0]
            print(f'here2 at step {i}')
        elif traj_point == 2 and tester.close_enough(state_real, grepC):
            traj_point = 3
            one_hot = [0, 0, 1, 0]
            print(f'here3 at step {i}')
        elif traj_point == 3 and tester.close_enough(state_real, putC):
            traj_point = 4
            one_hot = [0, 0, 0, 1]
            print(f'here4 at step {i}')

    return all_joints, traj_point, latent_reps, all_states


def init_image_buffers(image_topic_front='/camera/color/image_raw',
                       image_topic_side='/camera_side/color/image_raw'):
    """Initialize the global image buffers for both cameras."""
    global image_buffer_front, image_buffer_side
    image_buffer_front = ImageBuffer(
        num_history_images=config.num_history_images,
        image_size=config.image_size,
        image_topic=image_topic_front
    )
    image_buffer_side = ImageBuffer(
        num_history_images=config.num_history_images,
        image_size=config.image_size,
        image_topic=image_topic_side
    )
    return image_buffer_front, image_buffer_side


# Global target image capture (initialized when needed)
target_image_capture = None


def init_target_image_capture(image_topic='/camera/color/image_raw'):
    """Initialize target image capture for GeneralModel with use_image=True."""
    global target_image_capture
    target_image_capture = TargetImageCapture(
        image_size=config.image_size,
        crop_params=CROP_PARAMS,
        image_topic=image_topic
    )
    return target_image_capture


def setup_camera_view(torso_tilt=0.26, gaze_target=None):
    """
    Setup camera view by tilting torso and pointing head at workspace.
    Matches the setup used in data_collector_triangle.

    Args:
        torso_tilt: Torso tilt angle in radians (~0.26 rad = 15 degrees)
        gaze_target: [x, y, z] position to look at (default: center of workspace)
    """
    if gaze_target is None:
        # Default matches data_collector_triangle: center_x=0.475, center_y=-0.105
        gaze_target = np.array([0.475, -0.105, 0.865])

    # Tilt torso forward to help camera see the table
    # Torso has 2 joints: [joint0, tilt_joint]
    qtorso = [0.0, torso_tilt]
    comm.set_torso(qtorso)
    # Tell IK solver to use tilted torso (matches data_collector_triangle)
    kin.set_torso_for_ik(qtorso)
    print(f"Tilting torso forward by {torso_tilt * 180.0 / np.pi:.1f} degrees...")
    rospy.sleep(2.0)  # Wait for torso to reach position

    # Point head at gaze target using forward kinematics
    lookat(gaze_target)
    rospy.sleep(1.0)  # Wait for head to reach target position


def lookat(target_pos):
    """
    Point the head camera at a target position using forward kinematics.
    Replicates the lookat function from torcomm.py.

    Args:
        target_pos: [x, y, z] numpy array of target position in world frame
    """
    pfix = np.array(target_pos)

    # Get current head and torso joint positions
    q_head = comm.get_head_state()
    q_torso = comm.get_torso_state()

    # Wait for joint states to be available
    wait_count = 0
    while q_head is None or q_torso is None:
        rospy.sleep(0.1)
        q_head = comm.get_head_state()
        q_torso = comm.get_torso_state()
        wait_count += 1
        if wait_count > 50:  # 5 second timeout
            print("Warning: Could not get head/torso joint states, using defaults")
            q_head = [0.0, 0.0]
            q_torso = [0.0, 0.26]  # Default torso tilt
            break

    # Combine torso and head joints for forward kinematics
    # Head FK expects: [torso_j1, torso_j2, head_j1, head_j2]
    q4 = np.hstack([q_torso, q_head])

    # Compute forward kinematics for head to get eye position and rotation
    peye, Reye = kin.forwardkin(TOR._HEAD, q4)

    # Transform target to local eye coordinates
    pfix_loc = np.matmul(Reye.transpose(), pfix - peye)

    # Calculate delta pan and tilt
    dpan = np.arctan2(pfix_loc[0], pfix_loc[2])
    dtilt = np.arctan2(pfix_loc[1], pfix_loc[2])

    # Update head position (subtract dpan, add dtilt - matches torcomm.py)
    new_head = [q_head[0] - dpan, q_head[1] + dtilt]

    # Send head command
    comm.set_head(new_head)
    print(f"Looking at target: pan={new_head[0] * 180.0 / np.pi:.1f}°, tilt={new_head[1] * 180.0 / np.pi:.1f}°")


def get_dtw_metric(coords_dnfc, coords_basel, coords_gtruth):
    x_dnfc, y_dnfc, z_dnfc = coords_dnfc
    x_basel, y_basel, z_basel = coords_basel
    x_gtruth, y_gtruth, z_gtruth = coords_gtruth

    dnfc_cart_coords = np.array(list(zip(x_dnfc, y_dnfc, z_dnfc)))
    basel_cart_coords = np.array(list(zip(x_basel, y_basel, z_basel)))
    gtruth_cart_coords = np.array(list(zip(x_gtruth, y_gtruth, z_gtruth)))
    # optimal_path, dtw_score_dnfc = dtw_path(dnfc_cart_coords, gtruth_cart_coords)
    # optimal_path, dtw_score_basel = dtw_path(basel_cart_coords, gtruth_cart_coords)
    alignment_dnfc = dtw(dnfc_cart_coords, gtruth_cart_coords)
    dtw_dnfc = alignment_dnfc.distance
    dtw_norm_dnfc = alignment_dnfc.normalizedDistance

    alignment_basel = dtw(basel_cart_coords, gtruth_cart_coords)
    dtw_basel = alignment_basel.distance
    dtw_norm_basel = alignment_basel.normalizedDistance

    return dtw_dnfc, dtw_basel, dtw_norm_dnfc, dtw_norm_basel


def intrinsic_to_3d_cart(all_joints_vals):
    x, y, z = [], [], []
    for joints_vals in all_joints_vals:
        my_l = [0, 0]
        for j_val in joints_vals:
            my_l.append(float(j_val))
        p, R = kin.forwardkin(1, np.array(my_l))
        x.append(p[0])
        y.append(p[1])
        z.append(p[2])

    return x, y, z


# Plot results
def plot_results(tester:Tester, coords_dnfc, coords_basel, coords_gtruth,
                 eps_num, i_train, results_dir):
    x_dnfc, y_dnfc, z_dnfc = coords_dnfc
    x_basel, y_basel, z_basel = coords_basel
    x_gtruth, y_gtruth, z_gtruth = coords_gtruth

    elem = tester.dataset[eps_num]
    joint_size = tester.joint_size
    state_size = tester.state_size
    step_size = tester.step_size
    target_size = tester.target_size
    onehot_size = tester.onehot_size

    # Goal contains NORMALIZED 2D coordinates (x,y for 3 objects)
    goal_normalized = elem[0][step_size+state_size :
                   step_size+state_size+target_size
                   ].tolist()

    # Denormalize coordinates
    goal_real = (np.array(goal_normalized) * xy_std) + xy_mean

    # 6D coordinates: x,y for 3 objects (z is fixed at 0.865)
    z_fixed = 0.865
    obstA = torch.tensor([goal_real[0], goal_real[1], z_fixed])
    obstB = torch.tensor([goal_real[2], goal_real[3], z_fixed])
    obstC = torch.tensor([goal_real[4], goal_real[5], z_fixed])

    fig = plt.figure(figsize=(12.8, 9.6))
    ax = fig.add_subplot(111, projection='3d')

    ln_wd = 2
    ax.scatter(x_basel, y_basel, z_basel, c='g', s=1, label='Basel', linewidths=ln_wd)
    ax.scatter(x_dnfc, y_dnfc, z_dnfc, c='b', s=1, label='DNFC', linewidths=ln_wd)
    ax.scatter(x_gtruth, y_gtruth, z_gtruth, c='r', s=1, label='G.Truth', linewidths=ln_wd)

    ax.scatter([obstA[0]], [obstA[1]], [obstA[2]], c='k', marker='o')
    ax.scatter([obstB[0]], [obstB[1]], [obstB[2]], c='k', marker='o')
    ax.scatter([obstC[0]], [obstC[1]], [obstC[2]], c='k', marker='o')

    ax.text(obstA[0], obstA[1], obstA[2], 'p.A', color='black', fontsize=10, ha='center')
    ax.text(obstB[0], obstB[1], obstB[2], 'p.B', color='black', fontsize=10, ha='center')
    ax.text(obstC[0], obstC[1], obstC[2], 'p.C', color='black', fontsize=10, ha='center')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    # ax.set_title(f'DTW DNFC: {dtw_score_dnfc}, DTW Base: {dtw_score_basel}')
    ax.legend()
    
    plot_path = os.path.join(results_dir, f'plt_{eps_num}_{i_train}.png')
    plt.savefig(plot_path)
    # plt.show()
    plt.close()


def plot_latent_reps(latent_reps, states, results_dir, eps_num, i_train):
    # Transpose the data to separate each joint
    lat_rep_transp = list(zip(*latent_reps))
    states_transp = list(zip(*states))

    # Plot each joint's position over time
    time_steps = range(len(latent_reps))  # Assuming each inner list corresponds to a timestep

    plt.figure(figsize=(10, 6))
    
    colors = [
        'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 
        'black', 'orange', 'purple', 'brown', 'pink', 'lime', 
        'teal', 'gold'
    ]

    for idx, lat_rep in enumerate(lat_rep_transp[:7]):
        if idx < 7: label_st = f'Joint {idx + 1}'
        else: label_st = f'Vel. {(idx%7) + 1}'

        plt.plot(time_steps, lat_rep, 
                 label=f'Dim {idx + 1}', color=colors[idx])
        plt.plot(time_steps, states_transp[idx], 
                 label=label_st, color=colors[idx])

    # Add labels, legend, and title
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    # plt.title('Joint Positions Over Time')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(results_dir, f'latent_reps_{eps_num}_{i_train}.png')
    plt.savefig(plot_path)
    # plt.show()
    plt.close()


def log_loss(eps_num, dnfc_succ, basel_succ, dtw_dnfc, dtw_basel, 
             dtw_norm_dnfc, dtw_norm_basel, results_dir, file_name):
    
    print(f"DNFC perf. {dnfc_succ} & Basel perf. {basel_succ}")
    print(f"DTW DNFC. {dtw_dnfc} & DTW Basel. {dtw_basel}")

    perf_file_path = os.path.join(results_dir, file_name)
    with open(perf_file_path, 'a') as f:
        writer = csv.writer(f)
        row = [eps_num, dnfc_succ, basel_succ, dtw_dnfc, dtw_basel, 
               dtw_norm_dnfc, dtw_norm_basel]
        writer.writerow(row)


def store_states(states_dnfc, states_base, latent_reps):
    global all_states_dnfc
    global all_states_base
    global all_latent_reps

    all_states_dnfc.append(states_dnfc)
    all_states_base.append(states_base)
    all_latent_reps.append(latent_reps)


def save_list_to_file(lst, results_dir, file_name):
    file_path = os.path.join(results_dir, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(lst, f)


def create_results_dir(params_num):
    global epoch_no

    results_file = f'results/{config.dataset_name}_{params_num}K_{config.ds_ratio}'
    results_file += f'/ep:{epoch_no}/on_{config.v_name}_{config.C}_{config.use_custom_loss}_{config.v_name_base}'
    results_dir = os.path.join(cur_file_dir_path, results_file)
    if os.path.exists(results_dir):
        raise Exception(f"Result dir. exists. Are you testing again? Result dir: {results_dir}")
    else:
        os.makedirs(results_dir)
    print("results_dir", results_dir)

    perf_file_path = os.path.join(results_dir, "perf.csv")
    with open(perf_file_path, 'w') as f:
        writer = csv.writer(f)
        row = ["eps_num", "dnfc_succ", "basel_succ", "dnfc_dtw", "basel_dtw", 
            "dnfc_norm", "basel_norm"]
        writer.writerow(row)

    return results_dir


tester = Tester()
kin = TorKin()

# Configuration flags
use_two_stream = False  # Set to True to use TwoStreamBaseline
use_image = True        # Set to True to use image at t=0 as target representation
use_only_dnfc = True    # Only used when use_image=False and use_two_stream=False
epoch_no = 4400
train_num = 1

# Camera topic for image capture
image_topic_front = '/fixed_camera/image_raw'


def run_model_test(use_image, use_two_stream):
    """Unified testing loop for GeneralModel and TwoStreamBaseline."""
    # Determine model type name for logging
    model_type = "two_stream" if use_two_stream else "image_model" if use_image else "dnfc"

    # Initialize ROS node
    rospy.init_node(f'{model_type}_tester')

    # Initialize target image capture if using images
    if use_image:
        init_target_image_capture(image_topic_front)
        rospy.sleep(2)

    for model_complexity in ['high']:
        # Load model
        tester.load_model(0, epoch_no, config.use_custom_loss, model_complexity,
                          use_image=use_image, use_two_stream=use_two_stream)

        # Get model reference and params
        model = tester.two_stream_model if use_two_stream else tester.model
        params_num = tester.config.get_params_num(model)

        # Create results directory
        if use_two_stream:
            results_file = f'results/{config.dataset_name}_{params_num}K_{config.ds_ratio}'
            results_file += f'/ep:{epoch_no}/two_stream_{config.v_name_two_stream}'
        else:
            results_file = f'results/{config.dataset_name}_{params_num}K_{config.ds_ratio}'
            results_file += f'/ep:{epoch_no}/image_model_{config.v_name}'

        results_dir = os.path.join(cur_file_dir_path, results_file)
        if os.path.exists(results_dir):
            raise Exception(f"Result dir exists: {results_dir}")
        os.makedirs(results_dir)
        print("results_dir", results_dir)

        # Create performance CSV
        perf_file_path = os.path.join(results_dir, "perf.csv")
        with open(perf_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["eps_num", f"{model_type}_succ", f"{model_type}_dtw",
                           f"{model_type}_norm"])

        all_states = []
        all_latent_reps = []

        for eps_num in range(len(tester.dataset)):
            for i_train in range(train_num):
                # Reload model for this training run
                tester.load_model(i_train, epoch_no, config.use_custom_loss,
                                  model_complexity, use_image=use_image,
                                  use_two_stream=use_two_stream)
                model = tester.two_stream_model if use_two_stream else tester.model

                print(f'Testing {model_type} on episode {eps_num}, train {i_train}')
                comm.which(f'\n\n\n\n{model_type} start on path {eps_num}\n\n\n\n')

                # Run test
                all_joints, loss, latent_reps, states = online_test(
                    tester, eps_num, use_baseline=False, model=model,
                    use_image=use_image, target_image_capture=target_image_capture,
                    results_dir=results_dir, save_every_n_steps=10)

                # Calculate metrics
                coords = intrinsic_to_3d_cart(all_joints)
                coords_gtruth = tester.get_real_coordinates(eps_num)

                alignment = dtw(np.array(list(zip(*coords))),
                               np.array(list(zip(*coords_gtruth))))
                dtw_dist = alignment.distance
                dtw_norm = alignment.normalizedDistance

                succ = loss / 4
                print(f"{model_type} perf. {succ}, DTW: {dtw_dist}")

                with open(perf_file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([eps_num, succ, dtw_dist, dtw_norm])

                all_states.append(states)
                all_latent_reps.append(latent_reps)

        save_list_to_file(all_states, results_dir, f"all_states_{model_type}")
        save_list_to_file(all_latent_reps, results_dir, "all_latent_reps")


def run_dnfc_baseline_test():
    """Original testing loop for DNFC vs Baseline comparison."""
    rospy.init_node('dnfc_tester')

    for model_complexity in ['high']:
        tester.load_model(0, 0, config.use_custom_loss, model_complexity, use_image=False)
        params_num = tester.config.get_params_num(tester.model)
        results_dir = create_results_dir(params_num)

        all_states_dnfc = []
        all_states_base = []
        all_latent_reps = []

        for eps_num in range(len(tester.dataset)):
            for i_train in range(train_num):
                tester.load_model(i_train, epoch_no, config.use_custom_loss,
                                  model_complexity, use_image=False)
                print('waiting for DNFC')
                comm.which(f'\n\n\n\ndnfc start on path{eps_num}\n\n\n\n')
                all_joints_dnfc, loss_dnfc, latent_reps, states_dnfc = online_test(
                    tester, eps_num, use_baseline=False)

                if use_only_dnfc:
                    all_joints_base = all_joints_dnfc
                    loss_basel = loss_dnfc
                    states_base = states_dnfc
                else:
                    print('waiting for baseline')
                    comm.which(f'\n\n\n\nbaseline start on path{eps_num}\n\n\n\n')
                    all_joints_base, loss_basel, _, states_base = online_test(
                        tester, eps_num, use_baseline=True)

                coords_dnfc = intrinsic_to_3d_cart(all_joints_dnfc)
                coords_basel = intrinsic_to_3d_cart(all_joints_base)
                coords_gtruth = tester.get_real_coordinates(eps_num)

                dtw_dnfc, dtw_basel, dtw_norm_dnfc, dtw_norm_basel = get_dtw_metric(
                    coords_dnfc, coords_basel, coords_gtruth)

                plot_results(tester, coords_dnfc, coords_basel, coords_gtruth,
                            eps_num, i_train, results_dir)

                dnfc_succ = loss_dnfc / 4
                basel_succ = loss_basel / 4

                log_loss(eps_num, dnfc_succ, basel_succ, dtw_dnfc, dtw_basel,
                        dtw_norm_dnfc, dtw_norm_basel, results_dir, "perf.csv")
                plot_latent_reps(latent_reps, states_dnfc, results_dir,
                                eps_num, i_train)
                store_states(states_dnfc, states_base, latent_reps)

        save_list_to_file(all_states_dnfc, results_dir, "all_states_dnfc")
        save_list_to_file(all_states_base, results_dir, "all_states_base")
        save_list_to_file(all_latent_reps, results_dir, "all_latent_reps")


# Main entry point
if use_image or use_two_stream:
    run_model_test(use_image=use_image, use_two_stream=use_two_stream)
else:
    run_dnfc_baseline_test()

