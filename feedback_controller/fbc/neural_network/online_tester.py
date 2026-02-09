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
device = 'cpu'  # Force CPU to avoid CUDA library conflicts; change back to 'cuda' once resolved
cur_file_dir_path = os.path.dirname(__file__)
config = Config()
bridge = CvBridge()


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


# Global image buffer (initialized when needed)
image_buffer = None


def online_test(tester:Tester, eps_num, use_baseline):
    joint_size = tester.joint_size
    state_size = tester.state_size
    step_size = tester.step_size
    target_size = tester.target_size
    onehot_size = tester.onehot_size
    traj_step_size = 900

    elem = tester.dataset[eps_num]
    state = torch.tensor(elem[0][step_size : 
                                 step_size + state_size].tolist()
                                 ).to(device)
    
    # milestones = tester.get_changes_indexes(eps_num)
    one_hot = elem[0][step_size+state_size+target_size : 
                      step_size+state_size+target_size+onehot_size].tolist()
    goal = elem[0][step_size+state_size : 
                   step_size+state_size+target_size].tolist()
    
    obstA = torch.tensor(goal[0:3])
    obstB = torch.tensor(goal[3:6])
    obstC = torch.tensor(goal[6:9])
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
    comm.create_and_pub_msg(state[:7])
    rospy.sleep(5)

    # all_trajs_point = 0
    traj_point = 0

    all_joints = []
    latent_reps = []
    all_states = []
    for i in range(traj_step_size):
        comm.which('\n dnfc in on step'+str(i)+'\n')

        goal_tensor = torch.tensor(goal+one_hot).to(device)
        goal_nn = torch.unsqueeze(goal_tensor, 0)
        state_nn = torch.unsqueeze(state, 0)
        # delta = torch.tensor(elem[i][step_size + state_size + target_size + onehot_size: step_size + state_size + target_size + onehot_size + joint_size ].tolist())
        if use_baseline:
            basel_input = torch.cat((goal_nn, state_nn), dim=1)
            tester.baseline.eval()
            velocities_tensor = tester.baseline(basel_input)
        else:
            tester.model.eval()
            velocities_tensor, x_des, _ = tester.model(goal_nn, state_nn)
            x_des = torch.squeeze(x_des, 0)
            latent_reps.append(x_des.tolist())

        velocities_tensor = torch.squeeze(velocities_tensor, 0)
        state[:7] += (5*velocities_tensor)

        comm.create_and_pub_msg(state[:7])
        rospy.sleep(0.05)
        comm.jsLock.acquire()
        js = torch.tensor((list(comm.joint_state))).to(device)
        comm.jsLock.release()

        # TODO: Calculate velocity better. (!!!)
        state = torch.cat((js, velocities_tensor), dim=0).to(device)
        # state = torch.cat((state[:7],velocities_tensor),dim=0)
        all_joints.append(state[:7])
        all_states.append(state.tolist())

        pos = tester.get_end_eff(js.tolist())
        if one_hot[0]==1:
            pos[2] -= 0.02
            comm.move('point2', pos)
        elif one_hot[2]==1:
            pos[2]-= 0.02
            comm.move('point3', pos)

        # state += velocities_tensor
        if traj_point==0 and tester.close_enough(state, grepB):
            traj_point = 1
            one_hot = [1, 0, 0, 0]
            print('here1')
            print(i)
        elif traj_point==1 and tester.close_enough(state, putB):
            traj_point = 2
            one_hot = [0, 1, 0, 0]
            print('here2')
            print(i)
        elif traj_point==2 and tester.close_enough(state, grepC):
            traj_point = 3
            one_hot = [0, 0, 1, 0]   
            print('here3')
            print(i)  
        elif traj_point==3 and tester.close_enough(state, putC):
            traj_point = 4
            one_hot = [0, 0, 0, 1] 
            print('here4')
            print(i)

    # all_trajs_point += traj_point
    return all_joints, traj_point, latent_reps, all_states #all_trajs_point


def online_test_two_stream(tester: Tester, eps_num, results_dir=None, save_every_n_steps=10):
    """Online test using TwoStreamBaseline with image input.

    Args:
        tester: Tester instance
        eps_num: Episode number
        results_dir: Directory to save images (optional)
        save_every_n_steps: Save image every N steps (default: 10)
    """
    global image_buffer

    joint_size = tester.joint_size
    state_size = tester.state_size
    step_size = tester.step_size
    target_size = tester.target_size
    onehot_size = tester.onehot_size
    traj_step_size = 900

    elem = tester.dataset[eps_num]
    state = torch.tensor(elem[0][step_size:
                                 step_size + state_size].tolist()
                         ).to(device)

    one_hot = elem[0][step_size + state_size + target_size:
                      step_size + state_size + target_size + onehot_size].tolist()
    goal = elem[0][step_size + state_size:
                   step_size + state_size + target_size].tolist()

    obstA = torch.tensor(goal[0:3])
    obstB = torch.tensor(goal[3:6])
    obstC = torch.tensor(goal[6:9])
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
    comm.create_and_pub_msg(state[:7])
    rospy.sleep(7)

    traj_point = 0
    all_joints = []
    all_states = []

    # Create images directory if saving images
    images_dir = None
    if results_dir is not None:
        images_dir = os.path.join(results_dir, f"images_eps_{eps_num}")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        print(f"Saving images to: {images_dir}")

    # Ensure image buffer is initialized
    if image_buffer is None:
        raise RuntimeError("Image buffer not initialized. Call init_image_buffer() first.")

    # Wait for images to be available
    print("Waiting for images...")
    while image_buffer.get_image_stack() is None:
        rospy.sleep(0.1)
    print("Images available, starting test...")

    tester.two_stream_model.eval()

    for i in range(traj_step_size):
        comm.which('\n two_stream in on step' + str(i) + '\n')

        goal_tensor = torch.tensor(goal + one_hot).to(device).float()
        goal_nn = torch.unsqueeze(goal_tensor, 0)

        # Get current image stack
        image_stack = image_buffer.get_image_stack()
        if image_stack is None:
            rospy.logwarn("No images available, using previous action")
            continue

        # Save image periodically for debugging
        if images_dir is not None and i % save_every_n_steps == 0:
            with image_buffer.lock:
                if len(image_buffer.images) > 0:
                    # Save the most recent image
                    img_to_save = image_buffer.images[-1]
                    img_path = os.path.join(images_dir, f"step_{i:04d}.jpg")
                    img_to_save.save(img_path)

        with torch.no_grad():
            velocities_tensor = tester.two_stream_model(goal_nn, image_stack)

        velocities_tensor = torch.squeeze(velocities_tensor, 0)
        state[:7] += (5 * velocities_tensor)

        comm.create_and_pub_msg(state[:7])
        rospy.sleep(0.05)
        comm.jsLock.acquire()
        js = torch.tensor((list(comm.joint_state))).to(device)
        comm.jsLock.release()

        state = torch.cat((js, velocities_tensor), dim=0).to(device)
        all_joints.append(state[:7])
        all_states.append(state.tolist())

        pos = tester.get_end_eff(js.tolist())
        if one_hot[0] == 1:
            pos[2] -= 0.02
            comm.move('point2', pos)
        elif one_hot[2] == 1:
            pos[2] -= 0.02
            comm.move('point3', pos)

        if traj_point == 0 and tester.close_enough(state, grepB):
            traj_point = 1
            one_hot = [1, 0, 0, 0]
            print('here1')
            print(i)
        elif traj_point == 1 and tester.close_enough(state, putB):
            traj_point = 2
            one_hot = [0, 1, 0, 0]
            print('here2')
            print(i)
        elif traj_point == 2 and tester.close_enough(state, grepC):
            traj_point = 3
            one_hot = [0, 0, 1, 0]
            print('here3')
            print(i)
        elif traj_point == 3 and tester.close_enough(state, putC):
            traj_point = 4
            one_hot = [0, 0, 0, 1]
            print('here4')
            print(i)

    return all_joints, traj_point, all_states


def init_image_buffer(image_topic='/camera/color/image_raw'):
    """Initialize the global image buffer."""
    global image_buffer
    image_buffer = ImageBuffer(
        num_history_images=config.num_history_images,
        image_size=config.image_size,
        image_topic=image_topic
    )
    return image_buffer


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

    goal = elem[0][step_size+state_size : 
                   step_size+state_size+target_size
                   ].tolist()

    obstA = torch.tensor(goal[0:3])
    obstB = torch.tensor(goal[3:6])
    obstC = torch.tensor(goal[6:9])

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

use_only_dnfc = False
use_two_stream = True  # Set to True to use TwoStreamBaseline with images
epoch_no = 3000 #4000
train_num = 1
image_topic = '/head/camera/color/image_raw'  # Head camera (used during data collection)

if use_two_stream:
    # Initialize ROS node first for image subscription
    rospy.init_node('two_stream_tester')

    # Setup camera view (tilt torso and point head at workspace)
    # Adjust these values to match your data collection setup
    torso_tilt = 0.26  # radians (~15 degrees)
    # gaze_target matches data_collector_triangle: center_x=0.475, center_y=-0.105
    gaze_target = np.array([0.475, -0.105, 0.865])
    setup_camera_view(torso_tilt=torso_tilt, gaze_target=gaze_target)

    init_image_buffer(image_topic)
    rospy.sleep(2)  # Wait for image buffer to fill

    for model_complexity in ['high']:  # TwoStream typically uses 'high'
        tester.load_two_stream_model(0, epoch_no, model_complexity)
        params_num = tester.config.get_params_num(tester.two_stream_model)

        results_file = f'results/{config.dataset_name}_{params_num}K_{config.ds_ratio}'
        results_file += f'/ep:{epoch_no}/two_stream_{config.v_name_two_stream}'
        results_dir = os.path.join(cur_file_dir_path, results_file)
        if os.path.exists(results_dir):
            raise Exception(f"Result dir. exists. Are you testing again? Result dir: {results_dir}")
        else:
            os.makedirs(results_dir)
        print("results_dir", results_dir)

        perf_file_path = os.path.join(results_dir, "perf.csv")
        with open(perf_file_path, 'w') as f:
            writer = csv.writer(f)
            row = ["eps_num", "two_stream_succ", "two_stream_dtw", "two_stream_norm"]
            writer.writerow(row)

        all_states_two_stream = []
        for eps_num in range(len(tester.dataset)):
            for i_train in range(train_num):
                tester.load_two_stream_model(i_train, epoch_no, model_complexity)
                print(f'Testing TwoStream on episode {eps_num}, train {i_train}')
                comm.which(f'\n\n\n\ntwo_stream start on path {eps_num}\n\n\n\n')

                all_joints_ts, loss_ts, states_ts = online_test_two_stream(
                    tester, eps_num, results_dir=results_dir, save_every_n_steps=10)

                coords_ts = intrinsic_to_3d_cart(all_joints_ts)
                coords_gtruth = tester.get_real_coordinates(eps_num)

                # Calculate DTW metric (comparing to ground truth only)
                alignment_ts = dtw(np.array(list(zip(*coords_ts))),
                                   np.array(list(zip(*coords_gtruth))))
                dtw_ts = alignment_ts.distance
                dtw_norm_ts = alignment_ts.normalizedDistance

                ts_succ = loss_ts / 4
                print(f"TwoStream perf. {ts_succ}, DTW: {dtw_ts}")

                with open(perf_file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([eps_num, ts_succ, dtw_ts, dtw_norm_ts])

                all_states_two_stream.append(states_ts)

        save_list_to_file(all_states_two_stream, results_dir, "all_states_two_stream")

else:
    # Original testing loop for DNFC and Baseline
    for model_complexity in ['low', 'medium', 'high', 'xhigh']: #['medium']:
        # enc_hid, cont_hid, lin_hid, lin_out = config.get_model_dims(model_complexity)
        tester.load_model(0, 0, config.use_custom_loss, model_complexity)
        params_num = tester.config.get_params_num(tester.model)
        results_dir = create_results_dir(params_num)

        all_states_dnfc = []
        all_states_base = []
        all_latent_reps = []
        for eps_num in range(len(tester.dataset)): #random_idx: #range(27, 110):
            for i_train in range(train_num):
                tester.load_model(i_train, epoch_no, config.use_custom_loss, model_complexity)
                rospy.init_node('denz')
                print('waining for DNFC')
                comm.which('\n\n\n\ndnfc start on path'+str(eps_num)+'\n\n\n\n')
                all_joints_dnfc, loss_dnfc, latent_reps, states_dnfc = online_test(tester, eps_num, False)

                if use_only_dnfc:
                    all_joints_base, loss_basel, _, states_base = all_joints_dnfc, loss_dnfc, latent_reps, states_dnfc
                else:
                    print('waining for baseline')
                    comm.which('\n\n\n\nbaseline start on path'+str(eps_num)+'\n\n\n\n')
                    all_joints_base, loss_basel, _, states_base = online_test(tester, eps_num, True)

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

