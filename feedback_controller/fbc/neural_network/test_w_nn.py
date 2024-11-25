import os
import csv
import math
import time
random_seed = 1
import numpy as np
import matplotlib.pyplot as plt
import shutil

import torch
from torchvision import transforms
import torch.nn as nn

import gymnasium as gym
from PIL import Image

from poly3 import Poly3
from nn_models import GeneralModel, CustomLoss, MLPBaseline

# np.random.seed(random_seed)
# torch.manual_seed(random_seed)

plt.rcParams['font.size'] = 18
plt.rc('font', size=18)
plt.rc('legend', fontsize=14)


# total arm lenght 0.21
end = 2000 / 0.01

############ DECLARE PARAMETERS ############
m1 = 2.809173
m2 = 0.41335134

r1 = 0.01 / 2
r2 = 0.00001

l1 = 0.1
l2 = 0.1 + 0.01

d1 = l1 / 2
d2 = l2 / 2

I_1 = 1 / 12 * m1 * (3 * r1**2 + l1**2)
I_2 = 1 / 12 * m2 * (3 * r2**2 + l2**2)

Kp = np.array([[100, 0], [0, 100]])
Kd = np.array([[18, 0], [0, 18]])


def M(obs):
    c1 = obs[0]
    c2 = obs[1]
    s1 = obs[2]
    s2 = obs[3]

    alpha = m1 * d1**2 + I_1 + m2 * (l1**2 + d2**2) + I_2
    beta = m2 * l1 * d2
    gamma = m2 * d2**2 + I_2
    M_11 = alpha + 2 * beta * c2
    M_12 = gamma + beta * c2
    M_21 = gamma + beta * c2
    M_22 = gamma

    return np.array([[M_11, M_12], [M_21, M_22]])


def C(obs):
    c1 = obs[0]
    c2 = obs[1]
    s1 = obs[2]
    s2 = obs[3]

    d1 = l1 / 2
    d2 = l2 / 2
    alpha = m1 * d1**2 + I_1 + m2 * (l1**2 + d2**2) + I_2
    beta = m2 * l1 * d2
    gamma = m2 * d2**2 + I_2

    q1_p = obs[6]
    q2_p = obs[7]

    C_11 = -beta * s2 * q2_p
    C_12 = -beta * s2 * (q1_p + q2_p)
    C_21 = beta * s2 * q1_p
    C_22 = 0

    return np.array([[C_11, C_12], [C_21, C_22]])


def caluclateInvertKinematics(x, y, l1=0.1, l2=0.11):
    # q_r -> destinated angles ... calculate q_r using invert kinematicsx
    dist = math.sqrt(x**2 + y**2)
    cos1 = ((dist**2) + (l1**2) - (l2**2)) / (2 * dist * l1)
    cos1 = np.clip(cos1, -1, 1)
    angle1 = np.arccos(cos1)
    cos2 = ((l2**2) + (l1**2) - (dist**2)) / (2 * l2 * l1)
    cos2 = np.clip(cos2, -1, 1)
    angle2 = np.arccos(cos2)
    atan = np.arctan2(y, x)
    return angle1, angle2, atan


def setNewDestination(observation):
    global xTarget, yTarget, xFingertip, yFingertip, traj_gen
    xTarget = observation[4]
    yTarget = observation[5]

    xFingertip = observation[8] + xTarget
    yFingertip = observation[9] + yTarget

    ah1, ah2, ath = caluclateInvertKinematics(xTarget, yTarget)
    ad1, ad2, atd = caluclateInvertKinematics(xFingertip, yFingertip)

    if yTarget < 0:
        qh = [ath + ah1, ah2 - np.pi]
        qd = [atd + ad1, ad2 - np.pi]
    else:
        qh = [ath - ah1, np.pi - ah2]
        qd = [atd - ad1, np.pi - ad2]

    traj_gen = Poly3(np.array([qh[0], qh[1]]), np.array([qd[0], qd[1]]), end)


def average_absolute_difference_fast(torq1_cont, torq2_cont, torq1_pred, torq2_pred):
    torq1_cont = np.array(torq1_cont)
    torq2_cont = np.array(torq2_cont)
    torq1_pred = np.array(torq1_pred)
    torq2_pred = np.array(torq2_pred)

    # Calculate the absolute difference between the two arrays
    abs_diff_1 = np.abs(torq1_cont - torq1_pred)
    abs_diff_2 = np.abs(torq2_cont - torq2_pred)

    average_diff = (abs_diff_1 + abs_diff_2) / 2

    # Calculate the average of these absolute differences
    average_diff = np.mean(average_diff)
    
    return average_diff


def reset():
    global current_test_dir, episode_no, step_no
    global torq1_cont_steps, torq2_cont_steps, torq1_model_steps, torq2_model_steps
    global x_des_1, x_des_2, x_des_3, x_des_4
    global diff_1, diff_2, diff_3, diff_4
    global q1s, q2s, vel_1s, vel_2s
    global fingertip_traj_episodes, fingertip_traj_steps, i
    global nn_in_the_loop, torq_err_model_episodes
    if not nn_in_the_loop:
        global torq1_base_steps, torq2_base_steps, torq1_abl_steps, torq2_abl_steps
        global torq_err_base_episodes, torq_err_abl_episodes

    i = 0
    episode_no += 1
    step_no = 0

    # TODO when torqs reduced to 1 array each; we can remove this function
    average_diff = average_absolute_difference_fast(torq1_cont_steps, torq2_cont_steps, 
                                                     torq1_model_steps, torq2_model_steps)
    torq_err_model_episodes.append(average_diff)
    if not nn_in_the_loop:
        average_diff_base = average_absolute_difference_fast(torq1_cont_steps, torq2_cont_steps, 
                                                     torq1_base_steps, torq2_base_steps)
        torq_err_base_episodes.append(average_diff_base)

        average_diff_abl = average_absolute_difference_fast(torq1_cont_steps, torq2_cont_steps, 
                                                     torq1_abl_steps, torq2_abl_steps)
        torq_err_abl_episodes.append(average_diff_abl)

    fingertip_traj_episodes.append(fingertip_traj_steps)

    torq1_cont_steps, torq2_cont_steps, torq1_model_steps, torq2_model_steps = [], [], [], []
    if not nn_in_the_loop: 
        torq1_base_steps, torq2_base_steps = [], []
        torq1_abl_steps, torq2_abl_steps = [], []
    x_des_1, x_des_2, x_des_3, x_des_4 = [], [], [], []
    q1s, q2s, vel_1s, vel_2s = [], [], [], []
    diff_1, diff_2, diff_3, diff_4 = [], [], [], []
    fingertip_traj_steps = []

    create_dir_if_not_exists(current_test_dir + f"/videos/eps_{episode_no}")
    create_dir_if_not_exists(current_test_dir + "/reprs")
    create_dir_if_not_exists(current_test_dir + "/torques")
    create_dir_if_not_exists(current_test_dir + "/diffs")
    create_dir_if_not_exists(current_test_dir + "/trajectories")


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def crop_image(img):
    width, height = img.size

    # Define crop points
    left = width / 7
    top = height / 3.2
    right = 6 * width / 7
    bottom = 3.05 * height / 4

    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img


def preprocess_image_for_model(img):
    transform = transforms.Compose([
                # transforms.Resize([120, 120]),
                # transforms.RandomHorizontalFlip(), # Flip the data horizontally
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ])
    img = Image.fromarray(img) 
    img = crop_image(img)
    # img.show()
    # time.sleep(5)
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    return img


def save_image(arr, path):
    im = Image.fromarray(arr)
    im.save(path)  


# def get_traj_loss_as_string():
#     global traj_loss_custom, traj_loss_torques

#     title = ""
#     title += f' (traj_loss_custom={round(traj_loss_custom, 3)})'
#     title += f' (traj_loss_torques={round(traj_loss_torques, 3)})'
#     return title


def get_traj_case_as_string():
    global truncated

    title = ""
    if truncated: 
        title += "failure"  
    else: 
        title += "success"
    return title


def check_if_nn_in_the_loop():
    global nn_in_the_loop

    return nn_in_the_loop


def plot_diffs(diff_1, diff_2, diff_3, diff_4):
    global current_test_dir, episode_no, truncated

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(x_des_1)+1), diff_1, label='diff_1', color='blue')
    plt.plot(range(1, len(x_des_1)+1), diff_2, label='diff_2', color='green')
    plt.plot(range(1, len(x_des_1)+1), diff_3, label='diff_3', color='orange')
    plt.plot(range(1, len(x_des_1)+1), diff_4, label='diff_4', color='purple') 

    plt.xlabel('Timestep')
    # plt.ylabel('Diffs')
    # title = 'Diffs over Timestep'
    # title += get_traj_case_as_string()
    # title += get_traj_loss_as_string()
    # plt.title(title)
    plt.legend()
    case = get_traj_case_as_string()
    plt.savefig(current_test_dir + f"/diffs/diff_{episode_no}_{case}.png")
    # plt.show()
    plt.close()


def plot_reprs(x_des_1, x_des_2, x_des_3, x_des_4, q1s, q2s, vel_1s, vel_2s):
    global current_test_dir, episode_no, truncated
    # global run_no

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(x_des_1)+1), x_des_1, label='x_des_1', color='blue')
    plt.plot(range(1, len(x_des_1)+1), x_des_2, label='x_des_2', color='green')
    plt.plot(range(1, len(x_des_1)+1), x_des_3, label='x_des_3', color='orange')
    plt.plot(range(1, len(x_des_1)+1), x_des_4, label='x_des_4', color='purple')

    plt.plot(range(1, len(x_des_1)+1), q1s, label='q1', color='blue', 
             linestyle='--')
    plt.plot(range(1, len(x_des_1)+1), q2s, label='q2', color='green', 
             linestyle='--')
    plt.plot(range(1, len(x_des_1)+1), vel_1s, label='vel_1', color='orange', 
             linestyle='--')
    plt.plot(range(1, len(x_des_1)+1), vel_2s, label='vel_2', color='purple', 
             linestyle='--')

    # Add labels and legend
    plt.xlabel('Timestep')
    # plt.ylabel('Value')
    # title = 'Values over Timestep'
    # title += get_traj_case_as_string()
    # title += get_traj_loss_as_string()
    # plt.title(title)
    plt.legend()
    case = get_traj_case_as_string()
    plt.savefig(current_test_dir + f"/reprs/repr_{episode_no}_{case}.png")
    # plt.show()
    plt.close()


def plot_trajectory(fingertip_traj_steps):
    global current_test_dir, episode_no

    fingertip_traj_steps = np.array(fingertip_traj_steps)
    fingertip_traj_steps *= 100

    plt.figure(figsize=(12, 8))
    plt.scatter(fingertip_traj_steps[:, 0], fingertip_traj_steps[:, 1], color='green', 
                label='fingertip')
    plt.scatter(fingertip_traj_steps[-1, 0], fingertip_traj_steps[-1, 1], color='blue', 
                label='fingertip_end', marker='v')
    plt.scatter(fingertip_traj_steps[0, 0], fingertip_traj_steps[0, 1], color='blue', 
                label='fingertip_start', marker='P')
    plt.scatter(fingertip_traj_steps[:, 2], fingertip_traj_steps[:, 3], color='orange', 
                label='target')
    plt.scatter(fingertip_traj_steps[-1, 2], fingertip_traj_steps[-1, 3], color='red', 
                label='target_end', marker='X')
    plt.scatter(fingertip_traj_steps[0, 2], fingertip_traj_steps[0, 3], color='red', 
                label='target_start', marker='D')
    plt.xlabel('x(cm)')
    plt.ylabel('y(cm)')
    plt.legend()
    case = get_traj_case_as_string()
    plt.savefig(current_test_dir + f"/trajectories/traj_{episode_no}_{case}.png")
    # plt.show()
    plt.close()


def visualize_torques(torq1_cont_steps, torq2_cont_steps, torq1_model_steps, torq2_model_steps):
    global current_test_dir, episode_no, truncated
    global nn_in_the_loop

    timestep = list(range(len(torq1_cont_steps)))
    # Plotting each list over time
    plt.figure(figsize=(12, 8))  # Optional: Set the figure size
    plt.plot(timestep, torq1_cont_steps, label='controller_1', color="cyan")
    plt.plot(timestep, torq2_cont_steps, label='controller_2', color="olive")
    
    if nn_in_the_loop:
        plt.plot(timestep, torq1_model_steps, label='model_1', color="cyan", linestyle="dashed")
        plt.plot(timestep, torq2_model_steps, label='model_2', color="olive", linestyle="dashed")
    else:
        global torq1_base_steps, torq2_base_steps
        plt.plot(timestep, torq1_model_steps, label='DNFC_1', color="cyan", linestyle="dashed")
        plt.plot(timestep, torq2_model_steps, label='DNFC_2', color="olive", linestyle="dashed")
        plt.plot(timestep, torq1_base_steps, label='baseline_1', color="cyan", linestyle="dotted")
        plt.plot(timestep, torq2_base_steps, label='baseline_2', color="olive", linestyle="dotted")

    # Adding labels and title
    plt.xlabel('Timestep')
    plt.ylabel('Torque (N⋅m)')
    # title = f'Torques over timesteps of a trajectory'
    # title += get_traj_case_as_string()
    # title += get_traj_loss_as_string()
    # plt.title(title)
    plt.legend()  # Show legend to differentiate between the lists
    case = get_traj_case_as_string()
    plt.savefig(current_test_dir + f"/torques/trajectory_{episode_no}_{case}.png")
    plt.close()


def plot_hist(array, array_name, x_lim_low=None, x_lim_high=None, num_bins=100):
    global current_test_dir

    mean_value = np.mean(array)
    plt.figure(figsize=(12, 8))
    plt.hist(array, bins=num_bins, alpha=0.7, color='blue')  # Adjust bins as needed
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1.5, 
                label=f'Mean: {mean_value:.3f}')
    # plt.title(f'Histogram of {array_name}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    if x_lim_low is not None:
        plt.xlim(x_lim_low, x_lim_high)
        array_name += "_zoom"
    plt.legend()
    # plt.show()
    plt.savefig(current_test_dir + f"/hist_{array_name}.png")
    plt.close()


def calculate_euc_dist(point_1, point_2):
    distance = math.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)
    return distance


def get_models_names(use_custom_loss, nn_in_the_loop, use_image, use_baseline):
    model_name = ''
    model_name_base = ''
    model_name_abl = ''
    if use_custom_loss: 
        # model_name = f"cus_los_{C_const}_{D}_{E}"
        model_name = f"cus_los_const_mse_st"
        if not nn_in_the_loop: 
            model_name_base = "mse_los"
            model_name_abl = 'mse_los'
    else:
        model_name = "mse_los"

    model_name += "|st_vel_norm"
    if not nn_in_the_loop: 
        model_name_base += "|st_vel_norm"
        model_name_abl += "|st_vel_norm"

    if use_image:
        model_name += "|tar_img"
        if not nn_in_the_loop: 
            model_name_base += "|tar_img"
            model_name_abl += "|tar_img"
    else:
        model_name += "|tar_cart"
        if not nn_in_the_loop: 
            model_name_base += "|tar_cart"
            model_name_abl += "|tar_cart"

    if use_baseline:
        model_name += "|base"

    if not nn_in_the_loop: 
        # model_name += "|cont"
        model_name_base += "|base"
        # model_name_abl += "|ablat"

    return model_name, model_name_base, model_name_abl


# Custom Loss:
num_steps = 50
T = num_steps - 1
C_const = 1 #5
D = 1 #10
E = 1

# General:
current_dir_path = os.path.dirname(__file__)
episodes_num_ds = 4000 #466
max_num_steps = num_steps
use_baseline = False
use_custom_loss = True
# use_extrapolation = False
for_demonstration = False
store_image_freq = 5 #1e6
#####
nn_in_the_loop = True
compare_abl = True # compare with abl model too, if True.
use_dynamic_train = True #False
use_dynamic_targets = True #False
test_set_mode = 'random' #'random' #known, interpolation, extrapolation
num_train_sets = 2
num_test_sets = 1
max_episode = 500
use_image = True
use_model_saved_at_epoch = 1000 #1200 #TODO check this
trained_dataset_name = f"{episodes_num_ds}_manu_traj_dyna" #f"{episodes_num_ds}_man_tra"

if use_baseline:
    use_custom_loss = False

model_name, model_name_base, model_name_abl = get_models_names(use_custom_loss, 
                                                               nn_in_the_loop, use_image, 
                                                               use_baseline)
# Training:
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model:
encoded_space_dim = 4

if use_baseline:
    model = MLPBaseline(inp_dim=6, out_dim=2)
else:
    model = GeneralModel(encoded_space_dim=encoded_space_dim, use_image=use_image)
    if not nn_in_the_loop:
        model_base = MLPBaseline(inp_dim=6, out_dim=2)
        m = model_base.to(device)
        num_params = sum(p.numel() for p in m.parameters()) / 1e3
        model_name_base += f"|{num_params}K_params"

        model_abl = GeneralModel(encoded_space_dim=encoded_space_dim, use_image=use_image)
        m = model_abl.to(device)
        num_params = sum(p.numel() for p in m.parameters()) / 1e3
        model_name_abl += f"|{num_params}K_params"

m = model.to(device)
num_params = sum(p.numel() for p in m.parameters())/1e3
model_name += f"|{num_params}K_params"
    
test_results_root_dir = (current_dir_path + f"/test_results") + ("_demo" if for_demonstration else "")
test_results_root_dir += "/dynamic_train" if use_dynamic_train else "/static_train"
test_results_root_dir += "/dynamic_targets" if use_dynamic_targets else "/static_targets"
test_results_root_dir += "/image" if use_image else "/cartesian"
test_results_root_dir += "/online" if nn_in_the_loop else "/offline"
test_results_root_dir += f"/{test_set_mode}_test_set"
test_results_root_dir += f"/{trained_dataset_name}|{model_name}|epoch_{use_model_saved_at_epoch}"

counter_dir = 0
test_results_root_dir_tmp = test_results_root_dir
while os.path.exists(test_results_root_dir_tmp):
    test_results_root_dir_tmp = test_results_root_dir + f"_{counter_dir}"
    counter_dir += 1
test_results_root_dir = test_results_root_dir_tmp

# Mean & STD of training dataset: #TODO always check here

# Static 10K:
# train_ds_mean = np.array([-0.00188743, 0.00013914, -0.00315345, 0.01042289])
# train_ds_std = np.array([0.91649005, 1.31088051, 2.16946285, 3.51440869])

# # Dynamic: 
# train_ds_mean = np.array([0.19293386, 2.39774927, -0.0159914, 0.03152213])
# train_ds_std = np.array([0.74411076, 0.37488537, 0.72508472, 0.36188884])

# Image target representation: (4K linear trajectories)
train_ds_mean = np.array([0.20368284, 2.37490018, -0.0846421, 0.17627343])
train_ds_std = np.array([0.75501852, 0.39479032, 1.30610318, 0.60426782])

if use_custom_loss:
    criterion = CustomLoss(T, C_const, D, E)
else:
    criterion = nn.MSELoss()
criterion_base = nn.MSELoss()
criterion_abl = nn.MSELoss()

num_successes_trains = []
torq_err_model_trains = []
if not nn_in_the_loop: 
    torq_err_base_trains = []
    torq_err_abl_trains = []
fingertip_traj_trains = []

for i_train in range(num_train_sets):
    num_successes_tests = []
    torq_err_model_tests = []
    if not nn_in_the_loop: 
        torq_err_base_tests = []
        torq_err_abl_tests = []
    fingertip_traj_tests = []

    weight_path = current_dir_path + "/weights" + \
            f"/{trained_dataset_name}|{model_name}/train_no_{i_train}" + \
                f"/fbc_{use_model_saved_at_epoch}.pth"
    # print(weight_path)
    # err
    
    if not nn_in_the_loop:
        weight_path_base = current_dir_path + "/weights" + \
                f"/{trained_dataset_name}|{model_name_base}/train_no_{i_train}" + \
                    f"/fbc_{use_model_saved_at_epoch}.pth"
        weight_path_abl = current_dir_path + "/weights" + \
                f"/{trained_dataset_name}|{model_name_abl}/train_no_{i_train}" + \
                    f"/fbc_{use_model_saved_at_epoch}.pth"
    
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    if not nn_in_the_loop:
        model_base.load_state_dict(torch.load(weight_path_base))
        model_base.eval()
        print("Baseline loaded successfully from", weight_path_base)
        if compare_abl:
            model_abl.load_state_dict(torch.load(weight_path_abl))
            model_abl.eval()
            print("DNFC-ablation loaded successfully from", weight_path_abl)
    # err
    for i_test in range(num_test_sets):
        num_success_episodes = 0
        num_failure_episodes = 0
        torq_err_model_episodes = []
        if not nn_in_the_loop: 
            torq_err_base_episodes = []
            torq_err_abl_episodes = []
        fingertip_traj_episodes = []

        current_test_dir = test_results_root_dir + f"/train_no_{i_train}/test_no_{i_test}"

        print(f"=== Train no: {i_train}, Test no: {i_test} ===")
        print("device:", device)
        print("trained_dataset_name", trained_dataset_name)
        print("model_name", model_name)
        print(num_params, 'M parameters')
        print("type_of_criterion", type(criterion))
        print("model loaded successfully from the following path:", weight_path)
        print("current_test_dir", current_test_dir)

        create_dir_if_not_exists(current_test_dir)
        create_dir_if_not_exists(current_test_dir + "/videos/eps_0")
        create_dir_if_not_exists(current_test_dir + "/reprs")
        create_dir_if_not_exists(current_test_dir + "/torques")
        create_dir_if_not_exists(current_test_dir + "/diffs")
        create_dir_if_not_exists(current_test_dir + "/trajectories")

        episode_no = 0
        step_no = 0
        # env = gym.make('Reacher-v4', render_mode='human')
        env = gym.make("Reacher-v4", render_mode="rgb_array")

        observation, info = env.reset() #(seed=run_no)
        while np.isnan(observation).any():
            print(f"nan in episode {episode_no} reset", observation)
            observation, info = env.reset()

        goal_img = env.render()
        # im = plt.imshow(goal_img)
        goal_img = preprocess_image_for_model(goal_img)

        # setNewDestination(observation)
        # TODO torqs can be reduced to 1 array, as well as x_des', states, etc.
        i = 0
        torq1_cont_steps, torq2_cont_steps, torq1_model_steps, torq2_model_steps = [], [], [], []
        if not nn_in_the_loop: 
            torq1_base_steps, torq2_base_steps = [], []
            torq1_abl_steps, torq2_abl_steps = [], []
        x_des_1, x_des_2, x_des_3, x_des_4 = [], [], [], []
        diff_1, diff_2, diff_3, diff_4 = [], [], [], []
        q1s, q2s, vel_1s, vel_2s = [], [], [], []
        fingertip_traj_steps = []

        while True: # episode loop
            cur_img = env.render()
            if episode_no % store_image_freq == 0:
                save_image(cur_img, current_test_dir + 
                        f"/videos/eps_{episode_no}/step_{step_no}.png")
            cur_img = preprocess_image_for_model(cur_img)
            setNewDestination(observation)
            c1 = observation[0]
            c2 = observation[1]
            s1 = observation[2]
            s2 = observation[3]

            xTarget = observation[4]
            yTarget = observation[5]

            q1 = np.angle(c1 + 1j * s1)
            q2 = np.angle(c2 + 1j * s2)

            q = np.array([q1, q2])
            q_p = np.array([observation[6], observation[7]])
            xFingertip = observation[8] + xTarget
            yFingertip = observation[9] + yTarget
            fingertip_traj_steps.append([xFingertip, yFingertip, xTarget, yTarget])

            q_r, q_r_p, q_r_pp = traj_gen.generate(i)
            v = Kd @ (q_r_p - q_p) + Kp @ (q_r - q)
            tau = M(observation) @ v + C(observation) @ q_p
            tau = np.clip(tau, -1, 1)
            action_cont = (tau[0], tau[1])
            torq1_cont_steps.append(action_cont[0])
            torq2_cont_steps.append(action_cont[1])

            state = (q1, q2, observation[6], observation[7])
            state = (state - train_ds_mean) / train_ds_std   #normalize state
            state = torch.tensor(state).unsqueeze(0).to(device).float()

            target_pos = (xTarget, yTarget)
            target_pos = torch.tensor(target_pos).unsqueeze(0).to(device).float()

            model.eval()
            if not nn_in_the_loop:
                model_base.eval()
                model_abl.eval()
            with torch.no_grad():
                if use_image:
                    target_repr = cur_img #goal_img
                else:
                    target_repr = target_pos

                if use_baseline:
                    nn_input = torch.cat((target_repr, state), dim=1)
                    action_pred = model(nn_input)
                else:
                    # if step_no % 5 == 0:
                    #     print(state)
                    #     image = target_repr.squeeze()  # Remove the batch dimension
                    #     if image.max() <= 1:
                    #         image = image * 255  # Optionally scale if your tensor uses a 0-1 scale

                    #     # Convert to numpy and display using matplotlib
                    #     image = image.permute(1, 2, 0).detach().cpu().numpy()  # Rearrange dimensions to Height x Width x Channels
                    #     plt.imshow(image.astype('uint8'))  # Convert to uint8 if necessary
                    #     plt.axis('off')  # Turn off axis numbers and ticks
                    #     plt.show()

                    action_pred, x_des, diff = model(target_repr, state)
                    if not nn_in_the_loop:
                        nn_input = torch.cat((target_repr, state), dim=1)
                        action_pred_base = model_base(nn_input)
                        action_pred_abl, x_des_abl, diff_abl = model_abl(target_repr, state)

            action_cont = torch.tensor(action_cont).unsqueeze(0).to(device).float()
            if use_custom_loss:
                loss_custom, loss_torques = criterion(action_pred, action_cont, 
                            x_des, state,
                            torch.tensor(step_no).unsqueeze(0).to(device).float())
            else:
                loss_torques = criterion(action_pred, action_cont)
                loss_custom = loss_torques
            
            if use_baseline:
                pass
            else:
                diff = diff.squeeze(0).detach().cpu().numpy()
                diff_1.append(diff[0])
                diff_2.append(diff[1])
                diff_3.append(diff[2])
                diff_4.append(diff[3])

                x_des = x_des.squeeze(0).detach().cpu().numpy()
                x_des_1.append(x_des[0].item())
                x_des_2.append(x_des[1].item())
                x_des_3.append(x_des[2].item())
                x_des_4.append(x_des[3].item())
                q1s.append(q1)
                q2s.append(q2)
                vel_1s.append(observation[6])
                vel_2s.append(observation[7])

            action_pred = action_pred.squeeze(0).detach().cpu().numpy()
            torq1_model_steps.append(action_pred[0])
            torq2_model_steps.append(action_pred[1])
            if not nn_in_the_loop:
                action_pred_base = action_pred_base.squeeze(0).detach().cpu().numpy()
                torq1_base_steps.append(action_pred_base[0])
                torq2_base_steps.append(action_pred_base[1])

                action_pred_abl = action_pred_abl.squeeze(0).detach().cpu().numpy()
                torq1_abl_steps.append(action_pred_abl[0])
                torq2_abl_steps.append(action_pred_abl[1])

            if nn_in_the_loop:
                applied_action = action_pred
            else:
                applied_action = action_cont.squeeze(0).detach().cpu().numpy()
            # trajectory.append((q1, q2, observation[6], observation[7], 
            #                    action[0], action[1]))
            i += 0.01
            step_no += 1
            if step_no < max_num_steps:
                observation, reward, terminated, truncated, info = env.step(applied_action)
                if np.isnan(observation).any():
                    print(f"Got nan in train:{i_train}, test:{i_test}, episode:{episode_no}, step:{step_no}", 
                        observation)
            else:
                if not ((abs(xFingertip - xTarget) <= 0.01) and 
                        (abs(yFingertip - yTarget) <= 0.01)):
                    message_str = f"Couldn't find the solution in {max_num_steps} steps."
                    message_str += f" Train no: {i_train}, Episode no: {episode_no}."
                    print(message_str)
                    num_failure_episodes += 1
                    # plt.pause(0.001)
                else:
                    num_success_episodes += 1

                if episode_no%30==0:
                    print(f"episode {episode_no} finished.")

                if use_baseline:
                    pass
                else:
                    # plot_reprs(x_des_1, x_des_2, x_des_3, x_des_4, q1s, q2s, 
                    #         vel_1s, vel_2s)
                    # plot_diffs(diff_1, diff_2, diff_3, diff_4)
                    pass
                
                # visualize_torques(torq1_cont_steps, torq2_cont_steps, torq1_model_steps, 
                #                 torq2_model_steps)
                # plot_trajectory(fingertip_traj_steps)
                
                reset()
                if episode_no == max_episode:
                    break
                
                observation, info = env.reset()
                while np.isnan(observation).any():
                    print(f"nan in episode {episode_no} reset", observation)
                    observation, info = env.reset()        
                goal_img = env.render()
                goal_img = preprocess_image_for_model(goal_img)

        env.close()

        num_successes_tests.append(num_success_episodes)
        torq_err_model_tests.append(torq_err_model_episodes)
        if not nn_in_the_loop:
            torq_err_base_tests.append(torq_err_base_episodes)
            torq_err_abl_tests.append(torq_err_abl_episodes)
        fingertip_traj_tests.append(fingertip_traj_episodes)

        print(f"=\nTest no: {i_test}")
        print("num_successes", num_success_episodes)
        print("num_failures", num_failure_episodes)

    num_successes_trains.append(num_successes_tests)
    torq_err_model_trains.append(torq_err_model_tests)
    if not nn_in_the_loop:
        torq_err_base_trains.append(torq_err_base_tests)
        torq_err_abl_trains.append(torq_err_abl_tests)
    fingertip_traj_trains.append(fingertip_traj_tests)

    print(f"===\nTrain no: {i_train}")
    print("mean num_successes tests", np.mean(np.array(num_successes_tests)))

print(f"=====\nTesting Finished:")
print("mean num_successes trains (all)", np.mean(np.array(num_successes_trains)))

with open(test_results_root_dir + f"/num_successes_trains.npy", 'wb') as f:
    np.save(f, num_successes_trains)

with open(test_results_root_dir + f"/torq_err_model_trains.npy", 'wb') as f:
    np.save(f, torq_err_model_trains)

if not nn_in_the_loop:
    with open(test_results_root_dir + f"/torq_err_base_trains.npy", 'wb') as f:
        np.save(f, torq_err_base_trains)

    with open(test_results_root_dir + f"/torq_err_abl_trains.npy", 'wb') as f:
        np.save(f, torq_err_abl_trains)

import pickle
with open(test_results_root_dir + f'/fingertip_traj_trains.pkl', 'wb') as file:
    pickle.dump(fingertip_traj_trains, file)

print("Testing finished.")