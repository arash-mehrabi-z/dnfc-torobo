import torch
import rospy
from nn_models import GeneralModel, MLPBaseline
from std_msgs.msg import String
from config import Config
import socket
import time
import numpy as np
from torkin import TorKin
from threading import Lock
import torch.nn as nn
import matplotlib.pyplot as plt
from testers import Tester
from udp_comm import Comm
import os
import random
import csv
import pickle
# from tslearn.metrics import dtw_path
from dtw import *

plt.switch_backend('agg')

comm = Comm()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cur_file_dir_path = os.path.dirname(__file__)
config = Config()

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


tester = Tester()
use_only_dnfc = True
epoch_no = 2000
train_num = 3

results_file = f'results/{config.dataset_name}_{config.num_params}K'
results_file += f'/ep:{epoch_no}/on_{config.v_name}_{config.C}_{config.use_custom_loss}'
results_dir = os.path.join(cur_file_dir_path, results_file)
if os.path.exists(results_dir):
    print("Result dir. exists. Are you testing again?")
    sys.exit(1)
else:
    os.makedirs(results_dir)
print("results_dir", results_dir)

perf_file_path = os.path.join(results_dir, "perf.csv")
with open(perf_file_path, 'w') as f:
    writer = csv.writer(f)
    row = ["eps_num", "dnfc_succ", "basel_succ", "dnfc_dtw", "basel_dtw", 
           "dnfc_norm", "basel_norm"]
    writer.writerow(row)

kin = TorKin()
# random_idx = random.sample(range(0, 2000), 10)
all_states_dnfc = []
all_states_base = []
all_latent_reps = []
for eps_num in range(len(tester.dataset) // 3): #random_idx: #range(27, 110):
    for i_train in range(train_num):
        tester.load_model(i_train, epoch_no, config.use_custom_loss)
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

