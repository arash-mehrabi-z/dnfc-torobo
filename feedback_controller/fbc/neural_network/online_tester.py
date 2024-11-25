import torch
import rospy
from nn_models import GeneralModel, MLPBaseline
from std_msgs.msg import String
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

plt.switch_backend('agg')

comm = Comm()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cur_file_dir_path = os.path.dirname(__file__)

def online_test(use_baseline, dataset, traj_num, 
                num_params, num_params_base, epoch_num):
    joint_size = 7
    state_size = 2 * joint_size
    step_size = 1
    target_size = 9
    onehot_size = 4
    traj_step_size = 900

    dnfc_adr = f'weights/trajs:360_blocks:3_triangle|mse_los|tar_cart|v_init|{num_params}K_params' + \
        f'/train_no_0/fbc_{epoch_num}.pth'
    base_adr = f'weights/trajs:360_blocks:3_triangle|mse_los|tar_cart|base|v_init|{num_params_base}K_params' + \
        f'/train_no_0/fbc_{epoch_num}.pth'

    model = GeneralModel(state_size, (target_size+onehot_size),
                                 joint_size, use_image=False)
    baseline = MLPBaseline(state_size+(target_size+onehot_size), 
                           joint_size)
    m = model.to(device)
    m = baseline.to(device)
    
    if use_baseline:
        base_path = os.path.join(cur_file_dir_path, base_adr)
        baseline.load_state_dict(torch.load(base_path, 
                                         map_location=torch.device(device)))
        baseline.eval()

    else:
        dnfc_path = os.path.join(cur_file_dir_path, dnfc_adr)
        model.load_state_dict(torch.load(dnfc_path, 
                                         map_location=torch.device(device)))
        model.eval()

    point_loss = 0
    elem = dataset[traj_num]
    state = torch.tensor(elem[0][step_size : 
                                 step_size + state_size].tolist()
                                 ).to(device)
    
    milestones = t.get_changes_indexes(traj_num)
    print(milestones)
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

    path_point = 0
    all_joints = []
    loss = 0
    criterion = nn.L1Loss()

    for i in range(traj_step_size):
        comm.which('\n dnfc in on step'+str(i)+'\n')

        goal_tensor = torch.tensor(goal+one_hot).to(device)
        goal_nn = torch.unsqueeze(goal_tensor, 0)
        state_nn = torch.unsqueeze(state, 0)
        # delta = torch.tensor(elem[i][step_size + state_size + target_size + onehot_size: step_size + state_size + target_size + onehot_size + joint_size ].tolist())
        if use_baseline:
            basel_input = torch.cat((goal_nn, state_nn), dim=1)
            velocities_tensor = baseline(basel_input)
        else:
            velocities_tensor, x_des, _ = model(goal_nn, state_nn)

        velocities_tensor = torch.squeeze(velocities_tensor, 0)
        state[:7] += (5*velocities_tensor)

        comm.create_and_pub_msg(state[:7])
        rospy.sleep(0.05)
        comm.jsLock.acquire()
        js = torch.tensor((list(comm.joint_state))).to(device)
        comm.jsLock.release()

        state = torch.cat((js, velocities_tensor), dim=0).to(device)
        # state = torch.cat((state[:7],velocities_tensor),dim=0)
        all_joints.append(state[:7])

        pos = t.get_end_eff(js.tolist())
        if one_hot[0]==1:
            pos[2] -= 0.02
            comm.move('point2', pos)
        elif one_hot[2]==1:
            pos[2]-= 0.02
            comm.move('point3', pos)

        # state += velocities_tensor
        if path_point==0 and t.close_enough(state, grepB):
            path_point = 1
            one_hot = [1, 0, 0, 0]
            print('here1')
            print(i)
        elif path_point==1 and t.close_enough(state, putB):
            path_point = 2
            one_hot = [0, 1, 0, 0]
            print('here2')
            print(i)
        elif path_point==2 and t.close_enough(state, grepC):
            path_point = 3
            one_hot = [0, 0, 1, 0]   
            print('here3')
            print(i)  
        elif path_point==3 and t.close_enough(state, putC):
            path_point = 4
            one_hot = [0, 0, 0, 1] 
            print('here4')
            print(i)

    point_loss += path_point
    return all_joints, point_loss


# Plot results
def plot_results(all_joints, all_joints_general, 
                 plot_number, elem, results_dir):
    joint_size = 7
    state_size = 2 * joint_size
    step_size = 1
    target_size = 9
    onehot_size = 4
    goal = elem[0][step_size+state_size : 
                   step_size+state_size+target_size
                   ].tolist()

    obstA = torch.tensor(goal[0:3])
    obstB = torch.tensor(goal[3:6])
    obstC = torch.tensor(goal[6:9])

    fig = plt.figure(figsize=(12.8, 9.6))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = [], [], []
    x_general, y_general, z_general = [], [], []
    for new_angles in all_joints:
        my_l = [0, 0]
        for j in new_angles:
            my_l.append(float(j))
        p, R = kin.forwardkin(1, np.array(my_l))
        x.append(p[0])
        y.append(p[1])
        z.append(p[2])

    for new_angles in all_joints_general:
        my_l = [0, 0]
        for j in new_angles:
            my_l.append(float(j))
        p, R = kin.forwardkin(1, np.array(my_l))
        x_general.append(p[0])
        y_general.append(p[1])
        z_general.append(p[2])

    x_real, y_real, z_real = t.get_real_coordinates(plot_number)

    ln_wd = 2
    ax.scatter(x, y, z, c='g', s=1, label='Baseline', linewidths=ln_wd)
    ax.scatter(x_general, y_general, z_general, c='b', s=1, label='DNFC', linewidths=ln_wd)
    ax.scatter(x_real, y_real, z_real, c='r', s=1, label='G.Truth', linewidths=ln_wd)

    ax.scatter([obstA[0]], [obstA[1]], [obstA[2]], c='k', marker='o')
    ax.scatter([obstB[0]], [obstB[1]], [obstB[2]], c='k', marker='o')
    ax.scatter([obstC[0]], [obstC[1]], [obstC[2]], c='k', marker='o')

    ax.text(obstA[0], obstA[1], obstA[2], 'p.A', color='black', fontsize=10, ha='center')
    ax.text(obstB[0], obstB[1], obstB[2], 'p.B', color='black', fontsize=10, ha='center')
    ax.text(obstC[0], obstC[1], obstC[2], 'p.C', color='black', fontsize=10, ha='center')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    # ax.set_title('Model Trajectories')
    ax.legend()
    
    plot_path = os.path.join(results_dir, f'plt_{plot_number}.png')
    plt.savefig(plot_path)
    # plt.show()
    plt.close()

def log_loss(eps_num, dnfc_succ, basel_succ, results_dir):
    print("DNFC & Basel perf.", dnfc_succ, basel_succ)
    perf_file_path = os.path.join(results_dir, "perf.csv")
    with open(perf_file_path, 'a') as f:
        writer = csv.writer(f)
        row = [eps_num, dnfc_succ, basel_succ]
        writer.writerow(row)


t = Tester()
point_loss = 0
point_loss_model = 0
num_params = 7.541 #288.661 #25.301
num_params_base = 7.431
epoch_num = 1000
ds_name = 'trajs:360_blocks:3' + '_triangle'
dataset_path = os.path.join(cur_file_dir_path, 
                            f'data/torobo/{ds_name}/train_ds.npy')
dataset = np.load(dataset_path, allow_pickle=True, encoding='latin1')
print("Loaded dataset w/ shape", dataset.shape)

results_dir = os.path.join(cur_file_dir_path, 
                           f'results/{ds_name}_{num_params}K_ep:{epoch_num}')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

perf_file_path = os.path.join(results_dir, "perf.csv")
with open(perf_file_path, 'w') as f:
    writer = csv.writer(f)
    row = ["eps_num", "dnfc_succ", "basel_succ"]
    writer.writerow(row)

kin = TorKin()
# random_idx = random.sample(range(0, 2000), 10)
sum_dnfc_succ = 0
sum_basel_succ = 0
for eps_num in range(27, 149, 7): #random_idx: #range(27, 110):
    rospy.init_node('denz')
    print('waining for DNFC')
    comm.which('\n\n\n\ndnfc start on path'+str(eps_num)+'\n\n\n\n')
    all_joints_general, loss_general = online_test(False, dataset, eps_num, 
                                                   num_params, num_params_base, 
                                                   epoch_num)

    print('waining for baseline')
    comm.which('\n\n\n\nbaseline start on path'+str(eps_num)+'\n\n\n\n')
    all_joints, loss = online_test(True, dataset, eps_num, 
                                   num_params, num_params_base, 
                                   epoch_num)

    plot_results(all_joints, all_joints_general, 
                 eps_num, dataset[eps_num], results_dir)
    
    dnfc_succ = loss_general / 4
    sum_dnfc_succ += dnfc_succ

    basel_succ = loss / 4
    sum_basel_succ += basel_succ

    log_loss(eps_num, dnfc_succ, basel_succ, results_dir)

log_loss("sum", sum_dnfc_succ, sum_basel_succ, results_dir)