import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torkin import TorKin  # Assuming this is a custom module for kinematics calculations
from nn_models import GeneralModel  # Assuming this is a custom module for neural network models

# Load dataset
dataset = np.load('/home/deniz/Desktop/feedback_controller/fbc/neural_network/data/torobo/815_trajs_static/traj_normalized_test.npy', allow_pickle=True, encoding='latin1')

kin = TorKin()

# # Mean and std for normalization (replace with actual values if different)
# mean = [7.54634333e-01, 7.31245546e-01, 8.22160100e-01, 1.56559486e+00, -5.03539147e-01, 1.25944960e-01, 4.86361512e-01, -1.42900066e-02, -8.68937580e-03, -1.56803759e-03, 4.82522308e-03, 5.10058192e-04, -1.37347869e-02, -4.53504905e-03]
# mean_tensor = torch.tensor(mean)

# std = [0.18872306, 0.36291458, 0.22077918, 0.2650247, 0.23815572, 0.34502376, 0.24281919, 0.0819168, 0.08568684, 0.08324767, 0.07560839, 0.06727459, 0.15041522, 0.05504802]
# std_tensor = torch.tensor(std)

# Create a figure with 4 subplots (2 rows, 2 columns)
# fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})

# Flatten the axs array for easier indexing
axs = axs.flatten()

# Loop through the first 4 trajectories in the dataset (adjust range as needed)
for i in range(4):
    elem = dataset[i]  # Get the i-th trajectory

    obstA = torch.tensor(elem[1][22:25])
    obstB = torch.tensor(elem[1][25:28])
    obstC = torch.tensor(elem[1][28:31])


    x_points = []
    y_points = []
    z_points = []


    # Process each point in the trajectory
    input_s = torch.tensor(elem[0][1:15].tolist())
    input_non = (input_s * std_tensor) + mean_tensor
    state_tensor = input_non[:7]

    for j in elem:
        delta_pos = j[15:22]
        delta_pos_tensor = torch.tensor(delta_pos)
        state_tensor = state_tensor + delta_pos_tensor

        my_l = [0, 0]
        for s in state_tensor:
            my_l.append(float(s))

        p, R = kin.forwardkin(1, np.array(my_l))
        x_points.append(p[0])
        y_points.append(p[1])
        z_points.append(p[2])
    input_s=torch.tensor(elem[0][1:15].tolist())


    input_non=(input_s*std_tensor)+mean_tensor
    input_non=(input_s*std_tensor)+mean_tensor

    velocities=elem[0][8:15].tolist()
    velocities_tensor=input_non[7:]
    goal=elem[0][22:31].tolist()
    goal_tensor=torch.tensor(goal)
    
    joint_angles_tensor=input_non[:7]




    

    # Plot the trajectory on the corresponding subplot
    axs[i].scatter(x_points, y_points, z_points, c='y', s=5)
    axs[i].plot(x_points, y_points, z_points, c='k')
    axs[i].set_title(f'Trajectory {i+1}')
    axs[i].set_xlabel('X Label')
    axs[i].set_ylabel('Y Label')
    axs[i].set_zlabel('Z Label')

    axs[i].scatter([obstA[0]], [obstA[1]], [obstA[2]], c='r', marker='o', label='point A')
    axs[i].scatter([obstB[0]], [obstB[1]], [obstB[2]], c='g', marker='o',label='point B')
    axs[i].scatter([obstC[0]], [obstC[1]], [obstC[2]], c='b', marker='o',label='point C')
    axs[i].legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

