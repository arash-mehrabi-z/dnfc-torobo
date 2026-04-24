import matplotlib.pyplot as plt
from testers import Tester
import time
import os
import csv
from config import Config


cur_file_dir_path = os.path.dirname(__file__)
config = Config()

# Model parameters
model_complexity = 'high'
epoch_num = 0
use_custom_loss = config.use_custom_loss
use_image = True  # Use image at t=0 as target representation

# Create tester and load model
t = Tester()
t.load_model(train_no=0, epoch_no=epoch_num, use_custom_loss=use_custom_loss,
             model_complexity=model_complexity, use_image=use_image)

# Get model params for results directory
num_params = config.get_params_num(t.model)
ds_name = config.dataset_name

results_dir = os.path.join(cur_file_dir_path,
                           f'results/{ds_name}_{num_params}K/ep:{epoch_num}/emul')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

loss_file_path = os.path.join(results_dir, "perf.csv")
with open(loss_file_path, 'w') as f:
    writer = csv.writer(f)
    row = ["epoch", "dnfc"]
    writer.writerow(row)

# Calculate performance
dnfc_perf = t.calculate_cartesian_perform(use_baseline=False, use_image=use_image)
print('DNFC performance:', dnfc_perf)

loss_file_path = os.path.join(results_dir, "perf.csv")
with open(loss_file_path, 'a') as f:
    writer = csv.writer(f)
    row = [epoch_num, dnfc_perf]
    writer.writerow(row)

for num in range(0, t.dataset.shape[0]):
    x_model, y_model, z_model = t.get_coordinats(num, use_baseline=False, use_image=use_image)
    x_real, y_real, z_real = t.get_real_coordinates(num)

    obstA, obstB, obstC = t.get_obs_coordinates(num)

    # Create a figure and a 3D Axes
    fig = plt.figure(figsize=(12.8, 9.6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories
    ax.plot(x_real, y_real, z_real, c='r', label='G.Truth')
    ax.plot(x_model, y_model, z_model, c='b', label='DNFC')
    ax.scatter(x_real, y_real, z_real, c='r', s=5)
    ax.scatter(x_model, y_model, z_model, c='b', s=5)

    ax.scatter([obstA[0]], [obstA[1]], [obstA[2]], c='k', marker='o')
    ax.scatter([obstB[0]], [obstB[1]], [obstB[2]], c='k', marker='o')
    ax.scatter([obstC[0]], [obstC[1]], [obstC[2]], c='k', marker='o')

    # Annotate points A, B, C
    ax.text(obstA[0], obstA[1], obstA[2], 'p.A', color='black', fontsize=10, ha='center')
    ax.text(obstB[0], obstB[1], obstB[2], 'p.B', color='black', fontsize=10, ha='center')
    ax.text(obstC[0], obstC[1], obstC[2], 'p.C', color='black', fontsize=10, ha='center')

    # Set labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f'Trajectory {num} - Performance: {dnfc_perf:.2%}')
    ax.legend()

    plot_path = os.path.join(results_dir, f'plt_{num}.png')
    plt.savefig(plot_path)
    plt.close()

print(f"Results saved to {results_dir}")