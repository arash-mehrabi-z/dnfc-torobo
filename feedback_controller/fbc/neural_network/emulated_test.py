import matplotlib.pyplot as plt
from testers import Tester
import time
import os


cur_file_dir_path = os.path.dirname(__file__)
ds_name = 'trajs:2000_blocks:3_random'
results_dir = os.path.join(cur_file_dir_path, 
                           f'results/{ds_name}_7.541K_ep:675/emulated')
if not os.path.exists(results_dir):
      os.makedirs(results_dir)

t = Tester()
# t.get_perform3()

# print('emulated',t.get_base_loss('emulated'))
# print(t.get_base_loss('offline'))

# print('emulated',t.get_model_loss('emulated'))
# print(t.get_model_loss('offline'))

print('this is our perform', 
      t.calculate_cartesian_perform(use_baseline=False))
print('this is base perform', 
      t.calculate_cartesian_perform(use_baseline=True))

for num in range(5, 500, 7):
    x_model, y_model, z_model = t.get_coordinats(num, 
                                                 use_baseline=False)
    x_base, y_base, z_base = t.get_coordinats(num, 
                                              use_baseline=True)
    x_real, y_real, z_real = t.get_real_coordinates(num)

    obstA, obstB, obstC = t.get_obs_coordinates(num)

    # Create a figure and a 3D Axes
    fig = plt.figure(figsize=(12.8, 9.6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot real
    ax.plot(x_real, y_real, z_real, c='r', label='ground truth')
    ax.plot(x_model, y_model, z_model, c='b', label='our model')
    ax.plot(x_base, y_base, z_base, c='g', label='base')
    ax.scatter(x_base, y_base, z_base, c='g',s=5)
    ax.scatter(x_real, y_real, z_real, c='r',s=5)
    ax.scatter(x_model, y_model, z_model, c='b',s=5)

    ax.scatter([obstA[0]], [obstA[1]], [obstA[2]], c='k', marker='o')
    ax.scatter([obstB[0]], [obstB[1]], [obstB[2]], c='k', marker='o')
    ax.scatter([obstC[0]], [obstC[1]], [obstC[2]], c='k', marker='o')

    # Annotate points A, B, C
    ax.text(obstA[0], obstA[1], obstA[2], 'point A', color='black', fontsize=10, ha='center')
    ax.text(obstB[0], obstB[1], obstB[2], 'point B', color='black', fontsize=10, ha='center')
    ax.text(obstC[0], obstC[1], obstC[2], 'point C', color='black', fontsize=10, ha='center')

    # Set labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Trajectories')
    ax.legend()

    plot_path = os.path.join(results_dir, f'plt_{num}.png')
    plt.savefig(plot_path)
    plt.close()
    # Show plot
    # plt.show()