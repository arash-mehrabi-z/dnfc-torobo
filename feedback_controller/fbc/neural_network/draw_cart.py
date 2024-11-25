import matplotlib.pyplot as plt
import numpy as np
from testers import Tester

# Create an instance of Tester
t = Tester()
num=0

# Get coordinates
x_base, y_base, z_base = t.get_base_coordinats(num)
x_real, y_real, z_real = t.get_real_coordinates(num)
obstA, obstB, obstC = t.get_obs_coordinates(num)

# Generate a time array with 300 steps
time_steps = np.arange(299)  # Array with 300 steps (0, 1, 2, ..., 299)

# Create a figure with multiple subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid of subplots

# 3D Plot
ax3d = fig.add_subplot(2, 2, 1, projection='3d')
ax3d.plot(x_real, y_real, z_real, c='r', label='Ground Truth')
ax3d.plot(x_base, y_base, z_base, c='g', label='Base')
ax3d.scatter(x_base, y_base, z_base, c='g', s=5)
ax3d.scatter(x_real, y_real, z_real, c='r', s=5)
ax3d.scatter([obstA[0]], [obstA[1]], [obstA[2]], c='k', marker='o')
ax3d.scatter([obstB[0]], [obstB[1]], [obstB[2]], c='k', marker='o')
ax3d.scatter([obstC[0]], [obstC[1]], [obstC[2]], c='k', marker='o')
ax3d.text(obstA[0], obstA[1], obstA[2], 'Point A', color='black', fontsize=10, ha='center')
ax3d.text(obstB[0], obstB[1], obstB[2], 'Point B', color='black', fontsize=10, ha='center')
ax3d.text(obstC[0], obstC[1], obstC[2], 'Point C', color='black', fontsize=10, ha='center')
ax3d.set_xlabel('X Axis')
ax3d.set_ylabel('Y Axis')
ax3d.set_zlabel('Z Axis')
ax3d.set_title('3D Trajectories')
ax3d.legend()

# Time vs X
axs[0, 1].plot(time_steps, x_base, label='Base X', color='g')
axs[0, 1].plot(time_steps, x_real, label='Real X', color='r')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('X')
axs[0, 1].set_title('Time vs X')
axs[0, 1].legend()

# Time vs Y
axs[1, 0].plot(time_steps, y_base, label='Base Y', color='g')
axs[1, 0].plot(time_steps, y_real, label='Real Y', color='r')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Y')
axs[1, 0].set_title('Time vs Y')
axs[1, 0].legend()

# Time vs Z
axs[1, 1].plot(time_steps, z_base, label='Base Z', color='g')
axs[1, 1].plot(time_steps, z_real, label='Real Z', color='r')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Z')
axs[1, 1].set_title('Time vs Z')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
