import matplotlib.pyplot as plt
import numpy as np
from torkin import TorKin


kin = TorKin()
dataset = np.load('/Users/denizakkoc/Desktop/fbc_new/neural_network/data/torobo/815_trajs_static/obs_mixed_200_1.npy', allow_pickle=True, encoding='latin1')
print(dataset.shape)
print()
elem = dataset[7]
print(elem.shape)
print(len(elem[1][4]))

x_points_red = []
y_points_red = []
z_points_red = []

x_points_blue = []
y_points_blue = []
z_points_blue = []

time_points = []

obstA = elem[1][2]
obstB = elem[1][3]
obstC = elem[1][4]

obst_x = [obstA[0], obstB[0], obstC[0]]
obst_y = [obstA[1], obstB[1], obstC[1]]
obst_z = [obstA[2], obstB[2], obstC[2]]

# Create a figure and a 3D scatter plot
fig = plt.figure(figsize=(14, 10))

# 3D Scatter Plot
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(obst_x, obst_y, obst_z, c='b', marker='o')
ax1.scatter([obstB[0]], [obstB[1]], [obstB[2]], c='m', marker='o')
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
ax1.set_title('3D Scatter Plot')

# Time vs X Plot
ax2 = fig.add_subplot(222)
ax2.set_xlabel('Time')
ax2.set_ylabel('X')
ax2.set_title('Time vs X')

# Time vs Y Plot
ax3 = fig.add_subplot(223)
ax3.set_xlabel('Time')
ax3.set_ylabel('Y')
ax3.set_title('Time vs Y')

# Time vs Z Plot
ax4 = fig.add_subplot(224)
ax4.set_xlabel('Time')
ax4.set_ylabel('Z')
ax4.set_title('Time vs Z')

# Populate the plots with data
for k in range(0,len(elem)):
    j=elem[k]
    state_tensor = j[1][:]
    my_l = [0,0]
    for s in state_tensor:
        my_l.append(float(s))
    
    p, R = kin.forwardkin(1, np.array(my_l))
    

    x_points_blue.append(p[0])
    y_points_blue.append(p[1])
    z_points_blue.append(p[2])

time_points=[i for i in range(len(y_points_blue))] # Assuming a constant time of 300

# Update the 3D scatter plot
ax1.scatter(x_points_blue, y_points_blue, z_points_blue, c='b', marker='o', s=3)

# Update the 2D plots
ax2.scatter(time_points, x_points_blue, c='b', marker='o', s=3)

ax3.scatter(time_points, y_points_blue, c='b', marker='o', s=3)

ax4.scatter(time_points, z_points_blue, c='b', marker='o', s=3)

plt.tight_layout()
plt.show()
