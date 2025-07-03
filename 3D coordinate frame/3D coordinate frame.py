import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to create a rotation matrix for a given yaw, pitch, and roll
def rotation_matrix(yaw, pitch, roll):

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])


    R = Rz @ Ry @ Rx
    return R

def plot_coordinate_frame(ax, origin, R, length=1.0):

    colors = ['r', 'g', 'b']  # x, y, z axes colors
    for i in range(3):
        axis = R[:, i] * length
        ax.quiver(origin[0], origin[1], origin[2],
                  axis[0], axis[1], axis[2], color=colors[i],
                  arrow_length_ratio=0.1)

# Initialize plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Coordinate System with Orientation (Yaw, Pitch, Roll)')

origin = np.array([0, 0, 0])

yaw = np.radians(90)    
pitch = np.radians(60)  
roll = np.radians(60)   

R = rotation_matrix(yaw, pitch, roll)

plot_coordinate_frame(ax, origin, np.eye(3), length=1.0)
plot_coordinate_frame(ax, origin, R, length=1.0)

plt.show()
