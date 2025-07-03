import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets

# Function to compute the positions of the joints and end-effector
def compute_positions(link_lengths, joint_angles):
    positions = [np.array([0, 0])]  # Start at the origin
    total_angle = 0

    for link_length, joint_angle in zip(link_lengths, joint_angles):
        total_angle += joint_angle
        new_position = positions[-1] + link_length * np.array([np.cos(total_angle), np.sin(total_angle)])
        positions.append(new_position)

    return positions

# Function to plot the arm
def plot_arm(joint_angles):
    link_lengths = [1, 1]  # Lengths of the robot arm segments
    positions = compute_positions(link_lengths, joint_angles)

    # Extract x and y coordinates for plotting
    x_positions = [pos[0] for pos in positions]
    y_positions = [pos[1] for pos in positions]

    plt.figure(figsize=(6, 6))
    plt.plot(x_positions, y_positions, '-o', markersize=10, linewidth=5)  # Plot arm
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True)
    plt.title("2D Robotic Arm Simulation")
    plt.show()

# Create interactive sliders for the joint angles
interact(plot_arm, joint_angles=widgets.fixed([0, 0]),
         joint1=widgets.FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=0),
         joint2=widgets.FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=0))
