

from math import radians
import numpy as np
import pytransform3d.rotations as pyrot
from collections import defaultdict

import carla
from pytransform3d.transformations import concat
import orbslam3
import copy
import carla
from pynput import keyboard
import os
import numpy as np
import pytransform3d.rotations as pyrot
import cv2 as cv
from collections import defaultdict
from math import radians
from pytransform3d.transformations import concat
from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.pose import InertialApriltagEstimator
from maple.utils import *
from maple.pose.stereoslam import SimpleStereoSLAM
from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
import time
import csv
from maple.utils import carla_to_pytransform
from maple.pose.orbslam_utils import *
from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.pose import InertialApriltagEstimator
from maple.utils import *
from maple.surface.map import SurfaceHeight, sample_surface, sample_lander

""" Import the AutonomousAgent from the Leaderboard. """

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

def plot_poses_and_save(
    trajectory,
    orbslam_pose: np.ndarray,
    transformed_estimate: np.ndarray,
    real_pose: np.ndarray,
    frame_number: int,
    arrow_length: float = 0.5,
):
    """
    Plots and saves a 3D visualization of:
      - The raw ORB-SLAM pose (orbslam_pose)
      - The transformed/adjusted pose (transformed_estimate)
      - The real pose (real_pose)
    Each pose is shown as a point for the position plus three quiver arrows
    indicating its local x/y/z axes in red/green/blue.

    The axes are fixed from -6 to +6 in all directions.

    The figure is saved to 'pose_plot_{frame_number}.png'.

    :param orbslam_pose: 4x4 numpy array for ORB-SLAM pose.
    :param transformed_estimate: 4x4 numpy array for the corrected/adjusted pose.
    :param real_pose: 4x4 numpy array for the real/global pose (if available).
    :param frame_number: used to save the figure as 'pose_plot_{frame_number}.png'.
    :param arrow_length: length of each axis arrow (default 0.5).
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    def draw_pose(ax, T, label_prefix="Pose"):
        """
        Draw the coordinate axes for the transform T (4x4).
        We'll plot a small arrow for each local axis (x, y, z).
        Red for x, green for y, blue for z.
        """
        # Origin of this pose:
        origin = T[:3, 3]
        # Rotation part:
        R = T[:3, :3]

        # Local axes directions in world coords
        x_axis = R @ np.array([1, 0, 0]) * arrow_length
        y_axis = R @ np.array([0, 1, 0]) * arrow_length
        z_axis = R @ np.array([0, 0, 1]) * arrow_length

        # Plot the origin as a single point
        ax.scatter(origin[0], origin[1], origin[2])

        # Draw x, y, z quiver arrows in red, green, blue
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="red",
            label="_nolegend_",
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="green",
            label="_nolegend_",
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="blue",
            label="_nolegend_",
        )

        # Label near the origin
        ax.text(origin[0], origin[1], origin[2], label_prefix, size=8)

    # Draw the three poses (if they exist)
    if orbslam_pose is not None:
        draw_pose(ax, orbslam_pose, label_prefix="ORB-SLAM raw")
    if transformed_estimate is not None:
        draw_pose(ax, transformed_estimate, label_prefix="Transformed")
    if real_pose is not None:
        draw_pose(ax, real_pose, label_prefix="Real")

    # Set axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Fix the axes to range [-6, 6] in each dimension
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-6, 6)

    if not trajectory:
        print("No trajectory data to plot.")
        return

    positions = [pose[:3, 3] for pose in trajectory]  # Extract translation part
    positions = np.array(positions)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        positions[:, 0], positions[:, 1], positions[:, 2], label="Camera Trajectory"
    )
    ax.scatter(
        positions[0, 0], positions[0, 1], positions[0, 2], c="green", label="Start"
    )
    ax.scatter(
        positions[-1, 0], positions[-1, 1], positions[-1, 2], c="red", label="End"
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.set_title("ORB-SLAM3 Camera Trajectory")

    # Save by frame_number
    plt.savefig(f"/home/annikat/LAC_data/axis_vis/pose_plot_{frame_number}.png")
    plt.close(fig)


def correct_pose_orientation(pose):
    # Assuming pose is a 4x4 transformation matrix
    # Extract the rotation and translation components
    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    # Create a rotation correction matrix
    # To get: x-forward, y-left, z-up
    import numpy as np

    correction = np.array(
        [
            [0, 0, 1],  # New x comes from old z (forward)
            [1, 0, 0],  # New y comes from old x (left)
            [0, 1, 0],  # New z comes from old y (up)
        ]
    )

    # Apply the correction to the rotation part only
    corrected_rotation = np.dot(correction, rotation)

    # Reconstruct the transformation matrix
    corrected_pose = np.eye(4)
    corrected_pose[:3, :3] = corrected_rotation
    corrected_pose[:3, 3] = translation

    # change just rotation 

    return corrected_pose

def rotate_pose_in_place(pose_matrix, roll_deg=0, pitch_deg=0, yaw_deg=0):
    """
    Apply a local RPY rotation on the rotation part of the pose, keeping translation fixed.
    """
    import numpy as np

    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Compose rotation in local frame
    delta_R = Rz @ Ry @ Rx

    R_old = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]

    # Apply in local frame (right multiplication)
    R_new = R_old @ delta_R

    new_pose = np.eye(4)
    new_pose[:3, :3] = R_new
    new_pose[:3, 3] = t
    return new_pose


def transform_to_global_frame(local_pose, initial_global_pose):
    """
    Transform a pose from local frame to global frame.

    Parameters:
    - local_pose: 4x4 transformation matrix in local frame
    - initial_global_pose: 4x4 transformation matrix of the initial pose in global frame

    Returns:
    - global_pose: 4x4 transformation matrix in global frame
    """
    import numpy as np

    # First correct the orientation of the local pose
    corrected_local_pose = correct_pose_orientation(local_pose)

    # Transform to global frame by multiplying with the initial global pose
    global_pose = np.dot(initial_global_pose, corrected_local_pose)

    return global_pose
