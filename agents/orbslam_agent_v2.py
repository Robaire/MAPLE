#!/usr/bin/env python

# THIS AGENT IS INTEGRATING ALL PIECES
# TO INTEGRATE...
# BOULDER MAPPING: DONE
# BIG BOULDER MAPPING/RETURNING:
# LANDER AVOIDING:
# BIG BOULDER AVOIDING:
# SURFACE INTERPOLATION: DONE
# ADDING LANDER FEET TO SURFACE: DONE
# CHANGE INDEXING TO NOT BE BASED ON NUMBERS 13.425

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

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

from maple.pose import OrbslamEstimator

# from maple.pose.ORBSLAM_Interface import ORBSLAMInterface
from maple.utils import *

# from maple.pose.stereoslam import SimpleStereoSLAM
from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
import time
import csv
from maple.utils import carla_to_pytransform
import importlib.resources

""" Import the AutonomousAgent from the Leaderboard. """

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

""" Define the entry point so that the Leaderboard can instantiate the agent class. """


def get_entry_point():
    return "MITAgent"


""" Inherit the AutonomousAgent class. """


class MITAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """This method is executed once by the Leaderboard at mission initialization. We should add any attributes to the class using
        the 'self' Python keyword that contain data or methods we might need throughout the simulation. If you are using machine learning
        models for processing sensor data or control, you should load the models here. We encourage the use of class attributes in place
        of using global variables which can cause conflicts."""

        """ Add some attributes to store values for the target linear and angular velocity. """

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        self.current_v = 0
        self.current_w = 0

        # Initialize the sample list
        self.sample_list = []
        self.ground_truth_sample_list = []

        self._width = 1280
        self._height = 720

        self.good_loc = True

        # self._width = 1920
        # self._height = 1080

        # Store previous boulder detections
        self.previous_detections = []

        self.frame = 1
        self.trial = "orb_04"

        # set the trial number here
        self._active_side_cameras = False
        self._active_side_front_cameras = True

        self.estimator = InertialApriltagEstimator(self)
        self.navigator = Navigator(self)
        self.detector = BoulderDetector(
            self, carla.SensorPosition.FrontLeft, carla.SensorPosition.FrontRight
        )
        self.detectorBack = BoulderDetector(
            self, carla.SensorPosition.BackLeft, carla.SensorPosition.BackRight
        )

        self.g_map_testing = self.get_geometric_map()
        self.map_length_testing = self.g_map_testing.get_cell_number()

        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_height(i, j, 0)
                self.g_map_testing.set_cell_rock(i, j, 0)

        self.all_boulder_detections = []
        self.large_boulder_detections = [(0, 0, 2.5)]

        self.sample_list.extend(sample_lander(self))

        # self.orbslam = SimpleStereoSLAM(self.orb_vocab, self.orb_cams_config)
        self.orbslam = OrbslamEstimator(
            self,
            carla.SensorPosition.FrontLeft,
            carla.SensorPosition.FrontRight,
            mode="stereo_imu",
        )

        self.columns = ["frame", "gt_x", "gt_y", "gt_z", "x", "y", "z"]

        self.positions = []

        self.init_pose = carla_to_pytransform(self.get_initial_position())
        self.prev_pose = None

        # TODO: This should be in the orbslam class
        self.T_orb_to_global = None

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return True

    def sensors(self):
        """In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light."""

        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.Left: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.Right: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.BackRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.Back: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
        }
        return sensors

    def run_step(self, input_data):
        """Execute one step of navigation"""

        sensor_data_frontleft = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        sensor_data_frontright = input_data["Grayscale"][
            carla.SensorPosition.FrontRight
        ]

        # timestamp = time.time()

        if sensor_data_frontleft is not None:
            cv.imshow("Left front camera view", sensor_data_frontleft)
            cv.waitKey(1)

        if self.frame < 50:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))
            estimate = self.init_pose
            self.prev_pose = estimate

        elif (
            sensor_data_frontleft is not None
            and sensor_data_frontright is not None
            and self.frame >= 50
        ):
            print("trying to process frame")
            self.orbslam._estimate_stereo(input_data)
            print("processed frame")
            estimate_orbslamframe = self.orbslam.get_current_pose()
            trajectory_orbslam = self.orbslam.get_trajectory()
            print("estimate in orbslam frame: ", estimate_orbslamframe)

            orbslam_rotated = correct_pose_orientation(estimate_orbslamframe)

            # orbslam_rotated = rotate_pose_in_place(orbslam_rotated, 90, 0, 0)

            # orbslam_rotated = update_rpy(orbslam_rotated, 90, 0, 0)

            # estimate = transform_to_global_frame(orbslam_rotated, self.init_pose)
            estimate = orbslam_rotated

            if self.frame < 60:
                self.T_orb_to_global = self.init_pose @ np.linalg.inv(
                    orbslam_rotated
                )  # if you have rover->cam
                estimate = self.init_pose

            else:
                estimate = self.T_orb_to_global @ estimate_orbslamframe

            estimate = rotate_pose_in_place(estimate, 90, 270, 0)

            # T_global_map = T_global_cam0 @ np.linalg.inv(orbslam_pose_0)

            real_position = carla_to_pytransform(self.get_transform())

            # estimate_2 = np.linalg.inv(estimate) @ real_position

            # print("transform from estimate to true: ", estimate_true_transform)

            plot_poses_and_save(
                trajectory_orbslam,
                estimate_orbslamframe,
                estimate,
                real_position,
                self.frame,
            )

            print("true pose: ", real_position)
            print("orbslam returned transformed pose: ", estimate)

        elif (
            sensor_data_frontleft is None
            and sensor_data_frontright is None
            and self.frame >= 50
        ):
            estimate = self.prev_pose

        real_position = carla_to_pytransform(self.get_transform())

        self.frame += 1

        if self.frame < 100:
            goal_lin_vel = 0.0
            goal_ang_vel = 0.0

        elif self.frame < 200:
            goal_lin_vel = 0.2
            goal_ang_vel = 0.0

        elif self.frame < 300:
            goal_lin_vel = 0.0
            goal_ang_vel = 0.2

        elif self.frame < 400:
            goal_lin_vel = 0.0
            goal_ang_vel = 0.2

        else:
            goal_lin_vel = 0.2
            goal_ang_vel = 0.0

        # Finally, apply the resulting velocities
        control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)
        # This part is gathering info to be used later
        # transform = self.get_transform()
        # transform_location_x = transform.location.x
        # transform_location_y = transform.location.y
        # transform_location_z = transform.location.z

        # initial_transform = self.get_initial_position()

        # print("intial transform: ", initial_transform)

        # adding a bunch of info to save to a csv at the end
        # position_entry = [self.frame] + [
        #     transform_location_x,
        #     transform_location_y,
        #     transform_location_z,
        #     current_position_xyz[0],
        #     current_position_xyz[1],
        #     current_position_xyz[2],
        # ]

        # # Append to self.imu list to save at the end
        # self.positions.append(position_entry)

        return control

    def finalize(self):
        # Save the data to a CSV file
        output_filename_imu = f"/home/annikat/MAPLE/data/{self.trial}/position_data.csv"
        os.makedirs(os.path.dirname(output_filename_imu), exist_ok=True)

        # Write to CSV file
        with open(output_filename_imu, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.columns)  # Write header
            writer.writerows(self.positions)  # Write the IMU data rows

        print(f"Data saved to {output_filename_imu}")

        cv.destroyAllWindows()
        min_det_threshold = 2

        if self.frame > 15000:
            min_det_threshold = 3

        if self.frame > 35000:
            min_det_threshold = 5

        g_map = self.get_geometric_map()
        gt_map_array = g_map.get_map_array()

        N = gt_map_array.shape[
            0
        ]  # should be 179 if you are spanning -13.425 to 13.425 by 0.15
        x_min, y_min = gt_map_array[0][0][0], gt_map_array[0][0][0]
        resolution = 0.15

        # Calculate indices for center 2x2m region
        center_x_min_idx = int(round((-1 - x_min) / resolution))  # -.5m in x
        center_x_max_idx = int(round((1 - x_min) / resolution))  # +.5m in x
        center_y_min_idx = int(round((-1 - y_min) / resolution))  # -.5m in y
        center_y_max_idx = int(round((1 - y_min) / resolution))  # +.5m in y

        # setting all rock locations to 0
        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_rock(i, j, 0)

        clusters = defaultdict(list)
        filtered_detections = []

        # First pass: create clusters
        for x_rock, y_rock in self.all_boulder_detections:
            # Convert to grid coordinates
            i = int(round((x_rock - x_min) / resolution))
            j = int(round((y_rock - y_min) / resolution))

            # Create cluster key based on grid cell
            cluster_key = (i, j)
            clusters[cluster_key].append([x_rock, y_rock])

        final_clusters = []

        # Second pass: process clusters and filter outliers
        for (i, j), detections in clusters.items():
            # Skip clusters with less than 2 detections
            if len(detections) < min_det_threshold:
                continue

            final_clusters.extend(clusters[(i, j)])

            # Skip if in center region
            if (
                center_x_min_idx <= i <= center_x_max_idx
                and center_y_min_idx <= j <= center_y_max_idx
            ):
                continue

            # Sanity check: make sure we are within bounds
            if 0 <= i < N and 0 <= j < N:
                # Calculate cluster center
                x_center = float(np.mean([x for x, y in detections]))
                y_center = float(np.mean([y for x, y in detections]))

                # Convert back to grid coordinates for the map
                i_center = int(round((x_center - x_min) / resolution))
                j_center = int(round((y_center - y_min) / resolution))

                # Set rock location at cluster center
                self.g_map_testing.set_cell_rock(i_center, j_center, 1)

                # Store the cluster center as a simple list
                filtered_detections.append([x_center, y_center])

        # Initialize the data class to get estimates for all the squares
        surfaceHeight = SurfaceHeight(g_map)

        # Generate the actual map with the sample list
        if len(self.sample_list) > 0:
            surfaceHeight.set_map(self.sample_list)

    def on_press(self, key):
        """This is the callback executed when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear velocity. If the key pressed is either the left or right arrow, this method will set a target angular
        velocity of 0.6 radians per second."""

        if key == keyboard.Key.up:
            self.current_v += 0.1
            self.current_v = np.clip(self.current_v, 0, 0.3)
        if key == keyboard.Key.down:
            self.current_v -= 0.1
            self.current_v = np.clip(self.current_v, -0.3, 0)
        if key == keyboard.Key.left:
            self.current_w = 0.6
        if key == keyboard.Key.right:
            self.current_w = -0.6

    def on_release(self, key):
        """This method sets the angular or linear velocity to zero when the arrow key is released. Stopping the robot."""

        if key == keyboard.Key.up:
            self.current_v = 0
        if key == keyboard.Key.down:
            self.current_v = 0
        if key == keyboard.Key.left:
            self.current_w = 0
        if key == keyboard.Key.right:
            self.current_w = 0

        """ Press escape to end the mission. """
        if key == keyboard.Key.esc:
            self.mission_complete()
            cv.destroyAllWindows()


import numpy as np
import matplotlib

matplotlib.use("Agg")  # So it doesn't open a window
import matplotlib.pyplot as plt


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

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

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
