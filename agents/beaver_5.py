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

import carla
import numpy as np
from collections import defaultdict
from math import radians
from pytransform3d.transformations import concat, invert_transform
from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.pose import OrbslamEstimator

from maple.utils import *

from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
from maple.utils import carla_to_pytransform

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

        self.current_v = 0
        self.current_w = 0

        # Initialize the sample list
        self.sample_list = []
        self.ground_truth_sample_list = []

        self._width = 1280
        self._height = 720

        self.good_loc = True
        self.previous_detections = []

        self.frame = 1

        # set the trial number here
        self._active_side_cameras = False
        self._active_side_front_cameras = True

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
            mode="stereo",
        )

        self.positions = []

        self.init_pose = carla_to_pytransform(self.get_initial_position())
        self.prev_pose = None

        # TODO: This should be in the orbslam class
        self.T_orb_to_global = None
        self.T_world_correction = None

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return False

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
        try:
            return self.run_step_unsafe(input_data)
        except Exception as e:
            print(f"Error in run_step: {e}")
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

    def run_step_unsafe(self, input_data):
        """Execute one step of navigation"""

        sensor_data_frontleft = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        sensor_data_frontleft = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        sensor_data_frontright = input_data["Grayscale"][
            carla.SensorPosition.FrontRight
        ]

        camera_rover = carla_to_pytransform(
            self.get_camera_position(carla.SensorPosition.FrontLeft)
        )
        rover_camera = invert_transform(camera_rover)

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
            self.orbslam._estimate_stereo(input_data)
            estimate_orbslamframe = self.orbslam.get_current_pose()

            if estimate_orbslamframe is None:
                print("orbslam frame is none! Driving backwards")
                goal_lin_vel = -0.3
                goal_ang_vel = 0.0
                control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)
                self.frame += 1
                return control

            orbslam_rotated = correct_pose_orientation(estimate_orbslamframe)
            estimate = orbslam_rotated

            if self.frame < 60:
                self.T_orb_to_global = self.init_pose @ np.linalg.inv(
                    orbslam_rotated
                )  # if you have rover->cam
                estimate = self.init_pose

            else:
                estimate = self.T_orb_to_global @ estimate_orbslamframe
                estimate = rotate_pose_in_place(estimate, 90, 270, 0)

            camera_world = estimate
            camera_rover = carla_to_pytransform(
                self.get_camera_position(carla.SensorPosition.FrontLeft)
            )

            rover_camera = invert_transform(camera_rover)

            rover_world = concat(rover_camera, camera_world)
            estimate = rover_world
            self.prev_pose = estimate

        elif (
            sensor_data_frontleft is None
            and sensor_data_frontright is None
            and self.frame >= 50
        ):
            estimate = self.prev_pose

        if self.frame == 65:
            self.T_world_correction = self.init_pose @ np.linalg.inv(
                estimate
            )  # if you have rover->cam

        goal_location = self.navigator.goal_loc

        print("goal location: ", goal_location)

        if self.frame % 20 == 0 and self.frame > 65:
            try:
                print("attempting detections at frame ", self.frame)

                # This probably failed to unpack?
                detections, _ = self.detector(input_data)

                large_boulders_detections = self.detector.get_large_boulders()

                # Or this?
                detections_back, _ = self.detectorBack(input_data)

                # Get all detections in the world frame
                rover_world = estimate
                boulders_world = [
                    concat(boulder_rover, rover_world) for boulder_rover in detections
                ]

                boulders_world_back = [
                    concat(boulder_rover, rover_world)
                    for boulder_rover in detections_back
                ]

                large_boulders_detections = [
                    concat(boulder_rover, rover_world)
                    for boulder_rover in large_boulders_detections
                ]

                large_boulders_xyr = [
                    (b_w[0, 3], b_w[1, 3], 0.25) for b_w in large_boulders_detections
                ]
                print("large boulders ", large_boulders_xyr)

                # Now pass the (x, y, r) tuples to your navigator or wherever they need to go
                if len(large_boulders_xyr) > 0:
                    self.navigator.add_large_boulder_detection(large_boulders_xyr)
                    self.large_boulder_detections.extend(large_boulders_xyr)

                # If you just want X, Y coordinates as a tuple
                boulders_xyz = [
                    (b_w[0, 3], b_w[1, 3], b_w[2, 3]) for b_w in boulders_world
                ]
                boulders_xyz_back = [
                    (b_w[0, 3], b_w[1, 3], b_w[2, 3]) for b_w in boulders_world_back
                ]

                if len(boulders_xyz) > 0:
                    boulders_world_corrected = transform_points(
                        boulders_xyz, self.T_world_correction
                    )
                    self.all_boulder_detections.extend(boulders_world_corrected[:, :2])
                    print("len(boulders)", len(self.all_boulder_detections))

                if len(boulders_xyz_back) > 0:
                    boulders_world_back_corrected = transform_points(
                        boulders_xyz_back, self.T_world_correction
                    )
                    self.all_boulder_detections.extend(
                        boulders_world_back_corrected[:, :2]
                    )

            except Exception as e:
                print("error in detections: ", e)

        if self.frame > 80:
            goal_lin_vel, goal_ang_vel = self.navigator(estimate)
        else:
            goal_lin_vel, goal_ang_vel = 0.0, 0.0

        if self.frame % 10 == 0 and self.frame > 80:
            surface_points_uncorrected = sample_surface(estimate, 60)
            surface_points_corrected = transform_points(
                surface_points_uncorrected, self.T_world_correction
            )
            self.sample_list.extend(surface_points_corrected)

        control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)

        self.frame += 1

        return control

    def finalize(self):
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


def correct_pose_orientation(pose):
    # Assuming pose is a 4x4 transformation matrix
    # Extract the rotation and translation components
    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    # Create a rotation correction matrix
    # To get: x-forward, y-left, z-up

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


def transform_points(points_xyz, transform):
    """
    Apply a 4x4 transformation to a list or array of 3D points,
    with detailed debugging output if the input isn't as expected.
    """
    print("\n[transform_points] Starting transformation.")
    print(f"Original input type: {type(points_xyz)}")

    points_xyz = np.asarray(points_xyz)
    print(
        f"Converted to np.ndarray with shape: {points_xyz.shape}, dtype: {points_xyz.dtype}"
    )

    # Check the shape carefully
    # if points_xyz.ndim == 1:
    #     # print("[transform_points] Input is 1D (single point?), reshaping to (1,3).")
    #     points_xyz = points_xyz[None, :]
    # elif points_xyz.ndim == 2:
    #     # print("[transform_points] Input is 2D (probably correct). Shape:", points_xyz.shape)
    # elif points_xyz.ndim == 3:
    #     # print(f"[transform_points] Input is 3D with shape {points_xyz.shape}.")
    #     if points_xyz.shape[1] == 1 and points_xyz.shape[2] == 3:
    #         # print("[transform_points] (N,1,3) format detected. Squeezing dimension 1.")
    #         points_xyz = points_xyz.squeeze(1)
    #     else:
    #         # print("[transform_points] Unexpected 3D shape! Cannot continue.")
    #         raise ValueError(f"Expected (N,1,3) if 3D, got {points_xyz.shape}")
    # else:
    #     # print(f"[transform_points] Unexpected number of dimensions: {points_xyz.ndim}")
    #     raise ValueError(f"Input points must be (N,3) or (N,1,3) or (3,), got shape {points_xyz.shape}")

    # Final check
    if points_xyz.shape[1] != 3:
        raise ValueError(
            f"[transform_points] After processing, points must have shape (N,3). Got {points_xyz.shape}."
        )

    # Continue with transformation
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    points_homogeneous = np.hstack((points_xyz, ones))  # (N, 4)

    # print(f"[transform_points] Built homogeneous points with shape: {points_homogeneous.shape}")

    points_transformed_homogeneous = (transform @ points_homogeneous.T).T  # (N, 4)
    points_transformed = points_transformed_homogeneous[:, :3]

    # print(f"[transform_points] Finished transformation. Output shape: {points_transformed.shape}\n")

    return points_transformed


def rotate_pose_in_place(pose_matrix, roll_deg=0, pitch_deg=0, yaw_deg=0):
    """
    Apply a local RPY rotation on the rotation part of the pose, keeping translation fixed.
    """

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

    # First correct the orientation of the local pose
    corrected_local_pose = correct_pose_orientation(local_pose)

    # Transform to global frame by multiplying with the initial global pose
    global_pose = np.dot(initial_global_pose, corrected_local_pose)

    return global_pose
