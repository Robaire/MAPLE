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
import cv2
from math import hypot
import carla
from pynput import keyboard
import os
import numpy as np
import pytransform3d.rotations as pyrot
import cv2 as cv
from collections import defaultdict
from math import radians
from pytransform3d.transformations import concat, invert_transform
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
from maple.utils import carla_to_pytransform, extract_rock_locations
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
        self.previous_detections = []

        self.frame = 1
        self.trial = "orb_05"
        self.orb = cv2.ORB_create()

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
        self.orbslam_front = OrbslamEstimator(
            self,
            carla.SensorPosition.FrontLeft,
            carla.SensorPosition.FrontRight,
            mode="stereo",
        )

        self.orbslam_back = OrbslamEstimator(
            self,
            carla.SensorPosition.BackLeft,
            carla.SensorPosition.BackRight,
            mode="stereo",
        )

        self.DRIVE_BACKWARDS_FRAME = -1000000000000000

        self.columns = ["frame", "gt_x", "gt_y", "gt_z", "x", "y", "z"]

        self.positions = []

        self.init_pose = carla_to_pytransform(self.get_initial_position())
        self.prev_pose_front = None
        self.prev_pose_back = None

        # TODO: This should be in the orbslam class
        self.T_orb_to_global_front = None
        self.T_orb_to_global_back = None
        self.T_world_correction_front = None
        self.T_world_correction_back = None

        self.USE_FRONT_CAM = False
        self.USE_BACK_CAM = True

        self.prev_goal_location = None
        self.frames_since_goal_update = 0
        self.no_update_threshold = 3000

        # Tiered stuck detection parameters
        self.position_history = []
        self.is_stuck = False
        self.MILD_STUCK_FRAMES = 2000
        self.MILD_STUCK_THRESHOLD = 2.0  # If moved less than 3m in 2000 frames

        self.UNSTUCK_DISTANCE_THRESHOLD = (
            2.0  # How far to move to be considered unstuck
        )

        self.unstuck_sequence = [
            {"lin_vel": -0.45, "ang_vel": 0, "frames": 200},  # Backward
            {"lin_vel": 0, "ang_vel": 4, "frames": 100},  # Rotate clockwise
            {"lin_vel": 0.45, "ang_vel": 0, "frames": 200},  # Forward
            {"lin_vel": 0, "ang_vel": -4, "frames": 100},  # Rotate counter-clockwise
        ]

        self.gt_rock_locations = extract_rock_locations(
            "simulator/LAC/Content/Carla/Config/Presets/Preset_1.xml"
        )

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return False

    def sensors(self):
        """In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light."""

        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": False,
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
                "camera_active": False,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
        }
        return sensors
    

    def check_if_stuck(self, current_position):
        """
        Check if the rover is stuck using a tiered approach:
        1. Severe stuck: very little movement in a short period
        2. Mild stuck: limited movement over a longer period
        Returns True if stuck, False otherwise.

        Only performs the check every 10 frames to improve performance.
        """
        if current_position is None:
            return False

        # Add current position to history
        self.position_history.append(current_position)

        # Keep only enough positions for the longer threshold check
        if len(self.position_history) > self.MILD_STUCK_FRAMES:
            self.position_history.pop(0)

        # Only perform stuck detection every 10 frames to improve performance
        if self.frame % 10 != 0:
            return False

        # Check for mild stuck condition (longer timeframe)
        if len(self.position_history) >= self.MILD_STUCK_FRAMES:
            mild_check_position = self.position_history[0]  # Oldest position
            dx = current_position[0] - mild_check_position[0]
            dy = current_position[1] - mild_check_position[1]
            mild_distance_moved = np.sqrt(dx**2 + dy**2)

            if mild_distance_moved < self.MILD_STUCK_THRESHOLD:
                print(
                    f"MILD STUCK DETECTED! Moved only {mild_distance_moved:.2f}m in the last {self.MILD_STUCK_FRAMES} frames."
                )
                return True

        return False

    def run_step(self, input_data):
        """Execute one step of navigation"""
        print("Frame: ", self.frame)

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
            estimate_back = self.init_pose
            self.prev_pose_front = estimate
            self.prev_pose_back = estimate_back

        elif (
            sensor_data_frontleft is not None
            and sensor_data_frontright is not None
            and self.frame >= 50
        ):
            self.orbslam_front._estimate_stereo(input_data)
            self.orbslam_back._estimate_stereo(input_data)
            estimate_orbslamframe_front = self.orbslam_front.get_current_pose()
            estimate_orbslamframe_back = self.orbslam_back.get_current_pose()

            if estimate_orbslamframe_front is None:
                print("orbslam frame in front is none! Driving backwards")
                goal_lin_vel = -0.3
                goal_ang_vel = 0.0
                control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)
                self.frame += 1
                return control
            if estimate_orbslamframe_back is None:
                print("orbslam frame in back is none! Driving backwards")
                goal_lin_vel = -0.3
                goal_ang_vel = 0.0
                control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)
                self.frame += 1
                return control

            orbslam_reset_pose = np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
            )

            orbslam_rotated = correct_pose_orientation(estimate_orbslamframe_front)
            orbslam_rotated_back = correct_pose_orientation_back(
                estimate_orbslamframe_back
            )
            estimate = orbslam_rotated
            estimate_back = orbslam_rotated_back

            if self.frame < 60:
                self.T_orb_to_global_front = self.init_pose @ np.linalg.inv(
                    orbslam_rotated
                )
                self.T_orb_to_global_back = self.init_pose @ np.linalg.inv(
                    orbslam_rotated_back
                )
                estimate = self.init_pose
                estimate_back = self.init_pose

            # elif self.frame == 400:
            elif self.frame > 200 and np.allclose(
                estimate_orbslamframe_front.astype(float),
                orbslam_reset_pose,
                atol=0.001,
            ):
                print("ORBSLAM FRONT FAILED STILL NEED TO REIMPLEMENT FRONT")
                print("ORBLSAM FRONT FAILED SO DOING BACK CAMERAS")
                self.USE_FRONT_CAM = False
                self.USE_BACK_CAM = True
                self.DRIVE_BACKWARDS_FRAME = self.frame
                self.frame += 1
                return carla.VehicleVelocityControl(0.0, 0.0)

            elif self.frame > 200 and np.allclose(
                estimate_orbslamframe_back.astype(float), orbslam_reset_pose, atol=0.001
            ):
                # print("resetting transform since orbslam restarted!! THIS IS BAD AND DOES NOT FOR POSES BUT KEEPS IT RUNNING")
                print("ORBSLAM BACK FAILED STILL NEED TO REIMPLEMENT FRONT")
                self.mission_complete()
                return carla.VehicleVelocityControl(0.0, 0.0)

            elif (self.frame - self.DRIVE_BACKWARDS_FRAME) < 200:
                print(
                    "testing driiving backwards to overcome visual issues Driving backwards"
                )
                goal_lin_vel = -0.3
                goal_ang_vel = 0.0
                control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)
                self.frame += 1
                return control

            else:
                estimate = self.T_orb_to_global_front @ estimate_orbslamframe_front
                estimate_back = self.T_orb_to_global_back @ estimate_orbslamframe_back
                estimate = rotate_pose_in_place(estimate, 90, 270, 0)
                estimate_back = rotate_pose_in_place(estimate_back, 90, 270, 0)

            camera_world_back = estimate_back
            camera_world = estimate
            camera_rover = carla_to_pytransform(
                self.get_camera_position(carla.SensorPosition.FrontLeft)
            )
            camera_rover_back = carla_to_pytransform(
                self.get_camera_position(carla.SensorPosition.BackLeft)
            )

            rover_camera = invert_transform(camera_rover)
            rover_camera_back = invert_transform(camera_rover_back)

            rover_world = concat(rover_camera, camera_world)
            rover_world_back = concat(rover_camera_back, camera_world_back)

            estimate = rover_world
            estimate_back = rover_world_back

            self.prev_pose_front = estimate
            self.prev_pose_back = estimate_back

        elif (
            sensor_data_frontleft is None
            and sensor_data_frontright is None
            and self.frame >= 50
        ):
            estimate = self.prev_pose_front
            estimate_back = self.prev_pose_back

        if self.frame == 65:
            self.T_world_correction_front = self.init_pose @ np.linalg.inv(
                estimate
            )  # if you have rover->cam
            self.T_world_correction_back = self.init_pose @ np.linalg.inv(estimate_back)

        estimate_back_vis = estimate_back
        estimate_vis = estimate

        if self.USE_BACK_CAM and not self.USE_FRONT_CAM:
            estimate = estimate_back
            correction_T = self.T_world_correction_back
        elif self.USE_FRONT_CAM and not self.USE_BACK_CAM:
            estimate = estimate
            correction_T = self.T_world_correction_front

        # real_position = carla_to_pytransform(self.get_transform())
        real_position = None

        goal_location = self.navigator.goal_loc

        # This whole block of code is so the mission ends when we get stuck rather than continue giving bad measurements
        if self.prev_goal_location != goal_location:
            self.frames_since_goal_update = 0
            self.prev_goal_location = goal_location
        else: 
            self.frames_since_goal_update += 1

        if self.frames_since_goal_update >= self.no_update_threshold:
            print("finishing because it didn't reach the goal in time :(")
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        all_goals = self.navigator.static_path.get_full_path()
        # nearby_goals = self.navigator.static_path.find_nearby_goals([estimate[0,3], estimate[1,3]])
        nearby_goals = []

        if self.frame % 20 == 0 and self.frame > 65:
            # print("attempting detections at frame ", self.frame)

            detections, ground_points = self.detector(input_data)

            large_boulders_detections = self.detector.get_large_boulders()

            detections_back, _ = self.detectorBack(input_data)

            # Get all detections in the world frame
            rover_world = estimate
            boulders_world = [
                concat(boulder_rover, rover_world) for boulder_rover in detections
            ]

            ground_points_world = [
                concat(ground_point, rover_world) for ground_point in ground_points
            ]

            boulders_world_back = [
                concat(boulder_rover, rover_world) for boulder_rover in detections_back
            ]

            large_boulders_detections = [
                concat(boulder_rover, rover_world)
                for boulder_rover in large_boulders_detections
            ]

            large_boulders_xyr = [
                (b_w[0, 3], b_w[1, 3], 0.3) for b_w in large_boulders_detections
            ]

            nearby_large_boulders = []
            for large_boulder in large_boulders_xyr:
                print("large boulder: ", large_boulder)
                (bx, by, _) = large_boulder  # assuming large_boulder is (x, y)

                distance = hypot(bx - estimate[0, 3], by - estimate[1, 3])

                if distance <= 2.0:
                    nearby_large_boulders.append(large_boulder)
            print("large boulders ", nearby_large_boulders)

            # Now pass the (x, y, r) tuples to your navigator or wherever they need to go
            if len(nearby_large_boulders) > 0:
                # self.navigator.add_large_boulder_detection(nearby_large_boulders)
                self.large_boulder_detections.extend(nearby_large_boulders)

            # If you just want X, Y coordinates as a tuple
            boulders_xyz = [(b_w[0, 3], b_w[1, 3], b_w[2, 3]) for b_w in boulders_world]
            boulders_xyz_back = [
                (b_w[0, 3], b_w[1, 3], b_w[2, 3]) for b_w in boulders_world_back
            ]

            ground_points_xyz = [
                (b_w[0, 3], b_w[1, 3], b_w[2, 3]) for b_w in ground_points_world
            ]


            if len(boulders_xyz) > 0:
                boulders_world_corrected = transform_points(boulders_xyz, correction_T)
                self.all_boulder_detections.extend(boulders_world_corrected[:, :2])
                # print("len(boulders)", len(self.all_boulder_detections))

            if len(boulders_xyz_back) > 0:
                boulders_world_back_corrected = transform_points(
                    boulders_xyz_back, correction_T
                )
                self.all_boulder_detections.extend(boulders_world_back_corrected[:, :2])

            if len(ground_points_xyz) > 0:
                ground_points_xyz_corrected = transform_points(
                    ground_points_xyz, correction_T
                )
                self.sample_list.extend(ground_points_xyz_corrected)

        if self.frame > 80:
            goal_lin_vel, goal_ang_vel = self.navigator(estimate, input_data)
        else:
            goal_lin_vel, goal_ang_vel = 0.0, 0.0

        if self.frame % 20 == 0 and self.frame > 80:
            surface_points_uncorrected = sample_surface(estimate, 60)
            surface_points_corrected = transform_points(
                surface_points_uncorrected, correction_T
            )
            # self.sample_list.extend(surface_points_corrected)

            # surface_points_corrected is assumed to be a (N, 3) array or list of (x, y, z) points
            surface_points_corrected = np.asarray(surface_points_corrected)

            # Compute distance from origin in (x, y) plane
            xy = surface_points_corrected[:, :2]  # take x and y columns
            distances = np.linalg.norm(xy, axis=1)  # Euclidean distance

            # Mask points farther than 2 meters
            mask = distances > 1.5
            filtered_points = surface_points_corrected[mask]

            # Only extend the sample list with filtered points
            self.sample_list.extend(filtered_points)

        
        current_position = (
            (estimate[0, 3], estimate[1, 3]) if estimate is not None else None
        )

        if current_position is not None:
            # Always update position history
            self.position_history.append(current_position)

            # Keep only enough positions for the longer threshold check
            if len(self.position_history) > self.MILD_STUCK_FRAMES:
                self.position_history.pop(0)

            # Only check if stuck every 10 frames for performance
            if not self.is_stuck and self.frame % 10 == 0:
                self.is_stuck = self.check_if_stuck(current_position)
            elif self.is_stuck:
                # Check if we've moved enough to consider ourselves unstuck
                if len(self.position_history) > 0:
                    old_position = self.position_history[0]
                    dx = current_position[0] - old_position[0]
                    dy = current_position[1] - old_position[1]
                    distance_moved = np.sqrt(dx**2 + dy**2)

                    if distance_moved > self.UNSTUCK_DISTANCE_THRESHOLD:
                        print(
                            f"UNSTUCK! Moved {distance_moved:.2f}m - resuming normal operation."
                        )
                        # self.navigator.global_path_index_tracker = (
                        #     self.navigator.global_path_index_tracker + 1
                        # ) % len(self.navigator.global_path)
                        self.is_stuck = False
                        self.unstuck_phase = 0
                        self.unstuck_counter = 0
                        # Clear position history to reset stuck detection
                        self.position_history = []

        if self.frame >= 20000:
            self.mission_complete()

        control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)

        self.frame += 1

        return control

    def finalize(self):
        cv.destroyAllWindows()
        min_det_threshold = 2

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


def correct_pose_orientation_back(pose):
    # Assuming pose is a 4x4 transformation matrix
    # Extract the rotation and translation components
    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    # Create a rotation correction matrix for back camera
    # Accounting for the camera being mounted in the opposite direction
    import numpy as np

    correction = np.array(
        [
            [0, 0, -1],  # New x comes from negative old z (forward for back camera)
            [-1, 0, 0],  # New y comes from negative old x (right for back camera)
            [0, 1, 0],  # New z comes from old y (up)
        ]
    )

    # Apply the correction to the rotation part only
    corrected_rotation = np.dot(correction, rotation)

    # Reconstruct the transformation matrix
    corrected_pose = np.eye(4)
    corrected_pose[:3, :3] = corrected_rotation
    corrected_pose[:3, 3] = translation

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
    # Defensive checks
    if points_xyz is None:
        print("[transform_points] Warning: points_xyz is None")
        return np.empty((0, 3))

    points_xyz = np.asarray(points_xyz)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        print(f"[transform_points] Invalid shape: {points_xyz.shape}")
        return np.empty((0, 3))
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
