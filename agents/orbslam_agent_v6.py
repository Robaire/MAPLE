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

    def run_step(self, input_data):
        """Execute one step of navigation"""
        print("Frame: ", self.frame)

        sensor_data_frontleft = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        sensor_data_frontright = input_data["Grayscale"][carla.SensorPosition.FrontRight]

        # sensor_data_backleft = input_data["Grayscale"][carla.SensorPosition.BackLeft]
        # sensor_data_left = input_data["Grayscale"][carla.SensorPosition.Left]
        # sensor_data_right = input_data["Grayscale"][carla.SensorPosition.Right]

        # if sensor_data_frontleft is not None:
        #     keypoints_frontleft = self.orb.detect(sensor_data_frontleft, None)
        #     if keypoints_frontleft:
        #         responses = [kp.response for kp in keypoints_frontleft]
        #         avg_response = np.mean(responses)
        #         frontleft_score = avg_response
        #         print(f"{frontleft_score}: {len(keypoints_frontleft)} keypoints, avg response = {avg_response:.4f}")

        #     keypoints_backleft = self.orb.detect(sensor_data_backleft, None)
        #     if keypoints_backleft:
        #         responses = [kp.response for kp in keypoints_backleft]
        #         avg_response = np.mean(responses)
        #         backleft_score = avg_response
        #         print(f"{backleft_score}: {len(keypoints_backleft)} keypoints, avg response = {avg_response:.4f}")

        #     keypoints_left = self.orb.detect(sensor_data_left, None)
        #     if keypoints_left:
        #         responses = [kp.response for kp in keypoints_left]
        #         avg_response = np.mean(responses)
        #         left_score = avg_response
        #         print(f"{left_score}: {len(keypoints_left)} keypoints, avg response = {avg_response:.4f}")

        #     keypoints_right = self.orb.detect(sensor_data_right, None)
        #     if keypoints_right:
        #         responses = [kp.response for kp in keypoints_right]
        #         avg_response = np.mean(responses)
        #         right_score = avg_response
        #         print(f"{right_score}: {len(keypoints_right)} keypoints, avg response = {avg_response:.4f}")

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

            # kps_front = self.orbslam_front.slam.get_tracked_kp_qty()
            # kps_back = self.orbslam_back.slam.get_tracked_kp_qty()

            # print("keypoints in front frame: ", kps_front)
            # print("keypoints in back frame: ", kps_back)

            if self.frame < 60:
                self.T_orb_to_global_front = self.init_pose @ np.linalg.inv(
                    orbslam_rotated
                )
                self.T_orb_to_global_back = self.init_pose @ np.linalg.inv(
                    orbslam_rotated_back
                )
                estimate = self.init_pose
                estimate_back = self.init_pose
            elif self.frame > 200 and np.allclose(
                estimate_orbslamframe_front.astype(float),
                orbslam_reset_pose,
                atol=0.001,
            ):
                # print("resetting transform since orbslam restarted!! THIS IS BAD AND DOES NOT FOR POSES BUT KEEPS IT RUNNING")
                # TODO: Update this to reset orbslam with tranfsormed initial position from other orbslam output
                print("ORBLSAM FRONT FAILED SO DOING BACK CAMERAS")
                self.USE_FRONT_CAM = False
                self.USE_BACK_CAM = True
                self.DRIVE_BACKWARDS_FRAME = self.frame

                # TODO: Add logic to restart orbslam in 50 frames...
                # self.T_orb_to_global_front = self.prev_pose_front @ np.linalg.inv(
                #     orbslam_rotated
                # )
                # estimate = self.T_orb_to_global_front @ estimate_orbslamframe_front
                # estimate = rotate_pose_in_place(estimate, 90, 270, 0)
            elif self.frame > 200 and np.allclose(
                estimate_orbslamframe_back.astype(float), orbslam_reset_pose, atol=0.001
            ):
                # print("resetting transform since orbslam restarted!! THIS IS BAD AND DOES NOT FOR POSES BUT KEEPS IT RUNNING")
                print("ORBSLAM BACK FAILED STILL NEED TO REIMPLEMENT FRONT")
                self.mission_complete()
                # TODO: Update this to reset orbslam with tranfsormed initial position from other orbslam output
                # self.USE_FRONT_CAM = True
                # self.USE_BACK_CAM = False
                # self.DRIVE_BACKWARDS_FRAME = self.frame
                # self.T_orb_to_global_front = self.prev_pose_front @ np.linalg.inv(
                #     orbslam_rotated
                # )
                # estimate = self.T_orb_to_global_front @ estimate_orbslamframe_front
                # estimate = rotate_pose_in_place(estimate, 90, 270, 0)

            # So drive backwards for a bit, stop, then restart the bad orbslam?
            # TODO: Luke why does this turn while driving backwards??
            elif (self.frame - self.DRIVE_BACKWARDS_FRAME) < 150:
                print(
                    "testing driiving backwards to overcome visual issues Driving backwards"
                )
                goal_lin_vel = -0.3
                goal_ang_vel = 0.0
                control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)
                self.frame += 1
                return control

            # elif(150 <= self.frame - self.DRIVE_BACKWARDS_FRAME <= 199):
            #     print("stopping after driving backwards")
            #     goal_lin_vel = 0.0
            #     goal_ang_vel = 0.0
            #     control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)
            #     self.frame += 1
            #     return control
            # elif (self.frame - self.DRIVE_BACKWARDS_FRAME == 200):
            #     if not self.USE_FRONT_CAM:
            #         self.orbslam_front = OrbslamEstimator(
            #             self,
            #             carla.SensorPosition.FrontLeft,
            #             carla.SensorPosition.FrontRight,
            #             mode="stereo",
            #         )
            #         self.T_orb_to_global_front = self.prev_pose_front @ np.linalg.inv(
            #             orbslam_rotated
            #         )

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

            # print("world correction front: ", self.T_world_correction_front)
            # print("world correction back: ", self.T_world_correction_back)

        estimate_back_vis = estimate_back
        estimate_vis = estimate

        if self.USE_BACK_CAM and not self.USE_FRONT_CAM:
            estimate = estimate_back
            correction_T = self.T_world_correction_back
        elif self.USE_FRONT_CAM and not self.USE_BACK_CAM:
            estimate = estimate
            correction_T = self.T_world_correction_front

        real_position = carla_to_pytransform(self.get_transform())
        # real_position = None

        goal_location = self.navigator.goal_loc
        all_goals = self.navigator.static_path.get_full_path()
        # nearby_goals = self.navigator.static_path.find_nearby_goals([estimate[0,3], estimate[1,3]])
        nearby_goals = []

        # selected_goal = self.navigator.static_path.pick_goal(estimate, nearby_goals, input_data, self.orb)
        # self.navigator.goal_loc = selected_goal
        # goal_location = self.navigator.goal_loc

        # sensor_data_frontleft = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        # sensor_data_backleft = input_data["Grayscale"][carla.SensorPosition.BackLeft]
        # sensor_data_left = input_data["Grayscale"][carla.SensorPosition.Left]
        # sensor_data_right = input_data["Grayscale"][carla.SensorPosition.Right]

        # if sensor_data_frontleft is not None:
        #     keypoints_frontleft = self.orb.detect(sensor_data_frontleft, None)
        #     if keypoints_frontleft:
        #         responses = [kp.response for kp in keypoints_frontleft]
        #         avg_response = np.mean(responses)
        #         frontleft_score = avg_response
        #         print(f"{frontleft_score}: {len(keypoints_frontleft)} keypoints, avg response = {avg_response:.4f}")

        #     keypoints_backleft = self.orb.detect(sensor_data_backleft, None)
        #     if keypoints_backleft:
        #         responses = [kp.response for kp in keypoints_backleft]
        #         avg_response = np.mean(responses)
        #         backleft_score = avg_response
        #         print(f"{backleft_score}: {len(keypoints_backleft)} keypoints, avg response = {avg_response:.4f}")

        #     keypoints_left = self.orb.detect(sensor_data_left, None)
        #     if keypoints_left:
        #         responses = [kp.response for kp in keypoints_left]
        #         avg_response = np.mean(responses)
        #         left_score = avg_response
        #         print(f"{left_score}: {len(keypoints_left)} keypoints, avg response = {avg_response:.4f}")

        #     keypoints_right = self.orb.detect(sensor_data_right, None)
        #     if keypoints_right:
        #         responses = [kp.response for kp in keypoints_right]
        #         avg_response = np.mean(responses)
        #         right_score = avg_response
        #         print(f"{right_score}: {len(keypoints_right)} keypoints, avg response = {avg_response:.4f}")

        # print("goal location: ", goal_location)

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

            # large_boulders_xyr = [
            #     (b_w[0, 3], b_w[1, 3], 0.3) for b_w in large_boulders_detections
            # ]
            # print("large boulders ", large_boulders_xyr)

            # Now pass the (x, y, r) tuples to your navigator or wherever they need to go
            # if len(large_boulders_xyr) > 0:
            #     self.navigator.add_large_boulder_detection(large_boulders_xyr)
            #     self.large_boulder_detections.extend(large_boulders_xyr)

            # If you just want X, Y coordinates as a tuple
            boulders_xyz = [(b_w[0, 3], b_w[1, 3], b_w[2, 3]) for b_w in boulders_world]
            boulders_xyz_back = [
                (b_w[0, 3], b_w[1, 3], b_w[2, 3]) for b_w in boulders_world_back
            ]

            ground_points_xyz = [
                (b_w[0, 3], b_w[1, 3], b_w[2, 3]) for b_w in ground_points_world
            ]

            # print("boulders detected in front: ", len(boulders_xyz))
            # print("boulders detected in back: ", len(boulders_xyz_back))

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

        if self.frame % 10 == 0:
            plot_poses_and_nav(
                estimate_vis,
                estimate_back_vis,
                real_position,
                self.frame,
                goal_location,
                all_goals,
                nearby_goals,
                self.all_boulder_detections,
                self.large_boulder_detections,
                self.gt_rock_locations,
            )

        if self.frame > 80:
            
            goal_lin_vel, goal_ang_vel = self.navigator(estimate, input_data)
        else:
            goal_lin_vel, goal_ang_vel = 0.0, 0.0

        if self.frame % 10 == 0 and self.frame > 80:
            surface_points_uncorrected = sample_surface(estimate, 60)
            surface_points_corrected = transform_points(
                surface_points_uncorrected, correction_T
            )
            self.sample_list.extend(surface_points_corrected)

        goal_ang_vel = 0.4*goal_ang_vel
        goal_lin_vel = 0.6*goal_lin_vel

        control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)

        self.frame += 1

        return control

    def finalize(self):
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


def plot_poses_and_nav(
    transformed_estimate: np.ndarray,
    transformed_estimate_back: np.ndarray,
    real_pose: np.ndarray,
    frame_number: int,
    goal_location: np.ndarray,
    all_goals: list,
    nearby_goals: list,
    all_boulder_detections: list,  # Format: [(x, y), (x, y), ...]
    large_boulder_detections: list,  # Format: [(x, y, r), (x, y, r), ...]
    gt_boulder_detections: list,
    arrow_length: float = 0.5,
):
    """
    Plots and saves a 2D visualization of:
      - The transformed/adjusted pose (transformed_estimate)
      - The real pose (real_pose)
      - The goal location (in green)
      - RRT waypoints (as smaller red dots)
      - All boulder detections (as orange circles)
      - Large boulder detections (as red circles with their actual radius)

    Each pose is shown as a point for the position plus two quiver arrows
    indicating its local x/y axes in red/green.

    The axes are fixed from -12 to +12 in both directions.

    The figure is saved to 'pose_plot_{frame_number}.png'.

    :param transformed_estimate: 4x4 numpy array for the corrected/adjusted pose.
    :param real_pose: 4x4 numpy array for the real/global pose (if available).
    :param frame_number: used to save the figure as 'pose_plot_{frame_number}.png'.
    :param goal_location: numpy array [x, y] representing the goal position.
    :param rrt_waypoints: list of [x, y] coordinates for the RRT waypoints.
    :param all_boulder_detections: list of (x, y) tuples for all boulder detections.
    :param large_boulder_detections: list of (x, y, r) tuples for large boulder detections.
    :param arrow_length: length of each axis arrow (default 0.5).
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    legend_elements = []

    def draw_pose(ax, T, label_prefix="Pose"):
        """
        Draw the coordinate axes for the transform T (4x4) in 2D (x-y plane).
        We'll plot a small arrow for each local axis (x, y).
        Red for x, green for y.
        """
        # Origin of this pose (only x and y):
        origin = T[:3, 3][:2]  # Extract only x and y
        # Rotation part (for 2D, we only care about the rotation in the x-y plane):
        R = T[:3, :3]

        # Local axes directions in world coords (only x and y components)
        x_axis = (R @ np.array([1, 0, 0]))[:2] * arrow_length
        y_axis = (R @ np.array([0, 1, 0]))[:2] * arrow_length

        # Plot the origin as a single point
        ax.scatter(origin[0], origin[1], color="blue", s=40)

        # Draw x, y quiver arrows in red, green
        ax.quiver(
            origin[0],
            origin[1],
            x_axis[0],
            x_axis[1],
            color="red",
            scale=5,
            scale_units="inches",
            label="_nolegend_",
        )
        ax.quiver(
            origin[0],
            origin[1],
            y_axis[0],
            y_axis[1],
            color="green",
            scale=5,
            scale_units="inches",
            label="_nolegend_",
        )

        # Label near the origin
        ax.text(origin[0] + 0.1, origin[1] + 0.1, label_prefix, size=8)

    # Draw boulder detections (regular boulders)
    if all_boulder_detections is not None and len(all_boulder_detections) > 0:
        # Default radius for regular boulders if not specified
        default_radius = 0.15

        for boulder in all_boulder_detections:
            # Plot boulder center point
            ax.scatter(boulder[0], boulder[1], color="orange", s=20, alpha=0.7)
            # Plot circle representing boulder size
            circle = plt.Circle(
                (boulder[0], boulder[1]),
                default_radius,
                color="orange",
                fill=False,
                alpha=0.5,
            )
            ax.add_patch(circle)

        # Add to legend
        regular_boulder = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markersize=10,
            label="Regular Boulders",
        )
        legend_elements.append(regular_boulder)

    # Draw large boulder detections
    if large_boulder_detections is not None and len(large_boulder_detections) > 0:
        for boulder in large_boulder_detections:
            # Extract x, y, radius from the tuple
            x, y, radius = boulder

            # Plot large boulder center point
            ax.scatter(x, y, color="red", s=30, alpha=0.7)

            # Plot circle representing large boulder size (using the actual radius)
            circle = plt.Circle((x, y), radius, color="red", fill=False, alpha=0.5)
            ax.add_patch(circle)

        # Add to legend
        large_boulder = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Large Boulders",
        )
        legend_elements.append(large_boulder)

    gt_x, gt_y = zip(*[(float(x), float(y)) for x, y, _ in gt_boulder_detections])
    plt.scatter(gt_x, gt_y, c="black", marker="x", s=10, label="GT Rocks")

    # Draw the goal location as a green dot
    if goal_location is not None:
        ax.scatter(goal_location[0], goal_location[1], color="green", s=100, marker="*")
        goal_marker = plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="Goal",
        )
        legend_elements.append(goal_marker)

    # Draw goal waypoints as smaller magenta dots
    if all_goals is not None and len(all_goals) > 0:
        waypoints = np.array(all_goals)
        ax.scatter(waypoints[:, 0], waypoints[:, 1], color="magenta", s=20, alpha=0.7)
        waypoint_marker = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="magenta",
            markersize=8,
            label="All Goals",
        )
        legend_elements.append(waypoint_marker)


    # Draw goal waypoints as smaller magenta dots
    if nearby_goals is not None and len(nearby_goals) > 0:
        waypoints = np.array(nearby_goals)
        ax.scatter(waypoints[:, 0], waypoints[:, 1], color="blue", s=20, alpha=0.5)
        waypoint_marker = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=8,
            label="Nearby Goals",
        )
        legend_elements.append(waypoint_marker)

    # Draw the poses (if they exist)
    if transformed_estimate is not None:
        draw_pose(ax, transformed_estimate, label_prefix="Estimated")
        est_marker = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=8,
            label="Estimated Pose",
        )
        legend_elements.append(est_marker)

        # Draw the poses (if they exist)
    if transformed_estimate_back is not None:
        draw_pose(ax, transformed_estimate_back, label_prefix="Estimated")
        est_marker = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=8,
            label="Estimated Pose Back",
        )
        legend_elements.append(est_marker)

    if real_pose is not None:
        draw_pose(ax, real_pose, label_prefix="Real")
        real_marker = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=8,
            label="Real Pose",
        )
        legend_elements.append(real_marker)

    # Set axis labels
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    # Fix the axes to range [-12, 12] in each dimension
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.grid(True)

    # Add the legend with all elements
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_title("Navigation Trajectory (Frame {})".format(frame_number))
    ax.set_aspect("equal")

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
