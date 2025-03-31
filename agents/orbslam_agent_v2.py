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

import orbslam3
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
from maple.pose.ORBSLAM_Interface import ORBSLAMInterface
from maple.utils import *
from maple.pose.stereoslam import SimpleStereoSLAM
from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
import time
import csv
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
        self.trial = "orb_03"

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

        self.orb_vocab = (
            "/home/annikat/ORB-SLAM3-python/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"
        )
        self.orb_cams_config = "/home/annikat/ORB-SLAM3-python/third_party/ORB_SLAM3/Examples/Stereo/LAC_cam.yaml"
        self.orbslam = SimpleStereoSLAM(self.orb_vocab, self.orb_cams_config)

        self.columns = ["frame", "gt_x", "gt_y", "gt_z", "x", "y", "z"]

        self.positions = []

        self.init_pose = carla_to_pytransform(self.get_initial_position())

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

        timestamp = time.time()

        # Send to ORB-SLAM
        # self.orbslam.send_frame(sensor_data_frontleft, timestamp)

        if sensor_data_frontleft is not None:
            self.orbslam.send_frame(sensor_data_frontleft, timestamp)
            # self.orbslam.test_send_frame(timestamp)

            cv.imshow("Left front camera view", sensor_data_frontleft)
            cv.waitKey(1)
            # dir_frontleft = f'data/{self.trial}/FrontLeft/'

            # if not os.path.exists(dir_frontleft):
            #     os.makedirs(dir_frontleft)

            # cv.imwrite(dir_frontleft + str(self.frame) + '.png', sensor_data_frontleft)
            # print("saved image front left ", self.frame)

        if sensor_data_frontleft is not None and sensor_data_frontright is not None:
            self.orbslam.process_frame(
                sensor_data_frontleft, sensor_data_frontright, self.frame * 0.1
            )

        if self.frame == 1:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        # Get a position estimate for the rover
        # estimate, is_april_tag_estimate = self.estimator(input_data)
        estimate_orbslamframe = self.orbslam.get_current_pose()
        estimate = np.linalg.inv(self.init_pose) @ estimate_orbslamframe

        roll, pitch, yaw = pyrot.euler_from_matrix(
            estimate[:3, :3], i=0, j=1, k=2, extrinsic=True
        )
        if np.abs(pitch) > np.deg2rad(50) or np.abs(roll) > np.deg2rad(50):
            self.set_front_arm_angle(radians(0))
            self.set_back_arm_angle(radians(0))
        else:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        current_position = (
            (estimate[0, 3], estimate[1, 3]) if estimate is not None else None
        )

        current_position_xyz = (
            (estimate[0, 3], estimate[1, 3], estimate[2, 3])
            if estimate is not None
            else None
        )

        if current_position is not None:
            # Always update position history
            self.position_history.append(current_position)

        goal_locations_all = self.navigator.get_all_goal_locations()
        goal_locations_rrt = self.navigator.get_rrt_waypoints()

        # Determine where we are in the 150-frame cycle
        phase = self.frame % 200

        # Get the current goal
        goal_locations_rrt = self.navigator.get_rrt_waypoints()
        current_goal = None
        if goal_locations_rrt and self.navigator.global_path_index_tracker < len(
            goal_locations_rrt
        ):
            current_goal = goal_locations_rrt[self.navigator.global_path_index_tracker]

        # Check if we've changed goal
        if phase < 20:
            # Phase 1: Frames 0–49
            # ---------------------------------------
            # 1) We want to STOP here.

            nav_goal_lin_vel, nav_goal_ang_vel = self.navigator(estimate)

            goal_lin_vel = 2 * nav_goal_lin_vel / 3
            goal_ang_vel = 2 * nav_goal_ang_vel / 3

            stopped = False

        elif phase < 40:
            nav_goal_lin_vel, nav_goal_ang_vel = self.navigator(estimate)

            goal_lin_vel = nav_goal_lin_vel / 3
            goal_ang_vel = nav_goal_ang_vel / 3

            stopped = False

        elif phase < 60:
            goal_lin_vel = 0.0
            goal_ang_vel = 0.0

            stopped = False

        elif phase < 100:
            # Phase 2: Frames 50–99
            # ---------------------------------------
            # 2) We want to run boulder detection every 10 frames.
            #    (Keep velocity = 0.0 or whatever you'd like.)
            goal_lin_vel = 0.0
            goal_ang_vel = 0.0

            stopped = True

            if phase % 20 == 0:
                # Run boulder detection
                try:
                    detections, _ = self.detector(input_data)

                    large_boulders_detections = self.detector.get_large_boulders()

                    detections_back, _ = self.detectorBack(input_data)

                    # Get all detections in the world frame
                    rover_world = estimate
                    boulders_world = [
                        concat(boulder_rover, rover_world)
                        for boulder_rover in detections
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
                        (b_w[0, 3], b_w[1, 3], 0.25)
                        for b_w in large_boulders_detections
                    ]

                    # Now pass the (x, y, r) tuples to your navigator or wherever they need to go
                    self.navigator.add_large_boulder_detection(large_boulders_xyr)
                    self.large_boulder_detections.extend(large_boulders_xyr)

                    # If you just want X, Y coordinates as a tuple
                    boulders_xy = [(b_w[0, 3], b_w[1, 3]) for b_w in boulders_world]
                    boulders_xy_back = [
                        (b_w[0, 3], b_w[1, 3]) for b_w in boulders_world_back
                    ]

                    self.all_boulder_detections.extend(boulders_xy)
                    self.all_boulder_detections.extend(boulders_xy_back)

                except Exception as e:
                    print(f"Error processing detections: {e}")
                    print(f"Error details: {str(e)}")
                    # traceback.print_exc()  # This will print the full stack trace

        elif phase < 120:
            # Phase 1: Frames 0–49
            # ---------------------------------------
            # 1) We want to STOP here.

            nav_goal_lin_vel, nav_goal_ang_vel = self.navigator(estimate)

            goal_lin_vel = nav_goal_lin_vel / 3
            goal_ang_vel = nav_goal_ang_vel / 3

        elif phase < 140:
            nav_goal_lin_vel, nav_goal_ang_vel = self.navigator(estimate)

            goal_lin_vel = 2 * nav_goal_lin_vel / 3
            goal_ang_vel = 2 * nav_goal_ang_vel / 3

        else:
            # Phase 3: Frames 140=200
            # ---------------------------------------
            # 3) Go back to what the navigator says
            goal_lin_vel, goal_ang_vel = self.navigator(estimate)

        # After handling the phases, increment the frame counter
        self.frame += 1

        if self.frame < 200:
            goal_lin_vel = 0.0
            goal_ang_vel = 0.0

        # Finally, apply the resulting velocities
        control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)

        # Generate and add in the sample points
        if phase % 20 == 0:
            self.sample_list.extend(sample_surface(estimate, 60))

        # This part is gathering info to be used later
        transform = self.get_transform()
        transform_location_x = transform.location.x
        transform_location_y = transform.location.y
        transform_location_z = transform.location.z

        initial_transform = self.get_initial_position()

        print("intial transform: ", initial_transform)

        # adding a bunch of info to save to a csv at the end
        position_entry = [self.frame] + [
            transform_location_x,
            transform_location_y,
            transform_location_z,
            current_position_xyz[0],
            current_position_xyz[1],
            current_position_xyz[2],
        ]

        # Append to self.imu list to save at the end
        self.positions.append(position_entry)

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
