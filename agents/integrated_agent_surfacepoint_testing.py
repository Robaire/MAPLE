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

"""
This agent demonstrates how to structure your code and visualize camera data in
an OpenCV window and control the robot with keyboard commands with pynput
https://pypi.org/project/opencv-python/
https://pypi.org/project/pynput/

"""

import os
import random
from math import radians
import matplotlib.pyplot as plt
import traceback
import pickle
import numpy as np
import pytransform3d.rotations as pyrot
import signal
import sys
import csv

from collections import defaultdict

import carla
import cv2 as cv
from pynput import keyboard
from pytransform3d.transformations import concat

from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.navigation.navigator import generate_multi_angle_lawnmower
from maple.pose import InertialApriltagEstimator
from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
from maple.utils import carla_to_pytransform
""" Import the AutonomousAgent from the Leaderboard. """

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

""" Define the entry point so that the Leaderboard can instantiate the agent class. """


def get_entry_point():
    return "OpenCVagent"


""" Inherit the AutonomousAgent class. """


class OpenCVagent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """This method is executed once by the Leaderboard at mission initialization. We should add any attributes to the class using
        the 'self' Python keyword that contain data or methods we might need throughout the simulation. If you are using machine learning
        models, this would be a good place to load them."""


        # Initialize the frame counter
        self.frame = 1
        self.sample_list = []
        self.last_sample_list_length = 0  # Initialize the length tracker
        self.point_cloud_data = {'points': []}
        self.point_cloud_dir = os.path.join(os.path.dirname(__file__), "..", "data", "point_cloud")
        if not os.path.exists(self.point_cloud_dir):
            os.makedirs(self.point_cloud_dir)

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Initialize the camera sensors
        self._width = 1280
        self._height = 720

        self.good_loc = True

        # self._width = 1920
        # self._height = 1080

        # Store previous boulder detections
        self.previous_detections = []

        # Initialize the plot
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.show(block=False)  # Show the plot window without blocking

        """ Initialize a counter to keep track of the number of simulation steps. """

        # set the trial number here
        self.trial = "044"

        if not os.path.exists(f"./data/{self.trial}"):
            os.makedirs(f"./data/{self.trial}")

        self.checkpoint_path = f"./data/{self.trial}/boulders_frame{self.frame}.json"

        self._active_side_cameras = False
        self._active_side_front_cameras = True

        self.estimator = InertialApriltagEstimator(self)
        self.detector = BoulderDetector(
            self, carla.SensorPosition.FrontLeft, carla.SensorPosition.FrontRight
        )
        self.detectorBack = BoulderDetector(
            self, carla.SensorPosition.BackLeft, carla.SensorPosition.BackRight
        )

        # Remove the interactive plotting setup
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Create a directory for saving plots if it doesn't exist
        self.plots_dir = f"./data/{self.trial}/plots"
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        # Create a directory for saving plots if it doesn't exist
        self.surface_plots_dir = f"./data/{self.trial}/surface_plots"
        if not os.path.exists(self.surface_plots_dir):
            os.makedirs(self.surface_plots_dir)

        # Create point cloud directory
        self.point_cloud_dir = f"./data/{self.trial}/point_cloud"
        if not os.path.exists(self.point_cloud_dir):
            os.makedirs(self.point_cloud_dir)

        self.g_map_testing = self.get_geometric_map()
        self.map_length_testing = self.g_map_testing.get_cell_number()

        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_height(i, j, 0)
                self.g_map_testing.set_cell_rock(i, j, 0)

        self.all_boulder_detections = []
        self.large_boulder_detections = [(0, 0, 2.5)]

        # Load the pickled numpy array from the file
        file_path = "Moon_Map_01_0_rep0.dat"
        with open(file_path, "rb") as file:
            self.grid_data = pickle.load(file)

        # Get lander feet points and store them with confidence 1.0
        self.lander_points = sample_lander(self)
        self._save_initial_points()

        self.sample_list.extend(sample_lander(self))

        # Add position tracking for stuck detection
        self.position_history = []
        self.is_stuck = False
        self.unstuck_phase = 0
        self.unstuck_counter = 0

        # Tiered stuck detection parameters
        self.SEVERE_STUCK_FRAMES = 700
        self.SEVERE_STUCK_THRESHOLD = 0.4  # If moved less than 0.5m in 500 frames

        self.MILD_STUCK_FRAMES = 2000
        self.MILD_STUCK_THRESHOLD = 3.0  # If moved less than 3m in 1000 frames

        self.UNSTUCK_DISTANCE_THRESHOLD = (
            3.0  # How far to move to be considered unstuck
        )

        self.unstuck_sequence = [
            {"lin_vel": -0.45, "ang_vel": 0, "frames": 100},  # Backward
            {"lin_vel": 0, "ang_vel": 4, "frames": 60},  # Rotate clockwise
            {"lin_vel": 0.45, "ang_vel": 0, "frames": 150},  # Forward
            {"lin_vel": 0, "ang_vel": -4, "frames": 60},  # Rotate counter-clockwise
        ]

        # Add these variables for goal timeout tracking
        self.frames_since_goal_change = 0
        self.goal_timeout_threshold = 1000
        self.goal_timeout_active = False
        self.goal_timeout_counter = 0
        self.goal_timeout_duration = 200
        self.max_linear_velocity = 0.6  # Maximum linear velocity for timeout maneuver
        self.current_goal_index = 0  # Track which goal we're headed to

        # Initialize the keyboard listener
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ Add some attributes to store values for the target linear and angular velocity. """

        self.current_v = 0
        self.current_w = 0

        """ Initialize the navigator with custom path parameters """
        self.navigator = Navigator(self)
        # Override the default path with a multi-angle lawnmower pattern in a 5x5 region
        # Offset from lander by 3 meters to avoid collision
        # lander_x, lander_y = self.navigator.lander_x, self.navigator.lander_y
        # scan_center_x = lander_x + 4.0  # 4 meters to the right of lander
        # scan_center_y = lander_y + 4.0  # 4 meters forward of lander
        
        # # Generate a multi-angle lawnmower pattern for thorough coverage
        # self.navigator.global_path = generate_multi_angle_lawnmower(
        #     x0=scan_center_x,
        #     y0=scan_center_y,
        #     angles=[0, 45, 90],  # Scan at 0, 45, and 90 degrees for better coverage
        #     width=5.0,
        #     height=5.0,
        #     spacing=1.0  # 1 meter between passes for dense coverage
        # )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals by saving the .dat file before exiting."""
        print("\nSignal received. Saving point cloud data...")
        self._save_dat_file()
        sys.exit(0)

    def _save_dat_file(self):
        """Save the point cloud data to a .dat file."""
        dat_path = os.path.join(self.point_cloud_dir, "point_cloud_data.dat")
        with open(dat_path, 'wb') as f:
            pickle.dump(self.point_cloud_data, f)
        print(f"Point cloud data saved to {dat_path}")

    def _save_initial_points(self):
        """Save initial lander points to the point cloud CSV with confidence 1.0"""
        csv_path = os.path.join(self.point_cloud_dir, "point_cloud_data.csv")
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'x', 'y', 'z', 'confidence', 'source'])

        # Write lander points
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for point in self.lander_points:
                writer.writerow([0, point[0], point[1], point[2], 1.0, 'lander'])
                self.point_cloud_data['points'].append({
                    'frame': 0,
                    'point': point,
                    'confidence': 1.0,
                    'source': 'lander'
                })

    def _update_point_cloud_data(self):
        """Update point cloud data with any new points from sample_list"""
        if len(self.sample_list) > self.last_sample_list_length:
            # Get only the new points
            new_points = self.sample_list[self.last_sample_list_length:]
            
            csv_path = os.path.join(self.point_cloud_dir, "point_cloud_data.csv")
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for point in new_points:
                    writer.writerow([self.frame, point[0], point[1], point[2], 1.0, 'rover'])
                    self.point_cloud_data['points'].append({
                        'frame': self.frame,
                        'point': point,
                        'confidence': 1.0,
                        'source': 'rover'
                    })
            
            self.last_sample_list_length = len(self.sample_list)

    def destroy(self):
        """Called when the simulation ends to cleanup resources."""
        # Save point cloud data to .dat file
        output_file = os.path.join(self.point_cloud_dir, "point_cloud_data.dat")
        with open(output_file, 'wb') as f:
            pickle.dump(self.point_cloud_data, f)
        
        super().destroy()

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

        # Only perform stuck detection every 10 frames for performance
        if self.frame % 10 != 0:
            return False

        # Check for severe stuck condition (shorter timeframe)
        if len(self.position_history) >= self.SEVERE_STUCK_FRAMES:
            severe_check_position = self.position_history[-self.SEVERE_STUCK_FRAMES]
            dx = current_position[0] - severe_check_position[0]
            dy = current_position[1] - severe_check_position[1]
            severe_distance_moved = np.sqrt(dx**2 + dy**2)

            if severe_distance_moved < self.SEVERE_STUCK_THRESHOLD:
                print(
                    f"SEVERE STUCK DETECTED! Moved only {severe_distance_moved:.2f}m in the last {self.SEVERE_STUCK_FRAMES} frames."
                )
                return True

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

    def get_unstuck_control(self):
        # Same as before - no changes needed here
        current_phase = self.unstuck_sequence[self.unstuck_phase]
        lin_vel = current_phase["lin_vel"]
        ang_vel = current_phase["ang_vel"]
        self.unstuck_counter += 1

        if self.unstuck_counter >= current_phase["frames"]:
            self.unstuck_phase = (self.unstuck_phase + 1) % len(self.unstuck_sequence)
            self.unstuck_counter = 0
            print(f"Moving to unstuck phase {self.unstuck_phase}")

        return lin_vel, ang_vel

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

        # print("geometric map", self.g_map_testing.get_map_array())

        if self.frame == 1:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        # Get a position estimate for the rover
        estimate, is_april_tag_estimate = self.estimator(input_data)

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
                        self.navigator.global_path_index_tracker = (
                            self.navigator.global_path_index_tracker + 1
                        ) % len(self.navigator.global_path)
                        self.is_stuck = False
                        self.unstuck_phase = 0
                        self.unstuck_counter = 0
                        # Clear position history to reset stuck detection
                        self.position_history = []

        stopped = False

        goal_locations_all = self.navigator.get_all_goal_locations()
        goal_locations_rrt = self.navigator.get_rrt_waypoints()

        self.good_loc = False
        # check if some wild far off localization
        # TODO: make this reflect imu/controller expected position instead of this
        for goal_location in goal_locations_rrt:
            current_arr = np.array(current_position)
            goal_arr = np.array(goal_location)

            # Now subtract and compute the Euclidean distance
            distance = np.linalg.norm(current_arr - goal_arr)
            if distance < 1.0:
                # Skip this goal location if not within 1 meter
                self.good_loc = True

        if (
            not self.goal_timeout_active
            and current_position is not None
            and self.navigator.goal_loc is not None
        ):
            current_arr = np.array(current_position)
            goal_arr = np.array(self.navigator.goal_loc)
            distance = np.linalg.norm(current_arr - goal_arr)

            # If we're close enough to the goal, reset the timeout counter
            if distance < 1.0:
                self.frames_since_goal_change = 0

        # Determine where we are in the 150-frame cycle
        phase = self.frame % 200

        # Get the current goal
        goal_locations_rrt = self.navigator.get_rrt_waypoints()
        current_goal = None
        if goal_locations_rrt and self.navigator.global_path_index_tracker < len(
            goal_locations_rrt
        ):
            current_goal = goal_locations_rrt[self.navigator.global_path_index_tracker]

        # Check if we've changed goals
        if self.current_goal_index != self.navigator.global_path_index_tracker:
            self.current_goal_index = self.navigator.global_path_index_tracker
            self.frames_since_goal_change = 0
            print(f"New goal target: {current_goal}")
        else:
            self.frames_since_goal_change += 1

        # Check for goal timeout
        if (
            not self.is_stuck
            and not self.goal_timeout_active
            and self.frames_since_goal_change >= self.goal_timeout_threshold
        ):
            print(
                f"GOAL TIMEOUT: Haven't reached goal in {self.goal_timeout_threshold} frames!"
            )
            self.goal_timeout_active = True
            self.goal_timeout_counter = 0

        # Handle the phases like before, but add the timeout condition
        if self.is_stuck:
            # Existing stuck handling code...
            self.navigator.add_large_boulder_detection(
                (estimate[0, 3], estimate[1, 3], 0.7)
            )
            goal_lin_vel, goal_ang_vel = self.get_unstuck_control()
            print(
                f"UNSTUCK MANEUVER: lin_vel={goal_lin_vel}, ang_vel={goal_ang_vel}, phase={self.unstuck_phase}, counter={self.unstuck_counter}"
            )
        elif self.goal_timeout_active:
            # Handle goal timeout - maximum forward velocity for a set duration
            goal_lin_vel = 4.0
            goal_ang_vel = 0.0

            self.goal_timeout_counter += 1
            print(
                f"GOAL TIMEOUT MANEUVER: frame {self.goal_timeout_counter}/{self.goal_timeout_duration}"
            )

            if self.goal_timeout_counter >= self.goal_timeout_duration:
                print("GOAL TIMEOUT COMPLETE - resuming normal operation")
                self.goal_timeout_active = False
                # Increment the goal index to try the next goal
                self.navigator.global_path_index_tracker = (
                    self.navigator.global_path_index_tracker + 1
                ) % len(self.navigator.global_path)
                self.frames_since_goal_change = 0
        # if self.is_stuck:
        #     self.navigator.add_large_boulder_detection((estimate[0, 3], estimate[1, 3], 0.7))
        #     goal_lin_vel, goal_ang_vel = self.get_unstuck_control()
        #     print(f"UNSTUCK MANEUVER: lin_vel={goal_lin_vel}, ang_vel={goal_ang_vel}, phase={self.unstuck_phase}, counter={self.unstuck_counter}")
        else:
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
                        # Get boulder detections (ground points are automatically saved)
                        boulders, _ = self.detector(input_data, estimate)

                        large_boulders_detections = self.detector.get_large_boulders()

                        boulders_back, _ = self.detectorBack(input_data, estimate)

                        # Get all detections in the world frame
                        rover_world = estimate
                        boulders_world = [
                            concat(boulder_rover, rover_world)
                            for boulder_rover in boulders
                        ]

                        boulders_world_back = [
                            concat(boulder_rover, rover_world)
                            for boulder_rover in boulders_back
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
                        traceback.print_exc()  # This will print the full stack trace

            elif phase < 120:
                # Phase 1: Frames 0–49
                # ---------------------------------------
                # 1) We want to STOP here.

                nav_goal_lin_vel, nav_goal_ang_vel = self.navigator(estimate)

                goal_lin_vel = nav_goal_lin_vel / 3
                goal_ang_vel = nav_goal_ang_vel / 3

                stopped = False

            elif phase < 140:
                nav_goal_lin_vel, nav_goal_ang_vel = self.navigator(estimate)

                goal_lin_vel = 2 * nav_goal_lin_vel / 3
                goal_ang_vel = 2 * nav_goal_ang_vel / 3

                stopped = False

            else:
                # Phase 3: Frames 140=200
                # ---------------------------------------
                # 3) Go back to what the navigator says
                goal_lin_vel, goal_ang_vel = self.navigator(estimate)

                stopped = False

        # After handling the phases, increment the frame counter
        self.frame += 1

        # Update point cloud data with any new points
        self._update_point_cloud_data()

        # Finally, apply the resulting velocities
        control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)

        # Generate and add in the sample points
        if is_april_tag_estimate and stopped and phase % 20 == 0:
            self.sample_list.extend(sample_surface(estimate, 60))

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
