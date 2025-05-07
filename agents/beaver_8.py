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

import cv2
from math import hypot
import carla
from pynput import keyboard
import numpy as np
from collections import defaultdict
from math import radians
from pytransform3d.transformations import concat
from maple.boulder import BoulderDetector
from maple.navigation import Navigator

# from maple.pose import OrbslamEstimator
from maple.pose import DoubleSlamEstimator

from maple.utils import *

from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
from maple.utils import extract_rock_locations

""" Import the AutonomousAgent from the Leaderboard. """

from leaderboard.autoagents.autonomous_agent import AutonomousAgent


def get_entry_point():
    """Define the entry point so that the Leaderboard can instantiate the agent class."""
    return "MITAgent"


class MITAgent(AutonomousAgent):
    """Inherit the AutonomousAgent class."""

    def setup(self, path_to_conf_file):
        """This method is executed once by the Leaderboard at mission initialization. We should add any attributes to the class using
        the 'self' Python keyword that contain data or methods we might need throughout the simulation. If you are using machine learning
        models for processing sensor data or control, you should load the models here. We encourage the use of class attributes in place
        of using global variables which can cause conflicts."""

        """ Add some attributes to store values for the target linear and angular velocity. """

        # Required to end the mission with the escape key
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        # Camera resolution
        self._width = 1280
        self._height = 720

        # Initialize the frame counter
        self.frame = 0  # Frame gets stepped at the beginning of run_step

        # Initialize the sample list
        self.sample_list = []  # Surface samples
        self.sample_list.extend(sample_lander(self))  # Add samples from the lander feet

        # Initialize the ORB detector
        # self.orb = cv2.ORB_create()  # TODO: This doesn't get called anywhere?

        # Initialize the pose estimator
        self.estimator = DoubleSlamEstimator(self)

        # Initialize the navigator
        self.navigator = Navigator(self)
        self.goal_lin_vel = 0.0
        self.goal_ang_vel = 0.0

        # Initialize the boulder detectors
        self.front_detector = BoulderDetector(
            self, carla.SensorPosition.FrontLeft, carla.SensorPosition.FrontRight
        )
        self.rear_detector = BoulderDetector(
            self, carla.SensorPosition.BackLeft, carla.SensorPosition.BackRight
        )

        # Not sure what this is for but it get used in finalize?
        self.g_map_testing = self.get_geometric_map()
        self.map_length_testing = self.g_map_testing.get_cell_number()

        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_height(i, j, 0)
                self.g_map_testing.set_cell_rock(i, j, 0)

        self.all_boulder_detections = []
        self.large_boulder_detections = [(0, 0, 2.5)]

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

        # TODO: Remove if unused
        # Extract the rock locations from the preset file
        self.gt_rock_locations = extract_rock_locations(
            "simulator/LAC/Content/Carla/Config/Presets/Preset_1.xml"
        )

    def use_fiducials(self):
        """Not using fiducials for this agent"""
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
        try:
            self.frame += 1
            print("Frame: ", self.frame)
            return self.run_step_unsafe(input_data)
        except Exception as e:
            print(f"Error: {e}")
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

    def run_step_unsafe(self, input_data):
        """Execute one step of navigation"""

        # Wait for the rover to stabilize and arms to raise
        if self.frame < 30 * 20:  # Thirty seconds
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))
            return carla.VehicleVelocityControl(0.0, 0.0)

        ######################################
        # At this point we can begin driving #
        ######################################

        # On odd frames, we don't have images, so we can't estimate, just carry on with the next navigation step
        if self.frame % 2 != 0:
            # Just return the last command input
            return carla.VehicleVelocityControl(self.goal_lin_vel, self.goal_ang_vel)

        ######################################################################
        # At this point we have images, so we can estimate and do detections #
        ######################################################################

        # Get the pose
        # This will be none on frames without images (odd frames)
        # This will always be the rover in the global frame
        estimate = self.estimator.estimate(input_data)

        # Get the status of the estimator
        # This will be "no_images" if we are on a frame without images
        # This will be "last_any" if we are using the last valid pose (indicates a failure)
        # This will be "front" if we are using the front camera
        # This will be "rear" if we are using the back camera
        # This will be "combined" if we are using both cameras
        estimate_source = self.estimator.estimate_source

        # TODO: Decide what to do based on the estimate source
        if estimate_source == "front" or estimate_source == "rear":
            # TODO: Change navigation strategy since one of the orbslams is failing?
            pass

        if estimate_source == "last_any":
            # TODO: If this happens, both estimators are failing, we probably need to stop
            # We might want to wait for this to occur a few times before stopping
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        ##########################
        # Run boulder detections #
        ##########################

        # Run detections every 20 frames (1 Hz)
        if self.frame % 20 == 0:
            # Detections in the rover frame
            # TODO: Confirm that the rear detections are correctly in the rover frame
            front_detections, front_ground_points = self.front_detector(input_data)
            front_large_boulders = self.front_detector.get_large_boulders()

            rear_detections, rear_ground_points = self.rear_detector(input_data)
            rear_large_boulders = self.rear_detector.get_large_boulders()

            # Combine detections and convert to the global frame
            boulder_detections = front_detections + rear_detections
            large_boulder_detections = front_large_boulders + rear_large_boulders
            boulder_ground_points = front_ground_points + rear_ground_points

            # Convert to the global frame
            boulder_detections_world = [
                concat(boulder, estimate) for boulder in boulder_detections
            ]
            large_boulder_detections_world = [
                concat(boulder, estimate) for boulder in large_boulder_detections
            ]
            boulder_ground_points_world = [
                concat(ground_point, estimate) for ground_point in boulder_ground_points
            ]

            # Add the boulder detections to the all_boulder_detections list (only x, y)
            self.all_boulder_detections.extend(
                boulder[:2, 3].tolist() for boulder in boulder_detections_world
            )

            # Add large boulder detections to the all_boulder_detections list (only x, y)
            # TODO: Implement this, applying the correct filtering from Aleks
            # TODO: Implement distance filtering
            self.large_boulder_detections.extend(
                large_boulder[:2, 3].tolist()
                for large_boulder in large_boulder_detections_world
            )

            # Add ground points to the sample list (only x, y, z)
            self.sample_list.extend(
                ground_point[:2, 3].tolist()
                for ground_point in boulder_ground_points_world
            )

            # TODO: Add the random boulder code here...
            # Add the random boulder code here...
            # Really it should probably go in a function or something so it doesn't take
            # up so much space

        ########################
        # Run surface sampling #
        ########################
        # Surface points are in the global frame
        surface_points = sample_surface(estimate, 60)
        # TODO: For some reason points within 1.5m of the origin are filtered out?
        self.sample_list.extend(surface_points)

        #########################
        # Run navigation system #
        #########################
        self.goal_lin_vel, self.goal_ang_vel = self.navigator(estimate, input_data)
        return carla.VehicleVelocityControl(self.goal_lin_vel, self.goal_ang_vel)

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
            # Removed other boulder stuff, keeping for reference for the moment
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

        if self.frame > 80:
            goal_lin_vel, goal_ang_vel = self.navigator(estimate, input_data)
        else:
            goal_lin_vel, goal_ang_vel = 0.0, 0.0

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
        cv2.destroyAllWindows()
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

    def on_release(self, key):
        """Stop the display with the escape key"""
        """ Press escape to end the mission. """
        if key == keyboard.Key.esc:
            self.mission_complete()
