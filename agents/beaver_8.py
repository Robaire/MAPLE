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

import traceback
from collections import defaultdict, deque
from math import radians

import carla
import cv2
import numpy as np
from lac_data import Recorder
from pynput import keyboard
from pytransform3d.transformations import concat

from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.pose import DoubleSlamEstimator
from maple.surface.map import SurfaceHeight, sample_lander, sample_surface

# from maple.utils import *
from maple.utils import (
    extract_rock_locations,
    pytransform_to_tuple,
    carla_to_pytransform,
)


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
        # NOTE: Remove in submission
        listener = keyboard.Listener(on_release=self.on_release)
        listener.start()

        # Camera resolution
        self._width = 1280
        self._height = 720

        # Initialize the frame counter
        self.frame = 0  # Frame gets stepped at the beginning of run_step

        # Initialize the recorder (for testing only)
        # NOTE: Remove in submission
        self.recorder = Recorder(self, "/recorder/beaver_8.lac", 10)
        self.recorder.description("Beaver 8, images 10 Hz")

        # Initialize the sample list
        self.sample_list = []  # Surface samples (x, y, z)
        self.sample_list.extend(sample_lander(self))  # Add samples from the lander feet

        # Initialize the ORB detector
        # self.orb = cv2.ORB_create()  # TODO: This doesn't get called anywhere?

        # Initialize the pose estimator
        # self.estimator = DoubleSlamEstimator(self)
        self.estimator = None
        self.last_any_failures = 0

        # Initialize the navigator
        self.navigator = Navigator(self)
        self.goal_lin_vel = 0.0
        self.goal_ang_vel = 0.0

        self.prev_goal_location = None
        self.frame_goal_updated = 0
        self.no_update_threshold = 3000

        # Initialize the boulder detectors
        self.front_detector = BoulderDetector(
            self, carla.SensorPosition.FrontLeft, carla.SensorPosition.FrontRight
        )
        self.rear_detector = BoulderDetector(
            self, carla.SensorPosition.BackLeft, carla.SensorPosition.BackRight
        )

        # Initialize the boulder detection lists
        self.all_boulder_detections = []  # This is a list of (x, y) coordinates
        # This is a list of (x, y, r)
        self.large_boulder_detections = [[0, 0, 2.5]]  # Add the lander

        # Not sure what this is for but it get used in finalize?
        self.g_map_testing = self.get_geometric_map()
        self.map_length_testing = self.g_map_testing.get_cell_number()

        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_height(i, j, 0)
                self.g_map_testing.set_cell_rock(i, j, 0)

        # Stuck detection parameters
        self.is_stuck = False
        self.stuck_frames = 2000
        self.stuck_threshold = 2.0  # how much to move to be considered stuck
        self.unstuck_threshold = 2.0  # how far to move to be considered unstuck
        # This only gets extended every other frame, so in reality it represents 2 * stuck_frames
        # of time
        # The position history is only (x, y)
        self.position_history = deque(maxlen=self.stuck_frames)
        self.position_history.append(
            carla_to_pytransform(self.get_initial_position())[:2, 3].tolist()
        )

        # TODO: These values seem super high
        self.unstuck_sequence = [
            {"lin_vel": -0.45, "ang_vel": 0, "frames": 200},  # Backward
            {"lin_vel": 0, "ang_vel": 4, "frames": 100},  # Rotate clockwise
            {"lin_vel": 0.45, "ang_vel": 0, "frames": 200},  # Forward
            {"lin_vel": 0, "ang_vel": -4, "frames": 100},  # Rotate counter-clockwise
        ]

        # NOTE: Remove in submission
        # Extract the rock locations from the preset file
        # self.gt_rock_locations = extract_rock_locations(
        #     "simulator/LAC/Content/Carla/Config/Presets/Preset_1.xml"
        # )

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

    def check_if_stuck(self, rover_global) -> bool:
        """Checks if the rover is stuck"""

        # If for some reason this was called on a frame without an estimate, return False
        if rover_global is None:
            return False

        # If we have not reached the number of frames to start checking, return False
        if len(self.position_history) < self.stuck_frames:
            return False

        # Look at how far we have moved from the oldest tracked position
        distance = np.linalg.norm(
            rover_global[:2, 3] - np.array(self.position_history[0])
        )

        # If we have moved less than the threshold, we are stuck
        if distance < self.stuck_threshold:
            print(
                f"Stuck detected! Moved {distance:.2f}m in the last {self.stuck_frames} frames."
            )
            return True
        else:
            # If we have moved more than the threshold, we are not stuck
            return False

    def run_step(self, input_data):
        """Execute one step of navigation"""
        try:
            self.frame += 1
            print("Frame: ", self.frame)
            self.recorder.record_all(self.frame, input_data)
            return self.run_step_unsafe(input_data)
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

    def run_step_unsafe(self, input_data):
        """Execute one step of navigation"""

        ########################################
        # Check for an arbitrary end condition #
        ########################################

        if self.frame > 2_000:
            print(f"Reached {self.frame} frames, ending mission...")
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        ##################
        # Initialization #
        ##################

        if self.frame == 1:
            # This needs to occur here because we cannot initialize the estimator in setup since camera pose data is not available
            self.estimator = DoubleSlamEstimator(self)

            # Raise the arms
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        # Wait for the rover to stabilize and arms to raise
        if self.frame < 10 * 20:  # Ten seconds
            self.recorder.record_custom(
                self.frame, "control", {"linear": 0, "angular": 0}
            )
            return carla.VehicleVelocityControl(0.0, 0.0)

        ######################################
        # At this point we can begin driving #
        ######################################

        # On odd frames, we don't have images, so we can't estimate, just carry on with the next navigation step
        if self.frame % 2 != 0:
            # Log the control input
            self.recorder.record_custom(
                self.frame,
                "control",
                {"linear": self.goal_lin_vel, "angular": self.goal_ang_vel},
            )
            # Just return the last command input
            return carla.VehicleVelocityControl(self.goal_lin_vel, self.goal_ang_vel)

        ######################################################################
        # At this point we have images, so we can estimate and do detections #
        ######################################################################

        # Get the pose
        # This will be none on frames without images (odd frames)
        # This will always be the rover in the global frame
        rover_global = self.estimator.estimate(input_data)

        # Update the position history (only x, y)
        self.position_history.append(rover_global[:2, 3].tolist())

        # Get the status of the estimator
        # This will be "no_images" if we are on a frame without images
        # This will be "last_any" if we are using the last valid pose (indicates a failure)
        # This will be "front" if we are using the front camera
        # This will be "rear" if we are using the back camera
        # This will be "combined" if we are using both cameras
        estimate_source = self.estimator.estimate_source
        print(f"Pose estimate source: {estimate_source}")

        # Save the estimated pose data for testing
        # NOTE: Remove in submission
        x, y, z, roll, pitch, yaw = pytransform_to_tuple(rover_global)
        self.recorder.record_custom(
            self.frame,
            "estimate",
            {
                "x": x,
                "y": y,
                "z": z,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "source": estimate_source,
            },
        )

        # TODO: Decide what to do based on the estimate source (if anything)
        if estimate_source == "front" or estimate_source == "rear":
            pass

        # Track the number of times the last_any estimate fails
        # Only start tracking after 60 seconds so the rover has time to get moving
        if self.frame > 60 * 20:
            if estimate_source == "last_any":
                self.last_any_failures += 1

            if self.last_any_failures > 10:
                print(
                    f"Pose tracking failed {self.last_any_failures} times, ending mission..."
                )
                self.mission_complete()
                return carla.VehicleVelocityControl(0.0, 0.0)

        ##########################
        # Run boulder detections #
        ##########################

        # Run detections every 20 frames (1 Hz)
        if self.frame % 20 == 0:
            print("Running boulder detection...")

            # Detections in the rover frame
            # TODO: Confirm that the rear detections are correctly in the rover frame
            front_detections, front_ground_points = self.front_detector(input_data)

            # It looks like this returns boulder (x, y, z), we convert to (x, y, r) for the navigator
            front_large_boulders = self.front_detector.get_large_boulders()

            rear_detections, rear_ground_points = self.rear_detector(input_data)

            # Combine detections and convert to the global frame
            boulders_global = [
                concat(boulder_rover, rover_global)
                for boulder_rover in front_detections + rear_detections
            ]
            large_boulders_global = [
                concat(boulder_rover, rover_global)
                for boulder_rover in front_large_boulders
                # for boulder_rover in front_large_boulders + rear_large_boulders
            ]
            ground_points_global = [
                concat(ground_point, rover_global)
                for ground_point in front_ground_points + rear_ground_points
            ]

            # Add the boulder detections to the all_boulder_detections list (only x, y)
            self.all_boulder_detections.extend(
                boulder[:2, 3].tolist() for boulder in boulders_global
            )

            # Add large boulder detections to the all_boulder_detections list (x, y, r)
            # Filter large boulders to only included ones that are nearby (x, y distance)
            # TODO: Implement Alek's improved filtering
            large_boulders_global = [
                large_boulder[:2, 3].tolist() + [0.3]  # radius is 0.3 in beaver_7
                for large_boulder in large_boulders_global
                if np.linalg.norm(large_boulder[:2, 3] - rover_global[:2, 3]) <= 2
            ]
            self.large_boulder_detections.extend(large_boulders_global)

            # TODO: Add ground samples from ORBSLAM
            # Add ground points to the sample list (x, y, z)
            self.sample_list.extend(
                ground_point[:3, 3].tolist() for ground_point in ground_points_global
            )

        ########################
        # Run surface sampling #
        ########################
        # Surface points are in the global frame
        surface_points = sample_surface(rover_global, 60)
        self.sample_list.extend(surface_points)
        # TODO: For some reason points within 1.5m of the origin were filtered out?
        # self.sample_list.extend(
        #     [
        #         surface_point
        #         for surface_point in surface_points
        #         if np.linalg.norm(surface_point[:2, 3]) > 1.5
        #     ]
        # )

        ###############################
        # Check if the rover is stuck #
        ###############################

        # TODO: Implement latching of stuck state depending on stuck / unstuck criteria

        # Check the stuck status every 10 frames (2 Hz)
        if self.frame % 10 == 0:
            self.is_stuck = self.check_if_stuck(rover_global)

            if self.is_stuck:
                print("Rover is stuck!")

        # If the rover is stuck, we need to unstuck it
        # TODO: Implement unstuck sequence
        if self.is_stuck:
            pass

        ######################################
        # Check if the goal has been reached #
        ######################################

        # Check if the goal has been updated
        if self.prev_goal_location != self.navigator.get_goal_loc():
            # Save the frame the goal was updated
            self.frame_goal_updated = self.frame
            self.prev_goal_location = self.navigator.get_goal_loc()

        # Check out long it has been since the goal was updated (reached)
        elif self.frame - self.frame_goal_updated > self.no_update_threshold:
            print(
                f"Goal not reached in {self.frame - self.frame_goal_updated} frames, ending mission..."
            )
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        #########################
        # Run navigation system #
        #########################
        # Add boulder detections from this frame to the navigator list(x, y, r)
        self.navigator.add_large_boulder_detection(large_boulders_global)

        # Get the control inputs
        self.goal_lin_vel, self.goal_ang_vel = self.navigator(rover_global, input_data)

        # Log the control input
        # NOTE: Remove in submission
        self.recorder.record_custom(
            self.frame,
            "control",
            {"linear": self.goal_lin_vel, "angular": self.goal_ang_vel},
        )
        return carla.VehicleVelocityControl(self.goal_lin_vel, self.goal_ang_vel)

        ######################################################
        # TODO FIGURE OUT WHAT NEEDS TO BE KEPT FROM HERE ON #
        ######################################################

        current_position = (
            (estimate[0, 3], estimate[1, 3]) if estimate is not None else None
        )

        # TODO: THIS LOOKS LIKE IT SHOULD BE KEPT #
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

    def finalize(self):
        # NOTE: Remove in submission
        self.recorder.stop()
        cv2.destroyAllWindows()

        # Prep the surface and boulder maps
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
