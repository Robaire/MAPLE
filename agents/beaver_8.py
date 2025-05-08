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

import importlib
import traceback
from collections import defaultdict
from math import radians

import carla
import cv2
import matplotlib.pyplot as plt
import numpy as np

# from lac_data import Recorder
# from pynput import keyboard
from pytransform3d.transformations import concat

from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.pose import DoubleSlamEstimator
from maple.surface.map import SurfaceHeight, sample_lander, sample_surface
from maple.stuck import StuckDetector

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
        # listener = keyboard.Listener(on_release=self.on_release)
        # listener.start()

        # Camera resolution
        self._width = 1280
        self._height = 720

        # Initialize the frame counter
        self.frame = 0  # Frame gets stepped at the beginning of run_step

        # Initialize the recorder (for testing only)
        # NOTE: Remove in submission
        # self.recorder = Recorder(self, "/recorder/beaver_8.lac", 10)
        # self.recorder.description("Beaver 8, images 10 Hz")

        # Initialize the sample list
        self.sample_list = []  # Surface samples (x, y, z)
        self.sample_list.extend(sample_lander(self))  # Add samples from the lander feet

        # Initialize the ORB detector
        # self.orb = cv2.ORB_create()  # TODO: This doesn't get called anywhere?

        # Initialize the pose estimator
        # self.estimator = DoubleSlamEstimator(self)
        self.estimator = None
        self.last_rover_global = carla_to_pytransform(self.get_initial_position())
        self.last_gt_rover_global = self.last_rover_global
        self.last_any_failures = 0

        # Initialize the navigator
        self.navigator = Navigator(self)
        self.navigator.add_large_boulder_detection([[0, 0, 2.5]])  # Add the lander
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
        self.all_large_boulder_detections = []  # This is a list of (x, y, r)

        # The most recent set of large boulders detected
        self.large_boulder_detections = []  # This is a list of (x, y, r)

        # For plotting
        self.front_boulder_detections = []  # This is a list of (x, y) coordinates
        self.rear_boulder_detections = []  # This is a list of (x, y) coordinates

        # Not sure what this is for but it get used in finalize?
        self.g_map_testing = self.get_geometric_map()
        self.map_length_testing = self.g_map_testing.get_cell_number()

        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_height(i, j, 0)
                self.g_map_testing.set_cell_rock(i, j, 0)

        # Stuck detection parameters
        # Because this only checks on frames with images, the true frame count is 2 x 500 = 1000
        self.stuck_detector = StuckDetector(2000, 2.0, 2.0)
        self.stuck_detector.position_history.append(
            carla_to_pytransform(self.get_initial_position())[:2, 3].tolist()
        )

        # NOTE: Remove in submission
        # Extract the rock locations from the preset file
        # self.gt_rock_locations = extract_rock_locations("resources/Preset_1.xml")
        with importlib.resources.path("resources", "Preset_1.xml") as fpath:
            self.gt_rock_locations = extract_rock_locations(fpath)

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

    def run_step(self, input_data):
        """Execute one step of navigation"""
        try:
            self.frame += 1

            if self.frame % 2 == 0:
                print("Frame: ", self.frame)

            # self.recorder.record_all(self.frame, input_data)
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

        if self.frame % 3000 == 0:
            self.navigator.obstacles = [self.navigator.lander_obstacle]

        if self.frame > 3500:
            print(f"Reached {self.frame} frames, ending mission...")
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        # Save a plot every 500 frames
        if self.frame % 50 == 0:
            plot_poses_and_nav(
                self.last_rover_global,
                None,
                self.last_gt_rover_global,
                self.frame,
                self.navigator.get_goal_loc(),
                self.navigator.static_path.get_full_path(),
                self.front_boulder_detections,
                self.rear_boulder_detections,
                self.navigator.get_obstacle_locations(),
                self.gt_rock_locations,
            )

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
            # self.recorder.record_custom(
            #     self.frame, "control", {"linear": 0, "angular": 0}
            # )
            return carla.VehicleVelocityControl(0.0, 0.0)

        ######################################
        # At this point we can begin driving #
        ######################################

        # On odd frames, we don't have images, so we can't estimate, just carry on with the next navigation step
        if self.frame % 2 != 0:
            # Log the control input
            # self.recorder.record_custom(
            #     self.frame,
            #     "control",
            #     {"linear": self.goal_lin_vel, "angular": self.goal_ang_vel},
            # )
            # Just return the last command input
            return carla.VehicleVelocityControl(self.goal_lin_vel, self.goal_ang_vel)

        ######################################################################
        # At this point we have images, so we can estimate and do detections #
        ######################################################################

        # Get the pose
        # This will be none on frames without images (odd frames)
        # This will always be the rover in the global frame
        rover_global = self.estimator.estimate(input_data)
        # print("rover global: ", rover_global)
        self.last_rover_global = rover_global

        # If possible, get the ground truth pose
        try:
            self.last_gt_rover_global = carla_to_pytransform(self.get_transform())
        except Exception:
            pass

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
        # x, y, z, roll, pitch, yaw = pytransform_to_tuple(rover_global)
        # self.recorder.record_custom(
        #     self.frame,
        #     "estimate",
        #     {
        #         "x": x,
        #         "y": y,
        #         "z": z,
        #         "roll": roll,
        #         "pitch": pitch,
        #         "yaw": yaw,
        #         "source": estimate_source,
        #     },
        # )

        # TODO: Decide what to do based on the estimate source (if anything)
        if estimate_source == "front" or estimate_source == "rear":
            pass

        # Track the number of times the last_any estimate fails
        # Only start tracking after 60 seconds so the rover has time to get moving
        if self.frame > 60 * 20:
            if estimate_source == "last_any":
                self.last_any_failures += 1

            # TODO: This should probably be the number of failures in a row (or the last x frames)
            if self.last_any_failures > 1000:
                print(
                    f"Pose tracking failed {self.last_any_failures} times, ending mission..."
                )
                self.mission_complete()
                return carla.VehicleVelocityControl(0.0, 0.0)

        ##########################
        # Run boulder detections #
        ##########################

        # Run detections every 20 frames (1 Hz) unless the estimate is failing
        if self.frame % 20 == 0 and estimate_source != "last_any":
            print("Running boulder detection...")

            # Detections in the rover frame
            front_detections, front_ground_points = self.front_detector(input_data)
            rear_detections, rear_ground_points = self.rear_detector(input_data)

            # It looks like this returns boulder (x, y, z), we convert to (x, y, r) for the navigator
            front_large_boulders = self.front_detector.get_large_boulders()

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
            # We replace the current list of detections since the navigator compiles the list internally
            self.large_boulder_detections = [
                large_boulder[:2, 3].tolist() + [0.3]  # radius is 0.3 in beaver_7
                for large_boulder in large_boulders_global
                if np.linalg.norm(large_boulder[:2, 3] - rover_global[:2, 3]) <= 2
            ]

            # For plotting only
            self.all_large_boulder_detections.extend(self.large_boulder_detections)
            self.front_boulder_detections.extend(
                [concat(br, rover_global)[:2, 3].tolist() for br in front_detections]
            )
            self.rear_boulder_detections.extend(
                [concat(br, rover_global)[:2, 3].tolist() for br in rear_detections]
            )

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

        # Latches the stuck state if the rover is stuck for too long
        if self.stuck_detector.is_stuck(rover_global):
            print("Rover is stuck! Attempting to get free...")

            # Add the current location to the navigator as an obstacle
            # self.navigator.add_large_boulder_detection(
            #     [np.array(rover_global[:2, 3].tolist() + [0.7])]
            # )

            # Get the control input to get unstuck
            self.goal_lin_vel, self.goal_ang_vel = (
                self.stuck_detector.get_unstuck_control()
            )
            return carla.VehicleVelocityControl(self.goal_lin_vel, self.goal_ang_vel)

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
        self.navigator.add_large_boulder_detection(self.large_boulder_detections)

        # Get the control inputs
        self.goal_lin_vel, self.goal_ang_vel = self.navigator(rover_global, input_data)

        # Log the control input
        # NOTE: Remove in submission
        # self.recorder.record_custom(
        #     self.frame,
        #     "control",
        #     {"linear": self.goal_lin_vel, "angular": self.goal_ang_vel},
        # )
        return carla.VehicleVelocityControl(self.goal_lin_vel, self.goal_ang_vel)

    def finalize(self):
        # Save the plot
        plot_poses_and_nav(
            self.last_rover_global,
            None,
            self.last_gt_rover_global,
            self.frame,
            self.navigator.get_goal_loc(),
            self.navigator.static_path.get_full_path(),
            self.front_boulder_detections,
            self.rear_boulder_detections,
            self.navigator.get_obstacle_locations(),
            self.gt_rock_locations,
        )

        # NOTE: Remove in submission
        # self.recorder.stop()
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
        # if key == keyboard.Key.esc:
        #     self.mission_complete()


def plot_poses_and_nav(
    transformed_estimate: np.ndarray,
    transformed_estimate_back: np.ndarray,
    real_pose: np.ndarray,
    frame_number: int,
    goal_location: np.ndarray,
    all_goals: list,
    front_boulder_detections: list,  # Format: [(x, y), (x, y), ...]
    rear_boulder_detections: list,  # Format: [(x, y), (x, y), ...]
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

    default_radius = 0.15

    # Draw boulder detections
    if front_boulder_detections is not None:
        for boulder in front_boulder_detections:
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
            label="Front Boulders",
        )
        legend_elements.append(regular_boulder)

    # Draw boulder detections
    if rear_boulder_detections is not None:
        # Default radius for regular boulders if not specified
        default_radius = 0.15

        for boulder in rear_boulder_detections:
            # Plot boulder center point
            ax.scatter(boulder[0], boulder[1], color="blue", s=20, alpha=0.7)
            # Plot circle representing boulder size
            circle = plt.Circle(
                (boulder[0], boulder[1]),
                default_radius,
                color="blue",
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
            markerfacecolor="blue",
            markersize=10,
            label="Rear Boulders",
        )
        legend_elements.append(regular_boulder)

    # Draw large boulder detections
    if large_boulder_detections is not None:
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
    # if all_goals is not None and len(all_goals) > 0:
    #     waypoints = np.array(all_goals)
    #     ax.scatter(waypoints[:, 0], waypoints[:, 1], color="magenta", s=20, alpha=0.7)
    #     waypoint_marker = plt.Line2D(
    #         [0],
    #         [0],
    #         marker="o",
    #         color="w",
    #         markerfacecolor="magenta",
    #         markersize=8,
    #         label="All Goals",
    #     )
    #     legend_elements.append(waypoint_marker)
    if all_goals is not None and len(all_goals) > 0:
        waypoints = np.array(all_goals)

        # Extract x, y coordinates and weights
        x = waypoints[:, 0]
        y = waypoints[:, 1]
        weights = waypoints[:, 2]  # The third value is the weight

        # Create a colormap that goes from red (low values) to green (high values)
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap

        # Scatter plot with color based on weight values
        scatter = ax.scatter(
            x,
            y,
            c=weights,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,  # Set the range for the colormap
            s=40,
            alpha=0.9,
        )

        # Add a colorbar to show the mapping between colors and weights
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Weight Value")

        # Update the legend
        waypoint_marker = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(0.8),  # Use a high value in the colormap (green)
            markersize=8,
            label="All Goals",
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
    plt.savefig(f"recorder/{frame_number}.png")
    plt.close(fig)
