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

import csv
import os
import random
from math import radians
import matplotlib.pyplot as plt
import traceback
from numpy import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

import pandas as pd


import carla
import cv2 as cv
import numpy as np
from pynput import keyboard
from pytransform3d.transformations import concat

from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.pose import InertialApriltagEstimator, PoseGraph
from maple import utils
from maple.utils import *
from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
from maple.surface.post_processing import PostProcessor

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
        models for processing sensor data or control, you should load the models here. We encourage the use of class attributes in place
        of using global variables which can cause conflicts."""

        """ Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys. """

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ Add some attributes to store values for the target linear and angular velocity. """

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

        # Initialize the plot
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.show(block=False)  # Show the plot window without blocking

        """ Initialize a counter to keep track of the number of simulation steps. """

        self.frame = 1

        # set the trial number here
        self.trial = "032"

        if not os.path.exists(f"./data/{self.trial}"):
            os.makedirs(f"./data/{self.trial}")

        self.checkpoint_path = f"./data/{self.trial}/boulders_frame{self.frame}.json"

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

        self.g_map_testing = self.get_geometric_map()
        self.map_length_testing = self.g_map_testing.get_cell_number()

        print("map length:", self.map_length_testing)

        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_height(i, j, 0)
                self.g_map_testing.set_cell_rock(i, j, 0)

        self.all_boulder_detections = []
        self.large_boulder_detections = [(0, 0, 2.5)]

        self.gt_rock_locations = extract_rock_locations(
            "simulator/LAC/Content/Carla/Config/Presets/Preset_1.xml"
        )

        # Load the pickled numpy array from the file
        file_path = 'Moon_Map_01_0_rep0.dat'
        with open(file_path, 'rb') as file:
            self.grid_data = pickle.load(file)

        self.sample_list.extend(sample_lander(self))

        # Add position tracking for stuck detection
        self.position_history = []
        self.is_stuck = False
        self.unstuck_phase = 0
        self.unstuck_counter = 0
        self.MAX_STUCK_FRAMES = 300
        self.STUCK_DISTANCE_THRESHOLD = 0.5
        self.unstuck_sequence = [
            {"lin_vel": 0.45, "ang_vel": 0, "frames": 60},      # Forward
            {"lin_vel": -0.45, "ang_vel": 0, "frames": 60},     # Backward
            {"lin_vel": 0, "ang_vel": 4, "frames": 60},         # Rotate clockwise
            {"lin_vel": 0, "ang_vel": -4, "frames": 60}         # Rotate counter-clockwise
        ]

    def visualize_surface(self, predicted_array):
        """
        Save visualization of surface heights and their agreement.
        Shows ground truth, predicted heights, and agreement map side by side.
        
        Args:
            predicted_array: Array of shape (n, n, 4) containing [x, y, height, _] values
        """
        # Create figure with three subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(45, 15))
        plt.clf()  # Clear the current figure
        
        # Create a new figure after clearing
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(45, 15))
        
        # Create a range of ticks at 0.15 m intervals from -10 to 10
        x_ticks = np.arange(-10, 10.15, 0.15)
        y_ticks = np.arange(-10, 10.15, 0.15)
        
        # Function to set up common axis properties
        def setup_axis(ax, title):
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.grid(
                True,             # Turn on the grid
                which='both',     # Grid for both major and minor ticks
                color='lightgray',
                linestyle='-',
                linewidth=0.5,
                alpha=0.5         # Adjust alpha for desired lightness
            )
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)
        
        # Set up axes
        setup_axis(ax1, f"Frame {self.frame}: Ground Truth Surface Heights")
        setup_axis(ax2, f"Frame {self.frame}: Predicted Surface Heights")
        setup_axis(ax3, f"Frame {self.frame}: Height Agreement (±5cm)")
        
        # Extract the ground truth height data
        total_points = self.grid_data.shape[1]
        mid_point = total_points // 2
        
        # Calculate indices for -10 to 10 meter range
        cells_per_meter = 1 / 0.15  # cells per meter (15cm per cell)
        cells_to_edge = int(10 * cells_per_meter)
        start_index = mid_point - cells_to_edge
        end_index = mid_point + cells_to_edge
        
        # Extract the height data for our viewing window
        heights_gt = self.grid_data[start_index:end_index, start_index:end_index, 2]
        
        # Extract heights from predicted array and reshape to match ground truth
        heights_pred = predicted_array[:, :, 2]  # Get just the height values
        heights_pred = heights_pred[start_index:end_index, start_index:end_index]
        
        # Create agreement map (green where difference <= 5cm, red otherwise)
        height_diff = np.abs(heights_gt - heights_pred)
        agreement_map = np.where(height_diff <= 0.05, 1, 0)  # 1 for agreement (green), 0 for disagreement (red)
        
        # Create coordinate meshes for pcolor
        x_coords = np.arange(-10, 10.15, 0.15)[:heights_gt.shape[1]]
        y_coords = np.arange(-10, 10.15, 0.15)[:heights_gt.shape[0]]
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Plot all three visualizations
        mesh1 = ax1.pcolor(X, Y, heights_gt, cmap='viridis', alpha=0.7)
        mesh2 = ax2.pcolor(X, Y, heights_pred, cmap='viridis', alpha=0.7)
        mesh3 = ax3.pcolor(X, Y, agreement_map, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, alpha=0.7)
        
        # Add colorbars
        plt.colorbar(mesh1, ax=ax1, label='Height (m)')
        plt.colorbar(mesh2, ax=ax2, label='Height (m)')
        
        # Custom colorbar for agreement map
        cbar3 = plt.colorbar(mesh3, ax=ax3, ticks=[0, 1])
        cbar3.set_ticklabels(['> 5cm diff', '≤ 5cm diff'])
        
        # Calculate and display percentage of cells within tolerance
        agreement_percentage = np.mean(agreement_map) * 100
        ax3.set_title(f"Frame {self.frame}: Height Agreement (±5cm)\n{agreement_percentage:.1f}% within tolerance")
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the figure
        filename = f"{self.surface_plots_dir}/surface_grid_{self.frame:06d}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

    def visualize_detections(self, gt_pos, agent_pos, goal_loc, goal_locs_all, goal_locs_rrt, new_detections, old_detections, large_boulders):
        """
        Save visualization of agent position and boulder detections as matplotlib figures.
        """
        plt.figure(figsize=(15, 15))
        plt.clf()  # Clear the current figure

        # Create a range of ticks at 0.15 m intervals from -10 to 10
        x_ticks = np.arange(-10, 10.15, 0.15)
        y_ticks = np.arange(-10, 10.15, 0.15)

        # Set up the custom ticks and very light grid lines
        ax = plt.gca()
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])

        # Customize the grid to be lighter
        ax.grid(
            True,             # Turn on the grid
            which='both',     # Grid for both major and minor ticks
            color='lightgray',
            linestyle='-',
            linewidth=0.5,
            alpha=0.5         # Adjust alpha for desired lightness
        )


        # Plot ground truth rock locations as black X's
        if hasattr(self, "gt_rock_locations") and self.gt_rock_locations:
            gt_x, gt_y = zip(*[(float(x), float(y)) for x, y, _ in self.gt_rock_locations])
            plt.scatter(gt_x, gt_y, c="black", marker="x", s=20, label="GT Rocks")

        # Plot old detections in gray
        if old_detections:
            old_x, old_y = zip(*[(float(x), float(y)) for x, y in old_detections])
            plt.scatter(
                old_x,
                old_y,
                c="gray",
                marker="o",
                s=10,
                label="Previous Boulders",
                alpha=0.5,
            )

        # Plot new detections in red
        if new_detections:
            new_x, new_y = zip(*[(float(x), float(y)) for x, y in new_detections])
            plt.scatter(new_x, new_y, c="red", marker="o", s=10, label="New Boulders")

        if goal_locs_all:
            goal_x, goal_y = zip(*[(float(x), float(y)) for x, y in goal_locs_all])
            plt.scatter(goal_x, goal_y, c="green", marker="x", s=30, label="Goal Locations")

        if goal_locs_rrt:
            goal_x, goal_y = zip(*[(float(x), float(y)) for x, y in goal_locs_rrt])
            plt.scatter(goal_x, goal_y, c="purple", marker="x", s=30, label="Goal Locations")

        if large_boulders:
            # Unpack x, y, and radius from the tuples
            big_x, big_y, rad = zip(*[(float(x), float(y), float(r)) for x, y, r in large_boulders])
            
            # Plot the center points
            plt.scatter(big_x, big_y, c="red", marker="o", s=10, label="Large Boulders")
            
            # Get current axes to add circles
            ax = plt.gca()
            
            # For each boulder, create and add a circle patch
            for x_i, y_i, r_i in zip(big_x, big_y, rad):
                circle = plt.Circle((x_i, y_i), r_i, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(circle)

        # Plot agent position as a blue X
        if agent_pos is not None:
            plt.scatter(
                agent_pos[0], agent_pos[1], c="blue", marker="X", s=200, label="Agent"
            )
        if gt_pos is not None:
            plt.scatter(
                gt_pos[0], gt_pos[1], c="orange", marker="X", s=200, label="GT Position"
            )
        if goal_loc is not None:
            plt.scatter(
                goal_loc[0], goal_loc[1], c="green", marker="X", s=200, label="Goal Location"
            )

        plt.title(f"Frame {self.frame}: Boulder Detections")
        plt.legend()

        # Save the figure
        filename = os.path.join(self.plots_dir, f"frame_{self.frame:06d}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

    def check_if_stuck(self, current_position):
        """
        Check if the rover has been stuck for the last MAX_STUCK_FRAMES frames.
        Returns True if stuck, False otherwise.
        """
        if current_position is None:
            return False
            
        # Add current position to history
        self.position_history.append(current_position)
        
        # Keep only the last MAX_STUCK_FRAMES positions
        if len(self.position_history) > self.MAX_STUCK_FRAMES:
            self.position_history.pop(0)
            
        # Need at least MAX_STUCK_FRAMES positions to determine if stuck
        if len(self.position_history) < self.MAX_STUCK_FRAMES:
            return False
            
        # Get the oldest position in our history
        old_position = self.position_history[0]
        
        # Calculate distance moved
        dx = current_position[0] - old_position[0]
        dy = current_position[1] - old_position[1]
        distance_moved = np.sqrt(dx**2 + dy**2)
        
        # If we've moved less than the threshold, we're stuck
        if distance_moved < self.STUCK_DISTANCE_THRESHOLD:
            print(f"STUCK DETECTED! Moved only {distance_moved:.2f}m in the last {self.MAX_STUCK_FRAMES} frames.")
            return True
        
        return False

    def get_unstuck_control(self):
        """
        Execute the unstuck sequence and return appropriate velocity controls.
        Returns a tuple of (linear_velocity, angular_velocity)
        """
        # Get the current phase of the unstuck sequence
        current_phase = self.unstuck_sequence[self.unstuck_phase]
        
        # Apply the velocities for this phase
        lin_vel = current_phase["lin_vel"]
        ang_vel = current_phase["ang_vel"]
        
        # Increment the counter
        self.unstuck_counter += 1
        
        # If we've completed this phase, move to the next one
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

        sensor_data_frontleft = input_data["Grayscale"][carla.SensorPosition.FrontLeft]

        if sensor_data_frontleft is not None:
            cv.imshow("Left camera view", sensor_data_frontleft)
            cv.waitKey(1)
    
        agent_position = None

        # Get a position estimate for the rover
        estimate, is_april_tag_estimate = self.estimator(input_data)

        current_position = (estimate[0, 3], estimate[1, 3]) if estimate is not None else None

        if current_position is not None:
            if not self.is_stuck:
                self.is_stuck = self.check_if_stuck(current_position)
            else:
                # Check if we've moved enough to consider ourselves unstuck
                if len(self.position_history) > 0:
                    old_position = self.position_history[0]
                    dx = current_position[0] - old_position[0]
                    dy = current_position[1] - old_position[1]
                    distance_moved = np.sqrt(dx**2 + dy**2)
                    
                    if distance_moved > self.STUCK_DISTANCE_THRESHOLD:
                        print(f"UNSTUCK! Moved {distance_moved:.2f}m - resuming normal operation.")
                        self.is_stuck = False
                        self.unstuck_phase = 0
                        self.unstuck_counter = 0
                        # Clear position history to reset stuck detection
                        self.position_history = []

        # # TODO: this is just testing script
        # if self.frame%80==0:
        #     areas = self.detector.get_boulder_sizes()

        #     if areas is not None:
        #         print("areas: ", areas)

        # IMPORTANT NOTE: For developing using the exact location
        # real_position = carla_to_pytransform(self.get_transform())
        real_position = None
        

        print(f"Frame number: {self.frame}")

        stopped = False

        goal_loc = self.navigator.get_goal_loc()

        # obstacles = self.navigator.get_obstacle_locations()

        # if self.frame == 20:
        #     self.navigator.add_large_boulder_detection([(-4, 0, 1)])
        #     self.large_boulder_detections.extend([(0, 0, 2), (-1, -3.7, 0.5)])

        # print("obstacle locations", obstacles)

        goal_locations_all = self.navigator.get_all_goal_locations()
        goal_locations_rrt = self.navigator.get_rrt_waypoints()

        self.good_loc = False
        # check if some wild far off localization
        #TODO: make this reflect imu/controller expected position instead of this
        for goal_location in goal_locations_rrt:
            current_arr = np.array(current_position)
            goal_arr = np.array(goal_location)

            # Now subtract and compute the Euclidean distance
            distance = np.linalg.norm(current_arr - goal_arr)
            if distance < 1.0:
                # Skip this goal location if not within 1 meter
                self.good_loc = True

        if self.good_loc is False:
            print("bad localization - not within waypoints")


        if self.is_stuck:
            goal_lin_vel, goal_ang_vel = self.get_unstuck_control()
            print(f"UNSTUCK MANEUVER: lin_vel={goal_lin_vel}, ang_vel={goal_ang_vel}, phase={self.unstuck_phase}, counter={self.unstuck_counter}")
        else:
        # Determine where we are in the 150-frame cycle
            phase = self.frame % 150

            if phase < 30:
                # Phase 1: Frames 0–49
                # ---------------------------------------
                # 1) We want to STOP here.
                goal_lin_vel = 0.0
                goal_ang_vel = 0.0

                stopped = False

            elif phase < 80:
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
                        detections, ground_points = self.detector(input_data)

                        large_boulders_detections = self.detector.get_large_boulders()

                        detections_back, ground_points_back = self.detectorBack(input_data)
                        print(f"Boulder Detections: {len(detections)}")

                        # Get all detections in the world frame
                        # rover_world = utils.carla_to_pytransform(self.get_transform())
                        rover_world = estimate
                        boulders_world = [
                            concat(boulder_rover, rover_world) for boulder_rover in detections
                        ]

                        boulders_world_back = [
                            concat(boulder_rover, rover_world) for boulder_rover in detections_back
                        ]

                        large_boulders_detections = [
                            concat(boulder_rover, rover_world) for boulder_rover in large_boulders_detections
                        ]

                        large_boulders_xyr = [
                            (b_w[0, 3], b_w[1, 3], 0.25)
                            for b_w in large_boulders_detections
                        ]

                        # Now pass the (x, y, r) tuples to your navigator or wherever they need to go
                        if self.good_loc:
                            self.navigator.add_large_boulder_detection(large_boulders_xyr)
                            self.large_boulder_detections.extend(large_boulders_xyr)

                        # If you just want X, Y coordinates as a tuple
                        boulders_xy = [(b_w[0, 3], b_w[1, 3]) for b_w in boulders_world]
                        boulders_xy_back = [(b_w[0, 3], b_w[1, 3]) for b_w in boulders_world_back]

                        # TODO: Not sure what exactly you're trying to do here but I think this is it
                        if self.good_loc:
                            self.all_boulder_detections.extend(boulders_xy)
                            self.all_boulder_detections.extend(boulders_xy_back)
                        """
                        # add all boulders to boulder detection list
                        self.all_boulder_detections.append(boulders_xy)

                        for b_w in boulders_world:
                            self.all_boulder_detections.append((b_w[0, 3], b_w[1, 3]))

                        """

                        print(
                            "shape of all detections: ", np.shape(self.all_boulder_detections)
                        )

                        # The correct list is already in xy_boulders, no need for additional comprehension
                        new_boulder_positions = boulders_xy

                        if self.good_loc:
                            new_boulder_positions.extend(boulders_xy_back)


                        # Get agent position in world frame for visualization
                        # agent_position = (gt_pose[0, 3], gt_pose[1, 3])
                        agent_position = (estimate[0, 3], estimate[1, 3])
                        if real_position is not None:
                            gt_position = (real_position[0,3], real_position[1,3])
                        else:
                            gt_position = None
                        # Visualize the map with agent and boulder positions

                        if self.good_loc:
                            self.visualize_detections(
                                gt_position, agent_position, goal_loc, goal_locations_all, goal_locations_rrt, new_boulder_positions, self.all_boulder_detections, self.large_boulder_detections
                            )

                        # Update previous detections with a copy of the current detections
                        self.previous_detections = new_boulder_positions.copy()

                    except Exception as e:
                        print(f"Error processing detections: {e}")
                        print(f"Error details: {str(e)}")
                        traceback.print_exc()  # This will print the full stack trace


            else:
                # Phase 3: Frames 100–149
                # ---------------------------------------
                # 3) Go back to what the navigator says
                goal_lin_vel, goal_ang_vel = self.navigator(estimate)

                stopped = False

        # After handling the phases, increment the frame counter
        self.frame += 1

        # Finally, apply the resulting velocities
        control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)

        # Generate and add in the sample points
        if is_april_tag_estimate and stopped and phase%10==0 and self.good_loc:
            self.sample_list.extend(sample_surface(estimate))

            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection='3d')
            ax.clear()  # Clear the current figure

            # Extract all x, y, z for plotting
            xs = [xyz[0] for xyz in self.sample_list]
            ys = [xyz[1] for xyz in self.sample_list]
            zs = [xyz[2] for xyz in self.sample_list]
            ax.scatter(xs, ys, zs, c="black", marker="o", label="Ground Points")  # Plot ground points

            # # Extract all x, y, z for plotting
            # xs = [xyz[0] for xyz in self.ground_truth_sample_list]
            # ys = [xyz[1] for xyz in self.ground_truth_sample_list]
            # zs = [xyz[2] for xyz in self.ground_truth_sample_list]
            # ax.scatter(xs, ys, zs, c="green", marker="o", label="Ground Truth Ground Points")  # Plot ground points

            # ax.scatter(real_position[0, 3], real_position[1,3], real_position[2,3], c="green", marker="X", s=200, label="Ground Truth")
            ax.scatter(estimate[0, 3], estimate[1, 3], estimate[2, 3], c="black", marker="X", s=200, label="Agent")

            # Setting labels
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position')

            ax.set_title(f"Frame {self.frame}: Boulder Detections")
            ax.legend()

            # Save the figure
            filename = os.path.join(self.surface_plots_dir, f"frame_{self.frame:06d}.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
    
        return control
    
    def finalize(self):

        g_map = self.get_geometric_map()
        gt_map_array = g_map.get_map_array()

        N = gt_map_array.shape[0]  # should be 179 if you are spanning -13.425 to 13.425 by 0.15
        x_min, y_min = -13.425, -13.425
        resolution = 0.15

        # Calculate indices for center 2x2m region
        center_x_min_idx = int(round((-0.5 - x_min) / resolution))  # -.5m in x
        center_x_max_idx = int(round((0.5 - x_min) / resolution))   # +.5m in x
        center_y_min_idx = int(round((-0.5 - y_min) / resolution))  # -.5m in y
        center_y_max_idx = int(round((0.5 - y_min) / resolution))   # +.5m in y

        # setting all rock locations to 0
        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_rock(i, j, 0)

        # Create clusters based on grid cells
        from collections import defaultdict
        import numpy as np
        
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
            if len(detections) < 3:
                continue
            
            final_clusters.extend(clusters[(i, j)])
                
            # Skip if in center region
            if (center_x_min_idx <= i <= center_x_max_idx and 
                center_y_min_idx <= j <= center_y_max_idx):
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
        
        # Update all_boulder_detections with the filtered and clustered detections
        self.all_boulder_detections = [list(detection) for detection in filtered_detections]

        self.visualize_detections(
            None, None, None, None, None, final_clusters, self.all_boulder_detections, self.large_boulder_detections
        )

        # TODO set everything within a certain radius of the lander to 0 for the rocks

        # Initialize the data class to get estimates for all the squares
        surfaceHeight = SurfaceHeight(g_map)
        
        # Generate the actual map with the sample list
        if len(self.sample_list) > 0:
            surfaceHeight.set_map(self.sample_list)

        # Assuming self.sample_list is already populated with your data
        df = pd.DataFrame(self.sample_list)
        df.to_csv('output_sample_list.csv', index=False)  # This will save the dataframe to a CSV file without the index


        print(f'we are getting a map of {g_map.get_map_array()}')

        self.visualize_surface(g_map.get_map_array())

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
