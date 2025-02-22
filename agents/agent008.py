#!/usr/bin/env python

# THIS IS THE MOST FUNCTIONAL AGENT RIGHT NOW WITH EVALUATION
# IT NEEDS AN UPDATED NAV MODULE - FOR SOME REASON THAT PART IS WEIRD

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
from maple.surface.map import SurfaceHeight, sample_surface
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

        # Store previous boulder detections
        self.previous_detections = []

        # Initialize the plot
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.show(block=False)  # Show the plot window without blocking

        """ Initialize a counter to keep track of the number of simulation steps. """

        self.frame = 1

        self.columns = [
            "frame",
            "power",
            "input_v",
            "input_w",
            "gt_x",
            "gt_y",
            "gt_z",
            "gt_roll",
            "gt_pitch",
            "gt_yaw",
            "imu_accel_x",
            "imu_accel_y",
            "imu_accel_z",
            "imu_gyro_x",
            "imu_gyro_y",
            "imu_gyro_z",
        ]
        self.imu = []

        # set the trial number here
        self.trial = "016"

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

        self.g_map_testing = self.get_geometric_map()
        self.map_length_testing = self.g_map_testing.get_cell_number()

        print("map length:", self.map_length_testing)

        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_height(i, j, 0)
                self.g_map_testing.set_cell_rock(i, j, 0)

        self.all_boulder_detections = []

        self.gt_rock_locations = extract_rock_locations(
            "simulator/LAC/Content/Carla/Config/Presets/Preset_1.xml"
        )

    def visualize_detections(self, gt_pos, agent_pos, goal_loc, new_detections, old_detections):
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

        # print("Old detections data:", old_detections)
        # print("New detections data:", new_detections)
        # print("Number of old detections:", len(old_detections))
        # print("Number of new detections:", len(new_detections))

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

    # def visualize_detections(self, agent_pos, new_detections, old_detections):
    #     """
    #     Save visualization of agent position and boulder detections as matplotlib figures.
    #     """
    #     plt.figure(figsize=(15, 15))
    #     plt.clf()  # Clear the current figure

    #     # Set up the plot
    #     plt.grid(True)
    #     plt.xlim([-10, 10])
    #     plt.ylim([-10, 10])

    #     print("Old detections data:", old_detections)
    #     print("New detections data:", new_detections)
    #     print("Number of old detections:", len(old_detections))
    #     print("Number of new detections:", len(new_detections))

    #     # Plot ground truth rock locations as black X's
    #     if hasattr(self, "gt_rock_locations") and self.gt_rock_locations:
    #         gt_x, gt_y = zip(
    #             *[(float(x), float(y)) for x, y, _ in self.gt_rock_locations]
    #         )
    #         plt.scatter(gt_x, gt_y, c="black", marker="x", s=20, label="GT Rocks")

    #     # Plot old detections in gray
    #     if old_detections:
    #         old_x, old_y = zip(*[(float(x), float(y)) for x, y in old_detections])
    #         plt.scatter(
    #             old_x,
    #             old_y,
    #             c="gray",
    #             marker="o",
    #             s=10,
    #             label="Previous Boulders",
    #             alpha=0.5,
    #         )

    #     # Plot new detections in red
    #     if new_detections:
    #         new_x, new_y = zip(*[(float(x), float(y)) for x, y in new_detections])
    #         plt.scatter(new_x, new_y, c="red", marker="o", s=10, label="New Boulders")

    #     # Plot agent position as a blue X
    #     if agent_pos is not None:
    #         plt.scatter(
    #             agent_pos[0], agent_pos[1], c="blue", marker="X", s=200, label="Agent"
    #         )

    #     plt.title(f"Frame {self.frame}: Boulder Detections")
    #     plt.legend()

    #     # Save the figure
    #     filename = os.path.join(self.plots_dir, f"frame_{self.frame:06d}.png")
    #     plt.savefig(filename, dpi=300, bbox_inches="tight")
    #     plt.close()  # Close the figure to free memory

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
                "width": "1920",
                "height": "1080",
                "use_semantic": False,
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1920",
                "height": "1080",
                "use_semantic": False,
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1920",
                "height": "1080",
                "use_semantic": False,
            },
            carla.SensorPosition.Left: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1920",
                "height": "1080",
                "use_semantic": False,
            },
            carla.SensorPosition.Right: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1920",
                "height": "1080",
                "use_semantic": False,
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1920",
                "height": "1080",
                "use_semantic": False,
            },
            carla.SensorPosition.BackRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1920",
                "height": "1080",
                "use_semantic": False,
            },
            carla.SensorPosition.Back: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1920",
                "height": "1080",
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
    
        # front_data = input_data['Grayscale'][carla.SensorPosition.Front]  # Do something with this
        # if self._active_side_front_cameras:
        #     front_left_data = input_data["Grayscale"][
        #         carla.SensorPosition.FrontLeft
        #     ]  # Do something with this
        #     front_right_data = input_data["Grayscale"][
        #         carla.SensorPosition.FrontRight
        #     ]  # Do something with this
        # if self._active_side_cameras:
        #     left_data = input_data["Grayscale"][
        #         carla.SensorPosition.Left
        #     ]  # Do something with this
        #     right_data = input_data["Grayscale"][
        #         carla.SensorPosition.Right
        #     ]  # Do something with this
        # mission_time = round(self.get_mission_time(), 2)

        # Get a position estimate for the rover
        # estimate = self.estimator(input_data)
        agent_position = None

        # Get a position estimate for the rover
        estimate = self.estimator(input_data)
        # print(f'the estimator is estimating {estimate}')
        # IMPORTANT NOTE: For developing using the exact location
        # real_position = carla_to_pytransform(self.get_transform())
        real_position = None
        # print(f'the actual thing is {estimate}')
        # IMPORTANT NOTE: The estimate should never be NONE!!!,
        # Get a goal linear and angular velocity from navigation

        # Get the ground truth pose
        # gt_pose = utils.carla_to_pytransform(self.get_transform())

        # IF YOU WANT ELEMENTS OF THE POSE ITS
        # x, y, z, roll, pitch, yaw = utils.pytransform_to_tuple(gt_pose)

        # # IMPORTANT NOTE: The estimate should never be NONE!!!, this is test code to catch that
        # if estimate is None:
        #     goal_lin_vel, goal_ang_vel = 10, 0
        #     print(f"the estimate is returning NONE!!! that is a big problem buddy")
        # else:
        #     # Get a goal linear and angular velocity from navigation
        #     goal_lin_vel, goal_ang_vel = self.navigatior(estimate)
        #     agent_position = (estimate[0, 3], estimate[1, 3])

        #     # print(f"the estimate is {estimate}")
        #     imu_data = self.get_imu_data()
        #     # print(f"the imu data is {imu_data}")

        # Get a goal linear and angular velocity from navigation
        # goal_lin_vel, goal_ang_vel = self.navigator(estimate)
        # goal_lin_vel, goal_ang_vel = 0, 0
        # Set the goal velocities to be returned

        print(f"Frame number: {self.frame}")

        ground_points_world = []
        ground_points_back_world = []

        # # Check for detections
        # if self.frame % 20 == 0:  # Run at 1 Hz
        #     try:
        #         detections, ground_points = self.detector(input_data)
        #         detections_back, ground_points_back = self.detectorBack(input_data)
        #         print(f"Boulder Detections: {len(detections)}")

        #         # Get all detections in the world frame
        #         # rover_world = utils.carla_to_pytransform(self.get_transform())
        #         rover_world = estimate
        #         boulders_world = [
        #             concat(boulder_rover, rover_world) for boulder_rover in detections
        #         ]

        #         boulders_world_back = [
        #             concat(boulder_rover, rover_world) for boulder_rover in detections_back
        #         ]

        #         ground_points_world = [
        #             concat(point_rover, rover_world) for point_rover in ground_points
        #         ]

        #         ground_points_back_world = [
        #             concat(point_rover, rover_world) for point_rover in ground_points_back
        #         ]

        #         # If you just want X, Y coordinates as a tuple
        #         boulders_xy = [(b_w[0, 3], b_w[1, 3]) for b_w in boulders_world]
        #         boulders_xy_back = [(b_w[0, 3], b_w[1, 3]) for b_w in boulders_world_back]

        #         # TODO: Not sure what exactly you're trying to do here but I think this is it
        #         self.all_boulder_detections.extend(boulders_xy)
        #         self.all_boulder_detections.extend(boulders_xy_back)
        #         """
        #         # add all boulders to boulder detection list
        #         self.all_boulder_detections.append(boulders_xy)

        #         for b_w in boulders_world:
        #             self.all_boulder_detections.append((b_w[0, 3], b_w[1, 3]))

        #         """

        #         print(
        #             "shape of all detections: ", np.shape(self.all_boulder_detections)
        #         )

        #         # The correct list is already in xy_boulders, no need for additional comprehension
        #         new_boulder_positions = boulders_xy

        #         new_boulder_positions.extend(boulders_xy_back)

        #         # Debug prints to verify data
        #         # print("Raw xy_boulders (world frame):", boulders_xy)
        #         # print("New boulder positions (world frame):", new_boulder_positions)

        #         # Get agent position in world frame for visualization
        #         # agent_position = (gt_pose[0, 3], gt_pose[1, 3])
        #         agent_position = (estimate[0, 3], estimate[1, 3])
        #         if real_position is not None:
        #             gt_position = (real_position[0,3], real_position[1,3])
        #         else:
        #             gt_position = None
        #         # Visualize the map with agent and boulder positions
        #         self.visualize_detections(
        #             gt_position, agent_position, new_boulder_positions, self.all_boulder_detections
        #         )

        #         # Update previous detections with a copy of the current detections
        #         self.previous_detections = new_boulder_positions.copy()

        #     except Exception as e:
        #         print(f"Error processing detections: {e}")
        #         print(f"Error details: {str(e)}")
        #         traceback.print_exc()  # This will print the full stack trace

        # self.frame += 1

        # Set the goal velocities to be returned
        # control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)

        goal_loc = self.navigator.get_goal_loc()

        # Determine where we are in the 150-frame cycle
        phase = self.frame % 150

        if phase < 50:
            # Phase 1: Frames 0–49
            # ---------------------------------------
            # 1) We want to STOP here.
            goal_lin_vel = 0.0
            goal_ang_vel = 0.0

        elif phase < 100:
            # Phase 2: Frames 50–99
            # ---------------------------------------
            # 2) We want to run boulder detection every 10 frames.
            #    (Keep velocity = 0.0 or whatever you'd like.)
            goal_lin_vel = 0.0
            goal_ang_vel = 0.0

            if phase % 20 == 0:
                # Run boulder detection
                try:
                    detections, ground_points = self.detector(input_data)
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

                    ground_points_world = [
                        concat(point_rover, rover_world) for point_rover in ground_points
                    ]

                    ground_points_back_world = [
                        concat(point_rover, rover_world) for point_rover in ground_points_back
                    ]

                    # If you just want X, Y coordinates as a tuple
                    boulders_xy = [(b_w[0, 3], b_w[1, 3]) for b_w in boulders_world]
                    boulders_xy_back = [(b_w[0, 3], b_w[1, 3]) for b_w in boulders_world_back]

                    # TODO: Not sure what exactly you're trying to do here but I think this is it
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

                    new_boulder_positions.extend(boulders_xy_back)

                    # Debug prints to verify data
                    # print("Raw xy_boulders (world frame):", boulders_xy)
                    # print("New boulder positions (world frame):", new_boulder_positions)

                    # Get agent position in world frame for visualization
                    # agent_position = (gt_pose[0, 3], gt_pose[1, 3])
                    agent_position = (estimate[0, 3], estimate[1, 3])
                    if real_position is not None:
                        gt_position = (real_position[0,3], real_position[1,3])
                    else:
                        gt_position = None
                    # Visualize the map with agent and boulder positions
                    self.visualize_detections(
                        gt_position, agent_position, goal_loc, new_boulder_positions, self.all_boulder_detections
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
            # goal_lin_vel = 0.0
            # goal_ang_vel = 0.0

        # After handling the phases, increment the frame counter
        self.frame += 1

        # Finally, apply the resulting velocities
        control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)

        # Generate and add in the sample points
        self.sample_list.extend(sample_surface(estimate))

        for gp in ground_points_world:
            # Extract x,y,z from the homogeneous transform
            xyz = gp[:3, 3].tolist()  # e.g. [x, y, z]
            self.sample_list.append(xyz)

        for gp_back in ground_points_back_world:
            xyz_back = gp_back[:3, 3].tolist()
            self.sample_list.append(xyz_back)

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
            if len(detections) < 2:
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
            None, None, None, final_clusters, self.all_boulder_detections
        )

        # g_map = self.get_geometric_map()
        # gt_map_array = g_map.get_map_array()

        # N = gt_map_array.shape[0]  # should be 179 if you are spanning -13.425 to 13.425 by 0.15
        # x_min, y_min = -13.425, -13.425
        # resolution = 0.15

        # # Calculate indices for center 2x2m region
        # center_x_min_idx = int(round((-1 - x_min) / resolution))  # -1m in x
        # center_x_max_idx = int(round((1 - x_min) / resolution))   # +1m in x
        # center_y_min_idx = int(round((-1 - y_min) / resolution))  # -1m in y
        # center_y_max_idx = int(round((1 - y_min) / resolution))   # +1m in y

        # # setting all rock locations to 0
        # for i in range(self.map_length_testing):
        #     for j in range(self.map_length_testing):
        #         self.g_map_testing.set_cell_rock(i, j, 0)

        # # for (x_rock, y_rock, z_rock) in self.gt_rock_locations:
        # for (x_rock, y_rock) in self.all_boulder_detections:
        #     i = int(round((x_rock - x_min) / resolution))
        #     j = int(round((y_rock - y_min) / resolution))

        #     # Sanity check: make sure we are within bounds
        #     if 0 <= i < N and 0 <= j < N:
        #         # Check if the rock is outside the center 2x2m region
        #         if not (center_x_min_idx <= i <= center_x_max_idx and 
        #                 center_y_min_idx <= j <= center_y_max_idx):
        #             self.g_map_testing.set_cell_rock(i, j, 1)

        # # Visualize the map with agent and boulder positions
        # self.visualize_detections(
        #     [0,0], self.all_boulder_detections, self.all_boulder_detections
        # )

        # g_map = self.get_geometric_map()
        # gt_map_array = g_map.get_map_array()

        # N = gt_map_array.shape[0]  # should be 179 if you are spanning -13.425 to 13.425 by 0.15
        # x_min, y_min = -13.425, -13.425
        # resolution   = 0.15

        # # setting all rock locations to 0
        # for i in range(self.map_length_testing):
        #     for j in range(self.map_length_testing):
        #         self.g_map_testing.set_cell_rock(i, j, 0)

        # # for (x_rock, y_rock, z_rock) in self.gt_rock_locations:
        # for (x_rock, y_rock) in self.all_boulder_detections:

        #     i = int(round((x_rock - x_min) / resolution))
        #     j = int(round((y_rock - y_min) / resolution))

        #     # Sanity check: make sure we are within bounds
        #     if 0 <= i < N and 0 <= j < N:
        #         self.g_map_testing.set_cell_rock(i, j, 1)

        # TODO set everything within a certain radius of the lander to 0 for the rocks

        # Initialize the data class to get estimates for all the squares
        surfaceHeight = SurfaceHeight(g_map)
        
        # Generate the actual map with the sample list
        surfaceHeight.set_map(self.sample_list)

        print(f'we are getting a map of {g_map.get_map_array()}')

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
