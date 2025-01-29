#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import time
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent


def get_entry_point():
    return "DummyAgent"


class DummyAgent(AutonomousAgent):
    """
    Dummy agent to showcase the different functionalities of the agent
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self._active_side_cameras = False
        # As recommended by the challenge doc, we pull the initial rover and lander positions. Note that we still need to look into what these methods actually do.
        self.init_pos = self.get_initial_position() # A transform with the robot's initial pose: transform.location.[x,y,z] and transform.rotation[roll,pitch,yaw]
        self.robot_pos = self.init_pos
        self.lander_pos_R = self.get_initial_lander_position() # Transform from the robot's initial coordinate frame to the lander. Same attributes as get_initial_position()
        self.lander_pos_W = self.lander_pos_R
        lander_tw, lander_rw = self.transform_R_to_W(self.lander_pos_W)
        self.lander_pos.location.x, self.lander_pos.location.y, self.lander_pos.location.z = lander_tw
        self.lander_pos.rotation.roll ,self.lander_pos.rotation.pitch, self.lander_pos.rotation.yaw = lander_rw
        self.camera_list = ['Front','FrontLeft','FrontRight','Left','Right','BackLeft','BackRight','Back'] # List of possible camera names, for ease of function calling
        self.previous_positions = [] # A list of previous positions, to be used to detect if a collision has occurred
        current_power = self.get_current_power()

    def transform_robot_in_world(self, lander_pose_R):
        """
        We have the lander pose in the robot's frame, and the lander's pose in the world frame. This function calculate the robot's frame.
        """
        r_rl = [self.lander_pose_R.rotation.roll, self.lander_pose_R.rotation.pitch, self.lander_pose_R.rotation.yaw]
        r_wl = [self.lander_pos_W.rotation.roll, self.lander_pos_W.rotation.pitch, self.lander_pos_W.rotation.yaw]
        t_wl = [self.lander_pos_W.location.x, self.lander_pos_W.location.y, self.lander_pos_W.location.z]
        t_rl = [self.lander_pose_R.location.x, self.lander_pose_R.location.y, self.lander_pose_R.location.z]
        # Convert rotations to matrices
        R_WL = R.from_euler('xyz', r_wl).as_matrix()  # Rotation from W to L
        R_RL = R.from_euler('xyz', r_rl).as_matrix()  # Rotation from R to L

        # Invert R_RL and t_RL to get R_LR and t_LR
        R_LR = R_RL.T
        t_LR = -R_LR @ np.array(t_rl)

        # Compute R_WR and t_WR
        R_WR = R_WL @ R_LR
        t_WR = np.array(t_wl) + R_WL @ t_LR

        # Convert R_WR back to Euler angles
        r_wr = R.from_matrix(R_WR).as_euler('xyz')

        return t_WR, r_wr

    def transform_R_to_W(self, pose_in_R):
        """
        Transforms a pose in the robot's frame to the world frame.
        """
        r_wr = [self.robot_pos.rotation.roll, self.robot_pos.rotation.pitch, self.robot_pos.rotation.yaw]
        r_rl = [self.pose_in_R.rotation.roll, self.pose_in_R.rotation.pitch, self.pose_in_R.rotation.yaw]
        t_wr = [self.robot_pos.location.x, self.robot_pos.location.y, self.robot_pos.location.z]
        t_rl = [self.pose_in_R.location.x, self.pose_in_R.location.y, self.pose_in_R.location.z]
        R_WR = R.from_euler('xyz', r_wr).as_matrix()  # Rotation from W to R
        R_RL = R.from_euler('xyz', r_rl).as_matrix()  # Rotation from R to L

        # Compose rotations
        R_WL = R_WR @ R_RL

        # Compose translations
        t_WL = np.array(t_wr) + R_WR @ np.array(t_rl)

        # Convert R_WL back to Euler angles
        r_wl = R.from_matrix(R_WL).as_euler('xyz')

        return t_WL, r_wl

    def use_fiducials(self):
        """
        Override to return True if using the fiducials. Otherwise set to return False.
        """
        return True

    def sensors(self):
        """
        Define which sensors are going to be active at the start.
        The specifications and positions of the sensors are predefined and cannot be changed.
        """

        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": True,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Left: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Right: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.BackRight: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Back: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
        }
        return sensors
    
    def image_lander(self, camera):
        """ TODO
        Activates a specified camera and attempts to detect the lander fiducials. If the fiducials are detected, return True and update the lander position. If no fiducials are detected, return False.

        Input:
        Camera

        Output:
        True or False, depending on whether the fiducials were found
        """
        camera_data_list = []
        for name in camera:
            self.set_light_state(getattr(carla.SensorPosition, name), 1.0)
            camera_data =  input_data["Grayscale"][getattr(carla.SensorPosition,name)]
            camera_data_list.append(camera_data)
        for camera_data in camera_data_list:
            lander_position = search_for_fiducials(camera_data)
        self.lander_pos_R = lander_position # Update the lander's position in the robot frame
        return False
    
    def search_for_lander(self):
        """ TODO
        Performs a search for the lander fiducials using all of the robot's cameras. Failing a detection, the lander attempts to randomly move until the lander is located. Update the lander position if the fiducials are detected.
        
        Output:
        True or False, depending on whether the fiducials were found"""
        return False
    
    def estimate_current_height(self):
        """
        Estimates the current height of the robot using its pose estimation. This implementation soley relies on the fiducials."""
        # height = self.robot_pos.z
        t_WR, r_wr = self.transform_robot_in_world(self, self.lander_pos_R)
        height = t_WR[2] # z location of the robot in the world frame
        # If we wanted to, could update the geometric map based on the robot's transform in the world...
        return height
    
    def determine_spiral_direction(self):
        """
        Determines the desired direction of the robot motion, based upon the current lander position and robot pose. In this implementation, the robot simply drives in a clockwise spiral around the lander.

        Outputs:
        An orientation for the robot to travel in that moves along the spiral.

        Thoughts: This could be as simply as a square motion. Keep a counter for how far "out" the robot is from the lander and gradually move away.
        """
        return None
    
    def recharge_battery(self):
        """
        Commands the robot to move back towards the lander and recharge its batteries."""
        return None

    def run_step(self, input_data):
        """Execute one step of navigation"""

        # Based on the current expected position of the lander (transform from the rover to the lander), search for the fiducials using a camera
        camera = None # TODO: fill in with correct camera
        found_lander = self.image_lander(camera)
        if not found_lander:
            found_lander = self.search_for_lander()

        control = carla.VehicleVelocityControl(0, 0.5)
        front_data = input_data["Grayscale"][
            carla.SensorPosition.Front
        ]  # Do something with this
        imu_data = self.get_imu_data()
        if self._active_side_cameras:
            left_data = input_data["Grayscale"][
                carla.SensorPosition.Left
            ]  # Do something with this
            right_data = input_data["Grayscale"][
                carla.SensorPosition.Right
            ]  # Do something with this

        mission_time = round(self.get_mission_time(), 2)

        return control

    def finalize(self):
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(i, j, np.random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(np.random.randint(2)))
