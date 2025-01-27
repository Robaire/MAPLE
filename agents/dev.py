#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""
import time
import json
import math
from numpy import random
import numpy as np
import cv2

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from maple.pose.estimator import Estimator

from maple.navigation.simple_spiral import Navigation

def get_entry_point():
    return 'Dev'


class Dev(AutonomousAgent):

    """
    Spiral agent to start to develop from
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self._active_side_cameras = False
        self._active_side_front_cameras = True

        self.estimator = Estimator(self)
        self.navigatior = Navigation(self)

    def use_fiducials(self):
        return True
    
    def sensors(self):
        """
        Define which sensors are going to be active at the start.
        The specifications and positions of the sensors are predefined and cannot be changed.
        """
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': True, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Left: {
                'camera_active': True, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Right: {
                'camera_active': True, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': True, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.BackRight: {
                'camera_active': True, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Back: {
                'camera_active': True, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
        }
        return sensors

    def run_step(self, input_data):
        """Execute one step of navigation"""

        control = carla.VehicleVelocityControl(0, 0.5)
        front_data = input_data['Grayscale'][carla.SensorPosition.Front]  # Do something with this
        imu_data = self.get_imu_data()
        if self._active_side_front_cameras:
            front_left_data = input_data['Grayscale'][carla.SensorPosition.FrontLeft]  # Do something with this
            front_right_data = input_data['Grayscale'][carla.SensorPosition.FrontLeft]  # Do something with this
        if self._active_side_cameras:
            left_data = input_data['Grayscale'][carla.SensorPosition.Left]  # Do something with this
            right_data = input_data['Grayscale'][carla.SensorPosition.Right]  # Do something with this

        mission_time = round(self.get_mission_time(), 2)

        # Get a position estimate for the rover
        estimate = self.estimator(input_data)

        # Get a goal linear and angular velocity from navigation
        goal_lin_vel, goal_ang_vel = self.navigatior.dumb_spiral(estimate)

        # Set the goal velocities to be returned
        control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)
        
        return control

    def finalize(self):
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(i, j, random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(random.randint(2)))
