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

from maple.pose.pose_estimator import Estimator

from maple.navigation.navigator import Navigator

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
        self.navigatior = Navigator(self)

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

        front_data = input_data['Grayscale'][carla.SensorPosition.Front]  # Do something with this
        if self._active_side_front_cameras:
            front_left_data = input_data['Grayscale'][carla.SensorPosition.FrontLeft]  # Do something with this
            front_right_data = input_data['Grayscale'][carla.SensorPosition.FrontLeft]  # Do something with this
        if self._active_side_cameras:
            left_data = input_data['Grayscale'][carla.SensorPosition.Left]  # Do something with this
            right_data = input_data['Grayscale'][carla.SensorPosition.Right]  # Do something with this

        mission_time = round(self.get_mission_time(), 2)


        # Test code to end sim early
        # if mission_time > 45:
            # exit()
        # Test code to end sim early

        # Get a position estimate for the rover
        estimate = self.estimator(input_data)
        print(f'the estimator is estimating {estimate}')
        # IMPORTANT NOTE: For developing using the exact location
        from maple.utils import carla_to_pytransform
        estimate = carla_to_pytransform(self.get_transform())
        print(f'the actual thing is {estimate}')

        # IMPORTANT NOTE: The estimate should never be NONE!!!, this is test code to catch that
        if estimate is None:
            goal_lin_vel, goal_ang_vel = 10, 0
            print(f'the estimate is returning NONE!!! that is a big problem buddy')
        else:
            # Get a goal linear and angular velocity from navigation
            goal_lin_vel, goal_ang_vel = self.navigatior(estimate)

            print(f'the estimate is {estimate}\n at mission time {mission_time}')

        ##### This is test code
        # from maple.utils import pytransform_to_tuple
        # if estimate is not None:
        #     _, _, _, _, _, yaw = pytransform_to_tuple(estimate)
        #     print(f'the yaw is {yaw}')
        ##### This is test code

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
