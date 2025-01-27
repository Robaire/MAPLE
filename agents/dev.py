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

from maple.pose.apriltag import Estimator

from maple.pose.imu_Estimator import imu_Estimator

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
        self.imu_estimator = imu_Estimator(self)

        # This is goal angular and linear velocity that was last called (set to initialized values)
        self.goal_lin_vel = 10
        self.goal_ang_vel = 0
        self.prev_state = None

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


        # global rover estimate and global initial_lander_position in rotational/translation matrix form
        estimate = self.estimator(input_data)
        # TODO: We only have to call lander code once
        initial_lander_position = carla_to_pytransform(self.get_initial_lander_position())

        lander_carla = pytransform_to_carla(initial_lander_position)
        lander_x = lander_carla.location.x
        lander_y = lander_carla.location.y

        imu_state_est = None

        if self.prev_state is not None:
            # If there is a record of a previous state, perform an IMU estimate
            imu_state_est = self.imu_estimator.next_state()

        if estimate is not None:
            print(f'the estimate is not none')  
            estimate_carla = pytransform_to_carla(estimate)
            rover_x = estimate_carla.location.x
            rover_y = estimate_carla.location.y
            rover_yaw = estimate_carla.rotation.yaw

            self.goal_lin_vel, self.goal_ang_vel = april_tag_input_only(rover_x, rover_y, rover_yaw, lander_x, lander_y)

            self.prev_state = estimate
        elif imu_state_est is not None:
            estimate_carla = pytransform_to_carla(imu_state_est)
            # Use imu estimate if no apriltag detected
            rover_x = estimate_carla.location.x
            rover_y = estimate_carla.location.y
            rover_yaw = estimate_carla.rotation.yaw

            self.goal_lin_vel, self.goal_ang_vel = april_tag_input_only(rover_x, rover_y, rover_yaw, lander_x, lander_y)

        control = carla.VehicleVelocityControl(self.goal_lin_vel, self.goal_ang_vel)
        
        return control

    def finalize(self):
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(i, j, random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(random.randint(2)))
