#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module is for a simple testing agent that follows a predefined trajectory. It is intended to evaluate telementry data, such as that provided by the IMU.
"""
import time
import json
import math
from numpy import random
import numpy as np

import carla
import csv

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from maple.pose.inertial_apriltag import InertialApriltagEstimator

from maple.pose.inertial import InertialEstimator

from maple.pose.apriltag import ApriltagEstimator
from maple.utils import carla_to_pytransform


def get_entry_point():
    return 'DummyAgent'


class DummyAgent(AutonomousAgent):

    """
    Dummy agent to showcase the different functionalities of the agent
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self._active_side_cameras = False

        self.InertialAprilTagEstimator = InertialApriltagEstimator(self)

        self.gt_arr = []
        self.times = []
        self.powers = []
        self.ia_estimates = []
        self.imu_data_arr = []

    def use_fiducials(self):
        return False

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
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Left: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Right: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.BackRight: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Back: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048'
            },
        }
        return sensors

    def run_step(self, input_data):
        """Execute one step of navigation"""
        control = carla.VehicleVelocityControl(0, 0)
        end_time = 10
        front_data = input_data['Grayscale'][carla.SensorPosition.Front]  # Do something with this
        imu_data = self.get_imu_data()
        if self._active_side_cameras:
            left_data = input_data['Grayscale'][carla.SensorPosition.Left]  # Do something with this
            right_data = input_data['Grayscale'][carla.SensorPosition.Right]  # Do something with this

        mission_time = round(self.get_mission_time(), 2)
        if mission_time <= 3:
            # Allow the vehicle to settle
            control = carla.VehicleVelocityControl(0, 0)

        ia_estimate = self.InertialAprilTagEstimator(input_data)

        self.ia_estimates.append(ia_estimate[0])
        self.imu_data_arr.append(self.get_imu_data())
        self.times.append(mission_time)
        self.gt_arr.append(carla_to_pytransform(self.get_transform()))

        if mission_time > 3 and mission_time <= end_time:
            control = carla.VehicleVelocityControl(1., 0.2)

        elif mission_time > end_time:
            self.mission_complete()

        return control

    def finalize(self):
        """Code that is called after mission termination. Save any collected data, and generate the map."""
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        with open('data_output.csv', mode='w') as data_output:
            data_writer = csv.writer(data_output)
            data_writer.writerow(['Time', 'Actual', 'Estimated', 'IMU data'])
            for i in range(len(self.times)):
                data_writer.writerow([self.times[i], self.gt_arr[i], self.ia_estimates[i], self.imu_data_arr[i]])

        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(i, j, random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(random.randint(2)))
