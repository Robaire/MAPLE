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
        self.InertialEstimator = InertialEstimator(self)
        self.ApriltagEstimator = ApriltagEstimator(self)

        self.actual_positions = []
        self.estimated_positions = []
        self.apriltag_positions = []
        self.imu_positions = []
        self.times = []

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

        control = carla.VehicleVelocityControl(0, 0.5)
        end_time = 5
        front_data = input_data['Grayscale'][carla.SensorPosition.Front]  # Do something with this
        imu_data = self.get_imu_data()
        if self._active_side_cameras:
            left_data = input_data['Grayscale'][carla.SensorPosition.Left]  # Do something with this
            right_data = input_data['Grayscale'][carla.SensorPosition.Right]  # Do something with this

        mission_time = round(self.get_mission_time(), 2)
        if self.InertialEstimator.prev_state is None:
            self.InertialEstimator.prev_state = carla_to_pytransform(self.get_initial_position())
        if self.InertialAprilTagEstimator.prev_state is None:
            self.InertialAprilTagEstimator.prev_state = carla_to_pytransform(self.get_initial_position())
        ia_estimate = self.InertialAprilTagEstimator(input_data)
        i_estimate = self.InertialEstimator(input_data)
        a_estimate = self.ApriltagEstimator(input_data)
        #print("IMU Data:",self.get_imu_data())

        self.estimated_positions.append(ia_estimate)
        self.apriltag_positions.append(a_estimate)
        self.imu_positions.append(i_estimate)
        self.times.append(mission_time)
        self.actual_positions.append(carla_to_pytransform(self.get_transform()))

        if mission_time > 3 and mission_time <= end_time:
            control = carla.VehicleVelocityControl(0.3, 0)

        elif mission_time > end_time:
            self.mission_complete()

        return control

    def finalize(self):
        # print("Final data")
        # print("Actual positions:",self.actual_positions)
        # print("Estimated Positions:", self.estimated_positions)
        # print("apriltag_positions:", self.apriltag_positions)
        # print("imu_positions:", self.imu_positions)
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        with open('data_output.csv', mode='w') as data_output:
            data_writer = csv.writer(data_output)
            data_writer.writerow(['Time', 'Actual', 'Estimated', 'AprilTag', 'IMU'])
            for i in range(len(self.times)):
                data_writer.writerow([self.times[i], self.actual_positions[i], self.estimated_positions[i], self.apriltag_positions[i], self.imu_positions[i]])

        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(i, j, random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(random.randint(2)))
