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

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.pose import InertialApriltagEstimator, PoseGraph
from maple import utils
from maple.utils import *
from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
from maple.surface.post_processing import PostProcessor
from maple.navigation.simple_charge_nav import ChargingNavigator


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
        self._active_side_cameras = True
        self.charging_flag = True
        # self.estimator = InertialApriltagEstimator(self)
        self.charging_routine = ChargingNavigator(self)
        self.initial_step = True

    def use_fiducials(self):
        return False

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
                "camera_active": True,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Right: {
                "camera_active": True,
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

    def run_step(self, input_data):
        """Execute one step of navigation"""
        end_time = 180

        # estimate, is_april_tag_estimate = self.estimator(input_data)
        # We assume ground truth is always available
        estimate = carla_to_pytransform(self.get_transform())
        imu_data = self.get_imu_data()

        mission_time = round(self.get_mission_time(), 2)
        if self.initial_step == True:
            print("Charging Antenna pose:", self.charging_routine.antenna_pose)
            print("Initial rover pose:", self.charging_routine.rover_initial_position)
            self.initial_step = False

        if mission_time <= 3:
            self.charging_routine.battery_level = self.get_current_power()
            # Allow the vehicle to settle, and ensure lights are on
            self.set_light_state(carla.SensorPosition.Front, 1.0)
            self.set_light_state(carla.SensorPosition.Back, 1.0)
            self.set_light_state(carla.SensorPosition.Left, 1.0)
            self.set_light_state(carla.SensorPosition.Right, 1.0)
            control = carla.VehicleVelocityControl(0, 0)
            self.set_front_arm_angle(np.deg2rad(60))
            self.set_back_arm_angle(np.deg2rad(60))

        elif mission_time <= end_time and self.charging_flag:
            control, self.charging_flag = self.charging_routine.navigate(estimate)
            print("Control:", control)
            control = carla.VehicleVelocityControl(control[0], control[1])
            if self.charging_flag == False:
                print("Charging complete!")

        elif mission_time > end_time or self.charging_flag == False:
            self.mission_complete()

        return control

    def finalize(self):
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(i, j, random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(random.randint(2)))
