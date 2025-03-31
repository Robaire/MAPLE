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
from maple.pose.inertial_apriltag import InertialApriltagEstimator
from maple import utils
from maple.utils import *
from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
from maple.surface.post_processing import PostProcessor
from maple.navigation.charging_navigator_straightshot import ChargingNavigator


def get_entry_point():
    return "DummyAgent"


class DummyAgent(AutonomousAgent):
    """
    This is a simple agent to test if the new maple.pose InertialAngleEstimator works correctly.
    The agent will turn in place, drive straight using the full Estimator, and then turn in place again.
    During turns, the rover will just use the IMU.
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self._active_side_cameras = True
        self.charging_flag = True
        self.estimator = InertialApriltagEstimator(self)
        self.charging_routine = ChargingNavigator(self)
        self.initial_step = True
        self.stage = 0
        # self.ang_estimator = InertialAngleEstimator(self)

        self.gt_data = []
        self.est_data = []

    def use_fiducials(self):
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
        if self.stage == 0 or self.stage == 2 or self.stage == 4:
            estimate, is_april_tag_estimate = self.estimator(input_data)
            x, y, z, roll, pitch, yaw = pytransform_to_tuple(estimate)
        elif self.stage == 1 or self.stage == 3:
            estimate, is_april_tag_estimate = self.estimator(
                input_data, use_imu_ang=True
            )
            x, y, z, roll, pitch, yaw = pytransform_to_tuple(estimate)

        gt = self.get_transform()
        est = [x, y, z, roll, pitch, yaw]
        gt = [
            gt.location.x,
            gt.location.y,
            gt.location.z,
            gt.rotation.roll,
            gt.rotation.pitch,
            gt.rotation.yaw,
        ]
        self.gt_data.append(np.array(gt))
        self.est_data.append(np.array(est))

        control = carla.VehicleVelocityControl(0, 0)
        imu_data = self.get_imu_data()
        ang_threshold = np.deg2rad(20)
        ang_goal1 = np.deg2rad(90)
        ang_goal2 = np.deg2rad(-90)

        mission_time = round(self.get_mission_time(), 2)
        if self.initial_step == True:
            print("Charging Antenna pose:", self.charging_routine.antenna_pose)
            print("Initial rover pose:", self.charging_routine.rover_initial_position)
            self.initial_step = False

        if mission_time <= 3 and self.stage == 0:
            self.charging_routine.battery_level = self.get_current_power()
            # Allow the vehicle to settle, and ensure lights are on
            self.set_light_state(carla.SensorPosition.Front, 1.0)
            self.set_light_state(carla.SensorPosition.Back, 1.0)
            self.set_light_state(carla.SensorPosition.Left, 1.0)
            self.set_light_state(carla.SensorPosition.Right, 1.0)
            control = carla.VehicleVelocityControl(0, 0)
        elif self.stage == 0:
            self.stage = 1

        # Execute the first turn
        if np.abs(yaw - ang_goal1) > ang_threshold and self.stage == 1:
            if yaw < ang_goal1:
                ang_vel = 0.5
            else:
                ang_vel = -0.5
            control = carla.VehicleVelocityControl(0, ang_vel)
        elif self.stage == 1:
            self.stage = 2
            self.stage_1_end_time = mission_time

        if self.stage == 2 and np.abs(mission_time - self.stage_1_end_time) <= 10:
            # Drive straight for 10 seconds
            control = carla.VehicleVelocityControl(1, 0)
        elif self.stage == 2:
            self.stage = 3
            stage_2_end_time = mission_time

        # Execute the second turn
        if np.abs(yaw - ang_goal2) > ang_threshold and self.stage == 3:
            if yaw < ang_goal2:
                ang_vel = 0.5
            else:
                ang_vel = -0.5
            control = carla.VehicleVelocityControl(0, ang_vel)
        elif self.stage == 3:
            self.stage = 4
            stage_3_end_time = mission_time

        if mission_time > 60 or self.stage == 4:
            self.mission_complete()

        return control

    def finalize(self):
        # Save data
        np.save("gt_data.npy", np.array(self.gt_data))
        np.save("est_data.npy", np.array(self.est_data))
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(i, j, random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(random.randint(2)))
