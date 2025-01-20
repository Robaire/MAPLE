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

from Leaderboard.leaderboard.autoagents.autonomous_agent import AutonomousAgent

from pose.apriltag import Estimator, carla_to_pytransform, pytransform_to_carla

from navigation.simple_spiral import april_tag_input_only
from pose.imu_Estimator import imu_Estimator

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

        # print(f'the camera type is {type(front_data)}')
        # Make sure the data was gotten
        # TODO: This can be done in parallel using threading
        # if not front_data:
        #     front_scan = self.model(front_data)
        # if not front_left_data:
        #     front_left_scan = self.model(front_left_data)
        # if not front_right_data:
        #     front_left_scan = self.model(front_right_data)

        # print(f'the front scan is {front_scan} with type {type(front_scan)}')

        # # This is bad guess for if there is a rock in the front
        # if front_scan:
        #     for box in front_scan.boxes:
        #         # Extract coordinates (x1, y1, x2, y2) and confidence
        #         x1, y1, x2, y2 = box.xyxy[0]  # Get the top-left and bottom-right coordinates
        #         confidence = box.conf[0]      # Get the confidence score
                
        #         # Print the values
        #         print(f"Box Coordinates: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
        #         print(f"Confidence: {confidence:.2f}")

        mission_time = round(self.get_mission_time(), 2)

        # if mission_time == 15:
        #     self.set_light_state(carla.SensorPosition.Front, 1.0)
        #     self.set_light_state(carla.SensorPosition.Back, 1.0)
        #     self.set_light_state(carla.SensorPosition.Left, 1.0)
        #     self.set_light_state(carla.SensorPosition.Right, 1.0)

        # elif mission_time == 20:
        #     self.set_front_arm_angle(1.0)
        #     self.set_back_arm_angle(1.0)

        # elif mission_time > 20 and mission_time <= 30:
        #     control = carla.VehicleVelocityControl(0.3, 0)

        # elif mission_time > 30 and mission_time <= 40:
        #     control = carla.VehicleVelocityControl(0, 0.5)

        # elif mission_time == 40:
        #     self.set_radiator_cover_state(carla.RadiatorCoverState.Open)

        # elif mission_time == 50:
        #     self.set_camera_state(carla.SensorPosition.Left, True)
        #     self.set_camera_state(carla.SensorPosition.Right, True)
        #     self._active_side_cameras = True

        # elif mission_time > 50 and mission_time <= 60:
        #     control = carla.VehicleVelocityControl(0.3, 0.5)

        # elif mission_time > 60:
        #     self.mission_complete()

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
            rover_x = imu_state_est.location.x
            rover_y = imu_state_est.location.y
            rover_yaw = imu_state_est.rotation.yaw

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
