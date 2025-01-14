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
from ultralytics import YOLO
import cv2

import carla

from Leaderboard.leaderboard.autoagents.autonomous_agent import AutonomousAgent

from pose.apriltag import Estimator, carla_to_pytransform

def get_entry_point():
    return 'SpiralAgent'


class SpiralAgent(AutonomousAgent):

    """
    Spiral agent to showcase the different functionalities of the agent
    """

    current_goal_vel = 5
    current_goal_ang = 2.5

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self._active_side_cameras = False
        self._active_side_front_cameras = True
        self.model = YOLO('yolov8n.pt')  # Replace with your model if needed

        self.estimator = Estimator(self)

    def use_fiducials(self):
        return True
    
    # This is a chechy function to get started with the spiral that should be changed
    # IMPORTANT TODO: Fix the hell out of this function Luke, u shouldnt code this late
    initial_direction = np.array([1, 0, 0]) # TODO: Fix this to be less chechy
    goal_ang_vel_tracker = 0
    def chechy_shit_for_goal_ang_vel(self, estimate) -> float:
        """
        This function takes in an estimator current and lander initial rotational and translation matrix and returns a good angle, but does it chechly
        """

        # This is the x, y the rover is pointing assuming the positive y axis is forward
        rover_looking_x_dir, rover_looking_y_dir, _ = estimate[0][0], estimate[0][1], estimate[0][2]

        # This is the lnader initial position # NOTE: We dont need to run this every turn but ehh
        lander_transform = carla_to_pytransform(self.get_initial_lander_position())
        
        # This is the direction from rover to lander
        rover_to_lander_x_dir, rover_to_lander_y_dir = (lander_transform[2][0] - estimate[2][0]), (lander_transform[2][1] - estimate[2][1])

        # This is the angle from the rover looking direction to the lander TODO: Make this better
        def angle_helper(x, y):
            # This is a helper function to find angle from +x where going up is positive theta and down is negative theta
            return np.arctan(y/x) if x > 0 else (np.pi + np.arctan(y/x))


        return angle_helper(rover_looking_x_dir, rover_looking_y_dir) - angle_helper(rover_to_lander_x_dir, rover_to_lander_y_dir) - np.pi


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

        # global rover estimate and global initial_lander_position
        estimate = self.estimator(input_data)
        initial_lander_position = carla_to_pytransform(self.get_initial_lander_position())

        # print(f'the estimate is {estimate}')
        # print(f'the rover global position is {self.estimator.initial_rover_global} and {self.get_initial_position()} of type {type(self.estimator.initial_rover_global)}')
        # print(f'the initial lander position is {initial_lander_position} and {self.get_initial_lander_position()}')

        # Use the estimator to get goal lin_vel ang_vel

        if estimate is not None:
            self.goal_ang_vel_tracker = self.chechy_shit_for_goal_ang_vel(estimate)
            print(f'the goal angle is now {self.goal_ang_vel_tracker}')

        control = carla.VehicleVelocityControl(15, self.goal_ang_vel_tracker)
        
        return control

    def finalize(self):
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(i, j, random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(random.randint(2)))
