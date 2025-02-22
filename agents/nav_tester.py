#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
This agent is meant to use exact locations, it is meant for testing nav code
"""
import time
import json
import math
from numpy import random
import numpy as np
import cv2
import carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from maple.pose import InertialApriltagEstimator
from maple.navigation.navigator import Navigator
from maple.utils import carla_to_pytransform

from maple.boulder import BoulderDetector
from maple.boulder.map import BoulderMap
from pytransform3d.transformations import concat

def get_entry_point():
    return "Nav_test"
class Nav_test(AutonomousAgent):
    """
    Spiral agent to start to Nav_testelop from
    """
    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self._active_side_cameras = False
        self._active_side_front_cameras = True
        self.estimator = InertialApriltagEstimator(self)
        self.navigatior = Navigator(self)

        # Boulder Detectors
        self.front_detector = BoulderDetector(self, "FrontLeft", "FrontRight")
        self.rear_detector = BoulderDetector(self, "BackLeft", "BackRight")

        # Boulder Mapper
        self.boulder_mapper = BoulderMap(self.get_geometric_map())

        # Data Collection
        self.boulders_global = []
        self.boulders_global_large = []
        self.surface_global = []

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
                "camera_active": True,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
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
                "camera_active": True,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.BackRight: {
                "camera_active": True,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Back: {
                "camera_active": True,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
        }
        return sensors
    def run_step(self, input_data):
        """Execute one step of navigation"""
        front_data = input_data['Grayscale'][carla.SensorPosition.Front]  # Do something with this
        if self._active_side_front_cameras:
            front_left_data = input_data["Grayscale"][
                carla.SensorPosition.FrontLeft
            ]  # Do something with this
            front_right_data = input_data["Grayscale"][
                carla.SensorPosition.FrontLeft
            ]  # Do something with this
        if self._active_side_cameras:
            left_data = input_data["Grayscale"][
                carla.SensorPosition.Left
            ]  # Do something with this
            right_data = input_data["Grayscale"][
                carla.SensorPosition.Right
            ]  # Do something with this

        mission_time = round(self.get_mission_time(), 2)

        # Get a position estimate for the rover
        estimate = self.estimator(input_data)
        print(f'the estimator is estimating {estimate}')

        # IMPORTANT NOTE: For developing using the exact location
        estimate = carla_to_pytransform(self.get_transform())
        print(f'the actual thing is {estimate}')

        ## Data Collection ##
        # Rather than use the frame number to determine the sample frequency we use the mission time
        # because we may not have a pose estimate on the exact frame we are trying to sample on
        self.last_sample_time = self.get_mission_time()

        # Get boulder detections
        boulders_rover = []
        boulders_rover.extend(self.front_detector(input_data))
        boulders_rover.extend(self.rear_detector(input_data))

        # Convert the boulders to the global frame
        self.boulders_global.extend(
            [concat(b_r, estimate) for b_r in boulders_rover]
        )

        # TODO: If the navigation is using interim boulder map or surface mapping data it can be processes here
        # Although really this should be processed inside of the Navigator class
        # Maybe it should take a reference to the global boulder list and surface map lists?
        # self.navigator.update_boulders(self.boulders_global)

        # Theoretically, this code should identify large boulders via cluster mapping - Allison

        # TODO: ADJUST min_area TO MINIMUM SIZE OF PROBLEMATIC BOULDERS
        min_area = 30
        # Get boulder detections
        boulders_rover_large = []
        boulders_rover_large.extend(
            self.front_detector.get_large_boulders(min_area=min_area)
        )
        boulders_rover_large.extend(
            self.rear_detector.get_large_boulders(min_area=min_area)
        )
        # Convert the boulders to the global frame
        self.boulders_global_large.extend(
            [concat(b_r, estimate) for b_r in boulders_rover_large]
        )
        # Transforms to all large boulder detections and all large boulders
        boulders_global_large_clustered = self.boulder_mapper.generate_clusters(
            self.boulders_global_large
        )

        print(f'the boulders look like {boulders_global_large_clustered}')

        # IMPORTANT NOTE: The estimate should never be NONE!!!,
        # Get a goal linear and angular velocity from navigation
        goal_lin_vel, goal_ang_vel = self.navigatior(estimate)

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