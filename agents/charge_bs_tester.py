#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# uv run ./scripts/run_agent.py agents/charge_bs_tester.py --sim="./simulator" --xy="[-0.11222237348556519, -3]"

# This agent will attempt the foundling method for charging, the best location for start is near the location in the command above, feel free to alter slightly for testing

"""
This module provides a human agent to control the ego vehicle via keyboard
"""
import time
import json
import math
from numpy import random

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from maple.navigation.charging_bs import ChargingNavigator
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

        self.ChargeRoutine = ChargingNavigator(self)

    def use_fiducials(self):
        return True

    def sensors(self):
        """
        Define which sensors are going to be active at the start.
        The specifications and positions of the sensors are predefined and cannot be changed.
        """
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Left: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Right: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.BackRight: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Back: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '2448', 'height': '2048'
            },
        }
        return sensors

    def run_step(self, input_data):
        """Execute one step of navigation"""

        control, _ = self.ChargeRoutine.navigate(carla_to_pytransform(self.get_transform()))

        print(f'{control=}')
        # NOTE: Using the exact transform location
        control = carla.VehicleVelocityControl(*control)
        
        return control

    def finalize(self):
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        for i in range(map_length):
            for j in range(map_length):
                g_map.set_cell_height(i, j, random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(random.randint(2)))