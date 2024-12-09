#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import json
import carla

from leaderboard.agents.agent_utilities import AgentComponentsControl
from leaderboard.autoagents.autonomous_agent import AutonomousAgent


def get_entry_point():
    return 'PlaybackAgent'


class PlaybackAgent(AutonomousAgent):

    """
    Human agent to control the ego vehicle via keyboard
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self._use_fiducials = True
        self._control_generator = self.read_control(path_to_conf_file)

    def read_control(self, path_to_conf_file):

        with open(path_to_conf_file, 'r') as fd:
            records = json.load(fd)

        for record in records["records"]:
            control = carla.VehicleVelocityControl(
                record["control"]["linear_target_velocity"],
                record["control"]["angular_target_velocity"]
            )
            component_control = AgentComponentsControl.from_dict(record)
            yield control, component_control

    def sensors(self):
        """
        Define which sensors are going to be active at the start.
        The specifications and positions of the sensors are predefined and cannot be changed.
        """
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048'
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
        try:
            control, self._components_control = next(self._control_generator)

        except StopIteration as e:
            control = carla.VehicleVelocityControl()
            self.mission_complete()

        return control

    def destroy(self):
        """
        Cleanup
        """
        pass


