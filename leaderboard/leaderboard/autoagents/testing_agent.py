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


def get_entry_point():
    return 'TestingAgent'


class TestingAgent(AutonomousAgent):
    """
    Human agent to control the ego vehicle via keyboard
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        client = carla.Client()
        self.world = client.get_world()
        self._vehicle = None
        self._hazard_pos = []
        self._state = 0
        self._obstacle_max_distance = 10
        self._dist_threshold = 3

        with open(path_to_conf_file) as fd:
            data = json.load(fd)

        self._max_distance = data['behavior']['max_distance']
        self._linear_speed = data['behavior']['linear_speed']
        self._angular_speed = data['behavior']['angular_speed']
        self._turn_speed_factor = data['behavior']['turn_speed_factor']
        self._max_turn_angle = data['behavior']['max_turn_angle']
        self._duration = data['behavior']['duration']
        self._height_mean = data['mapping']['height_mean']
        self._height_std_dev = data['mapping']['height_std_dev']
        self._height_uncompleted_percentage = data['mapping']['height_uncompleted_percentage']
        self._rock_percentage = data['mapping']['rock_percentage']
        self._rock_uncompleted_percentage = data['mapping']['rock_uncompleted_percentage']
        self._sensors_data = data['sensors']

    def use_fiducials(self):
        return True

    def sensors(self):
        """
        Define which sensors are going to be active at the start.
        The specifications and positions of the sensors are predefined and cannot be changed.
        """
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': self._sensors_data[0][0], 'light_intensity': self._sensors_data[0][1],
                'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': self._sensors_data[1][0], 'light_intensity': self._sensors_data[1][1],
                'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': self._sensors_data[2][0], 'light_intensity': self._sensors_data[2][1],
                'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Left: {
                'camera_active': self._sensors_data[3][0], 'light_intensity': self._sensors_data[3][1],
                'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Right: {
                'camera_active': self._sensors_data[4][0], 'light_intensity': self._sensors_data[4][1],
                'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': self._sensors_data[5][0], 'light_intensity': self._sensors_data[5][1],
                'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.BackRight: {
                'camera_active': self._sensors_data[6][0], 'light_intensity': self._sensors_data[6][1],
                'width': '2448', 'height': '2048'
            },
            carla.SensorPosition.Back: {
                'camera_active': self._sensors_data[7][0], 'light_intensity': self._sensors_data[7][1],
                'width': '2448', 'height': '2048'
            }
        }
        return sensors

    def get_control(self):
        v_loc = self._vehicle.get_location()
        v_vec = self._vehicle.get_transform().get_forward_vector()
        v_vec = np.array((v_vec.x, v_vec.y, v_vec.z))

        for hazard_pos in self._hazard_pos:
            if hazard_pos.distance(v_loc) > self._obstacle_max_distance:
                continue

            hazard_vec = hazard_pos - self._vehicle.get_location()  # Suppossing vector to the (0,0,0)
            hazard_vec = hazard_vec.make_unit_vector()
            hazard_vec = np.array((hazard_vec.x, hazard_vec.y, hazard_vec.z))
            angle = math.degrees(math.acos(np.dot(hazard_vec, v_vec)))
            cross = np.cross(hazard_vec, v_vec)[2]
            if abs(angle) > self._max_turn_angle:
                continue

            return carla.VehicleVelocityControl(
                self._linear_speed, self._angular_speed if cross > 0 else -self._angular_speed)

        return carla.VehicleVelocityControl(self._linear_speed, 0)

    def run_step(self, input_data):
        """Execute one step of navigation"""
        if self._vehicle is None:
            self._vehicle = self.world.get_actors().filter('vehicle.ipex.ipex')[0]
            self.set_front_arm_angle(1.0)
            self.set_back_arm_angle(1.0)
            rocks = list(self.world.get_actors().filter('static.prop.sm_rock_*_xl'))
            rocks.extend(self.world.get_actors().filter('static.prop.sm_rock_*_l'))
            self._hazard_pos = [x.get_location() for x in rocks]
            self._hazard_pos.append(carla.Location(0,0,0))  # Lander

        if self.get_current_power() <= 25:
            self._vehicle.recharge_battery()

        control = carla.VehicleVelocityControl(0, 0)

        if self._state == 0:  # Going forward
            v_loc = self._vehicle.get_location()
            distance = v_loc.distance(carla.Location(0,0,0))
            if distance > self._max_distance:
                self._state = 1
            control = self.get_control()

        elif self._state == 1:  # Turning
            v_loc = carla.Location(0,0,0) - self._vehicle.get_location()  # Suppossing vector to the (0,0,0)
            v_vec = self._vehicle.get_transform().get_forward_vector()

            dot = (v_loc.x * v_vec.x + v_loc.y * v_vec.y + v_loc.z * v_vec.z) / (v_loc.length() * v_vec.length())
            angle = math.degrees(math.acos(dot))

            if abs(angle) < self._max_turn_angle:
                self._state = 2
            control = carla.VehicleVelocityControl(self._linear_speed * self._turn_speed_factor, self._angular_speed)

        elif self._state == 2:
            v_loc = self._vehicle.get_location()
            distance = v_loc.distance(carla.Location(0,0,0))
            if distance < self._max_distance:
                self._state = 0  # Back to forward if close enough
            elif distance > self._max_distance + self._dist_threshold:
                self._state = 1  # Redirect towards the lander if too far
            control = carla.VehicleVelocityControl(self._linear_speed, 0)

        else:
            # Just in case
            self._state = 0

        if self.get_mission_time() > self._duration:
            self.mission_complete()

        return control

    def finalize(self):
        """Add random values to the map"""
        g_map = self.get_geometric_map()
        for i in range(g_map.get_cell_number()):
            for j in range(g_map.get_cell_number()):
                if random.rand() > self._height_uncompleted_percentage:
                    g_map.set_cell_height(i,j, random.normal(self._height_mean, self._height_std_dev))
                if random.rand() > self._rock_uncompleted_percentage:
                    g_map.set_cell_rock(i,j, bool(random.rand() < self._rock_percentage))
