#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from numpy import random
from math import sin, cos, radians, degrees

import carla


MIN_EGO_DIST = 5        # m
MAX_EGO_DIST = 10       # m
MAX_DEGREE = 30      # deg


class MissionSpawner(object):
    """
    Class responsible for spawning all the necessary assets part of the mission
    """

    def __init__(self, world, mission_config, with_fiducials):
        """Spawns everything but the fiducials"""
        self._world = world
        self._bp_lib = self._world.get_blueprint_library()

        self._mission_config = mission_config
        self._with_fiducials = with_fiducials

        self.ego_vehicle = None
        self.lander = None

    def setup(self, seed):
        self._set_presetid()
        self._spawn_ego_vehicle(seed)
        self._get_lander_instance()

    def _spawn_ego_vehicle(self, seed):
        """Spawn the ego vehicle at random point around the center of the map"""
        random.seed(seed)

        radius = (MAX_EGO_DIST - MIN_EGO_DIST)*random.rand() + MIN_EGO_DIST
        base_angle = random.randint(0, 360)
        angle = radians(base_angle)
        yaw = base_angle + 180 + random.randint(-MAX_DEGREE, MAX_DEGREE + 1)

        ego_location = carla.Location(radius*cos(angle), radius*sin(angle), 0)  # Height is calculated automatically
        ego_transform = carla.Transform(ego_location, carla.Rotation(yaw=yaw))

        # Spawn the ego
        bp = self._bp_lib.find('vehicle.ipex.ipex')
        self.ego_vehicle = self._world.try_spawn_actor(bp, ego_transform)
        if not self.ego_vehicle:
            raise ValueError("Shutting down, couldn't spawn the IPEx")

        spectator = self._world.get_spectator()
        spectator.set_transform(carla.Transform(carla.Location(0, 0, 30), carla.Rotation(pitch=-90)))

        # Let the ego settle on the ground
        for _ in range(20):
            self._world.tick()

    def _set_presetid(self):
        """Spawns the mission preset which includes the lander and rocks"""
        self._world.set_presetid(int(self._mission_config.preset), self._with_fiducials)

    def _get_lander_instance(self):
        """Gets the lander instance from the server"""
        lander_actors = self._world.get_actors().filter('*lander*')
        if not lander_actors:
            raise ValueError("Shutting down, couldn't spawn the Lander")
        self.lander = lander_actors[0]

    def cleanup(self):
        """Remove all actors upon deletion"""
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
