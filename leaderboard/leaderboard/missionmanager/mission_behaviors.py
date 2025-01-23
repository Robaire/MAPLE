#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the Behaviors implementations.
"""

from leaderboard.utils.timer import GameTime
from math import acos, degrees

class MissionBehaviors(object):

    def __init__(self, world, fiducials_used, constants):

        self._world = world
        self._ego_vehicle = None
        self._lander = None
        self._current_power = 0
        self._max_power = 0  # This is only used to recharge the battery for a tick, so it works

        self._mission_duration = 0
        self._sim_duration = 0
        self._last_time_ego_moved = GameTime.get_time()

        self._running = True
        self._recharging = False

        self._constants = constants

        self.out_of_power = False
        self.out_of_mission_time = False
        self.out_of_sim_time = False
        self.out_of_bounds = False
        self.vehicle_blocked = False

        self.fiducials_used = fiducials_used

    def setup(self, ego_vehicle, lander):
        self._ego_vehicle = ego_vehicle
        self._lander = lander
        self._current_power = self._ego_vehicle.get_current_power()
        self._max_power = self._current_power

    def get_mission_duration(self):
        return self._mission_duration

    def get_current_power(self):
        return self._current_power

    def tick(self):
        """Updates all the behaviors and criteria"""
        self._current_power = self._ego_vehicle.get_current_power()
        self.refill_power()

        self.mission_duration_test()
        self.simulation_duration_test()

        self.out_of_power_test()
        self.out_of_bounds_test()
        self.vehicle_blocked_test()

    def refill_power(self):
        """
        Recharges the ego's power if it is close enough to it's power source.
        Instead of being a gradual increase, the power recharge is instantaneous,
        and expected time spent while recharging is also added to the simulation time
        """
        cover_transform = self._ego_vehicle.get_charging_port_position()
        lander_transform = self._lander.get_component_world_transform('charging-port')

        x_threshold = abs(cover_transform.location.x - lander_transform.location.x) < self._constants.refill_x_threshold
        y_threshold = abs(cover_transform.location.y - lander_transform.location.y) < self._constants.refill_y_threshold

        cover_vec = cover_transform.get_up_vector()
        lander_vec = lander_transform.get_forward_vector()
        angle_diff = abs(degrees(acos(cover_vec.dot(lander_vec))))
        angle_threshold = (180 - self._constants.refill_yaw_threshold) < angle_diff and angle_diff < (180 + self._constants.refill_yaw_threshold)

        refill = x_threshold and y_threshold and angle_threshold
        refill_out = cover_transform.location.distance(lander_transform.location) > self._constants.refill_reset_distance

        if refill and not self._recharging:
            self._recharging = True
            self._mission_duration += self._constants.refill_duration * (1 - self._current_power / self._max_power)
            self._current_power = self._max_power
            self._ego_vehicle.recharge_battery()
        elif refill_out and self._recharging:
            self._recharging = False

    def mission_duration_test(self):
        """Checks that the maximum mission duration hasn't been reached"""
        self._mission_duration += GameTime.get_delta_time()
        if self._mission_duration >= self._constants.max_mission_duration:
            self.stop()
            self.set_max_mission_duration()
            self.out_of_mission_time = True

    def simulation_duration_test(self):
        """Checks that the maximum simulation duration hasn't been reached"""
        self._sim_duration += GameTime.get_platform_delta_time()
        if self._sim_duration >= self._constants.max_simulation_duration:
            self.stop()
            self.out_of_sim_time = True

    def out_of_power_test(self):
        """Checks that the ego vehicle still has power, stopping the run if it has run out"""
        if self._current_power <= self._constants.min_vehicle_power:
            self.stop()
            self.out_of_power = True

    def out_of_bounds_test(self):
        """Checks that the ego vehicle hasn't gone outside the map"""
        ego_transform = self._ego_vehicle.get_transform()
        x = ego_transform.location.x
        y = ego_transform.location.y
        dist = self._constants.bounds_distance
        if x > dist or x < -dist or y > dist or y < -dist:
            self.stop()
            self.out_of_bounds = True

    def vehicle_blocked_test(self):
        """Detect whether or not the vehicle is stuck"""
        linear_speed = self._ego_vehicle.get_velocity().length()
        if linear_speed < self._constants.blocked_min_speed:
            if (GameTime.get_time() - self._last_time_ego_moved) >= self._constants.blocked_max_time:
                self.stop()
                self.vehicle_blocked = True
        else:
            self._last_time_ego_moved = GameTime.get_time()

    def stop(self):
        """Stops the simulation, called by the MissionManager"""
        self._running = False

    def set_max_mission_duration(self):
        """Set the duration to its max value, used for off-nominal terminations"""
        self._mission_duration = self._constants.max_mission_duration

    def is_running(self):
        """Checks if the simulation needs to keep running"""
        return self._running

    def cleanup(self):
        pass
