#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the MissionManager implementations.
It must not be modified and is for reference only!
"""

import time
import signal
import carla

from leaderboard.agents.agent_wrapper import AgentRuntimeError
from leaderboard.agents.sensor_interface import SensorReceivedNoData
from leaderboard.agents.geometric_map import create_terrain_map
from leaderboard.utils.timer import GameTime


class MissionManager(object):

    """
    Basic mission manager class. This class holds all functionality
    required to start, run and stop a mission.

    The user must not modify this class.

    To use the MissionManager:
    1. Create an object via manager = MissionManager()
    2. Load a mission via manager.load_mission()
    3. Trigger the execution of the mission manager.run_mission()
       This function is designed to explicitly control start and end of
       the mission execution
    4. If needed, cleanup with manager.stop_mission()
    """

    def __init__(self, timeout):
        """
        Setups up the parameters, which will be filled at load_mission()
        """
        self._timeout = float(timeout)
        self._world = None

        self._ego_vehicle = None
        self._agent_wrapper = None
        self._logger = None
        self._behaviors = None
        self._running = False

        self.terrain_map = None
        self.start_sys_time = 0.0
        self.start_sim_time = 0.0
        self.end_sys_time = 0.0
        self.end_sim_time = 0.0
        self.sys_duration = 0.0
        self.sim_duration = 0.0

        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate mission ticking when receiving a signal interrupt
        """
        self._running = False

    def setup(self, world, ego_vehicle, agent_wrapper, weather, behaviors, logger):
        """
        Load a new mission
        """
        GameTime.restart()
        self.start_sim_time = GameTime.get_time()
        self.start_sys_time = time.time()

        self._world = world
        self._ego_vehicle = ego_vehicle
        self._agent_wrapper = agent_wrapper
        self._logger = logger
        self._weather = weather
        self._behaviors = behaviors

    def run(self):
        """
        Trigger the start of the mission and wait for it to finish/fail
        """
        self._running = True
        GameTime.start(self._world.get_snapshot().timestamp)

        while self._running:
            self._tick()

    def _tick_carla(self):
        self._world.tick(self._timeout)

        # Update game time and actor information
        timestamp = self._world.get_snapshot().timestamp
        GameTime.on_carla_tick(timestamp)

    def _tick_agent(self):
        try:
            mission_duration = self._behaviors.get_mission_duration()
            velocity_control, component_control, vehicle_status = self._agent_wrapper.tick(mission_duration)

            self._ego_vehicle.apply_velocity_control(velocity_control)
            component_control.apply_control(self._ego_vehicle)

        # Special exception inside the agent that isn't caused by the agent
        except SensorReceivedNoData as e:
            raise RuntimeError(e)

        except Exception as e:
            raise AgentRuntimeError(e)

        return velocity_control, component_control, vehicle_status

    def _tick_logger(self, velocity_control, component_control, vehicle_status, ego_vehicle, mission_time):
        self._logger.tick(velocity_control, component_control, vehicle_status, ego_vehicle, mission_time)

    def _tick_weather(self):
        self._weather.tick()

    def _tick_behaviors(self):
        self._behaviors.tick()

    def _tick(self):
        """
        Run the next tick of the simulation. This is divided in functions to help with the profiling.
        """
        _ = self._tick_carla()
        _ = self._tick_weather()
        velocity_control, component_control, vehicle_status = self._tick_agent()
        _ = self._tick_logger(velocity_control, component_control, vehicle_status,
                              self._ego_vehicle, self._behaviors.get_mission_duration())
        _ = self._tick_behaviors()

        if not self._behaviors.is_running() or self._agent_wrapper.agent.has_finished():
            self._agent_wrapper.agent.finalize()
            self._running = False

    def stop(self, constants, development):
        """
        This function triggers a proper termination of a mission
        """
        self.end_sim_time = GameTime.get_time()
        self.end_sys_time = time.time()

        self.sim_duration = self.end_sim_time - self.start_sim_time
        self.sys_duration = self.end_sys_time - self.start_sys_time

        self.terrain_map = create_terrain_map(self._world, constants) if not development else None

    def cleanup(self):
        pass