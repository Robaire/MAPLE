#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


class MissionData(object):

    def __init__(self, config, crashed, failure_message, agent, manager, behaviors, show_results):
        self.name = config.name
        self.base_name = config.base_name
        self.repetition_index = config.rep_index
        self.crashed = crashed
        self.failure_message = failure_message
        self.ground_map = manager.terrain_map if manager else None

        self.agent_map = agent.get_agent_map() if agent else None
        self.fiducials_used = agent.agent_fiducials if agent else True
        self.sim_duration = manager.sim_duration if manager else 0
        self.sys_duration = manager.sys_duration if manager else 0
        self.start_sys_time = manager.start_sys_time if manager else 0
        self.end_sys_time = manager.end_sys_time if manager else 0

        self.mission_duration = behaviors.get_mission_duration() if behaviors else 0
        self.out_of_power = behaviors.out_of_power if behaviors else False
        self.out_of_mission_time = behaviors.out_of_mission_time if behaviors else False
        self.out_of_sim_time = behaviors.out_of_sim_time if behaviors else False
        self.out_of_bounds = behaviors.out_of_bounds if behaviors else False
        self.vehicle_blocked = behaviors.vehicle_blocked if behaviors else False

        self.show_results = show_results
