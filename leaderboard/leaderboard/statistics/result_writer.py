#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains the result gatherer and write for CARLA missions.
It shall be used from the MissionManager only.
"""

from __future__ import print_function

import time
from tabulate import tabulate


COLORED_STATUS = {
    "FAILURE": '\033[91mFAILURE\033[0m',
    "SUCCESS": '\033[92mSUCCESS\033[0m',
}


class ResultOutputProvider(object):
    """
    This module create a human readable table with the most important
    information about a run, printing it through the terminal
    """

    def __init__(self, data, record):

        self._name = data.name
        self._base_name = data.base_name
        self._repetition_index = data.repetition_index
        self._start_sys_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data.start_sys_time))
        self._end_sys_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data.end_sys_time))
        self._sys_duration = round(data.sys_duration, 2)
        self._sim_duration = round(data.sim_duration, 2)
        self._duration_ratio = round(data.sim_duration / data.sys_duration, 3)

        if data.out_of_power:
            self._termination = "Out of power"
        elif data.out_of_mission_time:
            self._termination = "Out of mission tim"
        elif data.out_of_sim_time:
            self._termination = "Out of simulation time"
        elif data.out_of_bounds:
            self._termination = "Out of bounds"
        elif data.vehicle_blocked:
            self._termination = "IPEx blocked"
        else:
            self._termination = ""

        self._geometric_score =  record.scores['geometric_map']
        self._rocks_score =  record.scores['rocks_map']
        self._productivity_score =  record.scores['mapping_productivity']
        self._localization_score =  record.scores['localization']

        self._create_output_table()

    def _create_output_table(self, ):
        """Creates the output message"""
    
        # Create the title
        output = "\n"
        output += "\033[1m========= Results of {} (repetition {}) \033[1m=========\033[0m\n".format(
            self._base_name, self._repetition_index)
        output += "\n"

        # Simulation part
        summary_list = [['Summary', '']]
        summary_list.extend([["Start Time", "{}".format(self._start_sys_time)]])
        summary_list.extend([["End Time", "{}".format(self._end_sys_time)]])
        summary_list.extend([["System Time (seconds)", "{}".format(self._sys_duration)]])
        summary_list.extend([["Game Time (seconds)", "{}".format(self._sim_duration)]])
        summary_list.extend([["Ratio (Game / System)", "{}x".format(self._duration_ratio)]])
        summary_list.extend([["Off-nominal termination", "{}".format(self._termination)]])

        output += tabulate(summary_list, tablefmt='fancy_grid')
        output += "\n\n"

        # Results part
        score_list = [['Scores', '']]

        score_list.extend([["Geometric map", self._geometric_score]])
        score_list.extend([["Rocks map", self._rocks_score]])
        score_list.extend([["Mapping productivity", self._productivity_score]])
        score_list.extend([["Localization", self._localization_score]])

        output += tabulate(score_list, tablefmt='fancy_grid')
        output += "\n"

        print(output)
