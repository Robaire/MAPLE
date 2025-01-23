#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a statistics manager for the CARLA AD leaderboard
"""

import os
import math
import numpy as np
from dictor import dictor

from leaderboard.agents.geometric_map import ROCK_UNCOMPLETED_VALUE, MAP_UNCOMPLETED_VALUE
from leaderboard.utils.checkpoint_tools import fetch_dict, save_dict, save_array, save_array_txt


ELIGIBLE_VALUES = {'Started': False, 'Finished': True, 'Crashed': False, 'Invalid': False}
ENTRY_STATUS_VALUES = ELIGIBLE_VALUES.keys()
STATUS_MESSAGES = {
    "Invalid_sensors": [False, "Invalid sensors"],
    "Simulation_crash" : [True, "Simulation crashed"],
    "Agent_setup_failure": [False, "Agent couldn't be set up"],
    "Agent_runtime_failure": [False, "Agent crashed"],
    "Finished_mission" : [False, ""]
}   # Crashed flag + message with details

ROUND_DIGITS = 1
DURATION_ROUND_DIGITS = 2

RESULTS_FILE = 'results.json'


class MissionRecord():
    def __init__(self):
        self.index = -1
        self.id = ''
        self.status = 'Started'

        self.scores = {
            "total": 0.0,
            "geometric_map": 0.0,
            "rocks_map": 0.0,
            "mapping_productivity": 0.0,
            "localization": 0.0
        }

        self.meta = {
            'simulation_duration': 0,
            'system_duration': 0,
            'duration_ratio': 0,
        }

    def to_json(self):
        """Return a JSON serializable object"""
        return vars(self)


class GlobalRecord():
    def __init__(self):
        self.status = 'Perfect'
        self.off_nominal_terminations = {
            "out_of_power": 0,
            "out_of_mission_time": 0,
            "out_of_simulation_time": 0,
            "out_of_bounds": 0,
            "vehicle_blocked": 0
        }

        self.scores_mean = {
            "total": 0.0,
            "geometric_map": 0.0,
            "rocks_map": 0.0,
            "mapping_productivity": 0.0,
            "localization": 0.0
        }

        self.scores_std_dev = self.scores_mean.copy()

        self.meta = {
            'simulation_duration': 0,
            'system_duration': 0,
            'duration_ratio': 0,
        }

    def to_json(self):
        """Return a JSON serializable object"""
        return vars(self)

class Checkpoint():

    def __init__(self):
        self.global_record = {}
        self.progress = []
        self.records = []

    def to_json(self):
        """Return a JSON serializable object"""
        d = {}
        d['global_record'] = self.global_record.to_json() if self.global_record else {}
        d['progress'] = self.progress
        d['records'] = []
        d['records'] = [x.to_json() for x in self.records if x.index != -1]  # Index -1 = Route in progress

        return d


class Results():

    def __init__(self):
        self.checkpoint = Checkpoint()
        self.entry_status = "Started"
        self.eligible = ELIGIBLE_VALUES[self.entry_status]
        self.values = []
        self.labels = []

    def to_json(self):
        """Return a JSON serializable object"""
        d = {}
        d['_checkpoint'] = self.checkpoint.to_json()
        d['entry_status'] = self.entry_status
        d['eligible'] = self.eligible
        d['values'] = self.values
        d['labels'] = self.labels

        return d


def to_mission_record(record_dict):
    record = MissionRecord()
    for key, value in record_dict.items():
        setattr(record, key, value)

    return record


class StatisticsManager(object):
    """
    This is the statistics manager for the CARLA leaderboard
    """

    def __init__(self, total_missions, endpoint, constants):
        self._current_index = 0
        self._results = Results()
        self._global_record = GlobalRecord()

        self._total_missions = total_missions
        self._endpoint = endpoint

        self._constants = constants

    def get_results_endpoint(self):
        return os.path.join(self._endpoint, RESULTS_FILE)

    def add_file_records(self):
        """Reads a file and saves its records onto the statistics manager"""
        data = fetch_dict(self.get_results_endpoint())
        if data:
            mission_records = dictor(data, '_checkpoint.records')
            if mission_records:
                for record in mission_records:
                    self._results.checkpoint.records.append(to_mission_record(record))

    def set_entry_status(self, entry_status):
        if entry_status not in ENTRY_STATUS_VALUES:
            raise ValueError("Found an invalid value for 'entry_status'")
        self._results.entry_status = entry_status
        self._results.eligible = ELIGIBLE_VALUES[entry_status]

    def set_progress(self, mission_index):
        self._results.checkpoint.progress = [mission_index, self._total_missions]

    def create_mission_record(self, name, index):
        """
        Creates the basic mission data.
        This is done at the beginning to ensure the data is saved, even if a crash occurs
        """
        self._current_index = index
        mission_record = MissionRecord()
        mission_record.id = name
        mission_record.index = index

        # Check if we have to overwrite an element (when resuming), or create a new one
        mission_records = self._results.checkpoint.records
        if index < len(mission_records):
            self._results.checkpoint.records[index] = mission_record
        else:
            self._results.checkpoint.records.append(mission_record)

    def compute_mission_statistics(self, data):
        """
        Computes the current statistics by evaluating the mission data.
        Additionally, saves both ground truth and agent maps to a file
        """

        def get_geometric_score(ground_map, agent_map):
            """Compare the calculated heights vs the real ones"""
            if agent_map is None or ground_map is None:
                return self._constants.geometric_map_min_score

            true_heights = ground_map[:,:,2]
            agent_heights = agent_map[:,:,2]
            error_heights = np.sum(np.abs(true_heights - agent_heights) < self._constants.geometric_map_threshold)
            score_rate = error_heights / true_heights.size
            return self._constants.geometric_map_max_score * score_rate

        def get_rocks_score(ground_map, agent_map):
            """
            Compare the number of rocks found vs the real ones using an F1 score. Uncompleted values
            will be supposed False, increasing the amount of false negatives.
            """
            if agent_map is None or ground_map is None:
                return self._constants.rock_min_score

            true_rocks = ground_map[:,:,3]
            if np.sum(true_rocks) == 0:
                # Special case, preset has no rocks, disable the
                return self._constants.rock_min_score

            agent_rocks = np.copy(agent_map[:,:,3])
            agent_rocks[agent_rocks == ROCK_UNCOMPLETED_VALUE] = False  # Uncompleted will 

            tp = np.sum(np.logical_and(agent_rocks == True, true_rocks == True))
            fp = np.sum(np.logical_and(agent_rocks == True, true_rocks == False))
            fn = np.sum(np.logical_and(agent_rocks == False, true_rocks == True))

            score_rate = (2 * tp) / (2 * tp + fp + fn)
            return self._constants.rock_max_score * score_rate

        def get_mapping_productivity_score(agent_map, mission_duration):
            """Linear interpolation between min and max score according to the time"""
            if agent_map is None:
                return self._constants.mapping_productivity_min_score

            total_completed_height = agent_map[:,:,2] != MAP_UNCOMPLETED_VALUE
            total_completed_rocks = agent_map[:,:,3] != ROCK_UNCOMPLETED_VALUE
            total_completed = np.sum(total_completed_height * total_completed_rocks)

            agent_rate = total_completed / mission_duration if mission_duration else 0
            score_rate = min(agent_rate / self._constants.mapping_productivity_score_rate, 1.0)
            return self._constants.mapping_productivity_max_score * score_rate

        def get_fiducials_score(fiducials_used):
            return self._constants.fiducials_min_score if fiducials_used else self._constants.fiducials_max_score

        mission_record = self._results.checkpoint.records[self._current_index]

        geometric_score = get_geometric_score(data.ground_map, data.agent_map)
        rocks_score = get_rocks_score(data.ground_map, data.agent_map)
        mapping_productivity = get_mapping_productivity_score(data.agent_map, data.mission_duration)
        localization_score = get_fiducials_score(data.fiducials_used)
        total_score = geometric_score + rocks_score + mapping_productivity + localization_score

        mission_record.scores['geometric_map'] = round(geometric_score, ROUND_DIGITS)
        mission_record.scores['rocks_map'] = round(rocks_score, ROUND_DIGITS)
        mission_record.scores['mapping_productivity'] = round(mapping_productivity, ROUND_DIGITS)
        mission_record.scores['localization'] = round(localization_score, ROUND_DIGITS)
        mission_record.scores['total'] = round(total_score, ROUND_DIGITS)

        # As mission records don't save the terminations, save them to the global record
        termination_message = ""
        if data.out_of_power:
            termination_message = 'Agent run out of power'
            self._global_record.off_nominal_terminations['out_of_power'] += 1
        elif data.out_of_mission_time:
            termination_message = 'Agent run out of mission time'
            self._global_record.off_nominal_terminations['out_of_mission_time'] += 1
        elif data.out_of_sim_time:
            termination_message = 'Agent run out of simulation time'
            self._global_record.off_nominal_terminations['out_of_simulation_time'] += 1
        elif data.out_of_bounds:
            termination_message = 'Agent went out of bounds'
            self._global_record.off_nominal_terminations['out_of_bounds'] += 1
        elif data.vehicle_blocked:
            termination_message = 'Agent got blocked'
            self._global_record.off_nominal_terminations['vehicle_blocked'] += 1

        if data.failure_message:
            mission_record.status = 'Failed - ' + data.failure_message
        elif termination_message:
            mission_record.status = 'Finished - ' + termination_message
        else:
            mission_record.status = 'Finished'

        ratio = data.sim_duration / data.sys_duration if data.sys_duration != 0 else 0
        mission_record.meta['simulation_duration'] = round(data.sim_duration, DURATION_ROUND_DIGITS)
        mission_record.meta['system_duration'] = round(data.sys_duration, DURATION_ROUND_DIGITS)
        mission_record.meta['duration_ratio'] = round(ratio, DURATION_ROUND_DIGITS)

        record_len = len(self._results.checkpoint.records)
        if self._current_index == record_len:
            self._results.checkpoint.records.append(mission_record)
        elif self._current_index < record_len:
            self._results.checkpoint.records[self._current_index] = mission_record
        else:
            raise ValueError("Not enough entries in the route record")

        base_endpoint = os.path.join(self._endpoint, data.name)
        if data.ground_map is not None:
            save_array(base_endpoint + ".dat", data.ground_map)
            save_array_txt(base_endpoint + ".txt", data.ground_map)
        if data.agent_map is not None:
            save_array(base_endpoint + "_agent.dat", data.agent_map)
            save_array_txt(base_endpoint + "_agent.txt", data.agent_map)

        return mission_record

    def compute_global_statistics(self):
        """Computes and saves the global statistics of the missions"""
        mission_records = self._results.checkpoint.records

        global_record = GlobalRecord()

        # Calculate the score's means and result
        for mission_record in mission_records:

            self._global_record.scores_mean['total'] += mission_record.scores['total'] / self._total_missions
            self._global_record.scores_mean['geometric_map'] += mission_record.scores['geometric_map'] / self._total_missions
            self._global_record.scores_mean['rocks_map'] += mission_record.scores['rocks_map'] / self._total_missions
            self._global_record.scores_mean['mapping_productivity'] += mission_record.scores['mapping_productivity'] / self._total_missions
            self._global_record.scores_mean['localization'] += mission_record.scores['localization'] / self._total_missions

            self._global_record.meta['simulation_duration'] += mission_record.meta['simulation_duration']
            self._global_record.meta['system_duration']  += mission_record.meta['system_duration']

            # Downgrade the global result if need be ('Perfect' -> 'Completed' -> 'Failed')
            if ' - ' in mission_record.status:
                self._global_record.status = 'Failed'
            elif global_record.status == 'Perfect' and mission_record.scores['total'] < 1000:
                self._global_record.status = 'Completed'

        if self._global_record.meta['system_duration'] != 0:
            ratio = self._global_record.meta['simulation_duration'] / self._global_record.meta['system_duration']
        else:
            ratio = 0.0
        self._global_record.meta['duration_ratio'] = ratio

        # Calculate the score's standard deviation
        if self._total_missions == 1:
            for key in self._global_record.scores_std_dev:
                self._global_record.scores_std_dev[key] = 0
        else:
            for mission_record in mission_records:
                for key in self._global_record.scores_std_dev:
                    diff = mission_record.scores[key] - self._global_record.scores_mean[key]
                    self._global_record.scores_std_dev[key] += math.pow(diff, 2)

            for key in self._global_record.scores_std_dev:
                value = math.sqrt(self._global_record.scores_std_dev[key] / float(self._total_missions - 1))
                self._global_record.scores_std_dev[key] = value

        # Round the scores
        for score in self._global_record.scores_mean.keys():
            self._global_record.scores_mean[score] = round(self._global_record.scores_mean[score], ROUND_DIGITS)
        for score in self._global_record.scores_std_dev.keys():
            self._global_record.scores_std_dev[score] = round(self._global_record.scores_std_dev[score], ROUND_DIGITS)
        for data in self._global_record.meta.keys():
            self._global_record.meta[data] = round(self._global_record.meta[data], DURATION_ROUND_DIGITS)

        # Save the global records
        self._results.checkpoint.global_record = self._global_record

        # Change the values and labels. These MUST HAVE A MATCHING ORDER
        self._results.values = [
            str(self._global_record.scores_mean['total']),
            str(self._global_record.scores_mean['geometric_map']),
            str(self._global_record.scores_mean['rocks_map']),
            str(self._global_record.scores_mean['mapping_productivity']),
            str(self._global_record.scores_mean['localization']),
            str(self._global_record.off_nominal_terminations['out_of_power']),
            str(self._global_record.off_nominal_terminations['out_of_mission_time']),
            str(self._global_record.off_nominal_terminations['out_of_simulation_time']),
            str(self._global_record.off_nominal_terminations['out_of_bounds']),
            str(self._global_record.off_nominal_terminations['vehicle_blocked']),
        ]

        self._results.labels = [
            "Avg. total score",
            "Avg. geometric map score",
            "Avg. rocks map score",
            "Avg. mapping productivity score",
            "Avg. localizations score",
            "Total out of power terminations",
            "Total out of mission time terminations",
            "Total out of simulation time terminations",
            "Total out of bounds terminations",
            "Total vehicle blocked terminations",
        ]

        # Change the entry status and eligible to finished
        self.set_entry_status('Finished')

    def validate_and_save_statistics(self):
        """
        Makes sure that all the relevant data is there.
        Changes the 'entry status' to 'Invalid' if this isn't the case
        """
        error_message = ""

        if not self._results.values:
            error_message = "Missing 'values' data"

        elif self._results.entry_status == 'Started':
            error_message = "'entry_status' has the 'Started' value"

        else:
            global_records = self._results.checkpoint.global_record
            progress = self._results.checkpoint.progress
            route_records = self._results.checkpoint.records

            if not global_records:
                error_message = "Missing 'global_records' data"

            if not progress:
                error_message = "Missing 'progress' data"

            elif (progress[0] != progress[1] or progress[0] != len(route_records)):
                error_message = "'progress' data doesn't match its expected value"

            else:
                for record in route_records:
                    if record.status == 'Started':
                        error_message = "Found a route record with missing data"
                        break

        if error_message:
            print("\n\033[91mThe statistics are badly formed. Setting their status to 'Invalid':")
            print("> {}\033[0m\n".format(error_message))

            self.set_entry_status('Invalid')

        self.save_statistics()

    def save_statistics(self):
        """
        Writes the results into the endpoint. Meant to be used only for partial evaluations,
        use 'validate_and_save_statistics' for the final one as it only validates the data.
        """
        save_dict(self.get_results_endpoint(), self._results.to_json())
