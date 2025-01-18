from collections import OrderedDict
from dictor import dictor

import copy

from leaderboard.utils.mission_parser import MissionParser
from leaderboard.utils.checkpoint_tools import fetch_dict


class MissionIndexer():
    def __init__(self, missions_file, repetitions, missions_subset):
        self._configs_dict = OrderedDict()
        self._configs_list = []
        self.index = 0

        mission_configurations = MissionParser.parse_missions_file(missions_file, missions_subset)
        self.total = len(mission_configurations) * repetitions

        for i, config in enumerate(mission_configurations):
            for repetition in range(repetitions):
                config.index = i * repetitions + repetition
                config.rep_index = repetition
                config.name = f"{config.base_name}_rep{config.rep_index}"
                self._configs_dict['{}.{}'.format(config.name, repetition)] = copy.copy(config)

        self._configs_list = list(self._configs_dict.values())

    def peek(self):
        return self.index < self.total

    def get_next_config(self):
        if self.index >= self.total:
            return None

        config = self._configs_list[self.index]
        self.index += 1

        return config

    def validate_and_resume(self, endpoint):
        """
        Validates the endpoint by comparing several of its values with the current running missions.
        If all checks pass, the simulation starts from the last mission.
        Otherwise, the resume is canceled, and the leaderboard goes back to normal behavior
        """
        data = fetch_dict(endpoint)
        if not data:
            print('Problem reading checkpoint. Found no data')
            return False

        entry_status = dictor(data, 'entry_status')
        if not entry_status:
            print("Problem reading checkpoint. Given checkpoint is malformed")
            return False
        if entry_status == "Invalid":
            print("Problem reading checkpoint. The 'entry_status' is 'Invalid'")
            return False

        checkpoint_dict = dictor(data, '_checkpoint')
        if not checkpoint_dict or 'progress' not in checkpoint_dict:
            print("Problem reading checkpoint. Given endpoint is malformed")
            return False

        progress = checkpoint_dict['progress']
        if progress[1] != self.total:
            print("Problem reading checkpoint. Endpoint's amount of missions does not match the given one")
            return False

        mission_data = dictor(checkpoint_dict, 'records')

        check_index = 0
        while check_index < progress[0]:
            mission_id = self._configs_list[check_index].name
            mission_id += "_rep" + str(self._configs_list[check_index].rep_index)
            checkpoint_mission_id = mission_data[check_index]['id']

            if mission_id != checkpoint_mission_id:
                print("Problem reading checkpoint. Checkpoint mission don't match the current ones")
                return False

            check_index += 1

        if entry_status == "Crashed":
            self.index = max(0, progress[0] - 1)  # Something went wrong, repeat the last route
        else: 
            self.index = max(0, progress[0])
        return True

