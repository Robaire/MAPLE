#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Module used to parse all the mission configuration parameters.
"""
import xml.etree.ElementTree as ET

import carla
from leaderboard.utils.mission_configuration import RockData, MissionConfiguration


def convert_elem_to_transform(elem):
    """Convert an ElementTree.Element to a CARLA transform"""
    return carla.Transform(
        carla.Location(
            float(elem.attrib.get('x')),
            float(elem.attrib.get('y')),
            float(elem.attrib.get('z'))
        ),
        carla.Rotation(
            roll=0.0,
            pitch=0.0,
            yaw=float(elem.attrib.get('yaw'))
        )
    )


class MissionParser(object):
    """
    Class used to parse all the mission configuration parameters.
    """

    @staticmethod
    def parse_missions_file(mission_filename, mission_subset=''):
        """
        Returns a list of mission configuration elements.
        :param mission_filename: the path to a set of missions.
        :return: List of dicts containing the waypoints, id and town of the missions
        """
        def get_missions_subset():
            """
            The mission subset can be indicated by single missions separated by commas,
            or group of missions separated by dashes (or a combination of the two)"""
            subset_ids = []
            subset_groups = mission_subset.replace(" ","").split(',')
            for group in subset_groups:
                if "-" in group:
                    # Group of missions, iterate from start to end, making sure both ids exist
                    start, end = group.split('-')
                    found_start, found_end = (False, False)

                    for mission in tree.iter("mission"):
                        mission_id = mission.attrib['id']
                        if not found_start and mission_id == end:
                            raise ValueError(f"Malformed mission subset '{group}', found the end id before the starting one")
                        elif not found_start and mission_id == start:
                            found_start = True
                        if not found_end and found_start:
                            if mission_id in subset_ids:
                                raise ValueError(f"Found a repeated mission with id '{mission_id}'")
                            else:
                                subset_ids.append(mission_id)
                            if mission_id == end:
                                found_end = True

                    if not found_start:
                        raise ValueError(f"Couldn\'t find the mission with id '{start}' inside the given missions file")
                    if not found_end:
                        raise ValueError(f"Couldn\'t find the mission with id '{end}' inside the given missions file")

                else:
                    # Just one mission, get its id while making sure it exists

                    found = False
                    for mission in tree.iter("mission"):
                        mission_id = mission.attrib['id']
                        if mission_id == group:
                            if mission_id in subset_ids:
                                raise ValueError(f"Found a repeated mission with id '{mission_id}'")
                            else:
                                subset_ids.append(mission_id)
                            found = True

                    if not found:
                        raise ValueError(f"Couldn't find the mission with id '{group}' inside the given missions file")

            subset_ids.sort()
            return subset_ids

        mission_configs = []
        tree = ET.parse(mission_filename)
        if mission_subset:
            subset_list = get_missions_subset()
        for mission in tree.iter("mission"):

            mission_id = mission.attrib['id']
            if mission_subset and mission_id not in subset_list:
                continue

            mission_config = MissionConfiguration()
            mission_config.map = mission.attrib['map']
            mission_config.base_name = "{}_{}".format(mission.attrib['map'], mission_id)
            mission_config.preset = mission.attrib['preset']
            mission_configs.append(mission_config)

        return mission_configs
