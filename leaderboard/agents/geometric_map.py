# Copyright (c) # Copyright (c) 2024 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


"""
This module provides GeometricMap implementation.
"""

import numpy as np
import carla

from leaderboard.agents.coordinate_conversion import toLHCSlocation


ROCK_UNCOMPLETED_VALUE = np.NINF
MAP_UNCOMPLETED_VALUE = np.NINF


def get_cell_data(world, x, y, cell_size):
    """
    Casts several rays onto the cell to get the height and rock data.
    The current amount of rays is 16, separated a distance of 3.75cm each
    (if the cell is 15cm) to ensure the 5cm rocks are always detected
    """
    locations = [
        (x - 3*cell_size/8, y - 3*cell_size/8),
        (x - 1*cell_size/8, y - 3*cell_size/8),
        (x + 1*cell_size/8, y - 3*cell_size/8),
        (x + 3*cell_size/8, y - 3*cell_size/8),
        (x - 3*cell_size/8, y - 1*cell_size/8),
        (x - 1*cell_size/8, y - 1*cell_size/8),
        (x + 1*cell_size/8, y - 1*cell_size/8),
        (x + 3*cell_size/8, y - 1*cell_size/8),
        (x - 3*cell_size/8, y + 1*cell_size/8),
        (x - 1*cell_size/8, y + 1*cell_size/8),
        (x + 1*cell_size/8, y + 1*cell_size/8),
        (x + 3*cell_size/8, y + 1*cell_size/8),
        (x - 3*cell_size/8, y + 3*cell_size/8),
        (x - 1*cell_size/8, y + 3*cell_size/8),
        (x + 1*cell_size/8, y + 3*cell_size/8),
        (x + 3*cell_size/8, y + 3*cell_size/8),
    ]
    heights = []
    rocks = []

    for x,y in locations:
        labelled_point = world.ground_projection(
            toLHCSlocation(carla.Location(x=x, y=y, z=25.0)), search_distance=50.0
        )
        if labelled_point is None:
            # raise ValueError("Failed to create the height map, couldn't find a point")
            return 0, False

        heights.append(labelled_point.location.z)
        rocks.append(True if labelled_point.label == carla.CityObjectLabel.Rock else False)

    height = sum(heights)/len(heights)
    rock = any(rocks)

    return height, rock

def create_base_map(constants):
    """
    Creates the base geometric map that will be given to the agent for its completion.
    It is a 2D numpy matrix where each element in them represents the [x,y, height, rock flag].
    """
    base_map = np.array(np.zeros((constants.cell_number, constants.cell_number, 4)))
    low = -constants.map_size / 2 + constants.cell_size / 2
    high = constants.map_size / 2 - constants.cell_size / 2
    values = np.arange(low, high + 0.05, constants.cell_size)  # Make sure float imprecision doesn't remove the last one
    indexes = np.arange(0, constants.cell_number, 1)

    for x_index in indexes:
        for y_index in indexes:
            base_map[x_index, y_index] = [values[x_index], values[y_index], MAP_UNCOMPLETED_VALUE, ROCK_UNCOMPLETED_VALUE]

    return base_map

def create_terrain_map(world, constants):
    """
    Creates the base geometric map that will be given to the agent for its completion.
    It is a 2D numpy matrix where each element in them represents the [x,y, height, rock flag].
    """
    print("Creating the ground truth: 0%", end="\r")

    terrain_map = np.array(np.zeros((constants.cell_number, constants.cell_number, 4)))
    low = -constants.map_size / 2 + constants.cell_size / 2
    high = constants.map_size / 2 - constants.cell_size / 2
    values = np.arange(low, high + 0.05, constants.cell_size)  # Make sure float imprecision doesn't remove the last one
    indexes = np.arange(0, constants.cell_number, 1)

    total_cells = constants.cell_number * constants.cell_number
    for x_index in indexes:
        if x_index % 10 == 0: world.tick()  # Minor performance drop but avoids hanging the simulation
        for y_index in indexes:
            percentage = round((x_index*constants.cell_number + y_index) / total_cells * 100, 2)
            print(f"Creating the ground truth: {percentage}%", end="\r")
            height, rock = get_cell_data(world, values[x_index], values[y_index], constants.cell_size)
            terrain_map[x_index, y_index] = [values[x_index], values[y_index], height, rock]

    print("Ground truth successfully created              ")
    return terrain_map


class GeometricMap(object):
    """
    This class provides a wrapper around the geometric map for ease of use.
    The map can still be accessed via the getter, in case users want to implement
    other helper functions, but make sure the 'self._map' is modifed, as that will
    be the variable used to emasure the agent's performance.
    """

    def __init__(self, constants):
        """Initialize the geometric map"""
        self._map = create_base_map(constants)
        self._map_size = constants.map_size
        self._cell_size = constants.cell_size
        self._cell_number = constants.cell_number

    def _is_cell_valid(self, x_index, y_index):
        """Returns whether the index is a valid one"""
        if x_index is None or x_index < 0 or x_index >= self._cell_number:
            return False
        if y_index is None or y_index < 0 or y_index >= self._cell_number:
            return False
        return True

    ###############################################################################
    #                                    General                                  #
    ###############################################################################
    def get_map_array(self):
        """Returns the geometric map. This returns the actual numpy array"""
        return self._map

    def get_map_size(self):
        """Returns the map size"""
        return self._map_size

    def get_cell_size(self):
        """Returns the cell size"""
        return self._cell_size

    def get_cell_number(self):
        """Returns the amount of cells per dimensions"""
        return self._cell_number

    def get_cell_indexes(self, x, y):
        """
        Given an x and y coordinates, returns the cell indexes that are closest to the given position.
        Returns None if the position is outside the mapping area.
        """
        cell_values = self._map[:,0,0]
        min_cell = self._map[0,0,0]
        max_cell = self._map[-1,0,0]
        max_distance = self.get_cell_size() / 2

        x_index = sum(cell_values < x - max_distance)
        if x_index == 0 and abs(min_cell - x) > max_distance:
            x_index = None
        if x_index == self._cell_number and abs(max_cell - x) > max_distance:
            x_index = None

        y_index = sum(cell_values < y - max_distance)
        if y_index == 0 and abs(min_cell - y) > max_distance:
            y_index = None
        if y_index == self._cell_number and abs(max_cell - y) > max_distance:
            y_index = None

        return (x_index, y_index)

    def get_cell_data(self, x_index, y_index):
        """Returns the cell center position and size. Returns None if out of range"""
        if not self._is_cell_valid(x_index, y_index):
            return (None, None)
        center = self._map[x_index, y_index, :2]
        return (center, self._cell_size)

    # Height helper functions
    def get_cell_height(self, x_index, y_index):
        """Gets the height of a given cell. Returns None if out of range"""
        if not self._is_cell_valid(x_index, y_index):
            return None
        return self._map[x_index, y_index, 2]

    def set_cell_height(self, x_index, y_index, height):
        """Returns the height of a given cell. Returns False if out of range, True otherwise"""
        if not self._is_cell_valid(x_index, y_index):
            return False
        self._map[x_index, y_index, 2] = height
        return True

    def get_height(self, x, y):
        """Returns the height of a given position. Returns None if out of range"""
        x_index, y_index = self.get_cell_indexes(x, y)
        if not self._is_cell_valid(x_index, y_index):
            return None
        return self.get_cell_height(x_index, y_index)

    def set_height(self, x, y, height):
        """Sets the height of a given position. Returns False if out of range, True otherwise"""
        x_index, y_index = self.get_cell_indexes(x, y)
        if not self._is_cell_valid(x_index, y_index):
            return False
        self.set_cell_height(x_index, y_index, height)
        return True

    # Rock structure helper functions
    def get_cell_rock(self, x_index, y_index):
        """Returns whether or not a given cell contains a rock. Returns None if out of range"""
        if not self._is_cell_valid(x_index, y_index):
            return None
        value = self._map[x_index, y_index, 3]
        return value if value == ROCK_UNCOMPLETED_VALUE else bool(value)

    def set_cell_rock(self, x_index, y_index, rock_flag):
        """Sets the rock flag of a given cell. Returns False if out of range, True otherwise"""
        if not self._is_cell_valid(x_index, y_index):
            return False
        self._map[x_index, y_index, 3] = rock_flag
        return True

    def get_rock(self, x, y):
        """Returns whether or not a given cell contains a rock. Returns None if out of range"""
        x_index, y_index = self.get_cell_indexes(x, y)
        if not self._is_cell_valid(x_index, y_index):
            return None
        return self.get_cell_rock(x_index, y_index)

    def set_rock(self, x, y, rock_flag):
        """Sets the rock flag of a given position. Returns False if out of range, True otherwise"""
        x_index, y_index = self.get_cell_indexes(x, y)
        if not self._is_cell_valid(x_index, y_index):
            return False
        self.set_cell_rock(x_index, y_index, rock_flag)
        return True