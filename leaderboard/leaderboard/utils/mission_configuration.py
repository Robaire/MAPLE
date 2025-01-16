#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the key configuration parameters for an XML-based mission
"""

import carla


class RockData(object):
    """
    This is a configuration class to hold the rocks attributes
    """

    def __init__(self, transform, scale, model):
        self.transform = transform
        self.scale = scale
        self.model = model

    @staticmethod
    def parse_from_node(node):
        """
        static method to initialize an RockData from a given ET tree
        """
        pos_x = float(node.attrib.get('x', 0))
        pos_y = float(node.attrib.get('y', 0))
        pos_z = float(node.attrib.get('z', 0))
        yaw = float(node.attrib.get('yaw', 0))

        transform = carla.Transform(
            carla.Location(x=pos_x, y=pos_y, z=pos_z),
            carla.Rotation(yaw=yaw)
        )

        scale = float(node.attrib.get('scale', 1))
        mesh = node.attrib.get('model')

        return RockData(transform, scale, mesh)


class MissionConfiguration(object):
    """
    Basic configuration of a mission
    """

    def __init__(self):
        self.map = None
        self.preset = None
        self.name = ""
        self.base_name = ""
        self.index = None
        self.rep_index = None
