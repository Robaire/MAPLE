#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a class to serve as a static timer
"""

import datetime

class GameTime(object):

    """
    This (static) class provides access to the CARLA game time
    """

    _carla_time = 0
    _carla_delta_time = 0
    _platform_previous_time = 0
    _platform_time = 0
    _wallclock_timestamp = 0
    _frame = 0  # This is the ACTUAL frame

    _init_time = 0
    _init_platform_time = 0

    @staticmethod
    def on_carla_tick(timestamp):
        """Callback receiving the CARLA time"""
        if GameTime._frame < timestamp.frame:

            GameTime._carla_time = timestamp.elapsed_seconds - GameTime._init_time
            GameTime._carla_delta_time = timestamp.delta_seconds
            GameTime._platform_time = timestamp.platform_timestamp - GameTime._init_platform_time
            GameTime._platform_delta_time = timestamp.platform_timestamp - GameTime._platform_previous_time
            GameTime._wallclock_timestamp = datetime.datetime.now()
            GameTime._frame = timestamp.frame

            GameTime._platform_previous_time = timestamp.platform_timestamp

    @staticmethod
    def start(timestamp):
        """Starts the timer, setting the initial values to 0"""
        GameTime._init_time = timestamp.elapsed_seconds
        GameTime._init_platform_time = timestamp.platform_timestamp
        GameTime._platform_previous_time = GameTime._init_platform_time

    @staticmethod
    def restart():
        """Reset game timer to 0"""
        GameTime._carla_time = 0
        GameTime._carla_delta_time = 0
        GameTime._platform_previous_time = 0
        GameTime._platform_time = 0
        GameTime._wallclock_timestamp = 0
        GameTime._frame = 0

        GameTime._init_time = 0
        GameTime._init_platform_time = 0

    @staticmethod
    def get_time():
        """Returns elapsed game time"""
        return GameTime._carla_time

    @staticmethod
    def get_delta_time():
        """Returns delta game time"""
        return GameTime._carla_delta_time

    @staticmethod
    def get_platform_time():
        """Returns elapsed game time"""
        return GameTime._platform_time

    @staticmethod
    def get_platform_delta_time():
        """Returns delta game time"""
        return GameTime._platform_delta_time

    @staticmethod
    def get_wallclocktime():
        """Returns elapsed game time"""
        return GameTime._wallclock_timestamp

    @staticmethod
    def get_frame():
        """Returns elapsed game time"""
        return GameTime._frame

