#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

import carla

from leaderboard.agents.coordinate_conversion import toRHCStransform

STR_TO_ENUM = {
    "Open": carla.RadiatorCoverState.Open,
    "Close": carla.RadiatorCoverState.Close,
    "Front": carla.SensorPosition.Front,
    "FrontLeft": carla.SensorPosition.FrontLeft,
    "FrontRight": carla.SensorPosition.FrontRight,
    "Left": carla.SensorPosition.Left,
    "Right": carla.SensorPosition.Right,
    "BackLeft": carla.SensorPosition.BackLeft,
    "BackRight": carla.SensorPosition.BackRight,
    "Back": carla.SensorPosition.Back,
}

CAMERA_NAME_TO_INDEX = {
    carla.SensorPosition.Front: 1,
    carla.SensorPosition.FrontLeft: 4,
    carla.SensorPosition.FrontRight: 6,
    carla.SensorPosition.Left: 14,
    carla.SensorPosition.Right: 12,
    carla.SensorPosition.BackLeft: 10,
    carla.SensorPosition.BackRight: 8,
    carla.SensorPosition.Back: 2,
}

LIGHT_NAME_TO_INDEX = {
    carla.SensorPosition.Front: 0,
    carla.SensorPosition.FrontLeft: 5,
    carla.SensorPosition.FrontRight: 7,
    carla.SensorPosition.Left: 15,
    carla.SensorPosition.Right: 13,
    carla.SensorPosition.BackLeft: 11,
    carla.SensorPosition.BackRight: 9,
    carla.SensorPosition.Back: 3,
}

class AgentComponentsControl(object):

    def __init__(self):
        self.front_arm_angle = None
        self.back_arm_angle = None
        self.front_drum_speed = None
        self.back_drum_speed = None
        self.radiator_cover_state = None
        self.sensor_state = {
            carla.SensorPosition.Front: [None, None],
            carla.SensorPosition.FrontLeft: [None, None],
            carla.SensorPosition.FrontRight: [None, None],
            carla.SensorPosition.Left:  [None, None],
            carla.SensorPosition.Right: [None, None],
            carla.SensorPosition.BackLeft: [None, None],
            carla.SensorPosition.BackRight: [None, None],
            carla.SensorPosition.Back: [None, None],
        }

    def apply_control(self, vehicle):
        if self.front_arm_angle is not None:
            vehicle.set_front_arm_angle(self.front_arm_angle)
        if self.back_arm_angle is not None:
            vehicle.set_back_arm_angle(self.back_arm_angle)
        if self.front_drum_speed is not None:
            vehicle.set_front_drums_target_speed(self.front_drum_speed)
        if self.back_drum_speed is not None:
            vehicle.set_back_drums_target_speed(self.back_drum_speed)
        if self.radiator_cover_state is not None:
            vehicle.set_radiator_cover_state(self.radiator_cover_state)

        for (sensor_id, (camera_state, light_state)) in self.sensor_state.items():
            if light_state is not None:
                vehicle.set_light_state(sensor_id, light_state)
            if camera_state is not None:
                vehicle.set_camera_state(sensor_id, camera_state)

    def __str__(self):
        str_ = f"AgentComponentControl(front_arm_angle={self.front_arm_angle}, "
        str_ += f"back_arm_angle={self.front_arm_angle}, "
        str_ += f"front_drum_speed={self.front_drum_speed}, "
        str_ += f"back_drum_speed={self.back_drum_speed}, "
        str_ += f"radiator_cover_state={self.radiator_cover_state}, "
        str_ += "sensor_state={"
        for sensor in self.sensor_state:
            str_ += f"{sensor}: {self.sensor_state[sensor]}, "
        str_ = str_[:-2]
        str_ += "})"
        return str_

    @staticmethod
    def from_dict(data):
        component_control = AgentComponentsControl()
        if data["front_arm_angle"] is not None:
            component_control.front_arm_angle = float(data["front_arm_angle"])
        if data["back_arm_angle"] is not None:
            component_control.back_arm_angle = float(data["back_arm_angle"])
        if data["front_drum_speed"] is not None:
            component_control.front_drum_speed = float(data["front_drum_speed"])
        if data["back_drum_speed"] is not None:
            component_control.back_drum_speed = float(data["back_drum_speed"])
        if data["radiator_cover_state"] is not None:
            component_control.radiator_cover_state = STR_TO_ENUM[data["radiator_cover_state"]]

        for (sensor_id, states) in data["sensor_state"].items():
            component_control.sensor_state[STR_TO_ENUM[sensor_id]] = states

        return component_control


class VehicleStatus(object):

    def __init__(self):
        self.odometry_linear_speed = None
        self.odometry_angular_speed = None
        self.front_arm_angle = None
        self.back_arm_angle = None
        self.front_drums_speed = None
        self.back_drums_speed = None
        self.radiator_cover_angle = None
        self.current_power = None
        self.consumed_power = None
        self.transform = None
        self.sensor_state = {
            carla.SensorPosition.Front: [None, None],
            carla.SensorPosition.FrontLeft: [None, None],
            carla.SensorPosition.FrontRight: [None, None],
            carla.SensorPosition.Left:  [None, None],
            carla.SensorPosition.Right: [None, None],
            carla.SensorPosition.BackLeft: [None, None],
            carla.SensorPosition.BackRight: [None, None],
            carla.SensorPosition.Back: [None, None],
        }

    @staticmethod
    def from_vehicle(vehicle, evaluation):
        status = VehicleStatus()
        status.odometry_linear_speed = vehicle.get_odometry_speed()
        status.odometry_angular_speed = vehicle.get_odometry_angular_speed()
        status.front_arm_angle = vehicle.get_front_arm_angle()
        status.back_arm_angle = vehicle.get_back_arm_angle()
        status.front_drums_speed = vehicle.get_front_drums_speed()
        status.back_drums_speed = vehicle.get_back_drums_speed()
        status.radiator_cover_angle = vehicle.get_radiator_cover_angle()
        status.current_power = vehicle.get_current_power()
        status.consumed_power = vehicle.get_consumed_power()
        if not evaluation: status.transform = toRHCStransform(vehicle.get_transform())

        socket_transforms = vehicle.get_socket_relative_transforms()
        for sensor_id in status.sensor_state.keys():
            camera_state = vehicle.get_camera_state(sensor_id)
            camera_position = toRHCStransform(socket_transforms[CAMERA_NAME_TO_INDEX[sensor_id]])
            light_state = vehicle.get_light_state(sensor_id)
            light_position = toRHCStransform(socket_transforms[LIGHT_NAME_TO_INDEX[sensor_id]])
            status.sensor_state[sensor_id] = [camera_state, light_state, camera_position, light_position]

        return status
