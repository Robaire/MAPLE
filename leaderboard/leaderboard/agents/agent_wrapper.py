#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for autonomous agents required for tracking and checking of used sensors
"""

import importlib
import os
import sys
import numpy as np

import carla

from leaderboard.agents.geometric_map import GeometricMap
from leaderboard.agents.sensor_interface import CallBack
from leaderboard.agents.agent_utilities import VehicleStatus
from leaderboard.agents.sensor_interface import SensorInterface
from leaderboard.agents.coordinate_conversion import toRHCStransform, get_lander_transform
from leaderboard.agents.imu import IMU
from leaderboard.utils.timer import GameTime

CAMERA_BONE_DICT = {
    carla.SensorPosition.Front: 'FrontArmCamera',
    carla.SensorPosition.FrontLeft: 'FrontLeftCamera',
    carla.SensorPosition.FrontRight: 'FrontRightCamera',
    carla.SensorPosition.Left: 'LeftCamera',
    carla.SensorPosition.Right: 'RightCamera',
    carla.SensorPosition.BackLeft: 'BackLeftCamera',
    carla.SensorPosition.BackRight: 'BackRightCamera',
    carla.SensorPosition.Back: 'BackArmCamera',
}

MAX_CAMERA_WIDTH = 2448
MAX_CAMERA_HEIGHT = 2048


class AgentSetupError(Exception):
    """
    Exceptions thrown when the agent returns an error during the simulation
    """

    def __init__(self, message):
        super(AgentSetupError, self).__init__(message)


class AgentRuntimeError(Exception):
    """
    Exceptions thrown when the agent returns an error during the simulation
    """

    def __init__(self, message):
        super(AgentRuntimeError, self).__init__(message)


class SensorConfigurationError(Exception):
    """
    Exceptions thrown when the agent tries to spawn semantic cameras during evaluation
    """

    def __init__(self, message):
        super(SensorConfigurationError, self).__init__(message)


def _get_agent_instance(agent_filename):
    module_name = os.path.basename(agent_filename).split('.')[0]
    sys.path.insert(0, os.path.dirname(agent_filename))
    module_agent = importlib.import_module(module_name)

    agent_class_name = getattr(module_agent, 'get_entry_point')()
    agent_class_obj = getattr(module_agent, agent_class_name)

    return agent_class_obj()


class AgentWrapper(object):
    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    """

    def __init__(self, world, agent_filename, agent_config, evaluation):
        """Initializer"""
        self._world = world
        self._vehicle = None
        self._sensor_interface = SensorInterface()

        self.agent = None   # Initialized here in case it fails
        self.agent_fiducials = None
        try:
            self.agent = _get_agent_instance(agent_filename)
            self.agent_fiducials = self.agent.use_fiducials() 
            assert isinstance(self.agent_fiducials, bool) # Make sure the fiducials flag is a boolean
        except Exception as e:
            raise AgentSetupError(e)
        self._agent_config = agent_config

        self._evaluation = evaluation
        self._cameras = {'Grayscale': {}, 'Semantic': {}}
        self._imu = None

    def setup(self, vehicle, lander, constants):
        """Prepares the agent and sensor"""
        self._vehicle = vehicle
        try:
            self.agent.set_geometric_map(GeometricMap(constants))
            self.agent.set_initial_position(toRHCStransform(vehicle.get_transform()))
            self.agent.set_initial_lander_position(get_lander_transform(vehicle.get_transform(), lander.get_transform()))
            self.agent.setup(self._agent_config)
        except Exception as e:
            raise AgentSetupError(e)
        self._setup_sensors()

    def _spawn_camera(self, sensor_id, transform, bone, width, height):
        """Spawns the cameras with their corresponding attributes"""
        bp_library = self._world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', width)
        bp.set_attribute('image_size_y', height)
        bp.set_attribute('fov', '70')
        bp.set_attribute('sensor_tick', '0.1')

        camera = self._world.spawn_actor(bp, transform, self._vehicle, bone=bone)
        if not camera:
            raise ValueError(f"Couldn't spawn camera {sensor_id}. Stopping the simulation")

        return camera

    def _spawn_semantic_camera(self, sensor_id, transform, bone, width, height):
        """Spawns the cameras with their corresponding attributes"""
        bp_library = self._world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', width)
        bp.set_attribute('image_size_y', height)
        bp.set_attribute('fov', '70')
        bp.set_attribute('sensor_tick', '0.1')

        semantic_camera = self._world.spawn_actor(bp, transform, self._vehicle, bone=bone)
        if not semantic_camera:
            raise ValueError(f"Couldn't spawn segmentation camera {sensor_id}. Stopping the simulation")

        return semantic_camera

    def _activate_camera(self, camera_id):
        """Make the cameras start listening"""
        if self._cameras['Grayscale'][camera_id][1]:
            print(f"Warning: Agent tried to activate camera 'carla.SensorPosition.{camera_id}' but it is already active")
            return
        camera = self._cameras['Grayscale'][camera_id][0]
        camera.listen(CallBack(camera_id, camera, self._sensor_interface, 'Grayscale'))
        self._cameras['Grayscale'][camera_id][1] = True

        if camera_id in self._cameras['Semantic']:
            semantic_camera = self._cameras['Semantic'][camera_id][0]
            semantic_camera.listen(CallBack(camera_id, camera, self._sensor_interface, 'Semantic'))
            self._cameras['Semantic'][camera_id][1] = True

    def _deactivate_camera(self, camera_id):
        """Make the cameras stop listening"""
        if not self._cameras['Grayscale'][camera_id][1]:
            print(f"Warning: Agent tried to deactivate camera 'carla.SensorPosition.{camera_id}' but it is already inactive")
            return
        camera = self._cameras['Grayscale'][camera_id][0]
        camera.stop()
        self._cameras['Grayscale'][camera_id][1] = False

        if camera_id in self._cameras['Semantic']:
            semantic_camera = self._cameras['Semantic'][camera_id][0]
            semantic_camera.stop()
            self._cameras['Semantic'][camera_id][1] = False

    def _check_attributes(self, sensor_data, sensor_id):
        """Checks that all the sensor attributes are in the agent's configuration"""
        if sensor_id not in sensor_data:
            raise AgentSetupError(f"Couldn't find the 'carla.SensorPosition.{sensor_id}' in the sensors configuration")
        attributes = sensor_data[sensor_id]
        if 'camera_active' not in attributes:
            raise AgentSetupError(f"Attribute 'camera_active' is missing from the sensor 'carla.SensorPosition.{sensor_id}'")
        if 'light_intensity' not in attributes:
            raise AgentSetupError(f"Attribute 'light_intensity' is missing from the sensor 'carla.SensorPosition.{sensor_id}'")
        if 'width' not in attributes:
            raise AgentSetupError(f"Attribute 'width' is missing from the sensor 'carla.SensorPosition.{sensor_id}'")
        if 'height' not in attributes:
            raise AgentSetupError(f"Attribute 'height' is missing from the sensor 'carla.SensorPosition.{sensor_id}'")

    def _setup_sensors(self):
        """Create the sensors defined by the user and attach them to the ego-vehicle"""

        # IMU
        self._imu = IMU(self._vehicle)

        # Cameras
        sensor_data = self.agent.sensors()
        for (camera_id, bone) in CAMERA_BONE_DICT.items():

            self._check_attributes(sensor_data, camera_id)

            camera_state = sensor_data[camera_id]['camera_active']
            light_state = sensor_data[camera_id]['light_intensity']
            width = str(sensor_data[camera_id]['width'])
            if int(width) > MAX_CAMERA_WIDTH:
                print(f"\033[93m'Camera '{camera_id}' has a higher width than the allowed one. Setting it to the maximum of {MAX_CAMERA_WIDTH}\033[0m")
                width = str(MAX_CAMERA_WIDTH)
            height = str(sensor_data[camera_id]['height'])
            if int(height) > MAX_CAMERA_HEIGHT:
                print(f"\033[93m'Camera '{camera_id}' has a higher height than the allowed one. Setting it to the maximum of {MAX_CAMERA_HEIGHT}\033[0m")
                height = str(MAX_CAMERA_HEIGHT)
            semantic = sensor_data[camera_id]['use_semantic'] if 'use_semantic' in sensor_data[camera_id] else False

            if semantic and self._evaluation:
                raise SensorConfigurationError("Detected semantic segmentation cameras during the evaluation")

            camera = self._spawn_camera(camera_id, carla.Transform(), bone, width, height)
            self._cameras['Grayscale'][camera_id] = [camera, False]

            if semantic:
                semantic_camera = self._spawn_semantic_camera(camera_id, carla.Transform(), bone, width, height)
                self._cameras['Semantic'][camera_id] = [semantic_camera, False]

            self._vehicle.set_camera_state(camera_id, camera_state)
            self._vehicle.set_light_state(camera_id, light_state)

            if camera_state:
                self._activate_camera(camera_id)

        # When using the 'sensor_tick', the first callback can be received at a wrong frame,
        # so tick several times on initialization
        for _ in range(2):
            self._world.tick()

    def _get_camera_data(self):
        """Returns the camera data of the current frame"""
        return self._sensor_interface.get_data(GameTime.get_frame(), self._cameras)

    def _get_imu_data(self):
        """Returns the imu data of the current frame"""
        return self._imu.get_data()

    def tick(self, mission_time):
        """Ticks the agent"""
        sensor_data = self._get_camera_data()
        self.agent.set_imu_data(self._get_imu_data())
        vehicle_status = VehicleStatus.from_vehicle(self._vehicle, self._evaluation)
        velocity_control, component_control = self.agent(mission_time, vehicle_status, sensor_data)
        self._update_sensors(component_control)
        return velocity_control, component_control, vehicle_status

    def _update_sensors(self, component_control):
        """Updates the state of the sensors according to the user changes"""
        for (sensor_id, (sensor_state, _)) in component_control.sensor_state.items():
            if sensor_state is not None:
                self._activate_camera(sensor_id) if sensor_state else self._deactivate_camera(sensor_id)

    def get_agent_map(self):
        """Returns the agent's calculated geometric map"""
        return self.agent.get_map_array()

    def stop(self):
        """Stops the sensors"""
        for (sensor, state) in self._cameras['Grayscale'].values():
            if state: sensor.stop()
        for (sensor, state) in self._cameras['Semantic'].values():
            if state: sensor.stop()

        self._world.tick()

    def cleanup(self):
        """Remove and destroy all sensors"""
        for (sensor, _) in self._cameras['Grayscale'].values():
            sensor.destroy()
        for (sensor, _) in self._cameras['Semantic'].values():
            sensor.destroy()

        self._world.tick()
