# Copyright (c) # Copyright (c) 2024 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


"""
This module provides GeometricMap implementation.
"""

import json

from leaderboard.utils.timer import GameTime
from leaderboard.agents.coordinate_conversion import toRHCStransform

SAVE_TO_DISK_PERIOD = 30 # Save every 30 seconds
CHUNK_PERIOD = 1800   # Split recorder into chunks of 30min each one (~150MB)

class MissionLogger(object):
    """
    This class handles the recording of all the simulation and agent data
    """

    def __init__(self, client, record, record_control, endpoint, name):
        """Initialize the geometric map"""
        self._client = client
        self._record = record
        self._record_control = record_control
        self._endpoint = endpoint
        self._agent_endpoint = None
        self._name = name

        self._chunk = 0
        self._agent_controls = {'records': []}

    def start(self):
        """Starts the CARLA recorder"""
        if self._record:
            recorder_endpoint = "{}/{}_chunk{}.log".format(self._endpoint, self._name, self._chunk)
            self._client.start_recorder(recorder_endpoint)

        if self._record_control:
            self._agent_endpoint = "{}/{}_agent_chunk{}.json".format(self._endpoint, self._name, self._chunk)

    def stop(self, chunk=False):
        """Stops the CARLA recorder"""
        if self._record:
            self._client.stop_recorder()

        if self._record_control:
            self.save_to_disk(force=True)
            if chunk:
                self._agent_controls['records'].clear()

    def tick(self, velocity_control, component_control, vehicle_status, vehicle, mission_time):
        """Updates the agent records"""

        if self._record_control:

            veh_transform = toRHCStransform(vehicle.get_transform())
            velocity_control.angular_target_velocity *= -1 # Back to RHCS

            new_record = {
                "control": {
                    "linear_velocity": velocity_control.linear_target_velocity,
                    "angular_velocity": velocity_control.angular_target_velocity,  
                },
                "current_power": round(vehicle_status.current_power, 3),
                "mission_time": mission_time,
                "transform": {
                    "x": veh_transform.location.x,
                    "y": veh_transform.location.y,
                    "z": veh_transform.location.z,
                    "roll": veh_transform.rotation.roll,
                    "pitch": veh_transform.rotation.pitch,
                    "yaw": veh_transform.rotation.yaw,
                }
            }

            front_angle = component_control.front_arm_angle
            if front_angle is not None: new_record["control"]["front_arm_angle"] = front_angle

            back_angle = component_control.back_arm_angle
            if back_angle is not None: new_record["control"]["back_arm_angle"] = back_angle

            front_speed = component_control.front_drum_speed
            if front_speed is not None: new_record["control"]["front_drum_speed"] = front_speed

            back_speed = component_control.back_drum_speed
            if back_speed is not None: new_record["control"]["back_drum_speed"] = back_speed

            if component_control.radiator_cover_state is not None:
                radiator_state = str(component_control.radiator_cover_state)
            else:
                radiator_state = None
            if radiator_state is not None: new_record["control"]["radiator_cover_state"] = radiator_state

            for (sensor, (camera, light)) in component_control.sensor_state.items():
                if camera is None and light is None: continue
                if camera: new_record["control"][f"camera_{str(sensor)}"] = camera
                if light: new_record["control"][f"light_{str(sensor)}"] = light

            self._agent_controls['records'].append(new_record)
            self.save_to_disk()

        # Split records into chunks to avoid large files
        current_time = round(GameTime.get_time(), 2)
        if current_time % CHUNK_PERIOD == 0:
            self.stop(chunk=True)
            self._chunk += 1
            self.start()

    def save_to_disk(self, force=False):
        if self._record_control and self._agent_controls:
            current_time = round(GameTime.get_time(), 2)
            if force or current_time % SAVE_TO_DISK_PERIOD == 0:
                with open(self._agent_endpoint, 'w') as fd:
                    json.dump(self._agent_controls, fd, indent=4, sort_keys=True)

    def cleanup(self):
        pass 

