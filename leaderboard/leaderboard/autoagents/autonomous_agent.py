#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

import carla
from leaderboard.agents.agent_utilities import AgentComponentsControl
from leaderboard.utils.timer import GameTime

MAX_LINEAR_SPEED = 0.49         # m/s
MAX_ANGULAR_SPEED = 4.13        # rad/s


class AutonomousAgent(object):
    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    ###############################################################################
    #                     Base functions, do not override them                    #
    ###############################################################################
    def __init__(self):
        self._components_control = None
        self._vehicle_status = None
        self._finished = False

        self._geometric_map = None
        self._initial_position = None
        self._initial_lander_position = None
        self._mission_time = 0

    def __call__(self, mission_time, vehicle_status, input_data):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        self._mission_time = mission_time
        self._vehicle_status = vehicle_status

        timestamp = GameTime.get_time()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = GameTime.get_platform_time()
        sim_ratio = 0 if wallclock_diff == 0 else timestamp / wallclock_diff

        print('=== [Agent] -- Wallclock = {} -- System time = {} -- Simulation time = {} -- Ratio = {}x'.format(
            str(wallclock)[:-3],
            format(wallclock_diff, '.3f'),
            format(timestamp, '.3f'),
            format(sim_ratio, '.3f')
        ))

        self._components_control = AgentComponentsControl()
        velocity_control = self.run_step(input_data)
        assert isinstance(velocity_control, carla.VehicleVelocityControl)  # Make sure it is a velocity control

        velocity_control2 = carla.VehicleVelocityControl()  # Actual output, clamped and in left-handed reference frame
        velocity_control2.linear_target_velocity = max(min(velocity_control.linear_target_velocity, MAX_LINEAR_SPEED), -MAX_LINEAR_SPEED)
        velocity_control2.angular_target_velocity = max(min(velocity_control.angular_target_velocity, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)
        velocity_control2.angular_target_velocity *= -1  # Change coordinate system frame

        return velocity_control2, self._components_control

    def set_geometric_map(self, geometric_map):
        """Saves the geomtric map"""
        self._geometric_map = geometric_map

    def set_initial_position(self, position):
        """Saves the agent's initial position for an easier transformation to CARLA coordinates"""
        self._initial_position = position

    def set_initial_lander_position(self, position):
        """Saves the lander's initial position with respect to the rover"""
        self._initial_lander_position = position

    def set_imu_data(self, imu_data):
        """Saves the IMU data of the current frame"""
        self._imu_data = imu_data

    def has_finished(self):
        """Returns whether or not the agent has finished"""
        return self._finished

    def get_map_array(self):
        """Returns the map array calculated by the agent"""
        if self._geometric_map is None:
            return None
        return self._geometric_map.get_map_array()

    ###############################################################################
    #                     Functions to be overriden by the user                   #
    ###############################################################################
    def setup(self, path_to_conf_file):
        """Initialize everything needed by your agent"""
        pass

    def use_fiducials(self):
        """Returns whether or not the lander fiducials will be used"""
        return True

    def sensors(self):
        """
        For all carla.SensorPositions of the ego vehicle, define the initial camera state
        (turned on or off), light intensity, and camera resolution. Note that the camera state
        and light intensity can also be changed in runtime.

        The 'use_semantic' is an optional argument (defaults to False) that creates an additional
        semantic segmentation camera with the same parameters and position as the RGB one, 
        automatically turning on and off when the RGB one does. Semantic segmentation cameras
        are ONLY ALLOWED DURING TRAINING and evaluated agents using them will be rejected.
        """
        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048', 'use_semantic': False
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048', 'use_semantic': False
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048', 'use_semantic': False
            },
            carla.SensorPosition.Left: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048', 'use_semantic': False
            },
            carla.SensorPosition.Right: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048', 'use_semantic': False
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048', 'use_semantic': False
            },
            carla.SensorPosition.BackRight: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048', 'use_semantic': False
            },
            carla.SensorPosition.Back: {
                'camera_active': False, 'light_intensity': 0, 'width': '2448', 'height': '2048', 'use_semantic': False
            },
        }

        return sensors

    def run_step(self, input_data):
        """
        Execute one step of navigation.

        input_data is a dictionary that contains the sensors data:
        - Active sensors will have their data represented as a numpy array
        - Active sensors without any data in this tick will instead contain 'None'
        - Inactive sensors will not be present in the dictionary.

        Example:

        input_data = {
            'Grayscale': {
                carla.SensorPosition.FrontLeft:  np.array(...),
                carla.SensorPosition.FrontRight:  np.array(...),
            },
            'Semantic':{
                carla.SensorPosition.FrontLeft:  np.array(...),
            }
        }
        """

        return carla.VehicleVelocityControl()

    def finalize(self):
        """
        Finalizes the agent. Called after the mission ends, which can occur
        due to the agent's mission_complete or any off-nominal terminations.
        """
        pass

    ###############################################################################
    #  Utility functions to 'expose' some functionalities of the AutonomousAgent  #
    ###############################################################################

    # Simulation getters
    def get_initial_position(self):
        """Returns the initial position of the agent (Transform)"""
        return self._initial_position

    def get_initial_lander_position(self):
        """Returns the initial position of the ladner with respect to the agent (Transform)"""
        return self._initial_lander_position

    def get_geometric_map(self):
        """Returns the geometric map"""
        return self._geometric_map

    def get_mission_time(self):
        """"Returns the mission time, which is the amount of time passed inside the simulation (s)"""
        return self._mission_time

    def get_current_power(self):
        """Returns the current power of the IPEx (Wh)"""
        return self._vehicle_status.current_power

    def get_consumed_power(self):
        """Returns the consumed power of the IPEx (Wh)"""
        return self._vehicle_status.consumed_power

    def get_imu_data(self):
        """Returns the IMU data of that frame (np.array())"""
        return self._imu_data

    # Finish function
    def mission_complete(self):
        """Tells the simulation that the agent has finished"""
        self._finished = True

    # Vehicle setters
    def set_front_arm_angle(self, angle):
        """Sets the front arm's target angle"""
        self._components_control.front_arm_angle = angle

    def set_back_arm_angle(self, angle):
        """Sets the back arm's target angle"""
        self._components_control.back_arm_angle = angle

    def set_front_drums_target_speed(self, speed):
        """Sets the front drum's target speed"""
        self._components_control.front_drum_speed = speed

    def set_back_drums_target_speed(self, speed):
        """Sets the back drum's target speed"""
        self._components_control.back_drum_speed = speed

    def set_radiator_cover_state(self, radiator_state):
        """Sets the radiator cover state"""
        self._components_control.radiator_cover_state = radiator_state

    def set_light_state(self, light, light_state):
        """Sets the light state"""
        self._components_control.sensor_state[light][1] = light_state

    def set_camera_state(self, camera, camera_state):
        """Sets the camera state"""
        self._components_control.sensor_state[camera][0] = camera_state

    # Vehicle getters
    def get_linear_speed(self):
        """Returns the vehicle's linear speed given by the odometry of the IPEx (m/s)"""
        return self._vehicle_status.odometry_linear_speed

    def get_angular_speed(self):
        """Returns the vehicle's angular speed given by the odometry of the IPEx (rad/s)"""
        return self._vehicle_status.odometry_angular_speed

    def get_front_arm_angle(self):
        """Returns the front arm's angle (rad)"""
        return self._vehicle_status.front_arm_angle

    def get_back_arm_angle(self):
        """Returns the back arm's angle (rad)"""
        return self._vehicle_status.back_arm_angle

    def get_front_drums_speed(self):
        """Returns the front drum's speed (rad/s)"""
        return self._vehicle_status.front_drums_speed

    def get_back_drums_speed(self):
        """Returns the back drum's speed (rad/s)"""
        return self._vehicle_status.back_drums_speed

    def get_radiator_cover_angle(self):
        """Returns the radiator cover angle (rad)"""
        return self._vehicle_status.radiator_cover_angle

    def get_light_state(self, light):
        """Returns the light state (0 to 100%)"""
        return self._vehicle_status.sensor_state[light][1]

    def get_camera_state(self, camera):
        """Returns the camera state (True / False)"""
        return self._vehicle_status.sensor_state[camera][0]

    def get_camera_position(self, camera):
        """Returns the camera position with respect to the vehicle (Transform)"""
        return self._vehicle_status.sensor_state[camera][2]

    def get_light_position(self, light):
        """Returns the light position with respect to the vehicle (Transform)"""
        return self._vehicle_status.sensor_state[light][3]

    def get_transform(self):
        """
        Returns the exact transform of the vehicle.
        THIS IS ONLY AVAILABLE FOR TRAINING and will return None during evaluation  (Transform).
        """
        return self._vehicle_status.transform
