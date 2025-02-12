#!/usr/bin/env python

# THIS AGENT CURRENTLY RUNS FASTSAM AND EXTRACTS BOULDER POSITIONS USING STEREO IMAGES FROM FRONT CAMERA
# IT RUNS WITH USER INPUTS USING ARROW KEYS
# IT SAVES DATA TO A "SELF.TRIAL" NUMBER THAT YOU HAVE TO SET

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This agent demonstrates how to structure your code and visualize camera data in
an OpenCV window and control the robot with keyboard commands with pynput
https://pypi.org/project/opencv-python/
https://pypi.org/project/pynput/

"""

import csv
import os
import random
from math import radians

import carla
import cv2 as cv
import numpy as np
from pynput import keyboard

from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.pose import InertialApriltagEstimator

""" Import the AutonomousAgent from the Leaderboard. """

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

""" Define the entry point so that the Leaderboard can instantiate the agent class. """


def get_entry_point():
    return "OpenCVagent"


""" Inherit the AutonomousAgent class. """


class OpenCVagent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """This method is executed once by the Leaderboard at mission initialization. We should add any attributes to the class using
        the 'self' Python keyword that contain data or methods we might need throughout the simulation. If you are using machine learning
        models for processing sensor data or control, you should load the models here. We encourage the use of class attributes in place
        of using global variables which can cause conflicts."""

        """ Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys. """

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ Add some attributes to store values for the target linear and angular velocity. """

        self.current_v = 0
        self.current_w = 0

        """ Initialize a counter to keep track of the number of simulation steps. """

        self.frame = 0

        self.columns = [
            "frame",
            "power",
            "input_v",
            "input_w",
            "gt_x",
            "gt_y",
            "gt_z",
            "gt_roll",
            "gt_pitch",
            "gt_yaw",
            "imu_accel_x",
            "imu_accel_y",
            "imu_accel_z",
            "imu_gyro_x",
            "imu_gyro_y",
            "imu_gyro_z",
        ]
        self.imu = []

        # set the trial number here
        self.trial = "008"

        if not os.path.exists(f"./data/{self.trial}"):
            os.makedirs(f"./data/{self.trial}")

        self.checkpoint_path = f"./data/{self.trial}/boulders_frame{self.frame}.json"

        self._active_side_cameras = False
        self._active_side_front_cameras = True

        self.estimator = InertialApriltagEstimator(self)
        self.navigatior = Navigator(self)
        self.detector = BoulderDetector(
            self, carla.SensorPosition.FrontLeft, carla.SensorPosition.FrontRight
        )

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return True

    def sensors(self):
        """In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light."""

        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1280",
                "height": "720",
                "use_semantic": True,
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1280",
                "height": "720",
                "use_semantic": True,
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1280",
                "height": "720",
                "use_semantic": True,
            },
            carla.SensorPosition.Left: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1280",
                "height": "720",
                "use_semantic": True,
            },
            carla.SensorPosition.Right: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1280",
                "height": "720",
                "use_semantic": True,
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1280",
                "height": "720",
                "use_semantic": True,
            },
            carla.SensorPosition.BackRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1280",
                "height": "720",
                "use_semantic": True,
            },
            carla.SensorPosition.Back: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": "1280",
                "height": "720",
                "use_semantic": True,
            },
        }
        return sensors

    def run_step(self, input_data):
        """Execute one step of navigation"""

        if self.frame == 0:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        sensor_data_frontleft = input_data["Grayscale"][carla.SensorPosition.FrontLeft]

        if sensor_data_frontleft is not None:
            cv.imshow("Left camera view", sensor_data_frontleft)
            cv.waitKey(1)
            # dir_frontleft = f'data/{self.trial}/FrontLeft/'

            # if not os.path.exists(dir_frontleft):
            #     os.makedirs(dir_frontleft)

            # # saving the semantic images and regular images
            # semantic = input_data['Semantic'][carla.SensorPosition.FrontLeft]
            # cv.imwrite(dir_frontleft + str(self.frame) + '_sem.png', semantic)

            # cv.imwrite(dir_frontleft + str(self.frame) + '.png', sensor_data_frontleft)
            # print("saved image front left ", self.frame)

        control = carla.VehicleVelocityControl(0, 0.5)
        front_data = input_data["Grayscale"][
            carla.SensorPosition.Front
        ]  # Do something with this
        if self._active_side_front_cameras:
            front_left_data = input_data["Grayscale"][
                carla.SensorPosition.FrontLeft
            ]  # Do something with this
            front_right_data = input_data["Grayscale"][
                carla.SensorPosition.FrontLeft
            ]  # Do something with this
        if self._active_side_cameras:
            left_data = input_data["Grayscale"][
                carla.SensorPosition.Left
            ]  # Do something with this
            right_data = input_data["Grayscale"][
                carla.SensorPosition.Right
            ]  # Do something with this

        mission_time = round(self.get_mission_time(), 2)

        # Get a position estimate for the rover
        estimate = self.estimator(input_data)

        # IMPORTANT NOTE: The estimate should never be NONE!!!, this is test code to catch that
        if estimate is None:
            goal_lin_vel, goal_ang_vel = 10, 0
            print(f"the estimate is returning NONE!!! that is a big problem buddy")
        else:
            # Get a goal linear and angular velocity from navigation
            goal_lin_vel, goal_ang_vel = self.navigatior(estimate)

            print(f"the estimate is {estimate}")
            imu_data = self.get_imu_data()
            print(f"the imu data is {imu_data}")

        if self.frame % 20 == 0:
            try:
                detections = self.detector(input_data)
                print(f"Boulder Detections: {len(detections)}")
            except:
                pass

        # Set the goal velocities to be returned
        control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)

        return control

    def finalize(self):
        """In the finalize method, we should clear up anything we've previously initialized that might be taking up memory or resources.
        In this case, we should close the OpenCV window."""

        # Save the data to a CSV file
        output_filename_imu = f"/home/annikat/LAC/MIT-MAPLE/LunarAutonomyChallenge/data/{self.trial}/imu_data.csv"

        # Write to CSV file
        with open(output_filename_imu, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.columns)  # Write header
            writer.writerows(self.imu)  # Write the IMU data rows

        print(f"Data saved to {output_filename_imu}")

        cv.destroyAllWindows()

        """ We may also want to add any final updates we have from our mapping data before the mission ends. Let's add some random values 
        to the geometric map to demonstrate how to use the geometric map API. The geometric map should also be updated during the mission
        in the run_step() method, in case the mission is terminated unexpectedly. """

        """ Retrieve a reference to the geometric map object. """

        geometric_map = self.get_geometric_map()

        map_array = self.get_map_array()

        print("Map array:", map_array)

        # # Save the data to a CSV file
        # output_filename_map_gt = f"/home/annikat/LAC/LunarAutonomyChallenge/data/{self.trial}/map_gt.csv"

        # np.savetxt(output_filename_map_gt, map_array, delimiter=",", fmt="%d")

        # print(f"Map saved to {output_filename_map_gt}")

        """ Set some random height values and rock flags. """

        for i in range(100):
            x = 10 * random.random() - 5
            y = 10 * random.random() - 5
            geometric_map.set_height(x, y, random.random())

            rock_flag = random.random() > 0.5
            geometric_map.set_rock(x, y, rock_flag)

        map_array = self.get_map_array()

        # print("Map array:", map_array)

        # # Save the data to a CSV file
        # output_filename_map_gt = f"/home/annikat/LAC/LunarAutonomyChallenge/data/{self.trial}/map_gt.csv"

        # np.savetxt(output_filename_map_gt, map_array, delimiter=",", fmt="%d")

        # print(f"Map saved to {output_filename_map_gt}")

    def on_press(self, key):
        """This is the callback executed when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear velocity. If the key pressed is either the left or right arrow, this method will set a target angular
        velocity of 0.6 radians per second."""

        if key == keyboard.Key.up:
            self.current_v += 0.1
            self.current_v = np.clip(self.current_v, 0, 0.3)
        if key == keyboard.Key.down:
            self.current_v -= 0.1
            self.current_v = np.clip(self.current_v, -0.3, 0)
        if key == keyboard.Key.left:
            self.current_w = 0.6
        if key == keyboard.Key.right:
            self.current_w = -0.6

    def on_release(self, key):
        """This method sets the angular or linear velocity to zero when the arrow key is released. Stopping the robot."""

        if key == keyboard.Key.up:
            self.current_v = 0
        if key == keyboard.Key.down:
            self.current_v = 0
        if key == keyboard.Key.left:
            self.current_w = 0
        if key == keyboard.Key.right:
            self.current_w = 0

        """ Press escape to end the mission. """
        if key == keyboard.Key.esc:
            self.mission_complete()
            cv.destroyAllWindows()
