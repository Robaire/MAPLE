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
import numpy as np
import csv
import carla
import cv2 as cv
import random
from math import radians
from pynput import keyboard
import os
import shutil
import matplotlib.pyplot as plt
import skimage
import json

""" Import the AutonomousAgent from the Leaderboard. """

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

""" Define the entry point so that the Leaderboard can instantiate the agent class. """

def get_entry_point():
    return 'OpenCVagent'

""" Inherit the AutonomousAgent class. """

class OpenCVagent(AutonomousAgent):

    def setup(self, path_to_conf_file):

        """ This method is executed once by the Leaderboard at mission initialization. We should add any attributes to the class using 
        the 'self' Python keyword that contain data or methods we might need throughout the simulation. If you are using machine learning 
        models for processing sensor data or control, you should load the models here. We encourage the use of class attributes in place
        of using global variables which can cause conflicts. """

        """ Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys. """

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ Add some attributes to store values for the target linear and angular velocity. """

        self.current_v = 0
        self.current_w = 0

        """ Initialize a counter to keep track of the number of simulation steps. """

        self.frame = 0

        self.columns = ['frame','gt_x', 'gt_y', 'gt_z', 'gt_roll', 'gt_pitch', 'gt_yaw', 'imu_accel_x', 'imu_accel_y', 'imu_accel_z', 'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z']
        self.imu = []

        # set the trial number here
        self.trial = '040'

        if not os.path.exists(f'./data/{self.trial}'):
                os.makedirs(f'./data/{self.trial}')

        self.checkpoint_path = f'./data/{self.trial}/boulders_frame{self.frame}.json'



    def use_fiducials(self):

        """ We want to use the fiducials, so we return True. """
        return True

    def sensors(self):

        """ In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048) 
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light. """

        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.Left: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.Right: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.BackRight: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.Back: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
        }
        return sensors

    def run_step(self, input_data):

        """ The run_step method executes in every simulation time-step. Your control logic should go here. """

        """ In the first frame of the simulation we want to raise the robot's excavating arms to remove them from the 
        field of view of the cameras. Remember that we are working in radians. """

        if self.frame == 0:
            self.set_front_arm_angle(radians(0))
            self.set_back_arm_angle(radians(0))
            self.set_radiator_cover_state(carla.RadiatorCoverState.Open)
        

        """ Let's retrieve the front left camera data from the input_data dictionary using the correct dictionary key. We want the 
        grayscale monochromatic camera data, so we use the 'Grayscale' key. """

        sensor_data_frontleft = input_data['Grayscale'][carla.SensorPosition.FrontLeft]
        sensor_data_frontright = input_data['Grayscale'][carla.SensorPosition.FrontRight]
        sensor_data_left = input_data['Grayscale'][carla.SensorPosition.Left]
        sensor_data_right = input_data['Grayscale'][carla.SensorPosition.Right]
        sensor_data_backleft = input_data['Grayscale'][carla.SensorPosition.BackLeft]
        sensor_data_backright = input_data['Grayscale'][carla.SensorPosition.BackRight]
        sensor_data_front = input_data['Grayscale'][carla.SensorPosition.Front]
        sensor_data_back = input_data['Grayscale'][carla.SensorPosition.Back]

        # This part is gathering info to be used later
        imu_data = self.get_imu_data()
        mission_time = round(self.get_mission_time(), 2)
        transform = self.get_transform()
        transform_location_x = transform.location.x
        transform_location_y = transform.location.y
        transform_location_z = transform.location.z
        transform_rotation_r = transform.rotation.roll
        transform_rotation_p = transform.rotation.pitch
        transform_rotation_y = transform.rotation.yaw
        input_v = self.current_v
        input_w = self.current_w

        initial_transform = self.get_initial_position()

        print("intial transform: ", initial_transform)

        # adding a bunch of info to save to a csv at the end
        imu_entry = [self.frame] + \
            [transform_location_x, transform_location_y, transform_location_z, transform_rotation_r, transform_rotation_p, transform_rotation_y] + \
            imu_data.tolist()  # Convert NumPy array to list

        # Append to self.imu list to save at the end
        self.imu.append(imu_entry)

        """ We need to check that the sensor data is not None before we do anything with it. The date for each camera will be 
        None for every other simulation step, since the cameras operate at 10Hz while the simulator operates at 20Hz. """

        # TODO: This is a bunch of repeat code (sorry) for saving all the images - need to make this a function or streamline it
        if sensor_data_frontleft is not None:

            cv.imshow('Left front camera view', sensor_data_frontleft)
            cv.waitKey(1)
            dir_frontleft = f'data/{self.trial}/FrontLeft/'

            if not os.path.exists(dir_frontleft):
                os.makedirs(dir_frontleft)

            cv.imwrite(dir_frontleft + str(self.frame) + '.png', sensor_data_frontleft)
            print("saved image front left ", self.frame)

        if sensor_data_frontright is not None:

            # cv.imshow('Right front camera view', sensor_data_frontright)
            # cv.waitKey(1)
            dir_frontright = f'data/{self.trial}/FrontRight/'

            if not os.path.exists(dir_frontright):
                os.makedirs(dir_frontright)

            cv.imwrite(dir_frontright + str(self.frame) + '.png', sensor_data_frontright)
            print("saved image front right ", self.frame)

        # if sensor_data_backleft is not None:

        #     cv.imshow('Left back camera view', sensor_data_backleft)
        #     cv.waitKey(1)
        #     dir_backleft = f'data/{self.trial}/BackLeft/'

        #     if not os.path.exists(dir_backleft):
        #         os.makedirs(dir_backleft)

        #     cv.imwrite(dir_backleft + str(self.frame) + '.png', sensor_data_backleft)
        #     print("saved image back left ", self.frame)

        # if sensor_data_backright is not None:

        #     cv.imshow('Right back camera view', sensor_data_backright)
        #     cv.waitKey(1)
        #     dir_backright = f'data/{self.trial}/BackRight/'

        #     if not os.path.exists(dir_backright):
        #         os.makedirs(dir_backright)

        #     cv.imwrite(dir_backright + str(self.frame) + '.png', sensor_data_backright)
        #     print("saved image back right ", self.frame)


        if sensor_data_left is not None:

            # cv.imshow('Left camera view', sensor_data_left)
            # cv.waitKey(1)
            dir_left = f'data/{self.trial}/Left/'

            if not os.path.exists(dir_left):
                os.makedirs(dir_left)

            cv.imwrite(dir_left + str(self.frame) + '.png', sensor_data_left)
            print("saved image left ", self.frame)

        if sensor_data_right is not None:

            # cv.imshow('Right camera view', sensor_data_right)
            # cv.waitKey(1)
            dir_right = f'data/{self.trial}/Right/'

            if not os.path.exists(dir_right):
                os.makedirs(dir_right)

            cv.imwrite(dir_right + str(self.frame) + '.png', sensor_data_right)
            print("saved image right ", self.frame)

        """ Now we prepare the control instruction to return to the simulator, with our target linear and angular velocity. """

        self.frame += 1

        # TODO: navigation stuff will come in here!
        control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        
        """ If the simulation has been going for more than 5000 frames, let's stop it. """
        if self.frame >= 5000:
            self.mission_complete()

        return control

    def finalize(self):

        """ In the finalize method, we should clear up anything we've previously initialized that might be taking up memory or resources. 
        In this case, we should close the OpenCV window. """

        # Save the data to a CSV file
        output_filename_imu = f"/home/annikat/MAPLE/data/{self.trial}/imu_data.csv"
        os.makedirs(os.path.dirname(output_filename_imu), exist_ok=True)

        # Write to CSV file
        with open(output_filename_imu, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.columns)  # Write header
            writer.writerows(self.imu)    # Write the IMU data rows

        print(f"Data saved to {output_filename_imu}")

        cv.destroyAllWindows()

        """ We may also want to add any final updates we have from our mapping data before the mission ends. Let's add some random values 
        to the geometric map to demonstrate how to use the geometric map API. The geometric map should also be updated during the mission
        in the run_step() method, in case the mission is terminated unexpectedly. """

        """ Retrieve a reference to the geometric map object. """

        geometric_map = self.get_geometric_map()

        map_array = self.get_map_array()

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

        """ This is the callback executed when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear velocity. If the key pressed is either the left or right arrow, this method will set a target angular 
        velocity of 0.6 radians per second. """

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

        """ This method sets the angular or linear velocity to zero when the arrow key is released. Stopping the robot. """

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