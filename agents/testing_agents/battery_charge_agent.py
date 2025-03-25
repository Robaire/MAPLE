#!/usr/bin/env python

"""
Steer with arrow keys. Take pictures and save location with space. Pictures are saved as sensor-missionTime. End sim with f1

"""



import numpy as np
import carla
import cv2 as cv
import random
from math import radians
from pynput import keyboard
import os

from maple.utils import pytransform_to_tuple, carla_to_pytransform

""" Import the AutonomousAgent from the Leaderboard. """

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

""" Define the entry point so that the Leaderboard can instantiate the agent class. """

def get_entry_point():
    return 'OpenCVagent'

""" Inherit the AutonomousAgent class. """

class OpenCVagent(AutonomousAgent):

    take_photo = False

    # Save to write to file at the end
    photos = []
    transforms = []

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

        # Keep track of previous power to see if we are charging
        self.prev_power = None

    def use_fiducials(self):

        """ We want to use the fiducials, so we return True. """
        return True

    def sensors(self):

        """ In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048) 
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light. """

        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'name': "front"
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'name': "front left"
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'name': "front right"
            },
            carla.SensorPosition.Left: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'name': "left"
            },
            carla.SensorPosition.Right: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'name': "right"
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'name': "back left"
            },
            carla.SensorPosition.BackRight: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'name': "back right"
            },
            carla.SensorPosition.Back: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'name': "back"
            },
        }
        return sensors

    def run_step(self, input_data):

        """ The run_step method executes in every simulation time-step. Your control logic should go here. """

        """ In the first frame of the simulation we want to raise the robot's excavating arms to remove them from the 
        field of view of the cameras. Remember that we are working in radians. """

        # Check if we are charging and print if we are
        if self.prev_power is not None and self.get_current_power() > self.prev_power:
            print(f'CHARGING')
        self.prev_power = self.get_current_power()

        if self.frame == 0:
            # Raise the arms
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

            # Open the flap
            self.set_radiator_cover_state(carla.RadiatorCoverState.Open)

        """ Let's retrieve the front left camera data from the input_data dictionary using the correct dictionary key. We want the 
        grayscale monochromatic camera data, so we use the 'Grayscale' key. """

        for position in self.sensors():
            sensor_data = input_data['Grayscale'][position]
            if sensor_data is not None:

                cv.imshow(str(self.sensors()[position]["name"]) + ' camera view', sensor_data)

                if self.take_photo:
                    # Create output directory if it doesn't exist
                    output_dir = 'data/battery'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # Create a filename with sensor name and timestamp
                    # import datetime
                    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    sensor_name = str(self.sensors()[position]["name"]).replace(" ", "_")
                    filename = f"{output_dir}/{sensor_name}_{self.get_mission_time()}.png"
                    
                    # Save the image
                    save_result = cv.imwrite(filename, sensor_data)
                    
                    if save_result:
                        print(f"Photo saved successfully as: {filename}")
                    else:
                        print(f"Failed to save photo as: {filename}")

                cv.waitKey(1)

                #cv.imwrite('out/' + str(self.frame) + '.png', self.sensor_data)

                self.frame += 1

                


        if self.take_photo:
            # Save the exact current location
            self.transforms.append((self.get_mission_time(), ) + pytransform_to_tuple(carla_to_pytransform(self.get_transform())))

        # Ensure we only take on snap shot
        self.take_photo = False

        control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        
        return control

    def finalize(self):

        """ In the finalize method, we should clear up anything we've previously initialized that might be taking up memory or resources. 
        In this case, we should close the OpenCV window. """

        cv.destroyAllWindows()

        """ We may also want to add any final updates we have from our mapping data before the mission ends. Let's add some random values 
        to the geometric map to demonstrate how to use the geometric map API. The geometric map should also be updated during the mission
        in the run_step() method, in case the mission is terminated unexpectedly. """

        """ Retrieve a reference to the geometric map object. """

        geometric_map = self.get_geometric_map()

        """ Set some random height values and rock flags. """

        # Create output directory if it doesn't exist
        output_dir = 'data/battery'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open('data/battery/transforms.csv', mode='w') as data_output:

            import csv

            data_writer = csv.writer(data_output)
 

            data_writer.writerow(['Time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
 

            for value in self.transforms:
 
                data_writer.writerow([value])

        for i in range(100):

            x = 10 * random.random() - 5
            y = 10 * random.random() - 5
            geometric_map.set_height(x, y, random.random())

            rock_flag = random.random() > 0.5
            geometric_map.set_rock(x, y, rock_flag)


    def on_press(self, key):

        """ This is the callback executed when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear velocity. If the key pressed is either the left or right arrow, this method will set a target angular 
        velocity of 0.6 radians per second. """

        print(f'the key is {key} of type {type(key)}')

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
        if key == keyboard.Key.space:
            self.take_photo = True
        if key == keyboard.Key.f1:
            print("exiting sim")
            self.mission_complete()

             

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
        if key == keyboard.Key.space:
            self.take_photo = False

        """ Press escape to end the mission. """
        if key == keyboard.Key.esc:
            self.mission_complete()
            cv.destroyAllWindows()

