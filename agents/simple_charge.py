#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import time
import json
import math
from numpy import random
import numpy as np
from PIL import Image
import pytransform3d.rotations as pyt_r
from pytransform3d.transformations import concat, invert_transform, transform_from
from datetime import datetime
from pathlib import Path
import os
from maple.pose.orbslam_utils import *
from maple.pose.stereoslam import SimpleStereoSLAM

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.pose import InertialApriltagEstimator, PoseGraph
from maple import utils
from maple.utils import *
from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
from maple.surface.post_processing import PostProcessor
from maple.navigation.simple_charge_nav import ChargingNavigator


def get_entry_point():
    return "DummyAgent"


class DummyAgent(AutonomousAgent):
    """
    Dummy agent to showcase the different functionalities of the agent
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.init_pose = carla_to_pytransform(self.get_initial_position())

        self._active_side_cameras = True
        self.charging_flag = True
        # self.estimator = InertialApriltagEstimator(self)
        self.charging_routine = ChargingNavigator(self)
        self.initial_step = True

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        DATASET_NAME = f"BATTERY_SIMPLE_CHARGING_{timestamp}"

        self.dataset_path = Path.cwd() / "data" / DATASET_NAME
        if self.dataset_path.exists():
            raise RuntimeError(f"{DATASET_NAME} already exists.")

        self.orb_vocab = "/home/annikat/MAPLE/resources/ORBvoc.txt"
        self.orb_cams_config = "/home/annikat/MAPLE/resources/orbslam_config.yaml"
        self.orbslam = SimpleStereoSLAM(self.orb_vocab, self.orb_cams_config)

        self.frame = 0
        


    def use_fiducials(self):
        return False

    def sensors(self):
        """
        Define which sensors are going to be active at the start.
        The specifications and positions of the sensors are predefined and cannot be changed.
        """
        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": True,
                "light_intensity": 1,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": True,
                "light_intensity": 1,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
                "light_intensity": 1,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Left: {
                "camera_active": True,
                "light_intensity": 1,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Right: {
                "camera_active": True,
                "light_intensity": 1,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": True,
                "light_intensity": 1,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.BackRight: {
                "camera_active": True,
                "light_intensity": 1,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Back: {
                "camera_active": True,
                "light_intensity": 1,
                "width": "2448",
                "height": "2048",
            },
        }
        return sensors

    def run_step(self, input_data):
        """Execute one step of navigation"""
        self.frame += 1
        end_time = 180

        if self.frame == 1:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))
    
        sensor_data_frontleft = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        sensor_data_frontright = input_data["Grayscale"][carla.SensorPosition.FrontRight]


        if self.frame < 50:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))
            estimate = self.init_pose
            self.prev_pose = estimate

        elif (
            sensor_data_frontleft is not None
            and sensor_data_frontright is not None
            and self.frame >= 50
        ):
            print("trying to process frame")
            self.orbslam.process_frame(
                sensor_data_frontleft, sensor_data_frontright, self.frame * 0.1
            )
            print("processed frame")
            estimate_orbslamframe = self.orbslam.get_current_pose()
            # trajectory_orbslam = self.orbslam.get_trajectory()
            # print("estimate in orbslam frame: ", estimate_orbslamframe)

            orbslam_rotated = correct_pose_orientation(estimate_orbslamframe)

            # orbslam_rotated = rotate_pose_in_place(orbslam_rotated, 90, 0, 0)

            # orbslam_rotated = update_rpy(orbslam_rotated, 90, 0, 0)

            # estimate = transform_to_global_frame(orbslam_rotated, self.init_pose)
            estimate = orbslam_rotated

            if self.frame < 60:
                self.T_orb_to_global = self.init_pose @ np.linalg.inv(
                    orbslam_rotated
                )  # if you have rover->cam
                estimate = self.init_pose

            else:
                estimate = self.T_orb_to_global @ estimate_orbslamframe

            estimate = rotate_pose_in_place(estimate, 90, 270, 0)

            print(f'the estimate it {estimate} whilt the ground truth is {self.init_pose}')

            # T_global_map = T_global_cam0 @ np.linalg.inv(orbslam_pose_0)

            # real_position = carla_to_pytransform(self.get_transform())

            # estimate_2 = np.linalg.inv(estimate) @ real_position

            # print("transform from estimate to true: ", estimate_true_transform)

            # plot_poses_and_save(
            #     trajectory_orbslam,
            #     estimate_orbslamframe,
            #     estimate,
            #     real_position,
            #     self.frame,
            # )

            # print("true pose: ", real_position)
            # print("orbslam returned transformed pose: ", estimate)

        elif (
            sensor_data_frontleft is None
            and sensor_data_frontright is None
            and self.frame >= 50
        ):
            estimate = self.prev_pose


        x, y, z, roll ,pitch, yaw = pytransform_to_tuple(estimate)


        mission_time = round(self.get_mission_time(), 2)
        if self.initial_step == True:
            print("Charging Antenna pose:", self.charging_routine.antenna_pose)
            print("Initial rover pose:", self.charging_routine.rover_initial_position)
            self.initial_step = False

        if mission_time <= 3:
            self.charging_routine.battery_level = self.get_current_power()
            # Allow the vehicle to settle, and ensure lights are on
            self.set_light_state(carla.SensorPosition.Front, 1.0)
            self.set_light_state(carla.SensorPosition.Back, 1.0)
            self.set_light_state(carla.SensorPosition.Left, 1.0)
            self.set_light_state(carla.SensorPosition.Right, 1.0)
            control = carla.VehicleVelocityControl(0, 0)
            self.set_front_arm_angle(np.deg2rad(60))
            self.set_back_arm_angle(np.deg2rad(60))

        elif mission_time <= end_time and self.charging_flag:
            control, self.charging_flag = self.charging_routine.navigate(estimate)
            print("Control:", control)
            control = carla.VehicleVelocityControl(control[0], control[1])
            if self.charging_flag == False:
                print("Charging complete!")

        elif mission_time > end_time or self.charging_flag == False:
            self.mission_complete()

        # NOTE: We have to make sure we have at least one image to ensure the rest of the images are good to get
        if input_data['Grayscale'][carla.SensorPosition.FrontRight] is not None:

            # Before saving the images, ensure the directory exists
            if not os.path.exists(self.dataset_path):
                os.makedirs(self.dataset_path)
                print(f"Created directory: {self.dataset_path}")

            front_img = input_data['Grayscale'][carla.SensorPosition.Front]
            left_img = input_data['Grayscale'][carla.SensorPosition.Left]
            right_img = input_data['Grayscale'][carla.SensorPosition.Right]
            backleft_img = input_data['Grayscale'][carla.SensorPosition.BackLeft]
            backright_img = input_data['Grayscale'][carla.SensorPosition.BackRight]
            back_img = input_data['Grayscale'][carla.SensorPosition.Back]
            frontleft_img = input_data['Grayscale'][carla.SensorPosition.FrontLeft]
            frontright_img = input_data['Grayscale'][carla.SensorPosition.FrontRight]

            Image.fromarray(front_img, mode="L").save(self.dataset_path / f"front-missiontime_{mission_time}.png")
            Image.fromarray(left_img, mode="L").save(self.dataset_path / f"left-missiontime_{mission_time}.png")
            Image.fromarray(right_img, mode="L").save(self.dataset_path / f"right-missiontime_{mission_time}.png")
            Image.fromarray(backleft_img, mode="L").save(self.dataset_path / f"back-left-missiontime_{mission_time}.png")
            Image.fromarray(backright_img, mode="L").save(self.dataset_path / f"back-right-missiontime_{mission_time}.png")
            Image.fromarray(back_img, mode="L").save(self.dataset_path / f"back-missiontime_{mission_time}.png")
            Image.fromarray(frontleft_img, mode="L").save(self.dataset_path / f"front-left-missiontime_{mission_time}.png")
            Image.fromarray(frontright_img, mode="L").save(self.dataset_path / f"front-right-missiontime_{mission_time}.png")

            transform = self.get_transform()

            rover_global = carla_to_pytransform(self.get_initial_position())
            lander_rover = carla_to_pytransform(self.get_initial_lander_position())
            self.lander_global = concat(lander_rover, rover_global)

            camera_correction = transform_from(
                matrix_from_euler([-np.pi / 2, 0, -np.pi / 2], 2, 1, 0, False), [0, 0, 0]
            )

            # Save each one relative to the camera


            for cam_name in ["Front", "Left", "Right", "BackLeft", "BackRight", "Back", "FrontLeft", "FrontRight"]:
                sensor_position = getattr(carla.SensorPosition, cam_name)
                transforms_path = self.dataset_path / f'transforms-{cam_name.lower()}-missiontime_{mission_time}.npy'

                # Ensure the dataset directory exists before saving
                self.dataset_path.mkdir(parents=True, exist_ok=True)

                try:
                    transforms = np.load(transforms_path)
                except (OSError):
                    transforms = np.empty((0, 7))

                # Convert to camera-transform
                camera_rover = carla_to_pytransform(self.get_camera_position(sensor_position))

                # Get rover in global frame
                rover_global = carla_to_pytransform(self.get_transform())

                # Transform camera from rover frame to the frame relative to the lander

                lander_global_inverse = invert_transform(self.lander_global)
                rover_lander = concat(rover_global, lander_global_inverse)

                camera_lander = concat(camera_rover, rover_lander)
                lander_camera = invert_transform(camera_lander)

                x, y, z, roll, pitch, yaw = pytransform_to_tuple(lander_camera)

                translation = np.array([x, y, z])
                euler = [yaw, pitch, roll]
                quat = pyt_r.quaternion_from_euler(euler, 2, 1, 0, False)  # w, x, y, z

                pose = np.concatenate([translation, quat])[np.newaxis, :]
                transforms = np.vstack([transforms, pose])
                print(f'saving the pose of translation as {translation} and the quat as {quat}')
                
                for pose in transforms:
                    translation = pose[:3]
                    quat = pose[3:]
                    print(f'after the {translation} and the quat as {quat}')

                print(f'the tranforms are {transforms}')
                np.save(transforms_path, transforms)

                # Now load the data back
                loaded_transforms = np.load(transforms_path)

                print(f'the loaded is {loaded_transforms}')

        return control

    def finalize(self):
        g_map = self.get_geometric_map()
        map_length = g_map.get_cell_number()
        for i in range(map_length):
            for j in range(map_length): 
                g_map.set_cell_height(i, j, random.normal(0, 0.5))
                g_map.set_cell_rock(i, j, bool(random.randint(2)))
