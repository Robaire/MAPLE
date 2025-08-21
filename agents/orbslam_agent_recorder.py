#!/usr/bin/env python

# ORB-SLAM Agent with Data Recording Capabilities
# This agent combines ORB-SLAM localization with comprehensive data recording
# Designed to run in a straight line while collecting data every other frame
# Automatically stops recording when dataset reaches 5GB

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This agent demonstrates ORB-SLAM integration with comprehensive data recording.
It runs in a straight line while collecting sensor data, camera feeds, and pose estimates
every other frame until the dataset reaches 5GB, then automatically stops recording.
"""

from math import radians
import numpy as np
import pytransform3d.rotations as pyrot
from collections import defaultdict
import os
import time
import csv

import carla
from pytransform3d.transformations import concat
import orbslam3
from pynput import keyboard
import cv2 as cv

from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.pose import InertialApriltagEstimator
from maple.utils import *
from maple.pose.stereoslam import SimpleStereoSLAM
from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
from maple.utils import carla_to_pytransform
from maple.pose.orbslam_utils import *

# Import the data recording functionality
from lac_data import Recorder

""" Import the AutonomousAgent from the Leaderboard. """

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

""" Define the entry point so that the Leaderboard can instantiate the agent class. """

def get_entry_point():
    return "ORBSLAMRecorderAgent"

""" Inherit the AutonomousAgent class. """

class ORBSLAMRecorderAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Initialize the agent with ORB-SLAM and data recording capabilities."""
        
        # Initialize keyboard listener for manual control
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        # Basic agent attributes
        self.current_v = 0
        self.current_w = 0
        self.frame = 1
        
        # Camera configuration
        self._width = 1280
        self._height = 720
        self._active_side_cameras = False
        self._active_side_front_cameras = True

        # Data collection parameters
        self.recording_active = True
        self.recording_frequency = 2  # Record every other frame
        self.max_dataset_size_gb = 5  # Stop recording at 5GB
        
        # Initialize data recording
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.recorder = Recorder(self, f"orbslam_straight_line_{timestamp}.lac", self.max_dataset_size_gb)
        self.recorder.description(f"ORB-SLAM straight line data collection - {timestamp}")
        
        print(f"🎯 Data recording initialized: {self.recorder.archive_path}")
        print(f"📊 Max dataset size: {self.max_dataset_size_gb}GB")
        print(f"🔄 Recording frequency: Every {self.recording_frequency} frames")

        # ORB-SLAM initialization
        self.orb_vocab = (
            "/home/annikat/ORB-SLAM3-python/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"
        )
        self.orb_cams_config = "/home/annikat/ORB-SLAM3-python/third_party/ORB_SLAM3/Examples/Stereo/LAC_cam.yaml"
        
        # Check if ORB-SLAM files exist, if not use default paths
        if not os.path.exists(self.orb_vocab):
            self.orb_vocab = "resources/ORBvoc.txt"
        if not os.path.exists(self.orb_cams_config):
            self.orb_cams_config = "resources/orbslam_config.yaml"
            
        try:
            self.orbslam = SimpleStereoSLAM(self.orb_vocab, self.orb_cams_config)
            print("✅ ORB-SLAM initialized successfully")
        except Exception as e:
            print(f"⚠️ ORB-SLAM initialization failed: {e}")
            print("🔄 Continuing without ORB-SLAM...")
            self.orbslam = None

        # Pose tracking
        self.init_pose = carla_to_pytransform(self.get_initial_position())
        self.prev_pose = None
        self.T_orb_to_global = None
        self.positions = []

        # Navigation system (simplified for straight line)
        self.navigator = Navigator(self)
        
        # Boulder detection (for obstacle avoidance)
        self.detector = BoulderDetector(
            self, carla.SensorPosition.FrontLeft, carla.SensorPosition.FrontRight
        )
        self.detectorBack = BoulderDetector(
            self, carla.SensorPosition.BackLeft, carla.SensorPosition.BackRight
        )

        # Surface mapping
        self.g_map_testing = self.get_geometric_map()
        self.map_length_testing = self.g_map_testing.get_cell_number()
        self.sample_list = []
        self.ground_truth_sample_list = []

        # Initialize map
        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_height(i, j, 0)
                self.g_map_testing.set_cell_rock(i, j, 0)

        # Boulder tracking
        self.all_boulder_detections = []
        self.large_boulder_detections = [(0, 0, 2.5)]
        self.sample_list.extend(sample_lander(self))

        # Straight line navigation parameters
        self.straight_line_velocity = 0.3  # m/s - conservative speed for data collection
        self.straight_line_angular_velocity = 0.0  # No turning
        self.mission_duration = 300  # seconds (5 minutes)
        self.start_time = None

        # Stuck detection
        self.position_history = []
        self.is_stuck = False
        self.unstuck_phase = 0
        self.unstuck_counter = 0
        
        self.SEVERE_STUCK_FRAMES = 700
        self.SEVERE_STUCK_THRESHOLD = 0.4
        self.MILD_STUCK_FRAMES = 2000
        self.MILD_STUCK_THRESHOLD = 3.0
        self.UNSTUCK_DISTANCE_THRESHOLD = 3.0
        
        self.unstuck_sequence = [
            {"lin_vel": -0.45, "ang_vel": 0, "frames": 100},
            {"lin_vel": 0, "ang_vel": 4, "frames": 60},
            {"lin_vel": 0.45, "ang_vel": 0, "frames": 150},
            {"lin_vel": 0, "ang_vel": -4, "frames": 60}
        ]

        print("🚀 ORB-SLAM Recorder Agent initialized successfully!")
        print(f"🎯 Mission duration: {self.mission_duration} seconds")
        print(f"🔄 Straight line velocity: {self.straight_line_velocity} m/s")

    def check_if_stuck(self, current_position):
        """Check if the rover is stuck using a tiered approach."""
        if current_position is None:
            return False
            
        # Add current position to history
        self.position_history.append(current_position)
        
        # Keep only enough positions for the longer threshold check
        if len(self.position_history) > self.MILD_STUCK_FRAMES:
            self.position_history.pop(0)
        
        # Only perform stuck detection every 10 frames to improve performance
        if self.frame % 10 != 0:
            return False
        
        # Check for severe stuck condition (shorter timeframe)
        if len(self.position_history) >= self.SEVERE_STUCK_FRAMES:
            severe_check_position = self.position_history[-self.SEVERE_STUCK_FRAMES]
            dx = current_position[0] - severe_check_position[0]
            dy = current_position[1] - severe_check_position[1]
            severe_distance_moved = np.sqrt(dx**2 + dy**2)
            
            if severe_distance_moved < self.SEVERE_STUCK_THRESHOLD:
                print(f"🚧 SEVERE STUCK DETECTED! Moved only {severe_distance_moved:.2f}m in the last {self.SEVERE_STUCK_FRAMES} frames.")
                return True
        
        # Check for mild stuck condition (longer timeframe)
        if len(self.position_history) >= self.MILD_STUCK_FRAMES:
            mild_check_position = self.position_history[0]
            dx = current_position[0] - mild_check_position[0]
            dy = current_position[1] - mild_check_position[1]
            mild_distance_moved = np.sqrt(dx**2 + dy**2)
            
            if mild_distance_moved < self.MILD_STUCK_THRESHOLD:
                print(f"⚠️ MILD STUCK DETECTED! Moved only {mild_distance_moved:.2f}m in the last {self.MILD_STUCK_FRAMES} frames.")
                return True
        
        return False

    def get_unstuck_control(self):
        """Execute unstuck sequence."""
        current_phase = self.unstuck_sequence[self.unstuck_phase]
        lin_vel = current_phase["lin_vel"]
        ang_vel = current_phase["ang_vel"]
        self.unstuck_counter += 1
        
        if self.unstuck_counter >= current_phase["frames"]:
            self.unstuck_phase = (self.unstuck_phase + 1) % len(self.unstuck_sequence)
            self.unstuck_counter = 0
            print(f"🔄 Moving to unstuck phase {self.unstuck_phase}")
        
        return lin_vel, ang_vel

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return False

    def sensors(self):
        """Define which sensors are active for data collection."""
        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": False,
                "light_intensity": 0.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.Left: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.Right: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.BackRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.Back: {
                "camera_active": False,
                "light_intensity": 0.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
        }
        return sensors

    def run_step(self, input_data):
        """Main execution loop with error handling."""
        try:
            return self.run_step_unsafe(input_data)
        except Exception as e:
            print(f"❌ FATAL ERROR: {e}")
            self.finalize()
            self.mission_complete()

    def run_step_unsafe(self, input_data):
        """Execute one step of the straight line navigation with data recording."""
        
        # Initialize start time on first frame
        if self.frame == 1:
            self.start_time = time.time()
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))
            print("🚀 Starting straight line data collection mission...")

        # Check mission duration
        if self.start_time and (time.time() - self.start_time) > self.mission_duration:
            print(f"⏰ Mission duration reached ({self.mission_duration}s) - completing mission")
            self.finalize()
            self.mission_complete()
            return carla.VehicleVelocityControl(0, 0)

        # Check if recording should stop due to file size
        if self.recording_active and self.recorder.is_done():
            print(f"💾 Dataset size limit reached ({self.max_dataset_size_gb}GB) - stopping recording")
            self.recording_active = False
            self.recorder.stop()

        # Data recording (every other frame)
        if self.recording_active and self.frame % self.recording_frequency == 0:
            try:
                self.recorder(self.frame, input_data)
                if self.frame % 100 == 0:  # Log every 100 frames
                    print(f"📊 Recording frame {self.frame} - Archive: {self.recorder.archive_path}")
            except Exception as e:
                print(f"⚠️ Recording error: {e}")

        # ORB-SLAM processing (every frame for localization)
        estimate = self.init_pose  # Default to initial pose
        
        if self.orbslam and self.frame >= 50:
            sensor_data_frontleft = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
            sensor_data_frontright = input_data["Grayscale"][carla.SensorPosition.FrontRight]
            
            if sensor_data_frontleft is not None and sensor_data_frontright is not None:
                try:
                    self.orbslam.process_frame(
                        sensor_data_frontleft, sensor_data_frontright, self.frame * 0.1
                    )
                    estimate_orbslamframe = self.orbslam.get_current_pose()
                    
                    if estimate_orbslamframe is not None:
                        orbslam_rotated = correct_pose_orientation(estimate_orbslamframe)
                        
                        if self.frame < 60:
                            self.T_orb_to_global = self.init_pose @ np.linalg.inv(orbslam_rotated)
                            estimate = self.init_pose
                        else:
                            estimate = self.T_orb_to_global @ estimate_orbslamframe
                        
                        estimate = rotate_pose_in_place(estimate, 90, 270, 0)
                        
                        if self.frame % 100 == 0:  # Log every 100 frames
                            print(f"📍 ORB-SLAM pose updated - Frame {self.frame}")
                except Exception as e:
                    print(f"⚠️ ORB-SLAM error: {e}")
                    estimate = self.prev_pose if self.prev_pose is not None else self.init_pose
            else:
                estimate = self.prev_pose if self.prev_pose is not None else self.init_pose
        
        self.prev_pose = estimate

        # Arm control based on terrain
        roll, pitch, yaw = pyrot.euler_from_matrix(estimate[:3, :3], i=0, j=1, k=2, extrinsic=True)
        if np.abs(pitch) > np.deg2rad(80) or np.abs(roll) > np.deg2rad(80):
            self.set_front_arm_angle(radians(0))
            self.set_back_arm_angle(radians(0))
        else:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        # Position tracking and stuck detection
        current_position = (estimate[0, 3], estimate[1, 3]) if estimate is not None else None
        
        if current_position is not None:
            self.position_history.append(current_position)
            if len(self.position_history) > self.MILD_STUCK_FRAMES:
                self.position_history.pop(0)
            
            # Check if stuck every 10 frames
            if not self.is_stuck and self.frame % 10 == 0:
                self.is_stuck = self.check_if_stuck(current_position)
            elif self.is_stuck:
                # Check if we've moved enough to consider ourselves unstuck
                if len(self.position_history) > 0:
                    old_position = self.position_history[0]
                    dx = current_position[0] - old_position[0]
                    dy = current_position[1] - old_position[1]
                    distance_moved = np.sqrt(dx**2 + dy**2)
                    
                    if distance_moved > self.UNSTUCK_DISTANCE_THRESHOLD:
                        print(f"✅ UNSTUCK! Moved {distance_moved:.2f}m - resuming normal operation.")
                        self.is_stuck = False
                        self.unstuck_phase = 0
                        self.unstuck_counter = 0
                        self.position_history = []

        # Boulder detection and obstacle avoidance (every 20 frames)
        if self.frame % 20 == 0:
            try:
                detections, _ = self.detector(input_data)
                detections_back, _ = self.detectorBack(input_data)
                
                if estimate is not None:
                    rover_world = estimate
                    boulders_world = [
                        concat(boulder_rover, rover_world) for boulder_rover in detections
                    ]
                    boulders_world_back = [
                        concat(boulder_rover, rover_world) for boulder_rover in detections_back
                    ]
                    
                    # Add large boulder detections to navigation
                    large_boulders_detections = self.detector.get_large_boulders()
                    large_boulders_xyr = [
                        (b_w[0, 3], b_w[1, 3], 0.25)
                        for b_w in large_boulders_detections
                    ]
                    
                    self.navigator.add_large_boulder_detection(large_boulders_xyr)
                    self.large_boulder_detections.extend(large_boulders_xyr)
                    
                    # Store boulder positions
                    boulders_xy = [(b_w[0, 3], b_w[1, 3]) for b_w in boulders_world]
                    boulders_xy_back = [(b_w[0, 3], b_w[1, 3]) for b_w in boulders_world_back]
                    self.all_boulder_detections.extend(boulders_xy)
                    self.all_boulder_detections.extend(boulders_xy_back)
                    
                    if len(self.all_boulder_detections) > 0 and self.frame % 100 == 0:
                        print(f"🪨 Boulder detections: {len(self.all_boulder_detections)} total")
                        
            except Exception as e:
                print(f"⚠️ Boulder detection error: {e}")

        # Navigation control
        if self.is_stuck:
            # Execute unstuck sequence
            goal_lin_vel, goal_ang_vel = self.get_unstuck_control()
            print(f"🔄 Unstuck maneuver: lin_vel={goal_lin_vel}, ang_vel={goal_ang_vel}")
        else:
            # Straight line navigation
            goal_lin_vel = self.straight_line_velocity
            goal_ang_vel = self.straight_line_angular_velocity

        # Surface sampling when stopped
        if goal_lin_vel == 0 and self.frame % 20 == 0:
            self.sample_list.extend(sample_surface(estimate, 60))

        # Progress logging
        if self.frame % 100 == 0:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            print(f"📊 Frame {self.frame} - Elapsed: {elapsed_time:.1f}s - Recording: {'✅' if self.recording_active else '⏹️'}")

        self.frame += 1
        
        # Return control command
        return carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)

    def finalize(self):
        """Clean up and finalize data collection."""
        print("🏁 Finalizing ORB-SLAM Recorder Agent...")
        
        # Stop recording if still active
        if self.recording_active:
            print("💾 Stopping data recording...")
            self.recorder.stop()
            self.recording_active = False
        
        # Finalize surface mapping
        try:
            g_map = self.get_geometric_map()
            surfaceHeight = SurfaceHeight(g_map)
            
            if len(self.sample_list) > 0:
                surfaceHeight.set_map(self.sample_list)
                print(f"🗺️ Surface map finalized with {len(self.sample_list)} samples")
            
            # Process boulder detections
            if len(self.all_boulder_detections) > 0:
                print(f"🪨 Final boulder detections: {len(self.all_boulder_detections)}")
                
        except Exception as e:
            print(f"⚠️ Error during finalization: {e}")
        
        print("✅ ORB-SLAM Recorder Agent finalized successfully!")

    def on_press(self, key):
        """Keyboard control for manual override."""
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
        """Reset velocities when keys are released."""
        if key == keyboard.Key.up:
            self.current_v = 0
        if key == keyboard.Key.down:
            self.current_v = 0
        if key == keyboard.Key.left:
            self.current_w = 0
        if key == keyboard.Key.right:
            self.current_w = 0
        
        # Press escape to end the mission
        if key == keyboard.Key.esc:
            print("🛑 Manual mission termination requested")
            self.finalize()
            self.mission_complete()
            cv.destroyAllWindows() 