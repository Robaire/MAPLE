import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import pandas as pd

# class IMUPositionTracker:
#     def __init__(self, initial_position=None,initial_orientation_rpy=None, sampling_rate=100):
#         """
#         Initialize the IMU position tracker.
        
#         Args:
#             initial_position: Initial position as [x, y, z] in meters. Default is origin.
#             initial_orientation_rpy: Initial orientation as [roll, pitch, yaw] in radians.
#                                      Default is [0, 0, 0].
#             sampling_rate: Sampling rate of the IMU data in Hz. Default is 100Hz.
#         """
#         # self.position = np.array(initial_position) if initial_position is not None else np.zeros(3)
#         self.position = np.array(initial_position, dtype=float) if initial_position is not None else np.zeros(3, dtype=float)
#         self.velocity = np.zeros(3)  # Initial velocity is zero
        
#         # Initial orientation as a rotation object from roll, pitch, yaw
#         if initial_orientation_rpy is not None:
#             roll, pitch, yaw = initial_orientation_rpy
#             self.orientation = Rotation.from_euler('xyz', [roll, pitch, yaw])
#         else:
#             self.orientation = Rotation.identity()
        
#         self.dt = 1.0 / sampling_rate  # Time step based on sampling rate
#         self.gravity = np.array([0, 0, -1.62])  # Gravity vector in m/s²
        
#         # Store trajectory history
#         self.position_history = [self.position.copy()]
#         self.time = 0
#         self.time_history = [0]
        
#     def update(self, imu_data):
#         """
#         Update position and orientation based on new IMU data.
        
#         Args:
#             imu_data: List of [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
#                       accel in m/s², gyro in rad/s
#         """
#         self.time += self.dt
        
#         # Extract IMU data
#         accel = np.array(imu_data[:3])
#         gyro = np.array(imu_data[3:])
        
#         # Update orientation using gyroscope data
#         # Convert angular velocity to rotation increment
#         angle_increment = gyro * self.dt
#         rotation_increment = Rotation.from_rotvec(angle_increment)
#         self.orientation = rotation_increment * self.orientation
        
#         # Remove gravity from acceleration by transforming gravity to body frame
#         # and subtracting from measured acceleration
#         gravity_body = self.orientation.inv().apply(self.gravity)
#         accel_without_gravity = accel - gravity_body
        
#         # Transform acceleration from body frame to world frame
#         accel_world = self.orientation.apply(accel_without_gravity)
        
#         # Integrate acceleration to get velocity
#         self.velocity += accel_world * self.dt
        
#         # Apply simple low-pass filter to velocity to reduce drift
#         # This is a basic approach - more sophisticated filtering would be better
#         self.velocity *= 0.99  # Damping factor
        
#         # Integrate velocity to get position
#         self.position += self.velocity * self.dt
        
#         # Store the new position
#         self.position_history.append(self.position.copy())
#         self.time_history.append(self.time)

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import block_diag



# class IMUPositionTracker:
#     def __init__(self, initial_position=[-2.263231, -4.454633, 1.404660], initial_orientation_rpy=None, sampling_rate=100):
#         """
#         Initialize the IMU position tracker with Kalman filtering.
        
#         Args:
#             initial_position: Initial position as [x, y, z] in meters.
#             initial_orientation_rpy: Initial orientation as [roll, pitch, yaw] in radians.
#             sampling_rate: Sampling rate of the IMU data in Hz.
#         """
#         # Initialize position and velocity
#         self.position = np.array(initial_position, dtype=float) if initial_position is not None else np.zeros(3, dtype=float)
#         self.velocity = np.zeros(3, dtype=float)
        
#         # Initialize orientation
#         if initial_orientation_rpy is not None:
#             roll, pitch, yaw = initial_orientation_rpy
#             self.orientation = Rotation.from_euler('xyz', [roll, pitch, yaw])
#         else:
#             self.orientation = Rotation.identity()
        
#         self.dt = 1.0 / sampling_rate
#         self.gravity = np.array([0, 0, -1.62])  # Lunar gravity in m/s²
        
#         # Store trajectory history
#         self.position_history = [self.position.copy()]
#         self.time = 0
#         self.time_history = [0]
        
#         # IMU bias estimation
#         self.accel_bias = np.zeros(3)  # Initial accelerometer bias
#         self.gyro_bias = np.zeros(3)   # Initial gyroscope bias
        
#         # Bias estimation parameters
#         self.bias_estimation_samples = 100  # Number of samples for initial bias estimation
#         self.calibration_samples = []
#         self.is_calibrated = False
        
#         # Initialize Kalman filter
#         # State vector: [position (3), velocity (3), orientation (3), accel_bias (3), gyro_bias (3)]
#         self.state = np.zeros(15)
#         self.state[0:3] = self.position
        
#         # State covariance matrix
#         self.P = np.eye(15) * 0.01
        
#         # Process noise settings - tuned for lunar environment
#         self.accel_noise = 0.01  # m/s²
#         self.gyro_noise = 0.001  # rad/s
#         self.accel_bias_noise = 0.0001  # m/s²
#         self.gyro_bias_noise = 0.00001  # rad/s
        
#     def calibrate_sensors(self, imu_data):
#         """
#         Collect data for sensor calibration during stationary periods.
        
#         Args:
#             imu_data: List of [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
#         """
#         if len(self.calibration_samples) < self.bias_estimation_samples:
#             self.calibration_samples.append(np.array(imu_data))
#             return False
#         elif not self.is_calibrated:
#             # Calculate mean bias values
#             samples = np.array(self.calibration_samples)
#             accel_samples = samples[:, :3]
#             gyro_samples = samples[:, 3:6]
            
#             # Accelerometer bias is the mean reading minus gravity (in body frame)
#             gravity_body = self.orientation.inv().apply(self.gravity)
#             self.accel_bias = np.mean(accel_samples, axis=0) - gravity_body
            
#             # Gyroscope bias is just the mean reading during stationary period
#             self.gyro_bias = np.mean(gyro_samples, axis=0)
            
#             self.is_calibrated = True
#             print(f"Calibration complete - Accel bias: {self.accel_bias}, Gyro bias: {self.gyro_bias}")
#             return True
        
#         return True
    
#     def build_process_noise_matrix(self):
#         """Build the process noise covariance matrix Q for the Kalman filter."""
#         # Position noise (derived from velocity uncertainty)
#         pos_noise = 0.5 * self.accel_noise * self.dt**2
        
#         # Velocity noise (derived from acceleration uncertainty)
#         vel_noise = self.accel_noise * self.dt
        
#         # Orientation noise (derived from gyro uncertainty)
#         ori_noise = self.gyro_noise * self.dt
        
#         # Create process noise matrix
#         Q_pos = np.eye(3) * pos_noise**2
#         Q_vel = np.eye(3) * vel_noise**2
#         Q_ori = np.eye(3) * ori_noise**2
#         Q_ab = np.eye(3) * self.accel_bias_noise**2
#         Q_gb = np.eye(3) * self.gyro_bias_noise**2
        
#         # Combine into block diagonal matrix
#         Q = block_diag(Q_pos, Q_vel, Q_ori, Q_ab, Q_gb)
#         return Q
    
#     def build_state_transition_matrix(self, accel, gyro):
#         """
#         Build the state transition matrix F for the Kalman filter.
        
#         Args:
#             accel: Corrected acceleration in world frame
#             gyro: Corrected gyro rates in body frame
#         """
#         # Basic state transition matrix
#         F = np.eye(15)
        
#         # Position is updated by velocity
#         F[0:3, 3:6] = np.eye(3) * self.dt
        
#         # Velocity is updated by orientation and acceleration
#         skew_accel = self.skew_symmetric(accel)
#         F[3:6, 6:9] = -skew_accel * self.dt
        
#         # Orientation is updated by gyro
#         skew_gyro = self.skew_symmetric(gyro)
#         F[6:9, 6:9] = np.eye(3) - skew_gyro * self.dt
        
#         # Velocity is affected by accelerometer bias
#         F[3:6, 9:12] = -self.orientation.as_matrix() * self.dt
        
#         # Orientation is affected by gyroscope bias
#         F[6:9, 12:15] = -np.eye(3) * self.dt
        
#         return F
    
#     def skew_symmetric(self, v):
#         """
#         Create a skew-symmetric matrix from a 3-element vector.
#         Used for cross product operations within matrices.
#         """
#         return np.array([
#             [0, -v[2], v[1]],
#             [v[2], 0, -v[0]],
#             [-v[1], v[0], 0]
#         ])
    
#     def update(self, imu_data):
#         """
#         Update position and orientation based on new IMU data.
        
#         Args:
#             imu_data: List of [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
#         """
#         self.time += self.dt
        
#         # Extract IMU data
#         accel_raw = np.array(imu_data[:3])
#         gyro_raw = np.array(imu_data[3:])
        
#         # Calibrate sensors if needed
#         if not self.is_calibrated and len(self.position_history) < 10:
#             self.calibrate_sensors(imu_data)
#             # During initial calibration, just store current position
#             self.position_history.append(self.position.copy())
#             self.time_history.append(self.time)
#             return
        
#         # Apply bias correction
#         accel = accel_raw - self.accel_bias
#         gyro = gyro_raw - self.gyro_bias
        
#         # Remove gravity from acceleration
#         gravity_body = self.orientation.inv().apply(self.gravity)
#         accel_without_gravity = accel - gravity_body
        
#         # Transform acceleration to world frame
#         accel_world = self.orientation.apply(accel_without_gravity)
        
#         # Get current state values
#         pos = self.state[0:3]
#         vel = self.state[3:6]
#         ori = self.state[6:9]  # Simplified orientation representation (small angle approximation)
        
#         # Build Kalman filter matrices
#         F = self.build_state_transition_matrix(accel_world, gyro)
#         Q = self.build_process_noise_matrix()
        
#         # Prediction step
#         self.state = F @ self.state
#         self.P = F @ self.P @ F.T + Q
        
#         # Update orientation separately using rotation objects (more accurate than small angle in state)
#         angle_increment = gyro * self.dt
#         rotation_increment = Rotation.from_rotvec(angle_increment)
#         self.orientation = rotation_increment * self.orientation
        
#         # Integrate acceleration to get velocity
#         self.velocity += accel_world * self.dt
        
#         # Update position
#         self.position += self.velocity * self.dt
        
#         # Update state vector with actual position, velocity and small-angle orientation
#         self.state[0:3] = self.position
#         self.state[3:6] = self.velocity
#         self.state[6:9] = self.orientation.as_rotvec()  # Small angle approximation
        
#         # Apply zero-velocity update when detected (simple threshold method)
#         accel_magnitude = np.linalg.norm(accel_without_gravity)
#         gyro_magnitude = np.linalg.norm(gyro)
        
#         if accel_magnitude < 0.1 and gyro_magnitude < 0.05:  # Thresholds need tuning for lunar environment
#             # Apply "soft" zero-velocity update
#             self.velocity *= 0.8
#             self.state[3:6] *= 0.8
            
#             # Update bias estimates during stationary periods
#             self.accel_bias = 0.98 * self.accel_bias + 0.02 * (accel_raw - gravity_body)
#             self.gyro_bias = 0.98 * self.gyro_bias + 0.02 * gyro_raw
        
#         # Store the new position
#         self.position_history.append(self.position.copy())
#         self.time_history.append(self.time)
        
#     def get_current_position(self):
#         """Return the current estimated position."""
#         return self.position.copy()
    
#     def get_current_orientation_rpy(self):
#         """Return the current orientation as roll, pitch, yaw angles in radians."""
#         return self.orientation.as_euler('xyz')
    
#     def get_trajectory(self):
#         """Return the full trajectory history."""
#         return np.array(self.position_history)

#     def plot_trajectory(self, save_path, gt_history):
#         """Plot the 3D trajectory."""
#         trajectory = np.array(self.position_history)
        
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
        
#         ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-')
#         ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='g', marker='o', s=100, label='Start')
#         ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='r', marker='o', s=100, label='End')

#         ax.plot(gt_history[:, 0], gt_history[:, 1], gt_history[:, 2], 'g-')
#         ax.scatter(gt_history[0, 0], gt_history[0, 1], gt_history[0, 2], c='g', marker='o', s=100, label='Start')
#         ax.scatter(gt_history[-1, 0], gt_history[-1, 1], gt_history[-1, 2], c='r', marker='o', s=100, label='End')
        
#         ax.set_xlabel('X (m)')
#         ax.set_ylabel('Y (m)')
#         ax.set_zlabel('Z (m)')
#         ax.set_title('IMU-based Position Trajectory')
#         ax.legend()
        
#         plt.tight_layout()
#         # plt.show()
#             # Save the figure instead of showing it
#         if save_path is None:
#             # Extract directory from the filename in the main script
#             import os
#             save_path = os.path.join(os.path.dirname(filename), 'imu_trajectory.png')
        
#         plt.savefig(save_path)
#         plt.close(fig)

# def process_imu_data(imu_data_frames, initial_position, initial_orientation_rpy, sampling_rate=100):
#     """
#     Process a sequence of IMU data frames to estimate position.
    
#     Args:
#         imu_data_frames: List of IMU data frames, each containing [ax, ay, az, gx, gy, gz]
#         initial_position: Initial position as [x, y, z] in meters
#         initial_orientation_rpy: Initial orientation as [roll, pitch, yaw] in radians
#         sampling_rate: IMU data sampling rate in Hz
        
#     Returns:
#         IMUPositionTracker object with the processed trajectory
#     """
#     tracker = IMUPositionTracker(
#         initial_position=initial_position,
#         initial_orientation_rpy= initial_orientation_rpy,
#         sampling_rate=sampling_rate
#     )
    
#     for imu_data in imu_data_frames:
#         tracker.update(imu_data)
    
#     return tracker

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import block_diag

class IMUPositionTracker:
    def __init__(self, initial_position=[-2.263231, -4.454633, 1.404660], initial_orientation_rpy=None, sampling_rate=100):
        """
        Initialize the IMU position tracker with Kalman filtering, optimized for lunar environment.
        
        Args:
            initial_position: Initial position as [x, y, z] in meters.
            initial_orientation_rpy: Initial orientation as [roll, pitch, yaw] in radians.
            sampling_rate: Sampling rate of the IMU data in Hz.
        """
        # Initialize position and velocity
        self.position = np.array(initial_position, dtype=float) if initial_position is not None else np.zeros(3, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        
        # Initialize orientation - note that we're using a specific initial orientation (1,0,0)
        if initial_orientation_rpy is not None:
            roll, pitch, yaw = initial_orientation_rpy
            self.orientation = Rotation.from_euler('xyz', [roll, pitch, yaw])
        else:
            # Default to IMU orientation (1,0,0) if not specified
            self.orientation = Rotation.from_euler('xyz', [np.pi/2, 0, 0])
        
        self.dt = 1.0 / sampling_rate
        self.gravity = np.array([0, 0, -1.62])  # Lunar gravity in m/s²
        
        # Store trajectory history
        self.position_history = [self.position.copy()]
        self.orientation_history = [self.orientation.as_euler('xyz')]
        self.time = 0
        self.time_history = [0]
        
        # IMU bias estimation - lower initial values for lunar environment
        self.accel_bias = np.zeros(3)  # Initial accelerometer bias
        self.gyro_bias = np.zeros(3)   # Initial gyroscope bias
        
        # Bias estimation parameters
        self.bias_estimation_samples = 50  # Reduce calibration period
        self.calibration_samples = []
        self.is_calibrated = False
        
        # Initialize Kalman filter
        # State vector: [position (3), velocity (3), orientation (3), accel_bias (3), gyro_bias (3)]
        self.state = np.zeros(15)
        self.state[0:3] = self.position
        self.state[6:9] = self.orientation.as_rotvec()
        
        # State covariance matrix - higher initial uncertainty for lunar environment
        self.P = np.eye(15) * 0.1
        # Higher position and orientation uncertainty
        self.P[0:3, 0:3] *= 5.0  # Position uncertainty
        self.P[6:9, 6:9] *= 2.0  # Orientation uncertainty
        
        # Process noise settings - tuned for lunar environment
        # Lower values due to smoother motion on the moon
        self.accel_noise = 0.005  # m/s² (reduced for moon)
        self.gyro_noise = 0.0005  # rad/s (reduced for moon)
        self.accel_bias_noise = 0.0001  # m/s²
        self.gyro_bias_noise = 0.00001  # rad/s
        
        # Increased sensitivity for turn detection in lunar environment
        self.zupt_accel_threshold = 0.05  # Much lower threshold for lunar environment
        self.zupt_gyro_threshold = 0.01   # Lower threshold for detecting rotations
        
        # Add turn detection sensitivity parameters
        self.min_turn_rate = 0.005  # Minimum angular velocity to detect as a turn (rad/s)
        self.turn_scaling = 2.0     # Scale factor to enhance detected turns

        # Track turning state
        self.is_turning = False
        self.turn_confidence = 0.0
        
    def calibrate_sensors(self, imu_data):
        """
        Collect data for sensor calibration during stationary periods.
        Adapted for lunar environment with lower thresholds.
        
        Args:
            imu_data: List of [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        """
        if len(self.calibration_samples) < self.bias_estimation_samples:
            self.calibration_samples.append(np.array(imu_data))
            return False
        elif not self.is_calibrated:
            # Calculate mean bias values
            samples = np.array(self.calibration_samples)
            accel_samples = samples[:, :3]
            gyro_samples = samples[:, 3:6]
            
            # Accelerometer bias calculation - account for lunar gravity
            # Transform gravity to expected orientation
            gravity_body = self.orientation.inv().apply(self.gravity)
            
            # In lunar environment, static acceleration should be close to lunar gravity
            self.accel_bias = np.mean(accel_samples, axis=0) - gravity_body
            
            # Gyroscope bias is the mean reading during stationary period
            # Lunar environment doesn't change this calculation
            self.gyro_bias = np.mean(gyro_samples, axis=0)
            
            print(f"Lunar Calibration complete - Accel bias: {self.accel_bias}, Gyro bias: {self.gyro_bias}")
            print(f"Gravity in body frame: {gravity_body}")
            
            self.is_calibrated = True
            return True
        
        return True
    
    def build_process_noise_matrix(self):
        """
        Build the process noise covariance matrix Q for the Kalman filter.
        Adapted for lunar environment with different noise characteristics.
        """
        # Position noise (derived from velocity uncertainty)
        pos_noise = 0.5 * self.accel_noise * self.dt**2
        
        # Velocity noise (derived from acceleration uncertainty)
        vel_noise = self.accel_noise * self.dt
        
        # Orientation noise (derived from gyro uncertainty)
        ori_noise = self.gyro_noise * self.dt
        
        # Increase process noise if turning is detected
        turn_factor = 1.0
        if self.is_turning:
            turn_factor = 3.0  # Increase process noise during turns
            
        # Create process noise matrix
        Q_pos = np.eye(3) * (pos_noise**2) * turn_factor
        Q_vel = np.eye(3) * (vel_noise**2) * turn_factor
        Q_ori = np.eye(3) * (ori_noise**2) * turn_factor
        Q_ab = np.eye(3) * self.accel_bias_noise**2
        Q_gb = np.eye(3) * self.gyro_bias_noise**2
        
        # Combine into block diagonal matrix
        Q = block_diag(Q_pos, Q_vel, Q_ori, Q_ab, Q_gb)
        return Q
    
    def build_state_transition_matrix(self, accel, gyro):
        """
        Build the state transition matrix F for the Kalman filter.
        
        Args:
            accel: Corrected acceleration in world frame
            gyro: Corrected gyro rates in body frame
        """
        # Basic state transition matrix
        F = np.eye(15)
        
        # Position is updated by velocity
        F[0:3, 3:6] = np.eye(3) * self.dt
        
        # Velocity is updated by orientation and acceleration
        skew_accel = self.skew_symmetric(accel)
        F[3:6, 6:9] = -skew_accel * self.dt
        
        # Orientation is updated by gyro
        skew_gyro = self.skew_symmetric(gyro)
        F[6:9, 6:9] = np.eye(3) - skew_gyro * self.dt
        
        # Velocity is affected by accelerometer bias
        F[3:6, 9:12] = -self.orientation.as_matrix() * self.dt
        
        # Orientation is affected by gyroscope bias
        F[6:9, 12:15] = -np.eye(3) * self.dt
        
        return F
    
    def skew_symmetric(self, v):
        """
        Create a skew-symmetric matrix from a 3-element vector.
        Used for cross product operations within matrices.
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def detect_turn(self, gyro):
        """
        Detect turning movements with higher sensitivity for lunar environment.
        
        Args:
            gyro: Gyroscope readings (bias-corrected)
            
        Returns:
            Boolean indicating if turning is detected
        """
        gyro_magnitude = np.linalg.norm(gyro)
        
        # Increase turn confidence if gyro magnitude exceeds threshold
        if gyro_magnitude > self.min_turn_rate:
            self.turn_confidence = min(1.0, self.turn_confidence + 0.2)
        else:
            self.turn_confidence = max(0.0, self.turn_confidence - 0.1)
            
        # Set turning state based on confidence
        self.is_turning = self.turn_confidence > 0.5
        
        return self.is_turning
    
    def update(self, imu_data):
        """
        Update position and orientation based on new IMU data.
        Optimized for lunar environment with enhanced turn detection.
        
        Args:
            imu_data: List of [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        """
        self.time += self.dt
        
        # Extract IMU data
        accel_raw = np.array(imu_data[:3])
        gyro_raw = np.array(imu_data[3:])
        
        # Calibrate sensors if needed
        if not self.is_calibrated and len(self.position_history) < 10:
            self.calibrate_sensors(imu_data)
            # During initial calibration, just store current position
            self.position_history.append(self.position.copy())
            self.orientation_history.append(self.orientation.as_euler('xyz'))
            self.time_history.append(self.time)
            return
        
        # Apply bias correction
        accel = accel_raw - self.accel_bias
        gyro = gyro_raw - self.gyro_bias
        
        # Detect turns with enhanced sensitivity for lunar environment
        is_turning = self.detect_turn(gyro)
        
        # Scale up gyro readings during turns to enhance sensitivity
        if is_turning:
            gyro = gyro * self.turn_scaling
            # print(f"Turn detected at t={self.time:.2f}s, gyro magnitude: {np.linalg.norm(gyro):.5f}")
        
        # Remove gravity from acceleration
        gravity_body = self.orientation.inv().apply(self.gravity)
        accel_without_gravity = accel - gravity_body
        
        # Transform acceleration to world frame
        accel_world = self.orientation.apply(accel_without_gravity)
        
        # Get current state values
        pos = self.state[0:3]
        vel = self.state[3:6]
        ori = self.state[6:9]  # Simplified orientation representation
        
        # Build Kalman filter matrices
        F = self.build_state_transition_matrix(accel_world, gyro)
        Q = self.build_process_noise_matrix()
        
        # Prediction step
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        
        # Update orientation separately using rotation objects (more accurate than small angle in state)
        angle_increment = gyro * self.dt
        rotation_increment = Rotation.from_rotvec(angle_increment)
        self.orientation = rotation_increment * self.orientation
        
        # Integrate acceleration to get velocity
        # Add a small bias to the velocity to counteract integration drift
        self.velocity += accel_world * self.dt
        
        # For lunar environment: very gentle velocity damping except during turns
        if is_turning:
            # During turns, less damping to capture movement
            self.velocity *= 0.98
        else:
            # Normal damping
            self.velocity *= 0.96
        
        # Update position
        self.position += self.velocity * self.dt
        
        # Update state vector with actual position, velocity and small-angle orientation
        self.state[0:3] = self.position
        self.state[3:6] = self.velocity
        self.state[6:9] = self.orientation.as_rotvec()  # Small angle approximation
        
        # Apply zero-velocity update when detected (simple threshold method)
        # Lower thresholds for lunar environment
        accel_magnitude = np.linalg.norm(accel_without_gravity)
        gyro_magnitude = np.linalg.norm(gyro)
        
        if accel_magnitude < self.zupt_accel_threshold and gyro_magnitude < self.zupt_gyro_threshold:
            # Apply "soft" zero-velocity update - gentler for lunar environment
            self.velocity *= 0.9
            self.state[3:6] *= 0.9
            
            # Update bias estimates during stationary periods
            # More aggressive bias update in lunar environment
            self.accel_bias = 0.95 * self.accel_bias + 0.05 * (accel_raw - gravity_body)
            self.gyro_bias = 0.95 * self.gyro_bias + 0.05 * gyro_raw
        
        # Store the new position and orientation
        self.position_history.append(self.position.copy())
        self.orientation_history.append(self.orientation.as_euler('xyz'))
        self.time_history.append(self.time)
        
    def get_current_position(self):
        """Return the current estimated position."""
        return self.position.copy()
    
    def get_current_orientation_rpy(self):
        """Return the current orientation as roll, pitch, yaw angles in radians."""
        return self.orientation.as_euler('xyz')
    
    def get_trajectory(self):
        """Return the full trajectory history."""
        return np.array(self.position_history)
    
    def plot_trajectory(self, save_path, gt_history):
        """
        Plot the 3D trajectory with coordinate frames at each point.
        
        Args:
            save_path: Path to save the plot
            gt_history: Ground truth trajectory data with shape (N, 3)
        """
        max_range = 10
        trajectory = np.array(self.position_history)
        # Extract orientation data
        orientations = np.array(self.orientation_history)  # Assuming this contains [roll, pitch, yaw] for each point
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2, label='IMU Estimated')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='g', marker='o', s=100, label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='r', marker='o', s=100, label='End')

        ax.plot(gt_history[:, 0], gt_history[:, 1], gt_history[:, 2], 'g-', linewidth=2, label='Ground Truth')
        ax.scatter(gt_history[0, 0], gt_history[0, 1], gt_history[0, 2], c='g', marker='o', s=100)
        ax.scatter(gt_history[-1, 0], gt_history[-1, 1], gt_history[-1, 2], c='r', marker='o', s=100)
        
        # Function to compute rotation matrix from roll, pitch, yaw
        def euler_to_rotation_matrix(roll, pitch, yaw):
            """Convert Euler angles to rotation matrix."""
            # Roll (x-axis rotation)
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
            
            # Pitch (y-axis rotation)
            R_y = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
            
            # Yaw (z-axis rotation)
            R_z = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            
            # Combine rotations: R = R_z * R_y * R_x
            R = np.dot(R_z, np.dot(R_y, R_x))
            return R
        
        # Plot coordinate frames at regular intervals
        n_points = len(trajectory)
        interval = max(1, n_points // 20)  # Show axes every N points to avoid overcrowding
        axis_length = max_range / 10.0  # Scale the axes appropriately
        
        for i in range(0, n_points, interval):
            position = trajectory[i]
            if i < len(orientations):
                roll, pitch, yaw = orientations[i]
                
                # Get rotation matrix
                R = euler_to_rotation_matrix(roll, pitch, yaw)
                
                # Define the three axes (x, y, z) in the local frame
                x_axis = R[:, 0] * axis_length
                y_axis = R[:, 1] * axis_length
                z_axis = R[:, 2] * axis_length
                
                # Plot the three axes
                ax.quiver(position[0], position[1], position[2], 
                        x_axis[0], x_axis[1], x_axis[2], 
                        color='r', linewidth=1.5, arrow_length_ratio=0.15)
                ax.quiver(position[0], position[1], position[2], 
                        y_axis[0], y_axis[1], y_axis[2], 
                        color='g', linewidth=1.5, arrow_length_ratio=0.15)
                ax.quiver(position[0], position[1], position[2], 
                        z_axis[0], z_axis[1], z_axis[2], 
                        color='b', linewidth=1.5, arrow_length_ratio=0.15)
        
        # Set equal aspect ratio for better visualization
        max_range = np.array([
            trajectory[:, 0].max() - trajectory[:, 0].min(),
            trajectory[:, 1].max() - trajectory[:, 1].min(),
            trajectory[:, 2].max() - trajectory[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (trajectory[:, 0].max() + trajectory[:, 0].min()) / 2
        mid_y = (trajectory[:, 1].max() + trajectory[:, 1].min()) / 2
        mid_z = (trajectory[:, 2].max() + trajectory[:, 2].min()) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Lunar IMU-based Position Trajectory with Orientation')
        
        # Add a legend entry for the axes
        red_line = Line2D([0], [0], color='r', lw=2)
        green_line = Line2D([0], [0], color='g', lw=2)
        blue_line = Line2D([0], [0], color='b', lw=2)
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([red_line, green_line, blue_line])
        labels.extend(['X axis', 'Y axis', 'Z axis'])
        ax.legend(handles, labels)
        
        plt.tight_layout()
        
        # Save the figure
        if save_path is None:
            import os
            save_path = os.path.join(os.path.dirname(filename), 'imu_trajectory_lunar.png')
        
        plt.savefig(save_path)
        plt.close(fig)
        
        # Plot error and orientation over time
        self.plot_trajectory_error(save_path.replace('.png', '_error.png'), gt_history)
        self.plot_orientation(save_path.replace('.png', '_orientation.png'))

    # def plot_trajectory(self, save_path, gt_history):
    #     """Plot the 3D trajectory."""
    #     trajectory = np.array(self.position_history)
        
    #     fig = plt.figure(figsize=(12, 10))
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2, label='IMU Estimated')
    #     ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='g', marker='o', s=100, label='Start')
    #     ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='r', marker='o', s=100, label='End')

    #     ax.plot(gt_history[:, 0], gt_history[:, 1], gt_history[:, 2], 'g-', linewidth=2, label='Ground Truth')
    #     ax.scatter(gt_history[0, 0], gt_history[0, 1], gt_history[0, 2], c='g', marker='o', s=100)
    #     ax.scatter(gt_history[-1, 0], gt_history[-1, 1], gt_history[-1, 2], c='r', marker='o', s=100)
        
    #     # Set equal aspect ratio for better visualization
    #     max_range = np.array([
    #         trajectory[:, 0].max() - trajectory[:, 0].min(),
    #         trajectory[:, 1].max() - trajectory[:, 1].min(),
    #         trajectory[:, 2].max() - trajectory[:, 2].min()
    #     ]).max() / 2.0
        
    #     mid_x = (trajectory[:, 0].max() + trajectory[:, 0].min()) / 2
    #     mid_y = (trajectory[:, 1].max() + trajectory[:, 1].min()) / 2
    #     mid_z = (trajectory[:, 2].max() + trajectory[:, 2].min()) / 2
        
    #     ax.set_xlim(mid_x - max_range, mid_x + max_range)
    #     ax.set_ylim(mid_y - max_range, mid_y + max_range)
    #     ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
    #     ax.set_xlabel('X (m)')
    #     ax.set_ylabel('Y (m)')
    #     ax.set_zlabel('Z (m)')
    #     ax.set_title('Lunar IMU-based Position Trajectory (Enhanced Turn Detection)')
    #     ax.legend()
        
    #     plt.tight_layout()
        
    #     # Save the figure
    #     if save_path is None:
    #         import os
    #         save_path = os.path.join(os.path.dirname(filename), 'imu_trajectory_lunar.png')
        
    #     plt.savefig(save_path)
    #     plt.close(fig)
        
    #     # Plot error and orientation over time
    #     self.plot_trajectory_error(save_path.replace('.png', '_error.png'), gt_history)
    #     self.plot_orientation(save_path.replace('.png', '_orientation.png'))
    
    def plot_trajectory_error(self, save_path, gt_history):
        """Plot the error between estimated and ground truth trajectory."""
        trajectory = np.array(self.position_history)
        
        # Make sure we're comparing the same number of points
        min_length = min(len(trajectory), len(gt_history))
        trajectory = trajectory[:min_length]
        gt_history = gt_history[:min_length]
        time_history = self.time_history[:min_length]
        
        # Calculate error at each point
        position_error = np.linalg.norm(trajectory - gt_history, axis=1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot overall position error
        ax1.plot(time_history, position_error, 'r-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position Error (m)')
        ax1.set_title('Lunar IMU Tracking Error vs Ground Truth')
        ax1.grid(True)
        
        # Plot component-wise error
        ax2.plot(time_history, trajectory[:, 0] - gt_history[:, 0], 'r-', label='X Error')
        ax2.plot(time_history, trajectory[:, 1] - gt_history[:, 1], 'g-', label='Y Error')
        ax2.plot(time_history, trajectory[:, 2] - gt_history[:, 2], 'b-', label='Z Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Component Error (m)')
        ax2.set_title('Component-wise Position Error')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
    
    def plot_orientation(self, save_path):
        """Plot the orientation over time to analyze turning behavior."""
        orientation_history = np.array(self.orientation_history)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.time_history, orientation_history[:, 0], 'r-', label='Roll')
        ax.plot(self.time_history, orientation_history[:, 1], 'g-', label='Pitch')
        ax.plot(self.time_history, orientation_history[:, 2], 'b-', label='Yaw')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rad)')
        ax.set_title('IMU Orientation Over Time')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

def process_imu_data(imu_data_frames, initial_position, initial_orientation_rpy, sampling_rate=100):
    """
    Process a sequence of IMU data frames to estimate position.
    
    Args:
        imu_data_frames: List of IMU data frames, each containing [ax, ay, az, gx, gy, gz]
        initial_position: Initial position as [x, y, z] in meters
        initial_orientation_rpy: Initial orientation as [roll, pitch, yaw] in radians
        sampling_rate: IMU data sampling rate in Hz
        
    Returns:
        IMUPositionTracker object with the processed trajectory
    """
    # Initialize the tracker with correct initial values
    tracker = IMUPositionTracker(
        initial_position=initial_position,
        initial_orientation_rpy=initial_orientation_rpy,
        sampling_rate=sampling_rate
    )
    
    # Process each IMU data frame
    for i, imu_data in enumerate(imu_data_frames):
        # Print occasional progress updates
        if i % 1000 == 0:
            print(f"Processing IMU frame {i}/{len(imu_data_frames)}")
        tracker.update(imu_data)
    
    return tracker

# Example usage
if __name__ == "__main__":
    # Load IMU data
    filename = "/home/annikat/MAPLE/data/035/imu_data.csv"
    df = pd.read_csv(filename)

    num_gt_cols = 7
    gt_data = df.iloc[:, :num_gt_cols]  # ground-truth columns
    gt_position_history = df.iloc[:, 1:4]
    gt_orientation_history = df.iloc[:, 4:7]
    print("Ground truth position data shape:", gt_orientation_history.shape)
    
    imu_data = df.iloc[:, num_gt_cols:]  # IMU columns
    imu_data_frames = imu_data.values.tolist()
    gt_position_history_np = gt_position_history.to_numpy()
    gt_orientation_history_np = gt_orientation_history.to_numpy()
    
    # Get initial position from ground truth data
    initial_position = gt_position_history_np[0, :].tolist()
    initial_orientation = gt_orientation_history_np[0,:].tolist()

    print(gt_orientation_history)

    initial_orientation_rpy = initial_orientation
    
    # Using the specified IMU orientation (1,0,0)
    # This corresponds to roll = π/2 (90 degrees), pitch = 0, yaw = 0
    # initial_orientation_rpy = [np.pi/2, 0, 0]

    # initial_orientation_rpy = 
    
    print(f"Starting lunar position tracking from initial position: {initial_position}")
    print(f"Initial orientation (roll, pitch, yaw): {initial_orientation_rpy}")
    
    # Process the data with the correct initial values
    tracker = process_imu_data(
        imu_data_frames,
        initial_position=initial_position,
        initial_orientation_rpy=initial_orientation_rpy,
        sampling_rate=100
    )
    
    # Plot the trajectory
    import os
    save_dir = os.path.dirname(filename)
    save_path = os.path.join(save_dir, 'imu_trajectory_lunar.png')
    tracker.plot_trajectory(save_path, gt_position_history_np)
    
    # Calculate final position error
    final_position = tracker.get_current_position()
    final_gt_position = gt_position_history_np[-1, :]
    final_error = np.linalg.norm(final_position - final_gt_position)
    
    print(f"Final position: {final_position}")
    print(f"Final ground truth position: {final_gt_position}")
    print(f"Final position error: {final_error:.2f} meters")

# # Example usage
# if __name__ == "__main__":
#     # Simulate IMU data for testing (walking in a circle)

#     filename = "/home/annikat/MAPLE/data/035/imu_data.csv"
#     df = pd.read_csv(filename)

#     num_gt_cols = 7

#     gt_data = df.iloc[:, :num_gt_cols]  # ground-truth columns
#     gt_position_history = df.iloc[:, 1:4]
#     print(gt_position_history)
#     imu_data = df.iloc[:, num_gt_cols:] # IMU columns

#     # 3. Convert IMU data to list of lists (if needed by your process_imu_data function)
#     imu_data_frames = imu_data.values.tolist()

#     gt_position_history_np = gt_position_history.to_numpy()
    

#     # Process the data with initial orientation in roll, pitch, yaw format
#     # Here we're assuming the initial orientation is level (0, 0, 0)
#     tracker = process_imu_data(
#         imu_data_frames,
#         initial_position=[-2.263231, -4.454633,  1.404660],
#         initial_orientation_rpy= [0.032353, 0.005363, 1.63065],
#         sampling_rate=100
#     )
    
#     # Plot the trajectory
#     # tracker.plot_trajectory()
#     import os
#     save_dir = os.path.dirname(filename)
#     save_path = os.path.join(save_dir, 'imu_trajectory.png')
#     tracker.plot_trajectory(save_path, gt_position_history_np)
    
    # # Example of using with real-time API
    # def real_time_example(duration=10, sampling_rate=100):
    #     """Example of real-time processing with the actual API."""
    #     # Initialize with roll, pitch, yaw angles (in radians)
    #     tracker = IMUPositionTracker(
    #         initial_position=[0, 0, 0],
    #         initial_orientation_rpy=[0, 0, 0],  # [roll, pitch, yaw] in radians
    #         sampling_rate=sampling_rate
    #     )
        
    #     # Number of samples to process
    #     n_samples = int(duration * sampling_rate)
        
    #     for _ in range(n_samples):
    #         # Get IMU data from API
    #         imu_data = get_imu_data()
            
    #         # Update position estimate
    #         tracker.update(imu_data)
            
    #         # Get current position and orientation
    #         current_position = tracker.get_current_position()
    #         current_orientation = tracker.get_current_orientation_rpy()
            
    #         print(f"Position: {current_position}")
    #         print(f"Orientation (roll, pitch, yaw): {current_orientation}")
            
    #         # Sleep to maintain sampling rate (not needed if API has timing control)
    #         # time.sleep(1/sampling_rate)
        
    #     # Plot the trajectory after completion
    #     tracker.plot_trajectory()
        
    #     return tracker