# Yea so claude just kind of did this all with some minor changes, going to look into making it better though

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class RobotStateEstimator:
    """
    Extended Kalman Filter for robot state estimation combining:
    - IMU data (acceleration, angular velocity)
    - Position measurements
    - Orientation measurements (as quaternions)
    - Linear velocity measurements
    
    State vector (16 elements):
    - Position (x, y, z): indices 0-2
    - Velocity (vx, vy, vz): indices 3-5
    - Orientation (quaternion qw, qx, qy, qz): indices 6-9
    - Angular velocity (wx, wy, wz): indices 10-12
    - Acceleration bias (ax_bias, ay_bias, az_bias): indices 13-15
    """
    
    def __init__(self, dt, 
                 process_noise,
                 pos_std, 
                 vel_std, 
                 orientation_std,
                 angular_vel_std,
                 accel_std,
                 gyro_bias_std):
        """
        Initialize the EKF for robot state estimation.
        
        Parameters:
        -----------
        dt : float
            Time step between filter updates
        process_noise : float
            Process noise parameter for state prediction
        pos_std : float
            Standard deviation of position measurement noise
        vel_std : float
            Standard deviation of velocity measurement noise
        orientation_std : float
            Standard deviation of orientation measurement noise (in radians)
        angular_vel_std : float
            Standard deviation of angular velocity measurement noise
        accel_std : float
            Standard deviation of acceleration measurement noise
        gyro_bias_std : float
            Process noise for gyroscope bias drift
        """
        # Create EKF with 16 state variables and potentially up to 13 measurement variables
        # (3 position, 3 velocity, 4 quaternion, 3 angular velocity)
        self.ekf = ExtendedKalmanFilter(dim_x=16, dim_z=13)
        
        # Store parameters
        self.dt = dt
        self.process_noise = process_noise
        
        # Initial state
        self.ekf.x = np.zeros(16)
        self.ekf.x[6] = 1.0  # Quaternion w component = 1 (identity rotation)
        
        # Initial state covariance
        self.ekf.P = np.eye(16) * 100  # High initial uncertainty
        self.ekf.P[6:10, 6:10] = np.eye(4) * 0.1  # Lower uncertainty for quaternion
        
        # Process noise covariance
        # We'll set this dynamically in the predict function
        self.ekf.Q = np.eye(16) * 0.1
        
        # Sensor noise parameters
        self.pos_std = pos_std
        self.vel_std = vel_std
        self.orientation_std = orientation_std
        self.angular_vel_std = angular_vel_std
        self.accel_std = accel_std
        self.gyro_bias_std = gyro_bias_std
        
        # For reliability estimation
        self.max_position_var = 1.0
        self.max_orientation_var = 0.1
        
    def normalize_quaternion(self):
        """Normalize the quaternion in the state vector to maintain unit length"""
        q = self.ekf.x[6:10]
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            self.ekf.x[6:10] = q / q_norm
    
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
        r = R.from_quat([q[1], q[2], q[3], q[0]])  # Note: scipy uses [x,y,z,w] order
        return r.as_euler('xyz', degrees=True)
    
    def update_process_model(self):
        """Update the process model based on current state"""
        # State transition matrix will be handled in the predict function
        # using a nonlinear model for the quaternion integration
        
        # Process noise - adjusted based on state
        # Higher noise for acceleration and angular velocity components
        self.ekf.Q = np.eye(16) * self.process_noise
        self.ekf.Q[13:16, 13:16] *= self.gyro_bias_std**2  # Acceleration bias drift
    
    def predict(self):
        """Predict the next state based on the current state"""
        # Extract current state
        pos = self.ekf.x[0:3]
        vel = self.ekf.x[3:6]
        quat = self.ekf.x[6:10]
        ang_vel = self.ekf.x[10:13]
        accel_bias = self.ekf.x[13:16]
        
        # Update process model
        self.update_process_model()
        
        # Save previous state
        x_prev = self.ekf.x.copy()
        
        # Forward Euler integration for position and velocity
        pos_new = pos + vel * self.dt
        # We'll update velocity with IMU data in the IMU update
        vel_new = vel
        
        # For quaternion, integrate angular velocity using first-order approximation
        # Convert angular velocity to quaternion rate
        ang_vel_quat = np.array([0, ang_vel[0], ang_vel[1], ang_vel[2]])
        quat_dot = 0.5 * self.quaternion_multiply(quat, ang_vel_quat)
        quat_new = quat + quat_dot * self.dt
        quat_new = quat_new / np.linalg.norm(quat_new)  # Normalize
        
        # Angular velocity remains the same (will be updated with IMU)
        ang_vel_new = ang_vel
        
        # Acceleration bias (slow drift model)
        accel_bias_new = accel_bias
        
        # Update state
        self.ekf.x[0:3] = pos_new
        self.ekf.x[3:6] = vel_new
        self.ekf.x[6:10] = quat_new
        self.ekf.x[10:13] = ang_vel_new
        self.ekf.x[13:16] = accel_bias_new
        
        # Update covariance
        # For a proper EKF, we would compute the Jacobian of the state transition function
        # Here we use a simplified approach assuming locally linear behavior
        F = np.eye(16)
        # Position affected by velocity
        F[0:3, 3:6] = np.eye(3) * self.dt
        # Quaternion affected by angular velocity (simplified)
        # This is a simplification, a proper EKF would use full quaternion dynamics
        F[6:10, 10:13] = np.eye(4, 3) * 0.5 * self.dt
        
        self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q
        
        self.normalize_quaternion()
    
    def update_imu(self, accel, gyro):
        """
        Update the filter with IMU measurements (acceleration and angular velocity)
        
        Parameters:
        -----------
        accel : numpy array, shape (3,)
            Linear acceleration measurements (ax, ay, az) in m/s²
        gyro : numpy array, shape (3,)
            Angular velocity measurements (wx, wy, wz) in rad/s
        """
        # Extract current state
        vel = self.ekf.x[3:6]
        quat = self.ekf.x[6:10]
        accel_bias = self.ekf.x[13:16]
        
        # Correct acceleration for gravity
        # Convert quaternion to rotation matrix
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x,y,z,w]
        rotation_matrix = r.as_matrix()
        
        # Gravity vector in world frame
        gravity = np.array([0, 0, 9.81])
        
        # Remove gravity from acceleration (assuming accelerometer measures proper acceleration)
        accel_world = rotation_matrix @ (accel - accel_bias) + gravity
        
        # Update velocity based on acceleration
        self.ekf.x[3:6] = vel + accel_world * self.dt
        
        # Update angular velocity from gyroscope
        self.ekf.x[10:13] = gyro
    
    def update_with_measurements(self, z, measurement_type):
        """
        Update the filter with measurements of position, orientation, or velocity
        
        Parameters:
        -----------
        z : numpy array
            Measurement values
        measurement_type : str
            Type of measurement: 'position', 'orientation', 'linear_velocity', 'angular_velocity',
            or 'all' for all measurements
        """
        if measurement_type == 'position':
            # Position measurement (x, y, z)
            H = np.zeros((3, 16))
            H[0:3, 0:3] = np.eye(3)  # Position components
            self.ekf.H = H
            self.ekf.R = np.eye(3) * self.pos_std**2
            self.ekf.update(z, HJacobian=lambda x: H, Hx=lambda x: H @ x)
            
        elif measurement_type == 'orientation':
            # Orientation measurement (quaternion)
            H = np.zeros((4, 16))
            H[0:4, 6:10] = np.eye(4)  # Quaternion components
            self.ekf.H = H
            self.ekf.R = np.eye(4) * self.orientation_std**2
            self.ekf.update(z, HJacobian=lambda x: H, Hx=lambda x: H @ x)
            self.normalize_quaternion()
            
        elif measurement_type == 'linear_velocity':
            # Linear velocity measurement (vx, vy, vz)
            H = np.zeros((3, 16))
            H[0:3, 3:6] = np.eye(3)  # Velocity components
            self.ekf.H = H
            self.ekf.R = np.eye(3) * self.vel_std**2
            self.ekf.update(z, HJacobian=lambda x: H, Hx=lambda x: H @ x)
            
        elif measurement_type == 'angular_velocity':
            # Angular velocity measurement (wx, wy, wz)
            H = np.zeros((3, 16))
            H[0:3, 10:13] = np.eye(3)  # Angular velocity components
            self.ekf.H = H
            self.ekf.R = np.eye(3) * self.angular_vel_std**2
            self.ekf.update(z, HJacobian=lambda x: H, Hx=lambda x: H @ x)
            
        elif measurement_type == 'all':
            # All measurements together (position, orientation, linear vel, angular vel)
            H = np.zeros((13, 16))
            H[0:3, 0:3] = np.eye(3)      # Position
            H[3:6, 3:6] = np.eye(3)      # Linear velocity
            H[6:10, 6:10] = np.eye(4)    # Orientation (quaternion)
            H[10:13, 10:13] = np.eye(3)  # Angular velocity
            
            self.ekf.H = H
            
            # Combined measurement noise covariance
            R = np.zeros((13, 13))
            R[0:3, 0:3] = np.eye(3) * self.pos_std**2           # Position
            R[3:6, 3:6] = np.eye(3) * self.vel_std**2           # Linear velocity
            R[6:10, 6:10] = np.eye(4) * self.orientation_std**2  # Orientation
            R[10:13, 10:13] = np.eye(3) * self.angular_vel_std**2  # Angular velocity
            
            self.ekf.R = R
            self.ekf.update(z, HJacobian=lambda x: H, Hx=lambda x: H @ x)
            self.normalize_quaternion()
    
    def get_state(self):
        """
        Returns the current estimated state.
        
        Returns:
        --------
        state_dict : dict
            Dictionary containing all state variables with labels
        """
        state_dict = {
            'position': self.ekf.x[0:3],
            'velocity': self.ekf.x[3:6],
            'quaternion': self.ekf.x[6:10],
            'euler_angles': self.quaternion_to_euler(self.ekf.x[6:10]),
            'angular_velocity': self.ekf.x[10:13],
            'accel_bias': self.ekf.x[13:16]
        }
        return state_dict
    
    def get_state_reliability(self):
        """
        Calculate reliability metrics for the state estimates based on covariance.
        
        Returns:
        --------
        reliability : dict
            Dictionary containing reliability metrics for different state components
        """
        # Get the diagonal of the covariance matrix (variances)
        variances = np.diag(self.ekf.P)
        
        # Position reliability
        pos_var = variances[0:3]
        pos_reliability = np.exp(-np.mean(pos_var) / self.max_position_var)
        pos_reliability = max(0, min(1, pos_reliability))
        
        # Orientation reliability
        quat_var = variances[6:10]
        ori_reliability = np.exp(-np.mean(quat_var) / self.max_orientation_var)
        ori_reliability = max(0, min(1, ori_reliability))
        
        # Velocity reliability
        vel_var = variances[3:6]
        vel_reliability = np.exp(-np.mean(vel_var) / (self.vel_std**2 * 10))
        vel_reliability = max(0, min(1, vel_reliability))
        
        # Overall reliability (weighted average)
        overall_reliability = 0.4 * pos_reliability + 0.4 * ori_reliability + 0.2 * vel_reliability
        
        reliability = {
            'position': pos_reliability,
            'orientation': ori_reliability,
            'velocity': vel_reliability,
            'overall': overall_reliability,
            'position_variance': pos_var,
            'orientation_variance': quat_var,
            'velocity_variance': vel_var
        }
        
        return reliability


