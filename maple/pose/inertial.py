import numpy as np
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr

from numpy.typing import NDArray
from maple.pose.estimator import Estimator
from maple.utils import carla_to_pytransform, pytransform_to_tuple

""" Potential TO-DO:
- Perform trapezoidal integration during calculations
- Keep track of past imu data for possible filtering
- I am assuming that the IMU is calibrated to take out gravity. If not, we need to account for it.
"""


# TODO: Check the moon gravity offset is working correctly
class InertialEstimator(Estimator):
    """Provides pose estimation using the IMU on the lander."""

    _g = 1.625  # Lunar acceleration due to gravity [m/s^2]

    def __init__(self, agent):
        """Initializes the estimator.

        Args:
            agent: The agent instance.
        """

        self._agent = agent
        self._mission_time = agent.get_mission_time()

        # At mission start we can get the position of the rover in the global coordinate frame
        x, y, z, roll, pitch, yaw = pytransform_to_tuple(
            carla_to_pytransform(agent.get_initial_position())
        )

        # [pos, vel, ang]
        self._state = np.array([x, y, z, 0, 0, 0, roll, pitch, yaw], dtype=np.float64)

    def set_pose(self, pose, velocity=None):
        """Reset the pose.
        This will reset the pose and state vector of the integrator. By default velocity is preserved.

        Args:
            pose: A (4, 4) transformation matrix to use as the new pose
            velocity: A (, 3) vector to override the state velocity [x, y, z]
        """

        # Update the state vector
        x, y, z, roll, pitch, yaw = pytransform_to_tuple(pose)
        self._state[:3] = [x, y, z]
        self._state[6:] = [roll, pitch, yaw]

        if velocity is not None:
            self._state[3:6] = velocity

    def estimate(self, input_data=None) -> NDArray:
        """Estimates the rover's position by integrating the IMU data

        Returns:
            A (4, 4) transformation matrix of the rover's position in the world frame
        """

        # Calculate the delta time
        delta_time = self._agent.get_mission_time() - self._mission_time
        self._mission_time = self._agent.get_mission_time()

        # Get the IMU data [acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z]
        imu_data = self._agent.get_imu_data()

        # Extract the acceleration and angular velocity from the IMU data
        gyro = imu_data[3:]
        acc = imu_data[:3]

        # Integrate angular rate into orientation
        # Do this before correcting for gravity in acceleration
        self._state[6:] += gyro * delta_time

        # Subtract the acceleration due to gravity based on the IMU's orientation
        rotation = pyrot.matrix_from_euler(self._state[6:][::-1], 2, 1, 0, False)
        acc -= rotation @ np.array([0, 0, self._g])

        # Integrate acceleration into velocity
        self._state[3:6] += acc * delta_time

        # Integrate velocity into position
        self._state[:3] += self._state[3:6] * delta_time

        # Return the pose
        return pytr.transform_from(rotation, self._state[:3])
