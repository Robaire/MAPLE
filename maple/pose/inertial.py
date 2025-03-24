import numpy as np
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr

from numpy.typing import NDArray
from maple.pose.estimator import Estimator
from maple.utils import carla_to_pytransform

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
        pose = carla_to_pytransform(agent.get_initial_position())

        # State Vector [pos, vel, quaternion(w, x, y, z)]
        self._state = np.zeros(10)
        self._state[:3] = pose[:3, 3]
        self._state[6:] = pyrot.quaternion_from_matrix(pose[:3, :3])

    def set_pose(self, pose, velocity=None):
        """Reset the pose.
        This will reset the pose and state vector of the integrator. By default velocity is preserved.

        Args:
            pose: A (4, 4) transformation matrix representing the rover in the global frame
            velocity: A (, 3) vector representing the rover's velocity in the rover body frame
        """

        # Update the state vector
        self._state[:3] = pose[:3, 3]
        self._state[6:] = pyrot.quaternion_from_matrix(pose[:3, :3])

        if velocity is not None:
            # Rotate the velocity into the global frame
            velocity = pyrot.q_prod_vector(pyrot.q_conj(self._state[6:]), velocity)
            self._state[3:6] = velocity

    @staticmethod
    def _omega(gyro: NDArray) -> NDArray:
        """Define the omega operator for the angular acceleration."""
        x, y, z = gyro
        return np.array(
            [
                [0, -x, -y, -z],
                [x, 0, z, -y],
                [y, -z, 0, x],
                [z, y, -x, 0],
            ]
        )

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
        gyro = imu_data[3:]  # [rad/s]
        acc = imu_data[:3]  # [m/s^2]

        # Integrate the gyroscope readings
        # \mathbf{q}_{t+1} = \Bigg[\mathbf{I}_4 + \frac{1}{2}\boldsymbol\Omega(\boldsymbol\omega)\Delta t\Bigg]\mathbf{q}_t
        q = np.eye(4) + (0.5 * self._omega(gyro) * delta_time) @ self._state[6:]
        self._state[6:] = q / np.linalg.norm(q)  # Normalize

        # Rotate the acceleration vector into the global frame
        # Since orientation represents the rover in the global frame, we need the conjugate
        acc = pyrot.q_prod_vector(pyrot.q_conj(self._state[6:]), acc)

        # Correct for gravity
        acc[2] += self._g

        # Integrate acceleration into velocity
        self._state[3:6] += acc * delta_time

        # Integrate velocity into position
        self._state[:3] += self._state[3:6] * delta_time

        # Return the pose as a 4x4 transformation matrix
        return pytr.transform_from_pq(np.hstack(self._state[:3], self._state[6:]))
