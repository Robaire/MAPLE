import numpy as np
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
from numpy.typing import NDArray

from maple.pose.estimator import Estimator

""" Potential TO-DO:
- Perform trapezoidal integration during calculations
- Keep track of past imu data for possible filtering
- I am assuming that the IMU is calibrated to take out gravity. If not, we need to account for it.
"""


# IMPORTANT TODO: Check this IMU estimator code (the acceleration due to gravity is def diff on moon)
# TODO: Refactor this
class InertialEstimator(Estimator):
    """Provides pose estimation using the IMU on the lander."""

    def __init__(self, agent):
        """Initializes the estimator.

        Args:
            agent: The agent instance.
        """

        self.agent = agent
        self.prev_state = None
        self.dt = (
            1 / 20
        )  # 20 Hz as defined by competition documentation. Could instead use the mission time function.
        self.g = 1.625  # m/s^2

    def change_in_state_imu_frame(self):
        """Estimates the change in the rover's state purely by integrating the imu data.

        Returns:
            The estimated change in state as a carla transform in the IMU's frame.
        """
        # imu_data return [ accelerometer.x, accelerometer.y, accelerometer.z, gyroscope.x, gyroscope.y, gyroscope.z]
        imu_data = self.agent.get_imu_data()

        # Extract the acceleration and angular velocity from the IMU data
        acc = np.array([imu_data[0], imu_data[1], imu_data[2]])
        gyro = np.array([imu_data[3], imu_data[4], imu_data[5]])

        # Integrate the acceleration to get the velocity
        vel = acc * self.dt

        # Integrate the velocity to get the position
        pos = vel * self.dt

        # Integrate the angular velocity to get the orientation. For now we do not use quaternions.
        ang = gyro * self.dt

        # Create a new transform with the updated state
        transl = pos
        # I believe the gyro will return extrinsic rotations, but this should be verified somehow
        rot = pyrot.active_matrix_from_extrinsic_roll_pitch_yaw(ang)
        state_delta = pytr.transform_from(rot, transl)

        # state_delta = carla_copy(pos[0], pos[1], pos[2], ang[0], ang[1], ang[2])
        return state_delta

    def estimate(self, input_data) -> NDArray:
        # TODO: THIS DOES NOT IMPLEMENT THE INTERFACE CORRECTLY
        """Estimates the rover's next state purely by concatenating the transform estimate from
        the imu with that of the previous state.

        Returns:
            The estimated next state as a carla transform in the world frame.
        """

        # If there is no previous state we cant perform this
        if self.prev_state is None:
            return None

        state_delta = self.change_in_state_imu_frame()

        # Transform the state delta to the world frame
        new_state_pytrans = pytr.concat(self.prev_state, state_delta)
        self.prev_state = new_state_pytrans if new_state_pytrans is not None else self.prev_state

        return new_state_pytrans
