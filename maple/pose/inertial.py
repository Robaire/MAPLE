import numpy as np
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
from numpy.typing import NDArray
from maple.utils import carla_to_pytransform

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

    agent: None
    prev_state: NDArray  # This is the transform of the rover in the global frame
    dt: float  # Simulation time step
    g: float  # Lunar acceleration due to gravity

    def __init__(self, agent):
        """Initializes the estimator.

        Args:
            agent: The agent instance.
        """

        self.agent = agent

        # At mission start we can get the position of the rover in the global coordinate frame
        self.prev_state = carla_to_pytransform(agent.get_initial_position())
        self.prev_vel = [0,0,0]

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

        # Subtract the acceleration due to gravity based on the IMU's orientation
        pq = pytr.pq_from_transform(self.prev_state)
        quat = pq[3:]
        grav_acc = pyrot.q_prod_vector(quat, [0, 0, self.g])
        #grav_acc = [0, 0, self.g]
        #prev_vel = pyrot.q_prod_vector(pyrot.q_conj(quat), self.prev_vel)
        prev_vel = self.prev_vel
        acc = acc - grav_acc
        # acc_old = acc.copy()
        # acc[0] = acc_old[1]
        # acc[1] = acc_old[0]
        acc[0] = 0.1
        acc[1] = 0
        acc[2] = 0

        # Integrate the angular velocity to get the orientation. For now we do not use quaternions.
        ang = gyro * self.dt

        #transl = [0,0,0]
        roll, pitch, yaw = ang
        # I believe the gyro will return intrinsic rotations, but this should be verified somehow
        rot = pyrot.matrix_from_euler([roll,pitch,yaw],0,1,2,extrinsic=False)

        # Integrate the acceleration to get the velocity
        print('prev_vel:', prev_vel)
        #prev_vel = pyrot.matrix_from_quaternion(pyrot.q_conj(quat)) @ rot.transpose() @ prev_vel
        vel = self.prev_vel + acc * self.dt
        #vel[0] = 0
        #vel[0] = 0
        #vel[1] = 0
        # vel[0] = 0.1
        # vel[1] = 0
        # vel[2] = 0
        self.prev_vel = vel.copy()

        # Integrate the velocity to get the position
        pos = vel * self.dt

        print("Pos:",pos)
        # Create a new transform with the updated state
        # transl = [0,0,0]
        # transl[0] = pos[1]
        # transl[1] = pos[0]
        #transl = pyrot.q_prod_vector(pyrot.q_conj(quat),pos) # Why rotate pos?
        transl = pos

        state_delta = pytr.transform_from(rot, transl)

        # state_delta = carla_copy(pos[0], pos[1], pos[2], ang[0], ang[1], ang[2])
        return state_delta

    def estimate(self, input_data=None) -> NDArray:
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
        print("Prev state:",self.prev_state)
        print("state delta:",state_delta)
        #state_delta = pytr.transform_from(np.eye(3),[0,0,0])
        new_state_pytrans = pytr.concat(state_delta, self.prev_state)
        self.prev_state = (
            new_state_pytrans if new_state_pytrans is not None else self.prev_state
        )

        return new_state_pytrans
