import numpy as np
import pytransform3d.transformations as pytr
import pytransform3d.rotations as pyrot
from maple.utils import carla_to_pytransform, pytransform_to_carla, carla_copy

""" Potential TO-DO:
- Perform trapezoidal integration during calculations
- Keep track of past imu data for possible filtering
- I am assuming that the IMU is calibrated to take out gravity. If not, we need to account for it.
"""

class imu_Estimator:
    """Provides pose estimation using the IMU on the lander."""

    def __init__(self, agent):
        """Initializes the estimator.
        
        Args:
            agent: The agent instance.
        """

        self.agent = agent
        self.dt = 1/20 # 20 Hz as defined by competition documentation. Could instead use the mission time function.
        self.g = 9.81 # m/s^2

    def change_in_state_imu_frame(self):
        """Estimates the change in the rover's state purely by integrating the imu data.
        
        Returns:
            The estimated change in state as a carla transform in the IMU's frame.
        """
        # imu_data return [ accelerometer.x, accelerometer.y, accelerometer.z, gyroscope.x, gyroscope.y, gyroscope.z]
        imu_data = self.agent.get_imu_data()
        print('imu_data:', imu_data)
        # Extract the acceleration and angular velocity from the IMU data
        acc = np.array([imu_data[0], imu_data[1], imu_data[2]])
        gyro = np.array([imu_data[3], imu_data[4], imu_data[5]])

        # Integrate the acceleration to get the velocity
        vel = acc * self.dt

        # Integrate the velocity to get the position
        pos = vel * self.dt
        print('pos:', pos)

        # Integrate the angular velocity to get the orientation. For now we do not use quaternions.
        ang = gyro * self.dt
        print('ang:', ang)

        # Create a new transform with the updated state
        transl = pos
        rot = pyrot.active_matrix_from_extrinsic_roll_pitch_yaw(ang) # I believe the gyro will return extrinsic rotations, but this should be verified somehow
        state_delta = pytr.transform_from(rot, transl)

        #state_delta = carla_copy(pos[0], pos[1], pos[2], ang[0], ang[1], ang[2])
        return state_delta
            

    def next_state(self):
        """Estimates the rover's next state purely by concatenating the transform estimate from 
        the imu with that of the previous state.
        
        Returns:
            The estimated next state as a carla transform in the world frame.
        """
        prev_state = self.agent.prev_state # Assume to be a pytransform

        state_delta = self.change_in_state_imu_frame()
        # Transform the state delta to the world frame
        new_state_pytrans = pytr.concat(prev_state, state_delta)

        return new_state_pytrans




