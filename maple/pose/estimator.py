# This is an abstraction of all position estimators so that we can call this outside of dev to get a gauranteed position estimate

from numpy.typing import NDArray

from maple.pose.apriltag import ApriltagEstimator
from maple.pose.imu import InertialEstimator

class Estimator:
    """Provides position estimate using other python files"""

    def __init__(self, agent):
        """Create the estimator.

        Args:
            agent: The Agent instance
        """

        self.agent = agent
        self.prev_state = None

        self.april_tag_estimator = ApriltagEstimator(agent)
        self.imu_estimator = InertialEstimator(agent)

    def __call__(self, input_data):
        """Equivalent to calling `estimate`."""
        return self.estimate(input_data)

    def estimate(self, input_data, estimator_type = 'ALL') -> NDArray:
        """
        Abstracts the other estimate functions to be able to only call one
        """
        position = None
        if estimator_type == 'ALL':
            position = self.april_tag_estimator(input_data)

            # if the april tag returns none use the imu, otherwise keep the position
            position = self.imu_estimator(self.prev_state) if position is None else position

            # if the position is not None then we can also update the previous state
            self.prev_state = position if position is not None else self.prev_state

        elif estimator_type == 'APRILTAG':
            position = self.april_tag_estimator(input_data)
            
        elif estimator_type == 'IMU':
            position = self.imu_estimator(self.prev_state)
            self.prev_state = position if position is not None else self.prev_state

        else:
            raise ValueError(f'Estimator type {estimator_type} not recognized')
        return position
    