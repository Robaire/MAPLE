# This is an abstraction of all position estimators so that we can call this outside of dev to get a gauranteed position estimate

from numpy.typing import NDArray

from maple.pose.apriltag import ApriltagEstimator
from maple.pose.imu_Estimator import imu_Estimator

from maple.utils import carla_to_pytransform

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
        self.imu_estimator = imu_Estimator(agent)

    def __call__(self, input_data):
        """Equivalent to calling `estimate`."""
        return self.estimate(input_data)

    def estimate(self, input_data) -> NDArray:
        """
        Abstracts the other estimate functions to be able to only call one
        """

        position = self.april_tag_estimator(input_data)

        # if the april tag returns none use the imu, otherwise keep the position
        position = self.imu_estimator(self.prev_state) if position is None else position

        # At this point the position is only None if there is no Apriltag and no previous state (implying we are at out start position)
        position = carla_to_pytransform(self.agent.get_initial_position()) if position is None else position

        # if the position is not None then we can also update the previous state
        self.prev_state = position if position is not None else self.prev_state

        return position
    