# This is an abstraction of all position estimators so that we can call this outside of dev to get a gauranteed position estimate

from numpy.typing import NDArray

from maple.pose.apriltag import ApriltagEstimator
from maple.pose.imu_estimator import imu_Estimator

from maple.utils import carla_to_pytransform

class Estimator:
    """Provides position estimate using other python files"""

    def __init__(self, agent):
        """Create the estimator.

        Args:
            agent: The Agent instance
        """

        self.agent = agent
        self.prev_state = carla_to_pytransform(self.agent.get_initial_position())

        self.april_tag_estimator = ApriltagEstimator(agent)
        self.imu_estimator = imu_Estimator(agent)

    def __call__(self, input_data):
        """Equivalent to calling `estimate`."""
        return self.estimate(input_data)

    def estimate(self, input_data) -> NDArray:
        """
        Abstracts the other estimate functions to be able to only call one
        """
        position = None

        # position = self.april_tag_estimator(input_data)

        # Remove this later
        printed = False
        if position is not None:
            print(f'using the april tag information')
            printed = True
        # Remove this later

        # IMPORTANT NOTE: The imu can be commented out for testing
        # if the april tag returns none use the imu, otherwise keep the position
        position = self.imu_estimator(self.prev_state) if position is None else position

        # Remove this later
        if not printed:
            print(f'using the imu data or nothing')
        # Remove this later

        # At this point if we dont know our position use the prev state as a fall back
        # IMPORTANT NOTE: the imu estimate should always return something so this line is a safey procation
        # position = self.prev_state if position is None else position

        # if the position is not None then we can also update the previous state
        self.prev_state = position if position is not None else self.prev_state

        return position
    