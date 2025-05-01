from maple.pose.estimator import Estimator
from maple.pose.orbslam import OrbslamEstimator
from numpy.typing import NDArray


class DoubleSlamEstimator(Estimator):
    def __init__(self, agent):
        self.agent = agent

        # Create the two ORB-SLAM estimators
        self.front = OrbslamEstimator(agent, "FrontLeft", "FrontRight", mode="stereo")
        self.rear = OrbslamEstimator(agent, "BackLeft", "BackRight", mode="stereo")

        # Store the last valid estimates from each estimator
        self.last_front_estimate = None
        self.last_rear_estimate = None

    def estimate(self, input_data) -> NDArray:
        # Get estimates from both ORB-SLAM estimators
        try:
            front_estimate = self.front.estimate(input_data)
        except RuntimeError:
            # TODO: Detect the failure and try to recover
            front_estimate = None

        try:
            rear_estimate = self.rear.estimate(input_data)
        except RuntimeError:
            # TODO: Detect the failure and try to recover
            rear_estimate = None

        # Store the last valid estimates
        if front_estimate is not None:
            self.last_front_estimate = front_estimate

        if rear_estimate is not None:
            self.last_rear_estimate = rear_estimate

        # TODO: Check if one of the estimates is near to failure and try to recover

        # If both estimates are valid, return the average
        if front_estimate is not None and rear_estimate is not None:
            return (front_estimate + rear_estimate) / 2

        # If only one estimate is valid, return it
        if front_estimate is not None:
            return front_estimate

        if rear_estimate is not None:
            return rear_estimate

        # If neither estimate is valid, return None
        # TODO: If this happens idk what to do :(
        return None
