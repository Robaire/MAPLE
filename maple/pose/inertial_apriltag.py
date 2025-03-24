from numpy.typing import NDArray

from maple.pose.apriltag import ApriltagEstimator
from maple.pose.estimator import Estimator
from maple.pose.inertial import InertialEstimator


class InertialApriltagEstimator(Estimator):
    """Provides a position estimate using both AprilTags and the IMU"""

    def __init__(self, agent):
        """Create the estimator.

        Args:
            agent: The Agent instance
        """

        self._april_tag_estimator = ApriltagEstimator(agent)
        self._imu_estimator = InertialEstimator(agent)

    def estimate(self, input_data) -> tuple[NDArray, bool]:
        """Return the current estimated pose, will use the AprilTags if available, otherwise will use IMU integration.

        Args:
            input_data: The input_data dictionary this time step

        Returns:
            A pytransform representing the rover in the global frame
        """

        # Generate pose estimates
        pose_april = self._april_tag_estimator(input_data)
        pose_imu = self._imu_estimator(input_data)  # Call this to ensure it integrates

        if pose_april is not None:
            # Update the imu pose
            self._imu_estimator.set_pose(pose_april)

            return pose_april, True

        else:
            return pose_imu, False