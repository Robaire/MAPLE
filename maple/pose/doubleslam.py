import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from maple.pose.estimator import Estimator
from maple.pose.orbslam import OrbslamEstimator
from maple.utils import pytransform_to_tuple, carla_to_pytransform, tuple_to_pytransform


class DoubleSlamEstimator(Estimator):
    def __init__(self, agent):
        self.agent = agent

        # The thresholds for the translation and rotation error before resetting
        self.translation_threshold = 2  # meters
        self.rotation_threshold = np.deg2rad(45)  # radians

        # Create the two ORB-SLAM estimators
        self.front = OrbslamEstimator(agent, "FrontLeft", "FrontRight", mode="stereo")
        self.rear = OrbslamEstimator(agent, "BackLeft", "BackRight", mode="stereo")

        # Store the last valid estimate
        rover_global = carla_to_pytransform(agent.get_initial_position())
        self.last_front_estimate = rover_global
        self.last_rear_estimate = rover_global
        self.last_combined_estimate = rover_global
        self.last_any_estimate = rover_global

        self.estimate_source = "last_any"

    def estimate(self, input_data) -> NDArray:
        # Get estimates from both ORB-SLAM estimators
        try:
            front_estimate = self.front.estimate(input_data)
        except RuntimeError:
            # Reset the front ORB-SLAM estimator
            # print("Front ORB-SLAM estimator crashed. Hard resetting...")
            # self.front.reset(self.last_combined_estimate)
            self.front.reset(self.last_any_estimate)
            front_estimate = None

        try:
            rear_estimate = self.rear.estimate(input_data)
        except RuntimeError:
            # Reset the rear ORB-SLAM estimator
            # print("Rear ORB-SLAM estimator crashed. Hard resetting...")
            # self.rear.reset(self.last_combined_estimate)
            self.rear.reset(self.last_any_estimate)
            rear_estimate = None

        # If called on a frame with no image data, both estimates will be None
        if front_estimate is None and rear_estimate is None:
            self.estimate_source = "no_images"
            return None

        if front_estimate is not None and rear_estimate is not None:
            # print("Both estimates directly from orbslam are not None")
            pass

        # Orbslam will return estimates that are None immediately after the map resets
        # However, when it loses tracking, it will return the last valid estimate until the map resets
        # We should check if an estimate is too close to the last valid estimate (1mm) if this is the case,
        # it is likely indicative of a tracking failure and we should ignore the estimate until the map resets,
        # or until the estimator recovers.
        # If the estimate recovers, it will be close ish to the last valid estimate
        # If the map resets, the estimate will be at the origin

        if front_estimate is not None:
            # Check if the front estimate is too close to the last front estimate
            distance, angle = self._pose_error(front_estimate, self.last_front_estimate)
            if distance < 0.001 and angle < np.deg2rad(1):
                # Too close, we should ignore this estimate
                # print(
                #     f"Front estimate is within {distance / 1000}mm of the last front estimate..."
                # )
                self.last_front_estimate = front_estimate
                front_estimate = None
                pass
            else:
                # Not too close, we can use this estimate
                pass

        if rear_estimate is not None:
            # Check if the rear estimate is too close to the last rear estimate
            distance, angle = self._pose_error(rear_estimate, self.last_rear_estimate)
            if distance < 0.001 and angle < np.deg2rad(1):
                # Too close, we should ignore this estimate
                # print(
                #     f"Rear estimate is within {distance / 1000}mm of the last rear estimate..."
                # )
                self.last_rear_estimate = rear_estimate
                rear_estimate = None
                pass
            else:
                # Not too close, we can use this estimate
                pass

        # TODO: For debug
        if front_estimate is not None and rear_estimate is not None:
            # print(
            #     "Both estimates have moved sufficiently far from the last valid estimate"
            # )
            pass

        # TODO: Check if any of the estimates are significantly different from the last valid estimate
        # TODO: Check both translation and rotation

        # TODO: Alternatively we may want to reset the estimator with the estimate from the other one
        # if one is available this frame. This might be better than resetting to the last valid estimate.

        # Check if the front estimate is too far away from any of the last valid estimates
        front_jumped = False
        if front_estimate is not None:
            if not self._within_threshold(front_estimate):
                front_jumped = True
                # print(
                #     "Front estimate is too far away from any of the last valid estimates"
                # )
                # print(f"Current front estimate: {pytransform_to_tuple(front_estimate)}")
                # print(
                #     f"Last front estimate: {pytransform_to_tuple(self.last_front_estimate)}"
                # )
                # print(
                #     f"Last any estimate: {pytransform_to_tuple(self.last_any_estimate)}"
                # )

                front_estimate = None  # Clear the front estimate since it is bad

        # Check the rear estimate
        rear_jumped = False
        if rear_estimate is not None:
            if not self._within_threshold(rear_estimate):
                rear_jumped = True
                print(
                    "Rear estimate is too far away from any of the last valid estimates"
                )
                print(f"Current rear estimate: {pytransform_to_tuple(rear_estimate)}")
                print(
                    f"Last rear estimate: {pytransform_to_tuple(self.last_rear_estimate)}"
                )
                print(
                    f"Last any estimate: {pytransform_to_tuple(self.last_any_estimate)}"
                )
                rear_estimate = None  # Clear the rear estimate since it is bad

                # If it looks like the rear estimate jumped to the origin (probably due to a map change)
                # reinitialize the origin to the front estimate if it is available
                # TODO: Do this after the front estimate is checked
                # if front_estimate is not None:
                #     print("Resetting with front-estimate")
                #     self.rear._set_orbslam_global(front_estimate)
                # else:
                #     print("Resetting with last-any-estimate")
                #     self.rear._set_orbslam_global(self.last_any_estimate)

        # At this point, we if an estimate is not None, it is valid
        if front_estimate is not None:
            self.last_front_estimate = front_estimate
        if rear_estimate is not None:
            self.last_rear_estimate = rear_estimate

        # If both estimates are valid, return the average
        if front_estimate is not None and rear_estimate is not None:
            # print("Both estimates are valid, averaging...")

            estimates = [front_estimate, rear_estimate]

            # TODO: Convert to quaternions and average

            # Take the circular mean of the rotation angles to avoid wrapping issues around -pi, pi
            _, _, _, f_roll, f_pitch, f_yaw = pytransform_to_tuple(front_estimate)
            _, _, _, r_roll, r_pitch, r_yaw = pytransform_to_tuple(rear_estimate)
            roll = self._circular_mean([f_roll, r_roll])
            pitch = self._circular_mean([f_pitch, r_pitch])
            yaw = self._circular_mean([f_yaw, r_yaw])

            # Average the translation
            x, y, z = np.mean([m[:3, 3] for m in estimates], axis=0)

            # Convert to a pytransform
            estimate = tuple_to_pytransform((x, y, z, roll, pitch, yaw))

            # Update the last valid estimates
            self.last_combined_estimate = estimate
            self.last_any_estimate = estimate
            self.estimate_source = "combined"
            return estimate

        # At this point, at least one of the estimates is not valid
        # Check the front estimate
        if front_estimate is not None:
            # print("Front estimate is valid, returning it...")
            self.last_any_estimate = front_estimate

            # Check if the rear estimate is invalid because it jumped
            if rear_jumped:
                # print("Rear estimate jumped, resetting with front estimate...")
                self.rear._set_orbslam_global(front_estimate)
            else:
                # The rear estimate did not jump, so it probably just lost tracking
                pass

            self.estimate_source = "front"
            return front_estimate

        # Check the rear estimate
        if rear_estimate is not None:
            # print("Rear estimate is valid, returning it...")
            self.last_any_estimate = rear_estimate

            # Check if the front estimate is invalid because it jumped
            if front_jumped:
                # print("Front estimate jumped, resetting with rear estimate...")
                self.front._set_orbslam_global(rear_estimate)
            else:
                # The front estimate did not jump, so it probably just lost tracking
                pass

            self.estimate_source = "rear"
            return rear_estimate

        # If both estimates are invalid, we should reset any that jumped to the last valid estimate
        if front_jumped:
            # print(
            #     "Front jumped, and no valid estimate, resetting to last valid estimate..."
            # )
            self.front._set_orbslam_global(self.last_any_estimate)

        if rear_jumped:
            # print(
            #     "Rear jumped, and no valid estimate, resetting to last valid estimate..."
            # )
            self.rear._set_orbslam_global(self.last_any_estimate)

        # If we get here, neither estimate is valid for some reason,
        # this could be because each has lost tracking, or each has jumped
        # print("No valid estimates, returning last valid estimate...")
        self.estimate_source = "last_any"
        return self.last_any_estimate

    def _circular_mean(self, angles):
        """Calculate the circular mean of a list of angles"""

        sin_average = np.sin(angles).mean()
        cos_average = np.cos(angles).mean()
        return np.arctan2(sin_average, cos_average)

    def _within_threshold(self, estimate):
        """Check if an estimate is within threshold to any of the last valid estimates."""

        valid_estimates = [
            self.last_front_estimate,
            self.last_rear_estimate,
            self.last_combined_estimate,
            self.last_any_estimate,
        ]

        for valid_estimate in valid_estimates:
            distance, angle = self._pose_error(estimate, valid_estimate)
            if (
                distance < self.translation_threshold
                and angle < self.rotation_threshold
            ):
                return True
        return False

    def _set_orbslam_global(self, rover_global):
        # Set the global for each ORB-SLAM estimator
        self.last_average_estimate = rover_global
        self.last_front_estimate = rover_global
        self.last_rear_estimate = rover_global
        self.last_any_estimate = rover_global
        self.front._set_orbslam_global(rover_global)
        self.rear._set_orbslam_global(rover_global)

    def _pose_error(self, estimate, last_estimate) -> tuple[float, float]:
        """Calculate the translation and rotation error between the estimate and the last valid estimate

        Returns:
            tuple[float, float]: The translation error (euclidean) and rotation error (angle)
        """

        # Calculate the translation error
        last_xyz = last_estimate[:3, 3]
        est_xyz = estimate[:3, 3]
        err_xyz = np.linalg.norm(est_xyz - last_xyz)

        # Calculate the rotation error
        last_rpy = last_estimate[:3, :3]
        est_rpy = estimate[:3, :3]
        err_angle = Rotation.from_matrix(est_rpy @ last_rpy.T).magnitude()

        return (err_xyz, err_angle)
