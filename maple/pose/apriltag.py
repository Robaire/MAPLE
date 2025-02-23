import numpy as np
from numpy.typing import NDArray
from pyapriltags import Detector
from pytransform3d.transformations import concat, invert_transform, transform_from
from pytransform3d.rotations import matrix_from_euler
from scipy.spatial.transform import Rotation

from maple.pose.estimator import Estimator
from maple import geometry
from maple.utils import camera_parameters, carla_to_pytransform


class ApriltagEstimator(Estimator):
    """Provides pose estimation using AprilTags on the lander."""

    agent: None
    fiducials: dict
    detector: Detector
    lander_global: NDArray

    def __init__(self, agent):
        """Create the estimator.

        Args:
            agent: The Agent instance
        """

        # # Check that the agent has fiducials enabled
        # if not agent.use_fiducials():
        #     raise ValueError("agent must have fiducials enabled")

        self.agent = agent
        self.detector = Detector()

        # At mission start we can get the position of the rover in the global coordinate frame
        # and the position of the lander in the rover's coordinate frame
        # using these we can determine the position of the lander in the global coordinate frame
        rover_global = carla_to_pytransform(agent.get_initial_position())
        lander_rover = carla_to_pytransform(agent.get_initial_lander_position())
        self.lander_global = concat(lander_rover, rover_global)

        # Correct for the apriltag coordinate convention
        tag_correction = transform_from(
            matrix_from_euler(
                [np.pi / 2, 0, -np.pi / 2],
                2,
                1,
                0,
                False,
            ),
            [0, 0, 0],
        )

        # We want a dict of tags (key: id, value: lander_to_tag)
        self.fiducials = {}
        for group, tag_group in geometry.lander["fiducials"].items():
            tag_rotations = {"a": -45, "b": 45, "c": 135, "d": -135}  # [degrees]
            rotation = matrix_from_euler(
                [np.deg2rad(tag_rotations[group]), 0, 0], 2, 1, 0, False
            )

            # For each tag determine its transform from the lander
            for tag in tag_group.values():
                translation = [tag["x"], tag["y"], tag["z"]]
                tag_lander = concat(
                    tag_correction, transform_from(rotation, translation)
                )

                # Add this tag to the dict
                self.fiducials[tag["id"]] = concat(tag_lander, self.lander_global)

    def estimate(self, input_data) -> NDArray:
        """Iterates through all active cameras and averages all detections.

        Args:
            input_data: The input data dictionary provided by the simulation

        Returns:
            An average pose estimate from all detections. None if no detections.
        """

        try:
            # "Grayscale" may not exist in the input_data dict if no cameras are active
            cameras = input_data["Grayscale"]
        except KeyError:
            return None

        estimates = []
        for camera, image in cameras.items():
            # image will be None if there is no image this tick
            if image is not None:
                # Get the estimates from this image
                estimates.extend(self._estimate_image(camera, image))

        if len(estimates) == 0:
            return None

        # Average the estimates
        return transform_from(
            Rotation.from_matrix([m[:3, :3] for m in estimates]).mean().as_matrix(),
            np.mean([m[:3, 3] for m in estimates], axis=0),
        )

    def _estimate_image(self, camera, image: NDArray[np.uint8]) -> list:
        """Process a single image to get a pose estimate.

        Args:
            camera: The camera the image came from
            image: A grayscale image to search for tags in

        Returns:
            A list of pose estimates for the camera in the global coordinate frame.
        """

        # camera is something like: carla.SensorPosition.Front
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(camera))
        rover_camera = invert_transform(camera_rover)

        # Due to the coordinate axis convention AprilTag uses we need to apply a rotation correction
        # Z is out of the camera, X is to the right of the image, Y is down in the image
        camera_correction = transform_from(
            matrix_from_euler([-np.pi / 2, 0, -np.pi / 2], 2, 1, 0, False), [0, 0, 0]
        )

        detections = self.detector.detect(
            image, True, camera_parameters(image.shape), 0.339
        )

        estimates = []
        for detection in detections:
            # Calculate camera in tag coordinates
            tag_camera = concat(
                transform_from(detection.pose_R, detection.pose_t.ravel()),
                camera_correction,
            )
            camera_tag = invert_transform(tag_camera)

            # Calculate the tag position in global coordinates
            try:
                tag_global = self.fiducials[detection.tag_id]
            except KeyError:
                continue

            camera_global = concat(camera_tag, tag_global)
            rover_global = concat(rover_camera, camera_global)
            estimates.append(rover_global)

        return estimates


class SafeApriltagEstimator(ApriltagEstimator):
    """A velocity limited Apriltag Estimator."""

    def __init__(self, agent, linear=0.01, angular=10):
        """Creates a SafeApriltagEstimator object

        Args:
            linear: The linear velocity threshold (m/s)
            angular: The angular velocity threshold (deg/s)
        """
        super().__init__(agent)
        self.linear_limit = linear
        self.angular_limit = np.deg2rad(angular)

    def estimate(self, input_data) -> NDArray:
        """Iterates through all active cameras and averages all detections.

        Args:
            input_data: The input data dictionary provided by the simulation

        Returns:
            An average pose estimate from all detections. None if no detections or velocity exceeds threshold.
        """

        # Check the linear velocity
        if abs(self.agent.get_linear_speed()) > self.linear_limit:
            return None

        # Check the angular velocity
        if abs(self.agent.get_angular_speed()) > self.angular_limit:
            return None

        # Check the magnitude of the IMU
        # imu_data = self.agent.get_imu_data()

        # Run Normal Estimation
        return super().estimate(input_data)
