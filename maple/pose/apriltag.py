# We can use apriltags on the lander to estimate our pose
# We need to:
#   - capture the geometry of the tags on the lander (from geometry.json)
#   - combine pose estimations from multiple detections (average? maybe with accuracy threshold)
#   - account for the potential offset of the lander from the global world coordinates

# NOTE: We get the rovers initial pose from the simulator so we can use this to test if we are getting reasonable results
# from looking at the apriltags on the lander (just in case something like the tagsize is wrong)
# NOTE: Use pytransform3d for transformations

import numpy as np
from numpy.typing import NDArray
from pyapriltags import Detector
from pytransform3d.transformations import concat, invert_transform, transform_from
from pytransform3d.rotations import matrix_from_euler

from maple import geometry
from maple.utils import camera_parameters, carla_to_pytransform


class Estimator:
    """Provides pose estimation using AprilTags on the lander."""

    agent: None
    fiducials: dict
    detector: Detector
    global_to_lander: NDArray

    def __init__(self, agent):
        """Create the estimator.

        Args:
            agent: The Agent instance
        """

        # Check that the agent has fiducials enabled
        if not agent.use_fiducials():
            raise ValueError("agent must have fiducials enabled")

        self.agent = agent
        self.detector = Detector()

        # At mission start we can get the position of the rover in the global coordinate frame
        # and the position of the lander in the rover's coordinate frame
        # using these we can determine the position of the lander in the global coordinate frame
        global_to_rover = carla_to_pytransform(agent.get_initial_position())
        rover_to_lander = carla_to_pytransform(agent.get_initial_lander_position())
        self.global_to_lander = concat(global_to_rover, rover_to_lander)

        # We want a dict of tags (key: id, value: lander_to_tag)
        self.fiducials = {}
        for group, tag_group in geometry.lander["fiducials"].items():
            tag_rotations = {"a": -45, "b": 45, "c": 135, "d": -135}  # [degrees]
            rotation = matrix_from_euler(
                [0, 0, np.deg2rad(tag_rotations[group])], 0, 1, 2, False
            )  # TODO: This angle might need to be reversed

            # For each tag determine its transform from the lander
            for tag in tag_group.values():
                translation = [tag["x"], tag["y"], tag["z"]]
                lander_to_tag = transform_from(rotation, translation)

                # Add this tag to the dict
                self.fiducials[tag["id"]] = lander_to_tag

        # The charging antenna locator is a special case we add
        # TODO:

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

        if not estimates:
            return None

        # Average all the estimates
        # Look at scipy rotation averaging
        # For position we should probably just average each [x, y, z]
        average = np.eye(4, 4)  # TODO: Implement
        return average

    def _global_to_tag(self, id) -> NDArray:
        """Get the transform of a tag in the global frame.
        Args:
            id: the tag id

        Returns:
            The transform from the global frame to the tag frame
        """

        # TODO: lookup the tag position using geometry
        lander_to_tag = np.eye(4, 4)

        # TODO: Figure out the correct concat ordering
        return concat(self.global_to_lander, lander_to_tag)

    def _estimate_image(self, camera, image: NDArray[np.uint8]) -> list:
        """Process a single image to get a pose estimate.

        Args:
            camera: The camera the image came from
            image: A grayscale image to search for tags in

        Returns:
            A list of pose estimates for the camera in the global coordinate frame.
        """

        # camera is something like: carla.SensorPosition.Front
        rover_to_camera = carla_to_pytransform(self.agent.get_camera_position(camera))
        camera_to_rover = invert_transform(rover_to_camera)

        detections = self.detector.detect(
            image, True, camera_parameters(image.shape), 0.339
        )

        estimates = []
        for detection in detections:
            # Calculate camera in tag coordinates
            # TODO: Beware of the apriltag coordinate frame
            # Z is out of the camera
            # X is to the right of the image
            # Y is down in the image
            tag_to_camera = np.eye(4, 4)  # TODO: Implement

            # Calculate the tag position in global coordinates
            global_to_tag = self._global_to_tag(detection.tag_id)

            # Concat transforms to get rover in world coordinates
            global_to_rover = concat(
                concat(global_to_tag, tag_to_camera), camera_to_rover
            )
            estimates.append(global_to_rover)

        return estimates
