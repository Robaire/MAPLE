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
from pytransform3d.transformations import concat

from maple.geometry import lander
from maple.utils import camera_parameters, carla_to_pytransform


class Estimator:
    """Provides pose estimation using AprilTags on the lander."""

    agent: None
    fiducials: dict
    detector: Detector
    lander_transform: None  # Position of the lander in the global coordinate frame

    def __init__(self, agent):
        """Create the estimator.

        Args:
            agent: The Agent instance
        """

        self.agent = agent
        self.detector = Detector()
        self.fiducials = lander["fiducials"]

        # At mission start we can get the position of the rover in the global coordinate frame
        # and the position of the lander in the rover's coordinate frame
        # using these we can determine the position of the lander in the global coordinate frame
        global_to_rover = carla_to_pytransform(agent.get_initial_position())
        rover_to_lander = carla_to_pytransform(agent.get_initial_lander_position())

        self.lander_transform = concat(global_to_rover, rover_to_lander)

    # To calculate the rotational offset of the apriltags we can fit calculate the normal
    # of a plane from three of the coordinates in each group

    def estimate(self, image: NDArray[np.uint8]) -> list:
        """Provide a pose estimation using a photo of the lander.

        Args:
            image: A grayscale image of the lander

        Returns:
            A pose estimate of the camera in the global coordinate frame. None if no tags are found.
        """
        # TODO: Should we transform from the camera coordinate frame to the rover frame here?
        # Yes, we have the agent class so we should extract the current state of the rover

        detections = self.detector.detect(
            image, True, camera_parameters(image.shape), 0.339
        )

        if len(detections) == 0:
            return None

        for detection in detections:
            # calculate camera pose in world coordinates
            pass

        return None
