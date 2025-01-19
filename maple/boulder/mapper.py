from maple.utils import carla_to_pytransform

from pytransform3d.transformations import concat

import torch
from fastsam import FastSAM


class BoulderMapper:
    """Estimates the position of boulders around the rover."""

    agent: None
    left: None  # This is the carla.SensorPosition object
    right: None  # This is the carla.SensorPosition object

    def __init__(self, agent, left, right):
        """Intializer.

        Args:
            agent: The Agent instance
            left: The left camera instance (string or object)
            right: The right camera instance (string or object)
        """
        self.agent = agent

        self.left = None
        self.right = None

        # Look through all the camera objects in the agent's sensors and save the ones for the stereo pair
        # We do this since the objects themselves are used as keys in the input_data dict and to check they exist
        for key in agent.sensors().keys():
            if str(key) == str(left):
                self.left = key

            if str(key) == str(right):
                self.right = key

        # Confirm that the two required cameras exist
        if self.left is None or self.right is None:
            raise ValueError("Required cameras are not defined in the agent's sensors.")

        # Setup fastsam
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.fastsam = FastSAM("./FastSAM-x.pt")

    def map(self, input_data) -> list:
        """Estimates the position of boulders in the scene.,

        Args:
            input_data: The input data dictionary provided by the simulation

        Returns:
            A list of boulder positions in the rover frame.
        """

        # Get camera images
        try:
            left_image = input_data["Grayscale"][self.left]
            right_image = input_data["Grayscale"][self.right]
        except KeyError:
            raise ValueError("Required cameras have no data.")

        # Run the FastSAM pipeline to detect boulders (blobs in the scene)
        self._find_boulders(left_image)

        # Run the stereo vision pipeline to get a depth map of the image

        # Combine the boulder positions in the scene with the depth map to get the boulder coordinates

        # Calculator the boulder positions in the rover frame
        boulders_camera = []

        # Get the camera positions
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))

        # Return the list of boulders in the rover frame
        boulders_rover = [
            concat(boulder_camera, camera_rover) for boulder_camera in boulders_camera
        ]
        return boulders_rover

    def _depth_map(self, left, right):
        """Generate a depth map of the image"""
        # Use openCV to generate a depth map using stereo vision
        pass

    def _find_boulders(self, image):
        """Get the boulder locations and covariance in the image."""
        # Uses fastSAM to identify boulders in the image

        pass
