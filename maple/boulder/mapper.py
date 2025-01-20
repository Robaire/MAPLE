import os

import numpy as np
import torch
from fastsam import FastSAM, FastSAMPrompt
from numpy.typing import NDArray
from pytransform3d.transformations import concat

from maple.utils import carla_to_pytransform


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
        if torch.backends.mps.is_built():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Check that the model weights are present
        if os.path.isfile("./FastSAM-x.pt"):
            self.fastsam = FastSAM("./FastSAM-x.pt")
        else:
            raise FileNotFoundError("FastSAM-x.pt not found.")

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
        (means, covariances) = self._find_boulders(left_image)

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

    def _find_boulders(self, image) -> tuple[list, list]:
        """Get the boulder locations and covariance in the image.

        Args:
            image: The image to search for boulders in

        Returns:
            A tuple containing the mean and covariance lists
        """

        # Run fastSAM on the input image
        results = self.fastsam(
            np.stack((image,) * 3, axis=-1),  # The image needs three channels
            device=self.device,
            retina_masks=True,
            # imgsz=1080,  # TODO: Where does this value come from? Should it just be the image width?
            imgsz=image.shape[1],
            conf=0.5,
            iou=0.9,
        )

        # TODO: Not sure whats going on here
        segmentation_masks = (
            FastSAMPrompt(image, results, device=self.device)
            .everything_prompt()
            .cpu()
            .numpy()
        )

        # Check if anything was segmented
        if len(segmentation_masks) == 0:
            return []

        means = []
        covs = []

        # Iterate over every mask and generate blobs
        # TODO: Add logic to prune objects (too large, too small, on boundary, etc...)
        for mask in segmentation_masks:
            # Compute the blob centroid and covariance from the mask
            mean, cov = self._compute_blob_mean_and_covariance(mask)

            print(f"Mask {mask.shape} | Mean: {mean} | Cov: {cov}")

            # Discard any blobs in the top half of the image
            if mean[1] < image.shape[0] / 2:
                continue

            # Discard any blobs on the left and right edges of the image
            margin = image.shape[1] * 0.05  # 5% on either side
            if mean[0] < margin or mean[0] > image.shape[1] - margin:
                continue

            # Append to lists
            means.append(mean)
            covs.append(cov)

        return means, covs

    @staticmethod
    def _compute_blob_mean_and_covariance(binary_image) -> tuple[NDArray, NDArray]:
        """Finds the mean and covariance of a segmentation mask.

        Args:
            binary_image: The segmentation mask

        Returns:
            The mean [x, y] and covariance matrix of the blob in pixel coordinates
        """

        # Create a grid of pixel coordinates.
        y, x = np.indices(binary_image.shape)

        # Threshold the binary image to isolate the blob.
        blob_pixels = (binary_image > 0).astype(int)

        # Compute the mean of pixel coordinates.
        mean_x = np.mean(x[blob_pixels == 1])
        mean_y = np.mean(y[blob_pixels == 1])
        mean = np.array([mean_x, mean_y])

        # Stack pixel coordinates to compute covariance using Scipy's cov function.
        pixel_coordinates = np.vstack((x[blob_pixels == 1], y[blob_pixels == 1]))

        # Compute the covariance matrix using numpy's covariance function
        covariance_matrix = np.cov(pixel_coordinates)

        return mean, covariance_matrix
