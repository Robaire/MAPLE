import importlib.resources

import cv2
import numpy as np
import torch
from fastsam import FastSAM, FastSAMPrompt
from numpy.typing import NDArray
from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import concat, transform_from

from maple.utils import camera_parameters, carla_to_pytransform


class BoulderMapper:
    """Estimates the position of boulders around the rover."""

    agent: None
    left: None  # This is the carla.SensorPosition object
    right: None  # This is the carla.SensorPosition object
    fastsam: None
    stereo: None

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
        elif torch.backends.mps.is_built():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Load the fastsam model weights from file
        with importlib.resources.path("resources", "FastSAM-x.pt") as fpath:
            self.fastsam = FastSAM(fpath)

        # Setup the stereo system
        window_size = 11
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=window_size,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            disp12MaxDiff=1,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
        )

    def __call__(self, input_data) -> list[NDArray]:
        """Equivalent to calling self.map()"""
        return self.map(input_data)

    def map(self, input_data) -> list[NDArray]:
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
        except (KeyError, TypeError):
            raise ValueError("Required cameras have no data.")

        # Run the FastSAM pipeline to detect boulders (blobs in the scene)
        centroids, _ = self._find_boulders(left_image)

        # Run the stereo vision pipeline to get a depth map of the image
        depth_map, _ = self._depth_map(left_image, right_image)

        # Combine the boulder positions in the scene with the depth map to get the boulder coordinates
        boulders_camera = self._get_positions(depth_map, centroids)

        # Get the camera position
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))

        # Calculate the boulder positions in the rover frame
        boulders_rover = [
            concat(boulder_camera, camera_rover) for boulder_camera in boulders_camera
        ]
        return boulders_rover

    def _get_positions(self, depth_map, centroids) -> list[NDArray]:
        """Calculate the position of objects in the left camera frame.

        Args:
            depth_map: The stereo depth map
            centroids: The object centroids in the left cameras frame

        Returns:
            The position of each centroid in the scene
        """

        focal_length, _, cx, cy = camera_parameters(depth_map.shape)

        boulders_camera = []

        window = 5
        for centroid in centroids:
            # Round to the nearest pixel coordinate
            u = round(centroid[0])
            v = round(centroid[1])

            # Find the average depth in window around centroid
            # Clamp the window to the edges of the image
            half_window = window // 2
            y_start = max(0, v - half_window)
            y_end = min(depth_map.shape[0], v + half_window + 1)
            x_start = max(0, u - half_window)
            x_end = min(depth_map.shape[1], u + half_window + 1)

            depth_window = depth_map[y_start:y_end, x_start:x_end]
            valid_depths = depth_window[depth_window > 0]

            # If there was no valid depth map around this centroid discard it
            if len(valid_depths) == 0:
                continue

            # Use median depth to be robust to outliers
            depth = np.median(valid_depths)

            # Get the position of the boulder in image coordinates
            x = ((u - cx) * depth) / focal_length
            y = ((v - cy) * depth) / focal_length
            z = depth

            # Discard boulders that are far away (> 5m)
            if z > 5:
                continue

            # TODO: The Z depth is to the surface of the boulder
            # We can estimate the size of the boulder using the variance
            # Assuming the boulders are roughly spherical we can add
            # approx 1/2 of the average variance to get the centerish

            boulder_image = transform_from(np.eye(3), [x, y, z])

            # Apply a rotation correction from the image coordinates to the camera coordinates
            # Z is out of the camera, X is to the right of the image, Y is down in the image
            image_camera = transform_from(
                matrix_from_euler([-np.pi / 2, 0, -np.pi / 2], 2, 1, 0, False),
                [0, 0, 0],
            )

            # Append it to the list
            boulders_camera.append(concat(boulder_image, image_camera))

        return boulders_camera

    def _depth_map(self, left_image, right_image) -> tuple[NDArray, NDArray]:
        """Generate a depth map of the scene

        Args:
            left_image: The image from the left camera
            right_image: The image from the right camera

        Returns:
            A tuple containing the depth map and the confidence map
        """

        # Check that the images are the same size
        if left_image.shape != right_image.shape:
            raise ValueError("Stereo mapping images must be the same size.")

        # Calculate the camera parameters
        focal_length, _, _, _ = camera_parameters(left_image.shape)

        # Get the camera positions in the robot frame
        left_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))
        right_rover = carla_to_pytransform(self.agent.get_camera_position(self.right))

        # Calculate the baseline between the cameras
        baseline = np.linalg.norm(left_rover[:3, 3] - right_rover[:3, 3])

        # Compute disparity map
        disparity = (
            self.stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        )

        # Calculate depth map
        depth_map = np.zeros_like(disparity)
        # Is this a boolean matrix used for indexing?
        valid_disparity = disparity > 0

        # Z = baseline * focal_length / disparity
        # TODO: What is this doing?
        depth_map[valid_disparity] = (baseline * focal_length) / disparity[
            valid_disparity
        ]

        # Computing confidence based on disparity and texture
        confidence_map = np.zeros_like(disparity)

        # Higher confidence for:
        # 1. Stronger disparity values
        # 2. Areas with good texture (using Sobel gradient magnitude)
        gradient_x = cv2.Sobel(left_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(left_image, cv2.CV_64F, 0, 1, ksize=3)
        texture_strength = np.sqrt(gradient_x**2 + gradient_y**2)

        # Normalize texture strength to 0-1
        texture_strength = cv2.normalize(texture_strength, None, 0, 1, cv2.NORM_MINMAX)

        # Combine disparity confidence and texture confidence
        disp_confidence = cv2.normalize(disparity, None, 0, 1, cv2.NORM_MINMAX)
        confidence_map = 0.7 * disp_confidence + 0.3 * texture_strength
        confidence_map[~valid_disparity] = 0

        return depth_map, confidence_map

    def _find_boulders(self, image) -> tuple[list[NDArray], list[NDArray]]:
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

        # TODO: Not sure whats going on here, but it generates segmentation masks
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
