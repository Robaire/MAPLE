import importlib.resources
import cv2
import numpy as np
import torch
from fastsam import FastSAM, FastSAMPrompt
from numpy.typing import NDArray
from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import concat, transform_from
import random
import os
import csv
import matplotlib.pyplot as plt

from maple.utils import camera_parameters, carla_to_pytransform


class BoulderDetector:
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

        # Last mapped boulder positions and boulder areas
        self.last_boulders = None
        self.last_areas = None

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
        """Get boulder detections"""
        return self.map(input_data)

    def map(self, input_data) -> list[NDArray]:
        """Estimates the position of boulders in the scene and saves ground points.

        Args:
            input_data: The input data dictionary provided by the simulation

        Returns:
            list[NDArray]: Boulder positions as 4x4 transforms in rover frame

        Note: Ground points are automatically saved to CSV/dat files and visualized
        """

        # Get camera images
        try:
            left_image = input_data["Grayscale"][self.left]
            right_image = input_data["Grayscale"][self.right]
        except (KeyError, TypeError):
            raise ValueError("Required cameras have no data.")

        # Run the FastSAM pipeline to detect boulders (blobs in the scene)
        centroids, covs, intensities = self._find_boulders(left_image)

        # TODO: I recommend moving this to _find_boulders instead since some filtering is already being done there
        areas = []
        for cov in covs:
            det_cov = np.linalg.det(cov)
            if det_cov <= 0:
                areas.append(float("nan"))
            else:
                area = np.pi * np.sqrt(det_cov)
                areas.append(area)

        # print("sizes:", areas)

        # Apply area filtering
        MIN_AREA = 50
        MAX_AREA = 2500
        elongation_threshold = 10
        pixel_intensity_threshold = 50

        centroids_to_keep = []
        areas_to_keep = []
        covs_to_keep = []

        for centroid, area, cov, intensity in zip(centroids, areas, covs, intensities):
            # print(pixel_intensities)
            eigen_vals = np.linalg.eigvals(cov)
            elongated = (eigen_vals.max() / eigen_vals.min()) > elongation_threshold
            bright = intensity > pixel_intensity_threshold
            if MIN_AREA <= area <= MAX_AREA and not elongated and bright:
                centroids_to_keep.append(centroid)
                covs_to_keep.append(cov)
                areas_to_keep.append(area)

        # Run the stereo vision pipeline to get a depth map of the image
        depth_map, _ = self._depth_map(left_image, right_image)
        if depth_map is None:
            print("DEBUG: Depth map generation failed")
            return []

        # Retrieve shape of depth map (assumes depth_map is 2D: height x width)
        height, width = depth_map.shape
        # print(f"DEBUG: Generated depth map with shape {depth_map.shape}")

        # Compute the start row (3/4 down the image)
        start_row = height * 3 // 4  # integer index for bottom 1/4

        # print(f"DEBUG: Searching for points in depth map of shape {depth_map.shape}")

        # Generate random points in the bottom 1/4, with distance threshold
        random_centroids = []
        max_distance = 15.0  # Maximum distance threshold in meters
        min_distance = 0.5  # Minimum distance threshold to filter invalid measurements

        for _ in range(20):
            x = random.randint(0, width - 1)
            y = random.randint(start_row, height - 1)
            depth = depth_map[y, x]
            if (
                min_distance < depth < max_distance
            ):  # Only include valid points within threshold
                random_centroids.append((x, y))

        # print(f"DEBUG: Found {len(random_centroids)} valid points with depth between {min_distance} and {max_distance} meters")

        # Combine the boulder positions in the scene with the depth map to get the boulder coordinates
        boulders_camera = self._get_positions(depth_map, centroids_to_keep)

        # Get the camera position
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))

        # Calculate the boulder positions in the rover frame
        boulders_rover = [
            concat(boulder_camera, camera_rover) for boulder_camera in boulders_camera
        ]
        self.last_boulders = boulders_rover

        # Adjust the boulder "areas" for depth
        adjusted_areas = []
        for centroid, area in zip(centroids_to_keep, areas_to_keep):
            adjusted_area = self._adjust_area_for_depth(depth_map, area, centroid)
            adjusted_areas.append(adjusted_area)
        self.last_areas = adjusted_areas

        # Retrieve shape of depth map (assumes depth_map is 2D: height x width)
        height, width = depth_map.shape

        # Compute the start row (2/3 down the image)
        start_row = height * 3 // 4  # integer index for bottom 1/3

        # Generate 20 random (x, y) pixel coordinates in the bottom 1/3
        # TODO: Do this more intelligently....
        random_centroids = []
        for _ in range(20):
            x = random.randint(0, width - 1)
            y = random.randint(start_row, height - 1)
            random_centroids.append((x, y))

        # Convert these random centroids into 3D camera-frame coordinates
        random_points_camera = self._get_positions(depth_map, random_centroids)

        # Transform points from camera to rover frame
        random_points_rover = [
            concat(point_camera, camera_rover) for point_camera in random_points_camera
        ]

        # Get current rover pose in global frame
        rover_global = carla_to_pytransform(self.agent.get_transform())

        # Transform points from rover frame to global frame
        random_points_global = [
            concat(point_rover, rover_global) for point_rover in random_points_rover
        ]

        # Save depth points to agent's point cloud data
        if hasattr(self.agent, "point_cloud_data"):
            # print("DEBUG: Starting point cloud save process")
            if not os.path.exists(self.agent.point_cloud_dir):
                os.makedirs(self.agent.point_cloud_dir)
                # print(f"DEBUG: Created point cloud directory at {self.agent.point_cloud_dir}")

            for point in random_points_global:
                self.agent.point_cloud_data["points"].append(
                    {
                        "frame": self.agent.frame,
                        "point": [point[0, 3], point[1, 3], point[2, 3]],
                        "confidence": 0.6,
                        "source": "depth",
                    }
                )
            # print(f"DEBUG: Added {len(random_points_global)} points to in-memory point cloud")

            # Write to CSV
            csv_path = os.path.join(self.agent.point_cloud_dir, "point_cloud_data.csv")
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for point in random_points_global:
                    writer.writerow(
                        [
                            self.agent.frame,
                            point[0, 3],
                            point[1, 3],
                            point[2, 3],
                            0.6,
                            "depth",
                        ]
                    )
            # print(f"DEBUG: Wrote {len(random_points_global)} points to CSV")

            # Create visualization
            viz_path = os.path.join(
                self.agent.point_cloud_dir, f"frame_{self.agent.frame:04d}"
            )
            self._visualize_point_cloud(None, viz_path)
            # print(f"DEBUG: Created visualization at {viz_path}")

        return boulders_rover, random_points_global

    def get_large_boulders(self, min_area: float = 40) -> list[NDArray]:
        """Get the last mapped boulder positions with adjusted area larger than min_area.

        Returns:
            A list of boulder positions
        """
        return [
            boulder
            for boulder, area in zip(self.last_boulders, self.last_areas)
            if area > min_area
        ]

    def get_boulder_sizes(self, min_area: float = 0.1) -> list[NDArray]:
        """Get the last mapped boulder positions with adjusted area larger than min_area.

        Returns:
            A list of boulder positions
        """
        return [
            area
            for boulder, area in zip(self.last_boulders, self.last_areas)
            if area > min_area
        ]

    def get_last_boulders(self) -> list[NDArray]:
        """Get the last mapped boulder positions.

        Returns:
            A list of boulder positions
        """
        return self.last_boulders

    def get_last_areas(self) -> list[float]:
        """Get the last mapped boulder areas.

        Returns:
            A list of boulder areas
        """
        return self.last_areas

    def _get_positions(self, depth_map, centroids) -> list[NDArray]:
        """Calculate the position of objects in the left camera frame.

        Args:
            depth_map: The stereo depth map
            centroids: The object centroids in the left cameras frame

        Returns:
            The position of each centroid in the scene
        """
        # print(f"DEBUG: Converting {len(centroids)} points to 3D coordinates")

        focal_length, _, cx, cy = camera_parameters(depth_map.shape)
        # print(f"DEBUG: Camera parameters - focal_length: {focal_length}, cx: {cx}, cy: {cy}")

        boulders_camera = []

        for centroid in centroids:
            depth = self._get_depth(depth_map, centroid)
            if depth == 0:
                continue

            u = round(centroid[0])
            v = round(centroid[1])

            # Get the position of the boulder in image coordinates
            x = ((u - cx) * depth) / focal_length
            y = ((v - cy) * depth) / focal_length
            z = depth

            # Discard boulders that are far away (> 5m)
            if z > 5:
                # print(f"DEBUG: Discarding point at ({x:.2f}, {y:.2f}, {z:.2f}) - too far")
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

        # print(f"DEBUG: Generated {len(boulders_camera)} valid 3D points")
        return boulders_camera

    def _get_depth(self, depth_map, centroid):
        """Get the depth of a small region around a centroid.

        Args:
            depth_map: The stereo depth map
            centroid: The centroid of the boulder

        Returns:
            The depth of the boulder
        """
        window = 5
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
            return 0

        # Use median depth to be robust to outliers
        depth = np.median(valid_depths)
        return depth

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

    # def _find_boulders(self, image) -> tuple[list[NDArray], list[NDArray]]:
    #     """Get the boulder locations and covariance in the image.

    #     Args:
    #         image: The image to search for boulders in

    #     Returns:
    #         A tuple containing the mean and covariance lists
    #     """

    #     # Run fastSAM on the input image
    #     results = self.fastsam(
    #         np.stack((image,) * 3, axis=-1),  # The image needs three channels
    #         device=self.device,
    #         retina_masks=True,
    #         # imgsz=1080,  # TODO: Where does this value come from? Should it just be the image width?
    #         imgsz=image.shape[1],
    #         conf=0.5,
    #         iou=0.9,
    #         verbose=False,
    #     )

    #     # TODO: Not sure whats going on here, but it generates segmentation masks
    #     segmentation_masks = (
    #         FastSAMPrompt(image, results, device=self.device)
    #         .everything_prompt()
    #         .cpu()
    #         .numpy()
    #     )

    #     # Check if anything was segmented
    #     if len(segmentation_masks) == 0:
    #         return []

    #     means = []
    #     covs = []

    #     # Iterate over every mask and generate blobs
    #     # TODO: Add logic to prune objects (too large, too small, on boundary, etc...)
    #     for mask in segmentation_masks:
    #         # Compute the blob centroid and covariance from the mask
    #         mean, cov = self._compute_blob_mean_and_covariance(mask)

    #         # Discard any blobs in the top half of the image
    #         if mean[1] < image.shape[0] / 2:
    #             continue

    #         # Discard any blobs on the left and right edges of the image
    #         margin = image.shape[1] * 0.05  # 5% on either side
    #         if mean[0] < margin or mean[0] > image.shape[1] - margin:
    #             continue

    #         # Append to lists
    #         means.append(mean)
    #         covs.append(cov)

    #     return means, covs

    def _find_boulders(
        self, image
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        """Get the boulder locations, covariance, and average pixel intensity in the image.

        Args:
            image: The grayscale image to search for boulders in
            image: The grayscale image to search for boulders in

        Returns:
            A tuple containing the means, covariances, and average intensities of each boulder region.
            A tuple containing the means, covariances, and average intensities of each boulder region.
        """

        # Run fastSAM on the input image (requires 3 channels, so we replicate the single channel).
        results = self.fastsam(
            np.stack((image,) * 3, axis=-1),
            device=self.device,
            retina_masks=True,
            imgsz=image.shape[1],
            conf=0.5,
            iou=0.9,
            verbose=False,
        )

        # Generate segmentation masks
        segmentation_masks = (
            FastSAMPrompt(image, results, device=self.device)
            .everything_prompt()
            .cpu()
            .numpy()
        )

        # If nothing was segmented, return empty lists
        if len(segmentation_masks) == 0:
            return [], [], []

        means = []
        covs = []
        avg_intensities = []

        # Iterate over each mask and extract relevant data
        for mask in segmentation_masks:
            # Compute centroid and covariance
            mean, cov = self._compute_blob_mean_and_covariance(mask)

            # Discard any blobs in the top third of the image
            if mean[1] < image.shape[0] / 3:
                continue

            # Discard any blobs on the left and right edges of the image (5% margin)
            margin = image.shape[1] * 0.05
            if mean[0] < margin or mean[0] > image.shape[1] - margin:
                continue

            # Calculate average pixel intensity for the region.
            # Assuming 'mask' is a binary mask with 1s for the boulder area.
            avg_pixel_value = np.mean(image[mask == 1])

            # Append results
            means.append(mean)
            covs.append(cov)
            avg_intensities.append(avg_pixel_value)

        return means, covs, avg_intensities

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

    @staticmethod
    def _rover_to_global(boulders_rover: list, rover_global: NDArray) -> list:
        """Converts the boulder locations from the rover frame to the global frame.

        Args:
            boulders_rover: A list of transforms representing points on the surface of boulders in the rover frame
            rover_global: The global transform of the rover

        Returns:
            A list of transforms representing points on the surface of boulders in the global frame
        """
        boulders_global = [
            concat(boulder_rover, rover_global) for boulder_rover in boulders_rover
        ]
        return boulders_global

    def _adjust_area_for_depth(
        self, depth_map: NDArray, pixel_area: float, centroid: tuple[float, float]
    ) -> float:
        """Adjusts the pixel area based on depth to estimate actual object size.
        If depth to pixel is unknown, it is assumed to be 1m, and size is likely underestimated.

        Args:
            pixel_area: The area in pixels from the segmentation mask
            depth_map: The stereo depth map
            centroid: The (u,v) pixel coordinates of the object centroid

        Returns:
            The adjusted area estimate accounting for perspective projection
        """
        # Scale up pixel area by 10000 to avoid floating point errors
        pixel_area = pixel_area * 10000
        # Get camera parameters
        focal_length, _, cx, cy = camera_parameters(depth_map.shape)

        depth = self._get_depth(depth_map, centroid)
        # If depth mapping fails, assume the object is 1m away (will likely underestimate size)
        if depth == 0:
            depth = 1.0

        # The scaling factor is proportional to depth squared
        # This accounts for perspective projection where apparent size decreases with distance
        depth_scaling = (depth**2) / (focal_length * focal_length)

        # Adjust the pixel area using the depth scaling
        adjusted_area = pixel_area * depth_scaling

        return adjusted_area

    def _visualize_point_cloud(self, depth_points, viz_dir):
        """Create 3D visualization of point cloud for current frame."""
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Plot all points from agent's point cloud data
        if hasattr(self.agent, "point_cloud_data"):
            for point_data in self.agent.point_cloud_data["points"]:
                point = point_data["point"]
                source = point_data["source"]
                if source == "lander":
                    ax.scatter(
                        point[0],
                        point[1],
                        point[2],
                        c="green",
                        marker="s",
                        s=100,
                        label="_nolegend_",
                    )
                elif source == "rover":
                    ax.scatter(
                        point[0],
                        point[1],
                        point[2],
                        c="blue",
                        marker="o",
                        s=50,
                        label="_nolegend_",
                    )
                elif source == "depth":
                    ax.scatter(
                        point[0],
                        point[1],
                        point[2],
                        c="red",
                        marker=".",
                        s=20,
                        label="_nolegend_",
                    )

        # Add legend
        ax.scatter([], [], c="green", marker="s", s=100, label="Lander Points")
        ax.scatter([], [], c="blue", marker="o", s=50, label="Rover Points")
        ax.scatter([], [], c="red", marker=".", s=20, label="Depth Points")
        ax.legend()

        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Point Cloud - Frame {self.agent.frame}")

        # Save plot
        plt.savefig(
            os.path.join(viz_dir, f"point_cloud_frame_{self.agent.frame:04d}.png")
        )
        plt.close()
