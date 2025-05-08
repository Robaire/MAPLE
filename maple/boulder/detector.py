import importlib.resources

import cv2
import numpy as np
import torch
from fastsam import FastSAM, FastSAMPrompt
from numpy.typing import NDArray
from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import concat, transform_from
import random

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
        self.last_masks = None  # Store masks for later analysis
        self.last_depth_map = None  # Store the depth map for later analysis

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
        
        # Add shape filtering parameters for eigenvalue-based analysis
        self.MIN_EIGENVALUE_RATIO = 0.4  # Minimum ratio (smaller/larger eigenvalue)
        self.MIN_EIGENVALUE = 5.0  # Minimum eigenvalue to ensure the blob has some size
        self.MAX_EIGENVALUE = 2500  # Increased to handle larger boulders
        self.RELAXED_MAX_EIGENVALUE = 4000  # Allow higher eigenvalues for excellent shapes
        
        # For exceptional shapes, we can relax other criteria
        self.EXCELLENT_SHAPE_RATIO = 0.5  # Ratio threshold for excellent circular shapes
        self.STANDARD_INTENSITY = 80  # Standard intensity threshold
        self.RELAXED_INTENSITY = 30  # Relaxed intensity threshold for excellent shapes
        
        # Size thresholds for boulders
        self.MIN_AREA = 5  # Minimum area to be considered a boulder
        self.MAX_AREA = 5000  # Maximum area to be considered a boulder
        self.MIN_LARGE_AREA = 75  # Minimum area to be considered a large boulder
        self.MAX_DEPTH_FOR_LARGE = 4  # Maximum depth (meters) for large boulder classification
        
        # Add compactness thresholds from export_mask_data_final.py
        self.MIN_COMPACTNESS = 0.75  # Minimum compactness for boulder detection
        self.RELAXED_COMPACTNESS = 0.65  # Relaxed compactness for boulders with excellent shape
        
        # Add weighted scoring parameters from export_mask_data_final.py
        self.BOULDER_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence score to classify as boulder
        self.LARGE_BOULDER_CONFIDENCE_THRESHOLD = 0.80  # Higher threshold for large boulders
        
        # Feature weights for scoring (must sum to 1.0)
        self.FEATURE_WEIGHTS = {
            'shape': 0.40,  # Eigenvalue ratio weight
            'compactness': 0.35,  # Circularity/roundness weight
            'depth': 0.05,  # Depth variation weight (less reliable)
            'intensity': 0.15,  # Brightness/intensity weight
            'size': 0.05    # Size/area weight (less important)
        }

    def __call__(self, input_data) -> list[NDArray]:
        """Equivalent to calling self.map()"""
        return self.map(input_data)

    def map(self, input_data) -> list[NDArray]:
        """Estimates the position of boulders in the scene.,

        Args:
            input_data: The input data dictionary provided by the simulation

        Returns:
            A list of boulder transforms in the rover frame.
        """

        # Get camera images
        try:
            left_image = input_data["Grayscale"][self.left]
            right_image = input_data["Grayscale"][self.right]
        except (KeyError, TypeError):
            raise ValueError("Required cameras have no data.")

        # Run the FastSAM pipeline to detect boulders (blobs in the scene)
        centroids, covs, intensities, xy_ground, masks = self._find_boulders(left_image)

        # TODO: I recommend moving this to _find_boulders instead since some filtering is already being done there
        areas = []
        for cov in covs:
            det_cov = np.linalg.det(cov)
            if det_cov <= 0:
                # If determinant is <= 0, it's not a valid positive-definite covariance,
                # so you might skip or set area to NaN
                areas.append(float("nan"))
            else:
                # Area of the 1-sigma ellipse
                area = np.pi * np.sqrt(det_cov)
                areas.append(area)

        # print("sizes:", areas)

        # TODO: Here is a place to prune big/small segments. For now picking kinda arbitrary values:
        centroids_to_keep = []
        areas_to_keep = []
        xy_ground_to_keep = []
        intensities_to_keep = []
        masks_to_keep = []

        for centroid, area,intensity, xy_ground_pix, mask in zip(centroids, areas, intensities, xy_ground, masks):
            if self.MIN_AREA <= area <= self.MAX_AREA:
                try:
                    # Calculate eigenvalues of covariance matrix for shape analysis
                    cov_index = centroids.index(centroid)
                    cov_matrix = covs[cov_index]
                    eigenvalues = np.sort(np.linalg.eigvals(cov_matrix))[::-1]  # Sort largest to smallest
                    
                    # Ensure we have valid eigenvalues (exactly 2 positive values)
                    if eigenvalues.shape[0] == 2 and eigenvalues[1] > 0 and eigenvalues[0] > 0:
                        # Calculate ratio of smaller/larger eigenvalue (0-1 scale, closer to 1 = more circular)
                        ratio = eigenvalues[1] / eigenvalues[0]
                        
                        if ratio >= self.MIN_EIGENVALUE_RATIO and eigenvalues[1] >= self.MIN_EIGENVALUE:
                            # For excellent shapes (more circular), use relaxed intensity threshold
                            if ratio >= self.EXCELLENT_SHAPE_RATIO and eigenvalues[0] <= self.RELAXED_MAX_EIGENVALUE:
                                if intensity > self.RELAXED_INTENSITY:
                                    centroids_to_keep.append(centroid)
                                    areas_to_keep.append(area)
                                    xy_ground_to_keep.append(xy_ground_pix)
                                    intensities_to_keep.append(intensity)
                                    masks_to_keep.append(mask)
                            # For good but not excellent shapes, use standard intensity threshold
                            elif intensity > self.STANDARD_INTENSITY and eigenvalues[0] <= self.MAX_EIGENVALUE:
                                centroids_to_keep.append(centroid)
                                areas_to_keep.append(area)
                                xy_ground_to_keep.append(xy_ground_pix)
                                intensities_to_keep.append(intensity)
                                masks_to_keep.append(mask)
                except Exception:
                    # Fall back to original intensity threshold if eigenvalue calculation fails
                    if intensity > 100:
                        centroids_to_keep.append(centroid)
                        areas_to_keep.append(area)
                        xy_ground_to_keep.append(xy_ground_pix)
                        intensities_to_keep.append(intensity)
                        masks_to_keep.append(mask)

        # Run the stereo vision pipeline to get a depth map of the image
        depth_map, _ = self._depth_map(left_image, right_image)

        # Combine the boulder positions in the scene with the depth map to get the boulder coordinates
        boulders_camera = self._get_positions(depth_map, centroids_to_keep)
        ground_pix_camera = self._get_positions(depth_map, xy_ground_to_keep)

        # Get the camera position
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))

        # Calculate the boulder positions in the rover frame
        boulders_rover = [
            concat(boulder_camera, camera_rover) for boulder_camera in boulders_camera
        ]
        ground_pix_rover = [
            concat(ground_xy_camera, camera_rover) for ground_xy_camera in ground_pix_camera
        ]

        self.last_boulders = boulders_rover

        # Adjust the boulder "areas" for depth
        adjusted_areas = []
        for centroid, area in zip(centroids_to_keep, areas_to_keep):
            adjusted_area = self._adjust_area_for_depth(depth_map, area, centroid)
            adjusted_areas.append(adjusted_area)
        self.last_areas = adjusted_areas

        # Store the segmentation masks and depth map for later analysis
        self.last_masks = masks_to_keep
        self.last_depth_map = depth_map

        # Return both the boulder positions and the random points
        return boulders_rover, ground_pix_rover

    def get_large_boulders(self, min_diameter_m=10, min_compactness=0.6, min_depth_std=0.1) -> list[NDArray]:
        """Return large, round, bulging boulders."""
        large_boulders = []
        for boulder, area, mask in zip(self.last_boulders, self.last_areas, self.last_masks):
            boulder_depth = boulder[2, 3]
            if boulder_depth <= self.MAX_DEPTH_FOR_LARGE:
                radius_m = np.sqrt(area / np.pi)
                diameter_m = radius_m * 2

                if diameter_m >= min_diameter_m:
                    # New checks:
                    compactness = self._compute_compactness(mask)
                    depth_std = self._compute_depth_std(self.last_depth_map, mask)

                    if compactness >= min_compactness and depth_std >= min_depth_std:
                        large_boulders.append(boulder)
        return large_boulders
    
    def _compute_compactness(self, mask: NDArray) -> float:
        """Calculate the compactness (circularity) of a mask.
        
        Args:
            mask: Binary mask of the object
            
        Returns:
            Compactness value between 0-1 (1 = perfect circle)
        """
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        cnt = contours[0]
        area = np.sum(mask)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return 0.0
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        return min(compactness, 1.0)  # Cap at 1.0 for perfect circles

    def _compute_depth_std(self, depth_map: NDArray, mask: NDArray) -> float:
        """Calculate the standard deviation of depth values within a mask.
        
        Args:
            depth_map: Depth map from stereo processing
            mask: Binary mask of the object
            
        Returns:
            Standard deviation of depth values (higher = more 3D structure)
        """
        valid_depths = depth_map[mask == 1]
        valid_depths = valid_depths[valid_depths > 0]  # remove invalid depths
        if len(valid_depths) == 0:
            return 0.0
        return np.std(valid_depths)

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

        focal_length, _, cx, cy = camera_parameters(depth_map.shape)

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

    def _find_boulders(self, image) -> tuple[list[NDArray], list[NDArray], list[float], list[NDArray], list[NDArray]]:
        """Get the boulder locations, covariance, and average pixel intensity in the image.

        Args:
            image: The grayscale image to search for boulders in

        Returns:
            A tuple containing the mean and covariance lists, intensities, bottom pixels, and masks
        """

        # Run fastSAM on the input image
        results = self.fastsam(
            np.stack((image,) * 3, axis=-1),  # The image needs three channels
            device=self.device,
            retina_masks=True,
            imgsz=image.shape[1],
            conf=0.5,
            iou=0.9,
            verbose=False,
        )

        # Generate segmentation masks safely
        prompt = FastSAMPrompt(image, results, device=self.device)
        segmentation_masks = prompt.everything_prompt()

        # Check if output is a tensor or not
        if isinstance(segmentation_masks, list):
            # No detections found, return an empty array
            segmentation_masks = np.zeros(
                (0, image.shape[0], image.shape[1]), dtype=np.uint8
            )
        else:
            segmentation_masks = segmentation_masks.cpu().numpy()

        # Check if anything was segmented
        if len(segmentation_masks) == 0:
            return [], [], [], [], []

        means = []
        covs = []
        bottom_pixes = []
        avg_intensities = []
        masks_to_return = []  # Masks that pass all filters
        
        for mask in segmentation_masks:
            # Compute centroid and covariance
            mean, cov, bottom_pix = self._compute_blob_mean_and_covariance(mask)

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
            
            # Calculate the compactness (circularity) of the mask
            try:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = contours[0]
                area = np.sum(mask)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                    
                compactness = (4 * np.pi * area) / (perimeter ** 2)
                # Cap compactness at 1.0 and validate
                if compactness > 1.5:  # Allow slight margin for computational errors
                    compactness = 0.0
                else:
                    compactness = min(compactness, 1.0)
            except Exception:
                # If compactness calculation fails, assign a low value
                compactness = 0.0
            
            # Ensure we have a valid covariance matrix for eigenvalue calculation
            if cov.size != 4 or np.linalg.det(cov) <= 0:
                continue
                
            # Calculate eigenvalues for shape analysis
            try:
                eigenvalues = np.sort(np.linalg.eigvals(cov))[::-1]  # Sort largest to smallest
                if eigenvalues.shape[0] != 2 or eigenvalues[1] <= 0 or eigenvalues[0] <= 0:
                    continue
                    
                # Calculate ratio of smaller/larger eigenvalue (0-1 scale, closer to 1 = more circular)
                eigenvalue_ratio = eigenvalues[1] / eigenvalues[0]
                
                # Calculate area (from covariance determinant)
                area = np.pi * np.sqrt(np.linalg.det(cov))
                
                # Apply initial filtering based on eigenvalues and area
                if (eigenvalues[1] < self.MIN_EIGENVALUE or 
                    area < self.MIN_AREA or 
                    area > self.MAX_AREA):
                    continue
                
                # Calculate feature scores for confidence calculation
                # Shape score (eigenvalue ratio - higher is better for boulders)
                shape_score = min(eigenvalue_ratio / self.EXCELLENT_SHAPE_RATIO, 1.0)
                
                # Compactness score (1.0 is a perfect circle)
                compactness_score = compactness
                
                # Intensity score - normalize based on thresholds
                if avg_pixel_value < self.RELAXED_INTENSITY:
                    intensity_score = 0.0
                elif avg_pixel_value < self.STANDARD_INTENSITY:
                    # Linear scaling between relaxed and standard threshold
                    intensity_score = (avg_pixel_value - self.RELAXED_INTENSITY) / \
                                     (self.STANDARD_INTENSITY - self.RELAXED_INTENSITY) * 0.5
                else:
                    # Higher than standard gets better score
                    intensity_score = 0.5 + min((avg_pixel_value - self.STANDARD_INTENSITY) / 100, 0.5)
                
                # Size score (bigger is better, up to a point)
                size_score = min(area / self.MIN_LARGE_AREA, 1.0)
                
                # Calculate overall confidence score without using depth information
                # Note: We redistribute the depth weight to shape and compactness
                adjusted_weights = {
                    'shape': self.FEATURE_WEIGHTS['shape'] + self.FEATURE_WEIGHTS['depth'] * 0.6,
                    'compactness': self.FEATURE_WEIGHTS['compactness'] + self.FEATURE_WEIGHTS['depth'] * 0.4,
                    'intensity': self.FEATURE_WEIGHTS['intensity'],
                    'size': self.FEATURE_WEIGHTS['size']
                }
                
                confidence_score = (
                    adjusted_weights['shape'] * shape_score +
                    adjusted_weights['compactness'] * compactness_score +
                    adjusted_weights['intensity'] * intensity_score +
                    adjusted_weights['size'] * size_score
                )
                
                # Now apply filtering logic - using three paths for boulder detection
                is_boulder = False
                
                # For excellent shapes (more circular), use relaxed intensity threshold
                if (eigenvalue_ratio >= self.EXCELLENT_SHAPE_RATIO and 
                    eigenvalues[0] <= self.RELAXED_MAX_EIGENVALUE and
                    compactness >= self.RELAXED_COMPACTNESS and
                    avg_pixel_value >= self.RELAXED_INTENSITY):
                    is_boulder = True
                
                # For good but not excellent shapes, use standard intensity threshold
                elif (eigenvalue_ratio >= self.MIN_EIGENVALUE_RATIO and 
                      eigenvalues[0] <= self.MAX_EIGENVALUE and
                      compactness >= self.MIN_COMPACTNESS and
                      avg_pixel_value >= self.STANDARD_INTENSITY):
                    is_boulder = True
                
                # Also accept high confidence boulders based on the scoring system
                elif confidence_score >= self.BOULDER_CONFIDENCE_THRESHOLD:
                    is_boulder = True
                    
                if is_boulder:
                    means.append(mean)
                    covs.append(cov)
                    bottom_pixes.append(bottom_pix)
                    avg_intensities.append(avg_pixel_value)
                    masks_to_return.append(mask)
            
            except Exception:
                # Fall back to original intensity threshold if eigenvalue calculation fails
                if avg_pixel_value > 100:
                    means.append(mean)
                    covs.append(cov)
                    bottom_pixes.append(bottom_pix)
                    avg_intensities.append(avg_pixel_value)
                    masks_to_return.append(mask)

        return means, covs, avg_intensities, bottom_pixes, masks_to_return

    @staticmethod
    def _compute_blob_mean_and_covariance(binary_image) -> tuple[NDArray, NDArray, NDArray]:
        """Finds the mean, covariance, and bottom-most pixel of a segmentation mask.

        Args:
            binary_image: The segmentation mask

        Returns:
            The mean [x, y], covariance matrix, and bottom-most pixel [x, y] of the blob in pixel coordinates
        """

        # Create a grid of pixel coordinates.
        y, x = np.indices(binary_image.shape)

        # Threshold the binary image to isolate the blob.
        blob_pixels = (binary_image > 0).astype(int)

        # Get all coordinates of pixels in the blob
        y_coords = y[blob_pixels == 1]
        x_coords = x[blob_pixels == 1]
        
        # Compute the mean of pixel coordinates.
        mean_x = np.mean(x_coords)
        mean_y = np.mean(y_coords)
        mean = np.array([mean_x, mean_y])

        # Stack pixel coordinates to compute covariance using Scipy's cov function.
        pixel_coordinates = np.vstack((x_coords, y_coords))

        # Compute the covariance matrix using numpy's covariance function
        covariance_matrix = np.cov(pixel_coordinates)
        
        # Find the bottom-most pixel (pixel with largest y-coordinate)
        if len(y_coords) > 0:
            max_y_index = np.argmax(y_coords)
            bottom_pixel = np.array([x_coords[max_y_index], y_coords[max_y_index]])
        else:
            # If no blob pixels are found, return zeros
            bottom_pixel = np.array([0, 0])

        return mean, covariance_matrix, bottom_pixel

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