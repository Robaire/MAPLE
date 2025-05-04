from lac_data import PlaybackAgent
from maple.utils import carla_to_pytransform, camera_parameters
import numpy as np
import os
import cv2
from pytransform3d.transformations import concat, transform_from
from pytransform3d.rotations import matrix_from_euler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from fastsam import FastSAM, FastSAMPrompt
import importlib.resources
import sys

def load_ground_truth_map(filename):
    """Load the ground truth map from .dat file."""
    map_data = np.load(filename, allow_pickle=True)
    print(f"Map data type: {type(map_data)}")
    print(f"Map data shape: {map_data.shape}")
    print(f"Map data dtype: {map_data.dtype}")
    return map_data

def get_map_z(ground_truth_map, x, y):
    """Get the z value from the map at the given x,y coordinates.
    Map is split into 15cm x 15cm grid.
    Map coordinates range from -13.425m to 13.425m in both x and y."""
    # Convert x,y to grid coordinates
    grid_size = 0.15  # 15cm
    map_min = -13.425  # meters
    
    # Offset coordinates to handle negative values
    x_offset = x - map_min
    y_offset = y - map_min
    
    # Convert to grid coordinates
    grid_x = int(x_offset / grid_size)
    grid_y = int(y_offset / grid_size)
    
    # Check if coordinates are within map bounds
    if 0 <= grid_x < ground_truth_map.shape[1] and 0 <= grid_y < ground_truth_map.shape[0]:
        return float(ground_truth_map[grid_y, grid_x, 2])  # Z value is at index 2
    return None

class BoulderGroundDetector:
    """Detects the ground points from boulders in images."""
    
    def __init__(self):
        # Setup stereo system
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
        
        # Setup FastSAM
        if torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA for FastSAM")
        elif torch.backends.mps.is_built():
            self.device = "mps"
            print("Using MPS for FastSAM")
        else:
            self.device = "cpu"
            print("Using CPU for FastSAM")
        
        # Load the FastSAM model
        print("Loading FastSAM model...")
        try:
            with importlib.resources.path("resources", "FastSAM-x.pt") as fpath:
                self.fastsam = FastSAM(fpath)
            print("FastSAM model loaded successfully")
        except Exception as e:
            print(f"Error loading FastSAM model: {e}")
            raise e
    
    def _find_boulders(self, image):
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
            return [], [], [], []

        means = []
        covs = []
        bottom_pixes = []
        avg_intensities = []
        for mask in segmentation_masks:
            # Compute centroid and covariance
            # Compute centroid and covariance
            mean, cov, bottom_pix = self._compute_blob_mean_and_covariance(mask)

            # Discard any blobs in the top third of the image
            # if mean[1] < image.shape[0] / 3:
            # Discard any blobs in the top third of the image
            if mean[1] < image.shape[0] / 3:
                continue

            # Discard any blobs on the left and right edges of the image (5% margin)
            margin = image.shape[1] * 0.05
            # Discard any blobs on the left and right edges of the image (5% margin)
            margin = image.shape[1] * 0.05
            if mean[0] < margin or mean[0] > image.shape[1] - margin:
                continue

            # Calculate average pixel intensity for the region.
            # Assuming 'mask' is a binary mask with 1s for the boulder area.
            avg_pixel_value = np.mean(image[mask == 1])

            # Append results
            # Calculate average pixel intensity for the region.
            # Assuming 'mask' is a binary mask with 1s for the boulder area.
            avg_pixel_value = np.mean(image[mask == 1])

            # Append results
            means.append(mean)
            covs.append(cov)
            bottom_pixes.append(bottom_pix)
            avg_intensities.append(avg_pixel_value)
            # avg_intensities.append(avg_pixel_value)

        return means, covs, avg_intensities, bottom_pixes
    
    def _compute_blob_mean_and_covariance(self, binary_image):
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
    
    def _depth_map(self, left_image, right_image):
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
    
    def get_depth(self, depth_map, point):
        """Get the depth of a small region around a point.
        
        Args:
            depth_map: The stereo depth map
            point: The image coordinates of the point
            
        Returns:
            The depth of the point
        """
        window = 5
        # Round to the nearest pixel coordinate
        u = round(point[0])
        v = round(point[1])
        
        # Find the average depth in window around point
        # Clamp the window to the edges of the image
        half_window = window // 2
        y_start = max(0, v - half_window)
        y_end = min(depth_map.shape[0], v + half_window + 1)
        x_start = max(0, u - half_window)
        x_end = min(depth_map.shape[1], u + half_window + 1)
        
        depth_window = depth_map[y_start:y_end, x_start:x_end]
        valid_depths = depth_window[depth_window > 0]
        
        # If there was no valid depth map around this point discard it
        if len(valid_depths) == 0:
            return 0
        
        # Use median depth to be robust to outliers
        depth = np.median(valid_depths)
        return depth



if __name__ == "__main__":
    agent = PlaybackAgent("/Users/aleksandergarbuz/Documents/MIT/NASAChallenge24/MAPLE/straight-line-1min.lac")
    
    print("Starting playback...")
    
    # Create detector
    detector = BoulderGroundDetector()
    
    # Create directories for output
    output_dir = "boulder_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Lists to store all boulder ground points for final visualization
    all_boulder_points_global = []
    
    frame = 1
    done = False
    while not done:
        # Get input data from the cameras
        input_data = agent.input_data()
        rover_global = carla_to_pytransform(agent.get_transform())

        # Check if we reached the end of the recording
        done = agent.at_end()

        # Get camera images
        left_camera = "FrontLeft"
        right_camera = "FrontRight"
        
        try:
            # Check if camera data exists in input_data
            if "Grayscale" not in input_data:
                print(f"Frame {int(frame)}: No Grayscale data found in input_data")
                frame = agent.step_frame()
                continue
                
            # Get camera images
            left_image = input_data["Grayscale"].get(left_camera)
            right_image = input_data["Grayscale"].get(right_camera)
            
            # Check if both images are valid
            if left_image is None or right_image is None:
                print(f"Frame {int(frame)}: Missing camera image. Left: {left_image is not None}, Right: {right_image is not None}")
                frame = agent.step_frame()
                continue
                
            # Print image shape for debugging
            print(f"Frame {int(frame)}: Image shapes - Left: {left_image.shape}, Right: {right_image.shape}")
            
        except Exception as e:
            print(f"Frame {int(frame)}: Error accessing camera images: {e}")
            frame = agent.step_frame()
            continue

        try:
            # Get camera positions in rover frame
            left_rover = carla_to_pytransform(agent.get_camera_position(left_camera))
            right_rover = carla_to_pytransform(agent.get_camera_position(right_camera))

            # Generate depth map
            depth_map = detector.compute_depth_map(left_image, right_image, agent, left_camera, right_camera)

            # Find boulders in the left image
            boulder_points, boulder_covs, intensities = detector.find_boulders(left_image)
            
            print(f"Frame {int(frame)}: Found {len(boulder_points)} boulder candidates")

            # Apply area filtering
            MIN_AREA = 50
            MAX_AREA = 2500
            elongation_threshold = 10
            pixel_intensity_threshold = 50

            points_to_keep = []
            covs_to_keep = []

            for point, cov, intensity in zip(boulder_points, boulder_covs, intensities):
                try:
                    det_cov = np.linalg.det(cov)
                    if det_cov <= 0:
                        continue
                        
                    area = np.pi * np.sqrt(det_cov)
                    
                    eigen_vals = np.linalg.eigvals(cov)
                    elongated = (eigen_vals.max() / eigen_vals.min()) > elongation_threshold
                    bright = intensity > pixel_intensity_threshold
                    
                    if MIN_AREA <= area <= MAX_AREA and not elongated and bright:
                        points_to_keep.append(point)
                        covs_to_keep.append(cov)
                except Exception as e:
                    print(f"Frame {int(frame)}: Error processing boulder point: {e}")
                    continue
                    
            print(f"Frame {int(frame)}: After filtering, {len(points_to_keep)} boulder points remain")

            # Convert boulder points to 3D positions
            boulder_points_camera = detector.get_positions(depth_map, points_to_keep, True, covs_to_keep)
            print(f"Frame {int(frame)}: Converted {len(boulder_points_camera)} points to 3D")

            # Transform from camera to rover frame
            boulder_points_rover = [concat(point, left_rover) for point in boulder_points_camera]

            # Transform from rover to global frame
            boulder_points_global = [concat(point, rover_global) for point in boulder_points_rover]

            # Store the boulder points for final visualization
            all_boulder_points_global.extend(boulder_points_global)

                
        except Exception as e:
            import traceback
            print(f"Frame {int(frame)}: Error processing frame: {str(e)}")
            traceback.print_exc()

        # Step the agent to the next frame
        frame = agent.step_frame()

    print("Playback complete!")

