import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import importlib.resources
from collections import namedtuple
import argparse
import glob

# Import the BoulderDetector class from your module
from maple.utils import camera_parameters, carla_to_pytransform
from maple.boulder import BoulderDetector

# Create a mock Agent class for testing purposes
class MockAgent:
    def __init__(self, left_camera_position, right_camera_position):
        self.left_camera_position = left_camera_position
        self.right_camera_position = right_camera_position
        self._sensors = {}
        
    def sensors(self):
        return self._sensors
    
    def get_camera_position(self, camera):
        if str(camera) == "left_camera":
            return self.left_camera_position
        elif str(camera) == "right_camera":
            return self.right_camera_position
        else:
            raise ValueError(f"Unknown camera: {camera}")

# Create a mock SensorPosition class
class SensorPosition:
    def __init__(self, name, location, rotation):
        self.name = name
        self.location = location
        self.rotation = rotation
    
    def __str__(self):
        return self.name

def main():
    parser = argparse.ArgumentParser(description="Detect boulders in stereo images and visualize results")
    parser.add_argument("--left_dir", type=str, default="/home/annikat/MAPLE/data/035/FrontLeft", help="Directory containing left images")
    parser.add_argument("--right_dir", type=str, default="/home/annikat/MAPLE/data/035/FrontRight", help="Directory containing right images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualized results")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up mock camera positions (with locations and rotations)
    Location = namedtuple('Location', ['x', 'y', 'z'])
    Rotation = namedtuple('Rotation', ['pitch', 'yaw', 'roll'])
    
    # Left camera at origin with no rotation
    left_location = Location(x=0, y=0, z=1.5)
    left_rotation = Rotation(pitch=0, yaw=0, roll=0)
    left_camera_position = SensorPosition("left_camera", left_location, left_rotation)
    
    # Right camera 0.2 meters to the right
    right_location = Location(x=0.2, y=0, z=1.5) 
    right_rotation = Rotation(pitch=0, yaw=0, roll=0)
    right_camera_position = SensorPosition("right_camera", right_location, right_rotation)
    
    # Create a mock agent with the camera positions
    mock_agent = MockAgent(left_camera_position, right_camera_position)
    mock_agent._sensors = {
        left_camera_position: None,
        right_camera_position: None
    }
    
    # Create a boulder detector
    detector = BoulderDetector(mock_agent, "left_camera", "right_camera")
    
    # Get all image files from the left and right directories
    left_images = sorted(glob.glob(os.path.join(args.left_dir, "*.png")))
    right_images = sorted(glob.glob(os.path.join(args.right_dir, "*.png")))
    
    # Make sure we have matching files (same number)
    if len(left_images) != len(right_images):
        print(f"Warning: Number of left images ({len(left_images)}) differs from right images ({len(right_images)})")
        
    # Process each pair of images with matching filenames
    processed_pairs = 0
    
    for left_path in left_images:
        # Extract the filename (e.g., "1.png", "3.png", etc.)
        left_filename = os.path.basename(left_path)
        
        # Construct the corresponding right image path
        right_path = os.path.join(args.right_dir, left_filename)
        
        # Check if the right image exists
        if not os.path.exists(right_path):
            print(f"Skipping {left_filename}: No matching right image found")
            continue
        
        print(f"Processing image pair {processed_pairs+1}: {left_filename}")
        
        # Read the images
        left_image = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        
        if left_image is None or right_image is None:
            print(f"Error reading images: {left_path} or {right_path}")
            continue
        
        # Create input data dictionary expected by the BoulderDetector
        input_data = {
            "Grayscale": {
                left_camera_position: left_image,
                right_camera_position: right_image
            }
        }
        
        # Run the boulder detection pipeline but capture intermediate results
        # Get boulder centroids and covariances
        centroids, covs, pixel_intensities = detector._find_boulders(left_image)
        
        # Filter boulders by area
        areas = []
        for cov in covs:
            det_cov = np.linalg.det(cov)
            if det_cov <= 0:
                areas.append(float("nan"))
            else:
                area = np.pi * np.sqrt(det_cov)
                areas.append(area)
        
        # Apply area filtering
        MIN_AREA = 50
        MAX_AREA = 2500
        elongation_threshold = 10
        pixel_intensity_threshold = 50

        centroids_to_keep = []
        areas_to_keep = []
        covs_to_keep = []

        for centroid, area, cov, intensity in zip(centroids, areas, covs, pixel_intensities):
            # print(pixel_intensities)
            eigen_vals = np.linalg.eigvals(cov)
            elongated = (eigen_vals.max() / eigen_vals.min()) > elongation_threshold
            bright = intensity > pixel_intensity_threshold
            if MIN_AREA <= area <= MAX_AREA and not elongated and bright:
                centroids_to_keep.append(centroid)
                covs_to_keep.append(cov)
                areas_to_keep.append(area)
        
        # Create output images with boulders highlighted
        # Convert grayscale to color for visualization
        left_vis = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
        right_vis = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
        
        # Draw ellipses for each boulder on the left image
        for centroid, cov, area in zip(centroids_to_keep, covs_to_keep, areas_to_keep):
            # Calculate eigenvectors and eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Get the angle of the largest eigenvector
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
            # Get width and height of ellipse (2 * standard deviation)
            width, height = 2 * np.sqrt(eigenvalues)
            
            # Draw the ellipse on the left image
            cv2.ellipse(
                left_vis, 
                (int(centroid[0]), int(centroid[1])),
                (int(width), int(height)),
                angle,
                0, 360,
                (0, 255, 0),  # Green color
                2
            )
            
            # Add text with area information
            cv2.putText(
                left_vis,
                f"{area:.1f}",
                (int(centroid[0]) + 10, int(centroid[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # Yellow color
                1
            )
        
        # Generate depth map for visualization
        depth_map, confidence_map = detector._depth_map(left_image, right_image)
        
        # Normalize depth map for visualization
        norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
        
        # Save the results
        base_name = os.path.splitext(left_filename)[0]  # just the number without extension
        
        cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_left_detected.jpg"), left_vis)
        # cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_right.jpg"), right_vis)
        # cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_depth.jpg"), depth_vis)
        
        # Also create a combined visualization
        # Resize images if they have different dimensions
        h_left, w_left = left_vis.shape[:2]
        h_depth, w_depth = depth_vis.shape[:2]
        
        # Create a canvas for the combined visualization
        combined_height = max(h_left, h_depth)
        combined_width = w_left + w_depth
        combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Place images on the canvas
        combined[:h_left, :w_left] = left_vis
        combined[:h_depth, w_left:w_left+w_depth] = depth_vis
        
        # Add labels
        cv2.putText(combined, "Detected Boulders", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Depth Map", (w_left + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_combined.jpg"), combined)
        
        # Try to get 3D positions if possible (this might fail depending on your test setup)
        try:
            boulder_positions, _ = detector.map(input_data)
            print(f"  Detected {len(boulder_positions)} boulders in 3D space")
        except Exception as e:
            print(f"  Could not compute 3D positions: {e}")
        
        processed_pairs += 1

    print(f"Processing complete. Processed {processed_pairs} image pairs.")

if __name__ == "__main__":
    main()