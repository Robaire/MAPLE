#!/usr/bin/env python

"""
Boulder Detection Mask Data Exporter

This script processes camera images from a .lac file using FastSAM to detect potential boulders,
and exports detailed data about the detected masks to CSV files periodically.

Key Features:
- Processes frames from a .lac file using FastSAM for segmentation
- Analyzes mask shape using eigenvalue analysis of covariance matrices
- Simulates depth estimation based on vertical position in the image
- Adjusts detected area based on simulated depth to account for perspective
- Exports CSV data every N frames with detailed metrics
- Creates visualizations of both all masks and filtered masks
- Saves individual mask images for further analysis

Filtering parameters:
- Position: Excludes masks in the top 1/3 of image and at edges
- Size: Uses depth-adjusted area between MIN_AREA and MAX_AREA
- Shape: Uses eigenvalue ratio to prefer circular over elongated shapes
- Intensity: Requires minimum average pixel intensity
"""

from lac_data import PlaybackAgent
import numpy as np
import os
import cv2
import torch
from fastsam import FastSAM, FastSAMPrompt
import importlib.resources
import matplotlib.cm as cm
import pandas as pd
from datetime import datetime

# Add camera parameters function from detector.py
def camera_parameters(image_shape):
    """Returns the approximate camera parameters for the simulation.
    
    Args:
        image_shape: The shape of the camera image
        
    Returns:
        The focal length, baseline, and image center
    """
    # Approximate values for the simulated camera
    fx = 1000.0 # focal length
    baseline = 0.2 # distance between cameras
    cx = image_shape[1] / 2 # image center x
    cy = image_shape[0] / 2 # image center y
    
    return fx, baseline, cx, cy

class MaskDataExporter:
    """Exports mask data from FastSAM segmentation and saves info to CSV files."""
    
    def __init__(self):
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
        
        # Define boulder filtering parameters
        self.MIN_AREA = 25
        self.MAX_AREA = 5000 
        self.MIN_INTENSITY = 50  
        
        # Shape filtering parameters based on eigenvalues
        # We're now using smaller/larger eigenvalue ratio (values between 0-1)
        # - Higher ratio (closer to 1) = more circular (both eigenvalues similar)
        # - Lower ratio (closer to 0) = more elongated (one eigenvalue much larger than other)
        self.MIN_EIGENVALUE_RATIO = 0.25  # Minimum ratio (smaller/larger eigenvalue)
        self.MIN_EIGENVALUE = 5.0  # Minimum eigenvalue to ensure the blob has some size
        
        # For exceptional shapes, we can relax other criteria
        self.EXCELLENT_SHAPE_RATIO = 0.5  # If ratio is above this, it's a very good circular shape
        self.RELAXED_INTENSITY = 30  # For excellent shapes, accept lower intensity
    
    def compute_blob_mean_and_covariance(self, binary_image, gray_image):
        """Finds the mean, covariance, and bottom-most pixel of a segmentation mask.
        Similar to detector.py implementation.
        
        Args:
            binary_image: The segmentation mask
            gray_image: The original grayscale image for intensity calculation
            
        Returns:
            The mean [x, y], covariance matrix, bottom-most pixel [x, y], and average intensity
        """
        # Create a grid of pixel coordinates
        y, x = np.indices(binary_image.shape)
        
        # Threshold the binary image to isolate the blob
        blob_pixels = (binary_image > 0).astype(int)
        
        # Get all coordinates of pixels in the blob
        y_coords = y[blob_pixels == 1]
        x_coords = x[blob_pixels == 1]
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            return None, None, None, 0
        
        # Compute the mean of pixel coordinates
        mean_x = np.mean(x_coords)
        mean_y = np.mean(y_coords)
        mean = np.array([mean_x, mean_y])
        
        # Stack pixel coordinates to compute covariance
        pixel_coordinates = np.vstack((x_coords, y_coords))
        
        # Compute the covariance matrix
        covariance_matrix = np.cov(pixel_coordinates)
        
        # Find the bottom-most pixel (pixel with largest y-coordinate)
        if len(y_coords) > 0:
            max_y_index = np.argmax(y_coords)
            bottom_pixel = np.array([x_coords[max_y_index], y_coords[max_y_index]])
        else:
            # If no blob pixels are found, return zeros
            bottom_pixel = np.array([0, 0])
        
        # Calculate average pixel intensity for the region
        # Assuming 'binary_image' is a binary mask with 1s for the boulder area
        avg_pixel_value = np.mean(gray_image[binary_image == 1])
        
        return mean, covariance_matrix, bottom_pixel, avg_pixel_value
    
    def _get_depth(self, depth_map, centroid):
        """Get the depth of a small region around a centroid.
        
        Args:
            depth_map: The stereo depth map (simulated in our case)
            centroid: The centroid of the boulder
        
        Returns:
            The depth of the boulder
        """
        # For simulation purposes, we'll generate a depth based on vertical position
        # In real application, this would use actual depth map data
        # Lower in the image typically means closer to the camera
        
        # Ensure the centroid is valid and within the image bounds
        if centroid is None or not isinstance(centroid, (list, tuple, np.ndarray)):
            return 1.0  # Default to 1.0m if invalid centroid
            
        height, width = depth_map.shape[:2] if len(depth_map.shape) > 1 else (0, 0)
        
        x, y = int(centroid[0]), int(centroid[1])
        if not (0 <= x < width and 0 <= y < height):
            return 1.0  # Default to 1.0m if out of bounds
            
        # For visualization purposes, we simulate depth based on vertical position
        # In a real system, you would sample the actual depth map here
        relative_vertical_pos = y / height
        
        # Simulate closer depth for objects lower in the frame (3-10 meters range)
        simulated_depth = 3.0 + relative_vertical_pos * 7.0
        
        return simulated_depth
        
    def _adjust_area_for_depth(self, depth_map, pixel_area, centroid):
        """Adjusts the pixel area based on depth to estimate actual object size.
        Similar to detector.py implementation.
        
        Args:
            pixel_area: The area in pixels from the segmentation mask
            depth_map: The stereo depth map (or simulated depth in our case)
            centroid: The (x,y) pixel coordinates of the object centroid
            
        Returns:
            The adjusted area estimate accounting for perspective projection
        """
        # Scale up pixel area by 10000 to avoid floating point errors
        pixel_area = pixel_area * 10000
        
        # Get camera parameters
        focal_length, _, cx, cy = camera_parameters(depth_map.shape)
        
        # Get depth at centroid (or simulate it in our case)
        depth = self._get_depth(depth_map, centroid)
        
        # If depth mapping fails, assume the object is 1m away
        if depth == 0:
            depth = 1.0
            
        # The scaling factor is proportional to depth squared
        # This accounts for perspective projection where apparent size decreases with distance
        depth_scaling = (depth**2) / (focal_length * focal_length)
        
        # Adjust the pixel area using the depth scaling
        adjusted_area = pixel_area * depth_scaling
        
        return adjusted_area
    
    def process_image(self, image, frame):
        """Process an image with FastSAM to get segmentation masks.
        
        Args:
            image: The grayscale image to process
            
        Returns:
            A tuple containing the original image, all masks, selected mask indices, mask data, and visualization
        """
        # Make sure image is grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
            
        # Convert to 3-channel for FastSAM
        color_image = np.stack((gray_image,) * 3, axis=-1)
        
        # Run FastSAM on the input image
        results = self.fastsam(
            color_image,
            device=self.device,
            retina_masks=True,
            imgsz=image.shape[1],
            conf=0.5,
            iou=0.9,
            verbose=False,
        )
        
        # Generate segmentation masks
        try:
            prompt = FastSAMPrompt(gray_image, results, device=self.device)
            segmentation_masks = prompt.everything_prompt()
            
            # Check if output is a tensor or not
            if isinstance(segmentation_masks, list):
                # No detections found, return an empty array
                segmentation_masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                segmentation_masks = segmentation_masks.cpu().numpy()
        except Exception as e:
            print(f"Error generating segmentation masks: {e}")
            return gray_image, [], [], [], None
        
        # Process all masks and collect data
        mask_data = []
        selected_indices = []
        
        for i, mask in enumerate(segmentation_masks):
            # Compute centroid, covariance, bottom point, and intensity
            mean, cov, bottom_pix, avg_intensity = self.compute_blob_mean_and_covariance(mask, gray_image)
            
            # Skip if computation failed
            if mean is None:
                continue
            
            # Calculate eigenvalues of covariance matrix
            try:
                eigenvalues = np.linalg.eigvals(cov)
                if np.any(eigenvalues <= 0):
                    # Not a valid positive-definite covariance matrix
                    continue
                    
                # Sort eigenvalues from largest to smallest
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                # Calculate area from covariance
                det_cov = np.linalg.det(cov)
                if det_cov <= 0:
                    continue
                    
                area = np.pi * np.sqrt(det_cov)
                
                # Adjust area for depth
                adjusted_area = self._adjust_area_for_depth(gray_image, area, mean)
                
                # Check if this mask should be selected as a boulder
                is_selected = False
                
                # Apply position criteria (not in top third, not at edges)
                if mean[1] >= gray_image.shape[0] / 3:  # Not in top third
                    margin = gray_image.shape[1] * 0.05
                    if not (mean[0] < margin or mean[0] > gray_image.shape[1] - margin):  # Not at edges
                        if self.MIN_AREA <= adjusted_area <= self.MAX_AREA:  # Size in range
                            ratio = eigenvalues[1] / eigenvalues[0]
                            if ratio >= self.MIN_EIGENVALUE_RATIO and eigenvalues[1] >= self.MIN_EIGENVALUE:
                                # For excellent shapes (more circular), use relaxed intensity threshold
                                if ratio >= self.EXCELLENT_SHAPE_RATIO:
                                    is_selected = avg_intensity >= self.RELAXED_INTENSITY
                                # For good but not excellent shapes, use standard intensity threshold
                                else:
                                    is_selected = avg_intensity >= self.MIN_INTENSITY
                                    
                                if is_selected:
                                    selected_indices.append(i)
                
                # Collect data for CSV export
                mask_data.append({
                    'mask_id': i,
                    'centroid_x': mean[0],
                    'centroid_y': mean[1],
                    'bottom_x': bottom_pix[0],
                    'bottom_y': bottom_pix[1],
                    'eigenvalue_1': eigenvalues[0],
                    'eigenvalue_2': eigenvalues[1],
                    'raw_area': area,  # Original pixel area
                    'adjusted_area': adjusted_area,  # Area adjusted for depth
                    'intensity': avg_intensity,
                    'is_selected': is_selected
                })
                
            except Exception as e:
                print(f"Error calculating mask metrics: {e}")
                continue
        
        print(f"Found {len(selected_indices)} selected boulder masks out of {len(mask_data)} total masks")
        
        # Create visualization with masks overlaid on original image
        overlay = self.create_overlay(gray_image, segmentation_masks, selected_indices)
        
        # Create filtered visualization
        filtered_overlay = self.create_filtered_overlay(gray_image, segmentation_masks, selected_indices)
        
        return gray_image, segmentation_masks, selected_indices, mask_data, overlay, filtered_overlay
    
    def create_overlay(self, image, masks, selected_indices):
        """Create a visualization with segmentation masks overlaid on the original image.
        
        Args:
            image: The original grayscale image
            masks: The segmentation masks from FastSAM
            selected_indices: Indices of masks that are selected as boulders
            
        Returns:
            A visualization image with colored masks overlaid on the original image
        """
        # Convert grayscale to color for overlay
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Draw region of interest boundaries
        height, width = image.shape[:2]
        
        # Draw line at 1/3 from the top (we ignore masks above this line)
        top_third_y = int(height / 3)
        cv2.line(vis_image, (0, top_third_y), (width, top_third_y), (0, 255, 0), 2)
        
        # Draw margin lines at 5% from left and right edges
        margin = int(width * 0.05)
        cv2.line(vis_image, (margin, 0), (margin, height), (0, 255, 0), 2)
        cv2.line(vis_image, (width - margin, 0), (width - margin, height), (0, 255, 0), 2)
        
        # Add labels for the region
        cv2.putText(vis_image, "Region of Interest", (width//2 - 80, top_third_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # If no masks, return image with just the region boundaries
        if masks is None or len(masks) == 0:
            return vis_image
        
        # Use different colors for each mask
        colors = cm.rainbow(np.linspace(0, 1, len(masks)))
        
        # Draw each mask with a different color
        for i, mask in enumerate(masks):
            # Use a different color for selected masks
            if i in selected_indices:
                color = (0, 255, 0)  # Green for selected boulders
                alpha = 0.6  # More opaque for selected
            else:
                color = (int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255))
                alpha = 0.3  # More transparent for non-selected
            
            # Overlay the mask on the image with transparency
            mask_layer = np.zeros_like(vis_image)
            mask_layer[mask == 1] = color
            vis_image = cv2.addWeighted(vis_image, 1, mask_layer, alpha, 0)
            
            # Draw contour around the mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color, 2)
            
            # Add mask ID
            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Add an indicator for selected masks
                label = f"{i+1}{'*' if i in selected_indices else ''}"
                cv2.putText(vis_image, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_image
    
    def create_filtered_overlay(self, image, masks, selected_indices):
        """Create a visualization with only the selected segmentation masks overlaid on the original image.
        
        Args:
            image: The original grayscale image
            masks: The segmentation masks from FastSAM
            selected_indices: Indices of masks that are selected as boulders
            
        Returns:
            A visualization image with colored masks overlaid on the original image
        """
        # Convert grayscale to color for overlay
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Draw region of interest boundaries
        height, width = image.shape[:2]
        
        # Draw line at 1/3 from the top (we ignore masks above this line)
        top_third_y = int(height / 3)
        cv2.line(vis_image, (0, top_third_y), (width, top_third_y), (0, 255, 0), 2)
        
        # Draw margin lines at 5% from left and right edges
        margin = int(width * 0.05)
        cv2.line(vis_image, (margin, 0), (margin, height), (0, 255, 0), 2)
        cv2.line(vis_image, (width - margin, 0), (width - margin, height), (0, 255, 0), 2)
        
        # Add labels for the region
        cv2.putText(vis_image, "Region of Interest", (width//2 - 80, top_third_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # If no masks, return image with just the region boundaries
        if masks is None or len(masks) == 0:
            return vis_image
        
        # Draw each selected mask with a different color
        for i, mask in enumerate(masks):
            if i in selected_indices:
                color = (0, 255, 0)  # Green for selected boulders
                alpha = 0.6  # More opaque for selected
                
                # Overlay the mask on the image with transparency
                mask_layer = np.zeros_like(vis_image)
                mask_layer[mask == 1] = color
                vis_image = cv2.addWeighted(vis_image, 1, mask_layer, alpha, 0)
                
                # Draw contour around the mask
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_image, contours, -1, color, 2)
                
                # Add mask ID
                M = cv2.moments(mask.astype(np.uint8))
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"{i+1}*"
                    cv2.putText(vis_image, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_image
    
    def save_mask_images(self, masks, frame_num, output_dir):
        """Save individual mask images.
        
        Args:
            masks: List of segmentation masks
            frame_num: Current frame number
            output_dir: Directory to save mask images
        """
        mask_dir = os.path.join(output_dir, f"frame_{frame_num:04d}_masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        for i, mask in enumerate(masks):
            mask_path = os.path.join(mask_dir, f"mask_{i:03d}.png")
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

def main():
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = f"mask_data_export_{timestamp}"
    vis_dir = os.path.join(output_base_dir, "visualizations")
    original_dir = os.path.join(output_base_dir, "original_images")
    csv_dir = os.path.join(output_base_dir, "csv_data")
    mask_dir = os.path.join(output_base_dir, "mask_images")
    filtered_vis_dir = os.path.join(output_base_dir, "filtered_visualizations")
    
    # Create directories
    for directory in [vis_dir, original_dir, csv_dir, mask_dir, filtered_vis_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Initialize exporter
    exporter = MaskDataExporter()
    
    # Configuration
    frame_interval = 10  # Process every 10th frame
    csv_interval = 50    # Export CSV data every 50 frames
    max_frames = 10000     # Maximum number of frames to process
    
    # Initialize playback agent with the binary file
    agent = PlaybackAgent("/Users/aleksandergarbuz/Documents/MIT/NASAChallenge24/MAPLE/beaver_6.lac")
    
    print("Starting playback...")
    print(f"Processing every {frame_interval}th frame, up to {max_frames} frames")
    print(f"Exporting CSV data every {csv_interval} frames")
    
    frame = 1
    processed_count = 0
    done = False
    all_mask_data = []
    
    while not done and processed_count < max_frames:
        # Get input data from the cameras
        input_data = agent.input_data()
        
        # Check if we reached the end of the recording
        done = agent.at_end()
        
        # Get camera image
        left_camera = "FrontLeft"
        
        try:
            # Check if camera data exists in input_data
            if "Grayscale" not in input_data:
                print(f"Frame {int(frame)}: No Grayscale data found in input_data")
                frame = agent.step_frame()
                continue
                
            # Get camera image
            left_image = input_data["Grayscale"].get(left_camera)
            
            # Check if image is valid
            if left_image is None:
                print(f"Frame {int(frame)}: Missing camera image")
                frame = agent.step_frame()
                continue
                
            # Process frames at specified interval
            if frame % frame_interval == 0:
                print(f"Processing frame {int(frame)}...")
                
                # Save original image
                original_output_path = os.path.join(original_dir, f"frame_{int(frame):04d}.png")
                cv2.imwrite(original_output_path, left_image)
                
                # Process image with FastSAM
                original, masks, selected_indices, mask_data, overlay, filtered_overlay = exporter.process_image(left_image, frame)
                
                # Add frame number to mask data
                for item in mask_data:
                    item['frame'] = int(frame)
                
                # Add to collection of all mask data
                all_mask_data.extend(mask_data)
                
                if len(masks) > 0:
                    # Save visualization
                    output_path = os.path.join(vis_dir, f"frame_{int(frame):04d}.png")
                    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    print(f"Frame {int(frame)}: Saved visualization to {output_path}")
                    
                    # Save filtered visualization
                    filtered_output_path = os.path.join(filtered_vis_dir, f"frame_{int(frame):04d}.png")
                    cv2.imwrite(filtered_output_path, cv2.cvtColor(filtered_overlay, cv2.COLOR_RGB2BGR))
                    print(f"Frame {int(frame)}: Saved filtered visualization to {filtered_output_path}")
                    
                    # Save individual mask images for this frame if it's a CSV export frame
                    if frame % csv_interval == 0:
                        exporter.save_mask_images(masks, int(frame), mask_dir)
                else:
                    print(f"Frame {int(frame)}: No masks detected")
                
                processed_count += 1
                
                # Export CSV data every csv_interval frames
                if frame % csv_interval == 0 and all_mask_data:
                    csv_path = os.path.join(csv_dir, f"mask_data_frame_{int(frame):04d}.csv")
                    pd.DataFrame(all_mask_data).to_csv(csv_path, index=False)
                    print(f"Exported mask data CSV to {csv_path}")
                    
                    # Reset all_mask_data for the next interval
                    all_mask_data = []
            
        except Exception as e:
            import traceback
            print(f"Frame {int(frame)}: Error processing frame: {str(e)}")
            traceback.print_exc()
        
        # Step the agent to the next frame
        frame = agent.step_frame()
    
    # Export any remaining mask data
    if all_mask_data:
        csv_path = os.path.join(csv_dir, f"mask_data_frame_{int(frame):04d}_final.csv")
        pd.DataFrame(all_mask_data).to_csv(csv_path, index=False)
        print(f"Exported final mask data CSV to {csv_path}")
    
    print("Playback complete!")
    print(f"Processed {processed_count} frames")
    print(f"All data exported to {os.path.abspath(output_base_dir)}")

if __name__ == "__main__":
    main()
