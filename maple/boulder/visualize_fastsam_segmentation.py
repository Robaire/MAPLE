#!/usr/bin/env python

from lac_data import PlaybackAgent
import numpy as np
import os
import cv2
import torch
from fastsam import FastSAM, FastSAMPrompt
import importlib.resources
import matplotlib.cm as cm


class FastSAMVisualizer:
    """Visualizes FastSAM segmentation masks."""
    
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
        self.MAX_AREA = 800
        self.MIN_INTENSITY = 100
        
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
    
    def process_image(self, image):
        """Process an image with FastSAM to get segmentation masks.
        
        Args:
            image: The grayscale image to process
            
        Returns:
            A tuple containing the original image, filtered boulder masks, and visualization
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
            return gray_image, None, None
        
        # Filter masks to only keep likely boulder masks
        boulder_masks = []
        for mask in segmentation_masks:
            # Compute centroid and covariance
            mean, cov, bottom_pix, avg_intensity = self.compute_blob_mean_and_covariance(mask, gray_image)
            
            if mean is None:
                continue
                
            # Discard blobs in the top third of the image
            if mean[1] < gray_image.shape[0] / 3:
                continue
                
            # Discard blobs on the left and right edges of the image (5% margin)
            margin = gray_image.shape[1] * 0.05
            if mean[0] < margin or mean[0] > gray_image.shape[1] - margin:
                continue
                
            # Calculate area from covariance
            try:
                det_cov = np.linalg.det(cov)
                if det_cov <= 0:
                    continue
                    
                area = np.pi * np.sqrt(det_cov)
                
                # Filter by area
                if self.MIN_AREA <= area <= self.MAX_AREA:
                    # Filter by intensity
                    if avg_intensity >= self.MIN_INTENSITY:
                        boulder_masks.append(mask)
            except Exception as e:
                print(f"Error calculating mask area: {e}")
                continue
        
        print(f"Found {len(boulder_masks)} boulder masks after filtering")
        
        # Create visualization with masks overlaid on original image
        overlay = self.create_overlay(gray_image, boulder_masks)
        
        return gray_image, boulder_masks, overlay
    
    def create_overlay(self, image, masks):
        """Create a visualization with segmentation masks overlaid on the original image.
        
        Args:
            image: The original grayscale image
            masks: The segmentation masks from FastSAM
            
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
            color = colors[i][:3]  # Convert RGBA to RGB
            color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Create a colored mask
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask == 1] = color
            
            # Overlay the mask on the image with transparency
            alpha = 0.5
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
                cv2.putText(vis_image, f"{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_image

def main():
    # Create output directories
    output_dir = "fastsam_visualizations"
    original_images_dir = "original_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(original_images_dir):
        os.makedirs(original_images_dir)
    
    # Initialize visualizer
    visualizer = FastSAMVisualizer()
    
    # Initialize playback agent with the binary file
    agent = PlaybackAgent("/Users/aleksandergarbuz/Documents/MIT/NASAChallenge24/MAPLE/beaver_6.lac")
    
    print("Starting playback...")
    
    frame = 1
    done = False
    while not done:
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
                
            # Process frames every 10 frames to reduce processing time
            if frame % 10 == 0:
                print(f"Processing frame {int(frame)}...")
                
                # Save original image
                original_output_path = os.path.join(original_images_dir, f"frame_{int(frame):04d}.png")
                cv2.imwrite(original_output_path, left_image)
                
                # Process image with FastSAM
                original, masks, visualization = visualizer.process_image(left_image)
                
                if masks is not None and len(masks) > 0:
                    print(f"Frame {int(frame)}: Found {len(masks)} segmented objects")
                    
                    # Save visualization
                    output_path = os.path.join(output_dir, f"frame_{int(frame):04d}.png")
                    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                    
                    print(f"Frame {int(frame)}: Saved visualization to {output_path}")
                else:
                    print(f"Frame {int(frame)}: No objects segmented")
            
        except Exception as e:
            import traceback
            print(f"Frame {int(frame)}: Error processing frame: {str(e)}")
            traceback.print_exc()
        
        # Step the agent to the next frame
        frame = agent.step_frame()
    
    print("Playback complete!")
    print(f"Visualizations saved to {os.path.abspath(output_dir)}")
    print(f"Original images saved to {os.path.abspath(original_images_dir)}")

if __name__ == "__main__":
    main()
