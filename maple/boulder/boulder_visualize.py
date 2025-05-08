#!/usr/bin/env python

"""
Boulder Visualization Script

This script generates a video from a directory of filtered boulder detection images.
The script allows for early termination with the ESC key, and will save the video
with the frames that have been processed up to that point.
"""

import os
import cv2
import glob
import numpy as np
from datetime import datetime

# Configuration (adjust these variables as needed)
input_dir = "/Users/aleksandergarbuz/Documents/MIT/NASAChallenge24/MAPLE/mask_data_export_20250507_160451/filtered_visualizations"
output_video = "boulder_detection.mp4"
fps = 24
max_frames = 5000  # Maximum number of frames to process
display_size = (1280, 720)  # Width, height for display and output video
show_preview = False  # Set to False to disable the preview window

def main():
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    # Get list of image files sorted by frame number
    image_files = sorted(
        glob.glob(os.path.join(input_dir, "*.png")),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    print(f"Output video will be saved to {output_video}")
    
    # Get first image to determine dimensions
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Error reading first image: {image_files[0]}")
        return
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(output_video, fourcc, fps, display_size)
    
    if not video_out.isOpened():
        print("Error opening video writer")
        return
    
    try:
        print("Processing images... Press ESC to stop early and save the video.")
        
        # Process frames
        for i, img_path in enumerate(image_files):
            if i >= max_frames:
                print(f"Reached maximum frame limit ({max_frames})")
                break
            
            # Read image
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Error reading image: {img_path}")
                continue
            
            # Resize frame for display and video
            frame_resized = cv2.resize(frame, display_size)
            
            # Add frame number and timestamp
            frame_num = os.path.basename(img_path).split('_')[1].split('.')[0]
            cv2.putText(frame_resized, f"Frame: {frame_num}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame to video
            video_out.write(frame_resized)
            
            # Display preview if enabled
            if show_preview:
                cv2.imshow('Boulder Detection Video', frame_resized)
                
                # Check for ESC key to quit early
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("Early termination requested")
                    break
            
            # Print progress every 100 frames
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
    
    finally:
        # Make sure to release the video writer and close windows
        video_out.release()
        cv2.destroyAllWindows()
        
        print(f"Video saved to {output_video}")

if __name__ == "__main__":
    main()
