"""This script is used to plot the detected boulders for testing."""

import numpy as np
import matplotlib.pyplot as plt

class BoulderPlotter:
    """A class to contain all of the plotting functions for testing the boulder detector. Does not include image plotting."""
    def __init__(self):
        pass

    def plot_boulders(self, all_boulder_detections: list, large_boulder_dections: list, gt_boulder_detections: list, frame_number: int, pose: np.ndarray):
        """
        Plot the detected boulders and ground truth boulders.

        Args:
            all_boulder_detections (list): List of all detected boulders. [(x, y), ...]
            large_boulder_dections (list): List of large detected boulders. [(x, y, r), ...]
            gt_boulder_detections (list): List of ground truth boulders. [(x, y), ...]
            frame_number (int): The frame number for the plot title.
            pose (np.ndarray): The pose of the rover in the world frame.
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot all detected boulders
        for detection in all_boulder_detections:
            ax.scatter(detection[0], detection[1], color='blue', label='Detected Boulder')

        # Plot large detected boulders
        for detection in large_boulder_dections:
            ax.scatter(detection[0], detection[1], color='red', label='Large Detected Boulder')

        # Plot ground truth boulders
        for detection in gt_boulder_detections:
            ax.scatter(detection[0], detection[1], color='green', label='Ground Truth Boulder')

        # Set plot title and labels
        ax.set_title(f'Boulder Detections - Frame {frame_number}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        
        # Create a grid for the plot, with a spacing of 0.15 x 0.15
        ax.set_xticks(np.arange(-10, 10, 0.15))
        ax.set_yticks(np.arange(-10, 10, 0.15))
        ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)

        # Show the plot
        plt.show()