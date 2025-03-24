import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import pandas as pd
import pyorbslam3

def rpy_to_matrix(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw Euler angles to a 3x3 rotation matrix.
    Uses the convention of rotating about x (roll), then y (pitch), then z (yaw).
    
    Args:
        roll (float): Rotation around x-axis in radians
        pitch (float): Rotation around y-axis in radians
        yaw (float): Rotation around z-axis in radians
        
    Returns:
        numpy.ndarray: 3x3 rotation matrix
    """
    # Roll (rotation around x-axis)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (rotation around y-axis)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (rotation around z-axis)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations: R = Rz * Ry * Rx (order matters!)
    R = Rz @ Ry @ Rx
    return R

def make_homogeneous_transform(x, y, z, roll, pitch, yaw):
    """
    Build a 4x4 homogeneous transformation matrix from 
    translation (x, y, z) and Euler angles (roll, pitch, yaw).
    
    Args:
        x, y, z (float): Translation components
        roll (float): Rotation around x-axis in radians
        pitch (float): Rotation around y-axis in radians
        yaw (float): Rotation around z-axis in radians
        
    Returns:
        numpy.ndarray: 4x4 homogeneous transformation matrix
    """
    # Build the 3x3 rotation matrix
    R = rpy_to_matrix(roll, pitch, yaw)
    
    # Create a 4x4 identity and insert R and translation
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [x, y, z]
    
    return T

def load_image(folder_path, image_number):
    """Load an image from the specified folder with the given number.
    
    Args:
        folder_path: Path to the image folder
        image_number: Number of the image to load (without extension)
        
    Returns:
        The loaded image as a numpy array or None if loading fails
    """
    image_path = os.path.join(folder_path, f"{image_number}.png")
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    print(f"Warning: Failed to load image {image_path}")
    return None

def main():
    trial = '035'

    # Define image folders
    left_folder = f"/home/annikat/MAPLE/data/{trial}/FrontLeft"
    right_folder = f"/home/annikat/MAPLE/data/{trial}/FrontRight"
    gt_file = f"/home/annikat/MAPLE/data/{trial}/imu_data.csv"

    # Path to ORB-SLAM3 vocabulary and config files
    # Make sure to adjust these paths to where you installed ORB-SLAM3
    vocab_path = "/path/to/ORBvoc.txt"  # Update this path
    config_path = "/path/to/stereo_config.yaml"  # Update this path

    # Load ground truth data
    df = pd.read_csv(gt_file)
    subset_df = df[['gt_x', 'gt_y', 'gt_z', 'gt_roll', 'gt_pitch', 'gt_yaw']]

    # Generate ground truth poses
    poses_4x4 = []
    for _, row in subset_df.iterrows():
        x_val = row['gt_x']
        y_val = row['gt_y']
        z_val = row['gt_z']
        roll = row['gt_roll']
        pitch = row['gt_pitch']
        yaw = row['gt_yaw']

        T = make_homogeneous_transform(x_val, y_val, z_val, roll, pitch, yaw)
        poses_4x4.append(T)

    # Convert to numpy array
    poses_4x4_array = np.stack(poses_4x4, axis=0)

    # Initialize ORB-SLAM3 system with stereo configuration
    slam = pyorbslam3.System(vocab_path, config_path, pyorbslam3.Sensor.STEREO)
    
    # Define image numbers to process (odd numbers from 1 to 399)
    image_numbers = list(range(1, 400, 2))
    
    # Initialize trajectory storage
    trajectory = []
    
    # Process image pairs
    for i, img_num in enumerate(image_numbers):
        print(f"Processing image {img_num}.png...")
        
        # Load stereo images
        left_img = load_image(left_folder, img_num)
        right_img = load_image(right_folder, img_num)
        
        if left_img is None or right_img is None:
            print(f"Error: Could not load images ({img_num}.png)")
            continue
        
        # Convert RGB to grayscale for ORB-SLAM3
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        
        # Pass stereo images to ORB-SLAM3 (using current timestamp)
        timestamp = float(i)  # Use frame number as timestamp if real timestamps aren't available
        pose = slam.process_stereo_image(left_gray, right_gray, timestamp)
        
        # Check if pose estimation was successful and add to trajectory
        if pose is not None:
            # ORB-SLAM3 returns camera-to-world transform matrix (4x4)
            trajectory.append(pose)
        else:
            print(f"Warning: Failed to estimate pose for image {img_num}")
            # If the first frame fails, we have a problem
            if i == 0:
                print("Error: Failed to initialize ORB-SLAM3 with the first frame")
                break
            # Otherwise, repeat the last known pose
            if trajectory:
                trajectory.append(trajectory[-1])
    
    # Shut down ORB-SLAM3 system
    slam.shutdown()
    
    # Convert trajectory to numpy array
    if trajectory:
        trajectory_array = np.array(trajectory)
        # Plot the estimated trajectory
        plot_trajectory(trajectory, poses_4x4_array, trial)
    else:
        print("Error: Failed to generate trajectory")

def plot_trajectory(trajectory, gt, trial):
    """Plot the 3D trajectory from the sequence of poses, together with the ground-truth (GT) path.
    
    Args:
        trajectory: List (or array) of 4x4 transformation matrices representing camera poses.
        gt: List (or array) of 4x4 transformation matrices representing the ground-truth poses.
    """
    # Extract positions from the estimated trajectory and ground truth
    positions = np.array([pose[:3, 3] for pose in trajectory])
    gt_positions = np.array([pose[:3, 3] for pose in gt])
    
    # Create a combined set of positions so we can properly find min/max
    combined_positions = np.concatenate([positions, gt_positions], axis=0)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the estimated trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=2, label='ORB-SLAM3 Path')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=50, marker='o')
    
    # Add start and end markers for the estimated trajectory
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=100, marker='o', label='Start (Est)')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, marker='o', label='End (Est)')

    # Plot the ground-truth path
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'g-', linewidth=2, label='GT Path')
    ax.scatter(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], c='purple', s=50, marker='o')
    
    # Add start and end markers for the ground-truth trajectory
    ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], c='green', s=100, marker='^', label='Start (GT)')
    ax.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], c='red', s=100, marker='^', label='End (GT)')
    
    # Set axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    # Add title and legend
    ax.set_title('ORB-SLAM3 vs Ground-Truth Trajectory')
    ax.legend()

    # Try to make the plot aspect ratio roughly equal using combined min/max
    max_range = np.array([
        combined_positions[:, 0].max() - combined_positions[:, 0].min(),
        combined_positions[:, 1].max() - combined_positions[:, 1].min(),
        combined_positions[:, 2].max() - combined_positions[:, 2].min()
    ]).max()
    
    mid_x = (combined_positions[:, 0].max() + combined_positions[:, 0].min()) * 0.5
    mid_y = (combined_positions[:, 1].max() + combined_positions[:, 1].min()) * 0.5
    mid_z = (combined_positions[:, 2].max() + combined_positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save the plot
    plt.savefig(f"orbslam3_trajectory_{trial}.png")
    print(f"Trajectory plot saved as 'orbslam3_trajectory_{trial}.png'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()