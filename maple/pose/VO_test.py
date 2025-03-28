"""
Test script for StereoVisualOdometer using stereo image pairs from specified folders.
Processes odd-numbered images from 1.png through 13.png.
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from visual_odometer import StereoVisualOdometer  # Assuming the previous code is saved as stereo_visual_odometer.py
from maple.utils import camera_parameters
import csv
import pandas as pd


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


    df = pd.read_csv(gt_file)

    # 2. Extract columns for x, y, z, roll, pitch, yaw (modify if your column names differ).
    #    Example: columns might be named 'gt', 'x', 'y', 'z', 'r', 'p', 'y'.
    #    If "gt" is just an index or something else, skip it. 
    #    Suppose the columns we need are actually x, y, z, r, p, y:
    subset_df = df[['gt_x', 'gt_y', 'gt_z', 'gt_roll', 'gt_pitch', 'gt_yaw']]

    # 3. Generate a 4x4 pose (homogeneous matrix) for each row
    poses_4x4 = []
    for _, row in subset_df.iterrows():
        # Extract numeric values (make sure they are in the correct units, e.g., radians)
        x_val = row['gt_x']
        y_val = row['gt_y']
        z_val = row['gt_z']
        roll  = row['gt_roll']
        pitch = row['gt_pitch']
        yaw   = row['gt_yaw']  # careful with column naming collisions

        # Build the 4x4 transform
        T = make_homogeneous_transform(x_val, y_val, z_val, roll, pitch, yaw)
        poses_4x4.append(T)

    # Now 'poses_4x4' is a list of 4x4 numpy arrays.
    # You can turn it into a 3D NumPy array if you prefer:
    poses_4x4_array = np.stack(poses_4x4, axis=0)  # shape will be (N, 4, 4)

    window_size = 11

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=window_size,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=1,
        P1=8 * 3 * window_size**2,
        P2=32 * 
        3 * window_size**2,
    )

    example_left_image = load_image(left_folder, 1)

    fx, fy, cx, cy = camera_parameters(example_left_image.shape)

    baseline = 0.16219154000292418  # Baseline in meters (distance between cameras)
    
    # Create intrinsics matrix
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Initialize stereo visual odometer
    odometer = StereoVisualOdometer(intrinsics, baseline, method_name="hybrid", device="cuda")

    
    # Check if folders exist
    if not os.path.exists(left_folder) or not os.path.exists(right_folder):
        print(f"Error: Image folders '{left_folder}' and/or '{right_folder}' do not exist.")
        return
    
    # Define image numbers to process (odd numbers from 1 to 13)
    # This will give you [1, 3, 5, 7, 9, 11, 13]
    image_numbers = list(range(1, 400, 2))

    
    # Initialize trajectory storage
    trajectory = []
    current_pose = poses_4x4_array[0] # Start with identity matrix (initial position)
    trajectory.append(current_pose.copy())
    
    # Process image pairs
    for i in range(len(image_numbers) - 1):
        current_num = image_numbers[i]
        next_num = image_numbers[i + 1]
        
        print(f"Processing images {current_num}.png and {next_num}.png...")
        
        # Load the first pair of images
        left_img1 = load_image(left_folder, current_num)
        right_img1 = load_image(right_folder, current_num)
        
        if left_img1 is None or right_img1 is None:
            print(f"Error: Could not load first pair of images ({current_num}.png)")
            continue
        
        # Initialize odometer with first pair
        if i == 0:
            odometer.update_last_frames(left_img1, right_img1)
        
        # Load the second pair of images
        left_img2 = load_image(left_folder, next_num)
        right_img2 = load_image(right_folder, next_num)
        
        if left_img2 is None or right_img2 is None:
            print(f"Error: Could not load second pair of images ({next_num}.png)")
            continue
        
        # Estimate pose change between frames
        rel_transform = odometer.estimate_rel_pose(left_img2, right_img2)
        
        # Update the global pose
        current_pose = current_pose @ rel_transform  # Matrix multiplication for pose update
        trajectory.append(current_pose.copy())
        
        print(f"Relative transform for images {current_num} to {next_num}:")
        print(rel_transform)
        print("---------------------------")
    
    # Plot the estimated trajectory
    plot_trajectory(trajectory, poses_4x4_array, trial)


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
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=2, label='Camera Path')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=50, marker='o')
    
    # Add start and end markers for the estimated trajectory
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=100, marker='o', label='Start (Est)')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, marker='o', label='End (Est)')

    # Plot the ground-truth path
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'r-', linewidth=2, label='GT Path')
    ax.scatter(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], c='purple', s=50, marker='o')
    
    # Add start and end markers for the ground-truth trajectory
    ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], c='green', s=100, marker='^', label='Start (GT)')
    ax.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], c='red', s=100, marker='^', label='End (GT)')
    
    # Set axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    # Add title and legend
    ax.set_title('Estimated vs Ground-Truth Trajectory')
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
    plt.savefig(f"stereo_odometry_trajectory_{trial}.png")
    print("Trajectory plot saved as 'stereo_odometry_trajectory_035.png'")
    
    # Show the plot
    plt.show()

# def plot_trajectory(trajectory, gt):
#     """Plot the 3D trajectory from the sequence of poses.
    
#     Args:
#         trajectory: List of 4x4 transformation matrices representing camera poses
#     """
#     # Extract positions from trajectory
#     positions = np.array([pose[:3, 3] for pose in trajectory])
#     gt_positions = np.array([pose[:3, 3] for pose in gt])
    
#     # Create a 3D plot
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Plot trajectory
#     ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=2, label='Camera Path')
#     ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=50, marker='o')
    
#     # Add start and end point markers
#     ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=100, marker='o', label='Start')
#     ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, marker='o', label='End')

#     ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'r-', linewidth=2, label='GT Path')
#     ax.scatter(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], c='purple', s=50, marker='o')
    
#     # Add start and end point markers
#     ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], c='green', s=100, marker='o', label='Start')
#     ax.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], c='red', s=100, marker='o', label='End')
    
#     # Set axis labels
#     ax.set_xlabel('X [m]')
#     ax.set_ylabel('Y [m]')
#     ax.set_zlabel('Z [m]')
    
#     # Add title and legend
#     ax.set_title('Estimated Camera Trajectory')
#     ax.legend()

#     # Try to make the plot aspect ratio equal
#     max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
#                           positions[:, 1].max() - positions[:, 1].min(),
#                           positions[:, 2].max() - positions[:, 2].min()]).max() #/ 2.0
    
#     mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
#     mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
#     mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
#     ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
#     # Save the plot
#     plt.savefig("stereo_odometry_trajectory_034.png")
#     print("Trajectory plot saved as 'stereo_odometry_trajectory.png'")
    
#     # Show the plot
#     plt.show()


if __name__ == "__main__":
    main()