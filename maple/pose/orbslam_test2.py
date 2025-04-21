import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil

def run_orbslam(orbslam_path, vocabulary_path, config_file, data_path, output_dir, mode='mono', timestamps_file=None):
    """
    Run ORB-SLAM3 as a subprocess
    
    Parameters:
    -----------
    orbslam_path : str
        Path to the ORB-SLAM executable
    vocabulary_path : str
        Path to the vocabulary file
    config_file : str
        Path to the configuration file
    data_path : str
        Path to the dataset
    output_dir : str
        Directory to save trajectory output
    mode : str
        ORB-SLAM mode: 'mono', 'stereo', or 'stereo-inertial'
    timestamps_file : str
        Path to timestamps file (required for mono mode with EuRoC dataset)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the command based on the mode
    cmd = [
        orbslam_path,
        vocabulary_path,
        config_file,
        data_path
    ]
    
    # Add timestamps file for mono mode if provided
    if mode == 'mono' and timestamps_file:
        cmd.append(timestamps_file)
    
    # Add output directory and optional name format
    output_name = f"dataset-{os.path.basename(data_path)}_{mode}"
    cmd.append(os.path.join(output_dir, output_name))
    
    print(f"Running ORB-SLAM3 in {mode} mode...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the process and capture output
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            check=False  # Don't raise exception on non-zero return
        )
        
        # Print output
        for line in process.stdout.splitlines():
            print(f"ORB-SLAM3: {line.strip()}")
        
        if process.returncode != 0:
            print(f"ORB-SLAM3 exited with code {process.returncode}")
            print(f"Error: {process.stderr}")
            return False
        
        print("ORB-SLAM3 completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error running ORB-SLAM3: {e}")
        return False
    
def prepare_custom_dataset(left_images, right_images, imu_data, output_path):
    """
    Prepare custom dataset in EuRoC format
    
    Parameters:
    -----------
    left_images : list
        List of paths to left camera images
    right_images : list
        List of paths to right camera images
    imu_data : DataFrame
        IMU data with timestamp and measurements
    output_path : str
        Path to save the prepared dataset
    """
    # Create EuRoC-like directory structure
    os.makedirs(f"{output_path}/mav0/cam0/data", exist_ok=True)
    os.makedirs(f"{output_path}/mav0/cam1/data", exist_ok=True)
    os.makedirs(f"{output_path}/mav0/imu0", exist_ok=True)
    
    # Create timestamp files
    with open(f"{output_path}/mav0/cam0/data.csv", 'w') as f:
        f.write("#timestamp [ns],filename\n")
        for i, img_path in enumerate(left_images):
            timestamp = i * (1/30) * 1e9  # Assuming 30Hz
            img_name = f"{int(timestamp)}.png"
            f.write(f"{int(timestamp)},{img_name}\n")
            # Copy or convert image
            shutil.copy(img_path, f"{output_path}/mav0/cam0/data/{img_name}")
    
    # Repeat for right images
    with open(f"{output_path}/mav0/cam1/data.csv", 'w') as f:
        f.write("#timestamp [ns],filename\n")
        for i, img_path in enumerate(right_images):
            timestamp = i * (1/30) * 1e9  # Assuming 30Hz
            img_name = f"{int(timestamp)}.png"
            f.write(f"{int(timestamp)},{img_name}\n")
            # Copy or convert image
            shutil.copy(img_path, f"{output_path}/mav0/cam1/data/{img_name}")
    
    # Save IMU data
    imu_data.to_csv(f"{output_path}/mav0/imu0/data.csv", index=False)
    
    return output_path

def read_trajectory(trajectory_file):
    """
    Read the trajectory file produced by ORB-SLAM3
    
    Parameters:
    -----------
    trajectory_file : str
        Path to the trajectory file
        
    Returns:
    --------
    DataFrame
        DataFrame with trajectory data
    """
    try:
        # First try TUM format
        df = pd.read_csv(trajectory_file, sep=' ', header=None)
        
        if df.shape[1] == 8:  # TUM format
            df.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
            return df
        elif df.shape[1] == 12:  # KITTI format
            # Extract the translation components
            df = pd.DataFrame({
                'timestamp': range(len(df)),
                'x': df[3],
                'y': df[7],
                'z': df[11]
            })
            return df
        else:
            print(f"Unknown trajectory format with {df.shape[1]} columns")
            return None
    
    except Exception as e:
        print(f"Error reading trajectory: {e}")
        return None


# Define file paths
ORBSLAM_PATH = "/home/annikat/Dev/ORB_SLAM3/Examples/Monocular/mono_euroc"
VOCABULARY_PATH = "/home/annikat/Dev/ORB_SLAM3/Vocabulary/ORBvoc.txt"
CONFIG_FILE = "/home/annikat/Dev/ORB_SLAM3/Examples/Monocular/EuRoC.yaml"
DATASET_PATH = "/home/annikat/EuRoC_data/V101"  # EuRoC dataset
OUTPUT_DIR = "./results"
# Add this line for the timestamps file
TIMESTAMPS_FILE = "/home/annikat/Dev/ORB_SLAM3/Examples/Monocular/EuRoC_TimeStamps/V101.txt"

# Run ORB-SLAM3
success = run_orbslam(
    orbslam_path=ORBSLAM_PATH,
    vocabulary_path=VOCABULARY_PATH,
    config_file=CONFIG_FILE,
    data_path=DATASET_PATH,
    output_dir=OUTPUT_DIR,
    mode="mono",
    timestamps_file=TIMESTAMPS_FILE  # Pass the timestamps file
)

if success:
    # Find the trajectory file
    # ORB-SLAM3 typically saves it as CameraTrajectory.txt
    trajectory_file = os.path.join(OUTPUT_DIR, "CameraTrajectory.txt")
    
    if os.path.exists(trajectory_file):
        # Read the trajectory
        trajectory = read_trajectory(trajectory_file)
        
        if trajectory is not None:
            # Plot 2D trajectory (top-down view)
            plt.figure(figsize=(10, 8))
            plt.plot(trajectory['x'], trajectory['z'], 'b-')
            plt.plot(trajectory['x'].iloc[0], trajectory['z'].iloc[0], 'go', markersize=8, label='Start')
            plt.plot(trajectory['x'].iloc[-1], trajectory['z'].iloc[-1], 'ro', markersize=8, label='End')
            plt.grid(True)
            plt.axis('equal')
            plt.title('ORB-SLAM3 Trajectory (Top-Down View)')
            plt.xlabel('X (m)')
            plt.ylabel('Z (m)')
            plt.legend()
            plt.savefig(os.path.join(OUTPUT_DIR, "trajectory_2d.png"))
            plt.show()
            
            # Plot 3D trajectory
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(trajectory['x'], trajectory['z'], trajectory['y'], 'b-')
            ax.scatter(trajectory['x'].iloc[0], trajectory['z'].iloc[0], trajectory['y'].iloc[0], 
                      c='g', marker='o', s=100, label='Start')
            ax.scatter(trajectory['x'].iloc[-1], trajectory['z'].iloc[-1], trajectory['y'].iloc[-1], 
                      c='r', marker='o', s=100, label='End')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.set_zlabel('Y (m)')
            ax.set_title('ORB-SLAM3 3D Trajectory')
            ax.legend()
            plt.savefig(os.path.join(OUTPUT_DIR, "trajectory_3d.png"))
            plt.show()
    else:
        print(f"Trajectory file not found at {trajectory_file}")
else:
    print("ORB-SLAM3 failed to run successfully")