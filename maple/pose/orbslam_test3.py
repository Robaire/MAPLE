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
    
def prepare_custom_dataset(input_path, output_path, mode='mono', img_extension='.png', fps=30):
    """
    Prepare custom dataset in a format compatible with ORB-SLAM3
    
    Parameters:
    -----------
    input_path : str
        Path to input dataset
    output_path : str
        Path to save the prepared dataset
    mode : str
        'mono', 'stereo', or 'stereo-inertial'
    img_extension : str
        Image file extension
    fps : float
        Frames per second of the dataset
    
    Returns:
    --------
    tuple
        (processed_data_path, timestamps_file_path)
    """
    os.makedirs(output_path, exist_ok=True)
    print(output_path)
    print('maple/pose/prepared_mav0')
    
    if mode == 'mono':
        # Create directories
        image_output_dir = os.path.join(output_path, "images/")
        os.makedirs(image_output_dir, exist_ok=True)

        image_path = os.path.join(input_path, 'images')
        
        # Find all images in the input directory
        image_files = sorted([f for f in os.listdir(image_path) 
                             if f.endswith(img_extension)])
        
        print("number of images: ", len(image_files))
        
        # Create timestamp file
        timestamp_file = os.path.join(output_path, "timestamps.txt")
        with open(timestamp_file, 'w') as f:
            for i, img_file in enumerate(image_files):
                # Generate timestamp (in seconds)
                timestamp = i * (1.0 / fps)
                f.write(f"{timestamp:.6f}\n")
                
                # Copy image to output directory
                src = os.path.join(image_path, img_file)
                dst = os.path.join(image_output_dir, f"{i:06d}{img_extension}")
                shutil.copy(src, dst)
        
        print(f"Prepared {len(image_files)} images for monocular SLAM")
        return output_path, timestamp_file
    
    elif mode == 'stereo':
        # Create directories
        left_output_dir = os.path.join(output_path, "left")
        right_output_dir = os.path.join(output_path, "right")
        os.makedirs(left_output_dir, exist_ok=True)
        os.makedirs(right_output_dir, exist_ok=True)
        
        # Assuming input_path has left/ and right/ subdirectories
        left_path = os.path.join(input_path, "left")
        right_path = os.path.join(input_path, "right")
        
        if not (os.path.exists(left_path) and os.path.exists(right_path)):
            print(f"Error: Could not find left/ and right/ directories in {input_path}")
            return None, None
        
        # Find all images
        left_images = sorted([f for f in os.listdir(left_path) if f.endswith(img_extension)])
        right_images = sorted([f for f in os.listdir(right_path) if f.endswith(img_extension)])
        
        # Create timestamp file
        timestamp_file = os.path.join(output_path, "timestamps.txt")
        with open(timestamp_file, 'w') as f:
            for i in range(min(len(left_images), len(right_images))):
                # Generate timestamp (in seconds)
                timestamp = i * (1.0 / fps)
                f.write(f"{timestamp:.6f}\n")
                
                # Copy left and right images
                shutil.copy(
                    os.path.join(left_path, left_images[i]), 
                    os.path.join(left_output_dir, f"{i:06d}{img_extension}")
                )
                shutil.copy(
                    os.path.join(right_path, right_images[i]), 
                    os.path.join(right_output_dir, f"{i:06d}{img_extension}")
                )
        
        print(f"Prepared {min(len(left_images), len(right_images))} stereo pairs")
        return output_path, timestamp_file
    
    elif mode == 'stereo-inertial':
        # Similar to stereo, but also process IMU data
        # This is more complex and depends on your IMU data format
        print("Stereo-inertial mode preparation requires IMU data processing.")
        print("Please add your IMU data processing code based on your specific format.")
        return None, None
    
    else:
        print(f"Unsupported mode: {mode}")
        return None, None

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

def create_custom_yaml(output_path, camera_params):
    """
    Create a custom YAML configuration file for ORB-SLAM3
    
    Parameters:
    -----------
    output_path : str
        Path to save the YAML file
    camera_params : dict
        Dictionary with camera parameters
        
    Returns:
    --------
    str
        Path to the created YAML file
    """
    yaml_content = f"""
%YAML:1.0

# Camera Parameters. Adjust these according to your camera!

# Camera calibration and distortion parameters (OpenCV)
Camera.fx: {camera_params.get('fx', 458.654)}
Camera.fy: {camera_params.get('fy', 457.296)}
Camera.cx: {camera_params.get('cx', 367.215)}
Camera.cy: {camera_params.get('cy', 248.375)}

Camera.k1: {camera_params.get('k1', -0.28340811)}
Camera.k2: {camera_params.get('k2', 0.07395907)}
Camera.p1: {camera_params.get('p1', 0.00019359)}
Camera.p2: {camera_params.get('p2', 1.76187114e-05)}

# Camera frames per second 
Camera.fps: {camera_params.get('fps', 30.0)}

# Color order of the images (0: BGR, 1: RGB. Only used with color sensors)
Camera.RGB: {camera_params.get('rgb', 1)}

# ORB Extractor Parameters
ORBextractor.nFeatures: {camera_params.get('nFeatures', 1000)}
ORBextractor.scaleFactor: {camera_params.get('scaleFactor', 1.2)}
ORBextractor.nLevels: {camera_params.get('nLevels', 8)}
ORBextractor.iniThFAST: {camera_params.get('iniThFAST', 20)}
ORBextractor.minThFAST: {camera_params.get('minThFAST', 7)}

# Viewer Parameters
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
    """
    
    yaml_file = os.path.join(output_path, "custom_camera.yaml")
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created custom YAML configuration at {yaml_file}")
    return yaml_file

def run_with_custom_dataset(input_dataset_path, mode='mono', camera_params=None):
    """
    Run ORB-SLAM3 with a custom dataset
    
    Parameters:
    -----------
    input_dataset_path : str
        Path to the input dataset
    mode : str
        ORB-SLAM mode: 'mono', 'stereo', or 'stereo-inertial'
    camera_params : dict
        Dictionary with camera parameters
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # You'll need to adjust these paths to match your ORB-SLAM3 installation
    ORBSLAM_BASE = "/home/annikat/Dev/ORB_SLAM3"
    
    if mode == 'mono':
        ORBSLAM_PATH = f"{ORBSLAM_BASE}/Examples/Monocular/mono_euroc"
    elif mode == 'stereo':
        ORBSLAM_PATH = f"{ORBSLAM_BASE}/Examples/Stereo/stereo_euroc"
    elif mode == 'stereo-inertial':
        ORBSLAM_PATH = f"{ORBSLAM_BASE}/Examples/Stereo-Inertial/stereo_inertial_euroc"
    else:
        print(f"Unsupported mode: {mode}")
        return False
    
    VOCABULARY_PATH = f"{ORBSLAM_BASE}/Vocabulary/ORBvoc.txt"
    OUTPUT_DIR = os.path.join(base_dir, "results")
    
    # Prepare dataset directory
    dataset_name = os.path.basename(input_dataset_path)
    print("input dataset path", input_dataset_path)
    prepared_dataset_dir = os.path.join(base_dir, f"{dataset_name}")
    
    # Default camera parameters if none provided
    if camera_params is None:
        camera_params = {
            'fx': 458.654, 'fy': 457.296, 'cx': 367.215, 'cy': 248.375,
            'k1': -0.28340811, 'k2': 0.07395907, 'p1': 0.00019359, 'p2': 1.76187114e-05,
            'fps': 30.0, 'rgb': 1, 'nFeatures': 1000, 'scaleFactor': 1.2,
            'nLevels': 8, 'iniThFAST': 20, 'minThFAST': 7
        }
    
    # Create custom YAML file
    config_file = create_custom_yaml(prepared_dataset_dir, camera_params)
    
    # Prepare the dataset
    prepared_data_path, timestamps_file = prepare_custom_dataset(
        input_dataset_path, prepared_dataset_dir, mode=mode
    )
    
    if prepared_data_path is None:
        print("Dataset preparation failed.")
        return False
    
    # Run ORB-SLAM3
    success = run_orbslam(
        orbslam_path=ORBSLAM_PATH,
        vocabulary_path=VOCABULARY_PATH,
        config_file=config_file,
        data_path=prepared_data_path,
        output_dir=OUTPUT_DIR,
        mode=mode,
        timestamps_file=timestamps_file
    )
    
    if success:
        # Find the trajectory file
        trajectory_file = os.path.join(OUTPUT_DIR, "CameraTrajectory.txt")
        if not os.path.exists(trajectory_file):
            # Try with the dataset name in the filename
            trajectory_file = os.path.join(
                OUTPUT_DIR, 
                f"dataset-{os.path.basename(prepared_data_path)}_{mode}_KeyFrameTrajectory.txt"
            )
        
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
    
    return success

# Example usage
if __name__ == "__main__":
    # Path to your custom dataset
    CUSTOM_DATASET = "/home/annikat/MAPLE/maple/pose/mav0"
    
    # Optional: Camera parameters (calibration values)
    # These should be replaced with your actual camera calibration values
    my_camera_params = {
        'fx': 458.654,  # Focal length x
        'fy': 457.296,  # Focal length y
        'cx': 367.215,  # Principal point x
        'cy': 248.375,  # Principal point y
        'k1': -0.28340811,  # Distortion parameter k1
        'k2': 0.07395907,   # Distortion parameter k2
        'p1': 0.00019359,   # Distortion parameter p1
        'p2': 1.76187114e-05,  # Distortion parameter p2
        'fps': 30.0,         # Camera FPS
        'rgb': 1,            # Color order (1 for RGB, 0 for BGR)
        'nFeatures': 1000,   # Number of ORB features
        'scaleFactor': 1.2,  # Scale factor between levels
        'nLevels': 8,        # Number of pyramid levels
        'iniThFAST': 20,     # Initial FAST threshold
        'minThFAST': 7       # Minimum FAST threshold
    }
    
    # Run with custom dataset (choose mode: 'mono', 'stereo', or 'stereo-inertial')
    run_with_custom_dataset(CUSTOM_DATASET, mode='mono', camera_params=my_camera_params)