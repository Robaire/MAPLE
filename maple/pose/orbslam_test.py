#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import subprocess
from tqdm import tqdm
import glob
import pandas as pd
import threading
import queue
from datetime import datetime

class OrbSlamRunner:
    def __init__(self, orbslam_path, config_file, output_dir, mode='mono'):
        """
        Initialize ORB-SLAM runner
        
        Parameters:
        -----------
        orbslam_path : str
            Path to the ORB-SLAM executable
        config_file : str
            Path to the ORB-SLAM configuration file
        output_dir : str
            Directory to save trajectory output
        mode : str
            ORB-SLAM mode: 'mono', 'stereo', or 'stereo-inertial'
        """
        self.orbslam_path = orbslam_path
        self.config_file = config_file
        self.output_dir = output_dir
        self.mode = mode
        self.trajectory_file = os.path.join(output_dir, "trajectory.txt")
        self.orbslam_process = None
        self.image_queue = queue.Queue(maxsize=100)
        self.imu_queue = queue.Queue(maxsize=100)
        self.running = False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def start_orbslam_process(self):
        """Start ORB-SLAM as a subprocess"""
        vocabulary_path = os.path.join(os.path.dirname(self.orbslam_path), "../Vocabulary/ORBvoc.txt")
        
        # Build the ORB-SLAM command based on mode
        if self.mode == 'mono':
            cmd = [
                self.orbslam_path,
                vocabulary_path,
                self.config_file,
                "0",  # Use camera ID 0 for live input
                self.output_dir
            ]
        elif self.mode == 'stereo':
            cmd = [
                self.orbslam_path,
                vocabulary_path,
                self.config_file,
                "0",  # Use camera ID 0 for left camera
                "1",  # Use camera ID 1 for right camera
                self.output_dir
            ]
        elif self.mode == 'stereo-inertial':
            # For stereo-inertial, we'll communicate through a socket or pipe
            # This is simplified for demonstration purposes
            cmd = [
                self.orbslam_path,
                vocabulary_path,
                self.config_file,
                "--use_socket",  # Custom flag for socket communication
                self.output_dir
            ]
        
        print(f"Starting ORB-SLAM in {self.mode} mode...")
        try:
            self.orbslam_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            # Start a thread to monitor the output
            threading.Thread(target=self._monitor_output, daemon=True).start()
            return True
        except Exception as e:
            print(f"Error starting ORB-SLAM: {e}")
            return False
    
    def _monitor_output(self):
        """Monitor the output of the ORB-SLAM process"""
        while self.orbslam_process and self.orbslam_process.poll() is None:
            output = self.orbslam_process.stdout.readline()
            if output:
                print(f"ORB-SLAM: {output.strip()}")
    
    def stop_orbslam_process(self):
        """Stop the ORB-SLAM process"""
        if self.orbslam_process:
            print("Stopping ORB-SLAM process...")
            self.orbslam_process.terminate()
            self.orbslam_process.wait(timeout=5)
            self.orbslam_process = None
    
    def run_with_camera(self, camera_id_left=0, camera_id_right=1, frequency=30.0):
        """
        Run ORB-SLAM with live camera input
        
        Parameters:
        -----------
        camera_id_left : int
            Camera ID for left camera (or mono camera)
        camera_id_right : int
            Camera ID for right camera (only used in stereo modes)
        frequency : float
            Target frame rate in Hz
        """
        self.running = True
        
        # Start ORB-SLAM process
        if not self.start_orbslam_process():
            self.running = False
            return False
        
        # Open cameras
        cap_left = cv2.VideoCapture(camera_id_left)
        if not cap_left.isOpened():
            print(f"Error: Could not open camera {camera_id_left}")
            self.stop_orbslam_process()
            self.running = False
            return False
        
        cap_right = None
        if self.mode in ['stereo', 'stereo-inertial']:
            cap_right = cv2.VideoCapture(camera_id_right)
            if not cap_right.isOpened():
                print(f"Error: Could not open camera {camera_id_right}")
                cap_left.release()
                self.stop_orbslam_process()
                self.running = False
                return False
        
        # Set up display window
        cv2.namedWindow('ORB-SLAM Camera Feed', cv2.WINDOW_NORMAL)
        
        # Timing variables
        period = 1.0 / frequency
        last_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                elapsed = current_time - last_time
                
                # Sleep to maintain target frequency
                if elapsed < period:
                    time.sleep(period - elapsed)
                
                # Capture frame from left camera
                ret_left, frame_left = cap_left.read()
                if not ret_left:
                    print("Error reading from left camera")
                    break
                
                if self.mode in ['stereo', 'stereo-inertial']:
                    # Capture frame from right camera
                    ret_right, frame_right = cap_right.read()
                    if not ret_right:
                        print("Error reading from right camera")
                        break
                    
                    # Display combined image
                    combined = np.hstack((frame_left, frame_right))
                    cv2.imshow('ORB-SLAM Camera Feed', combined)
                    
                    # TODO: Send images to ORB-SLAM via socket/pipe
                    # This would require customizing ORB-SLAM to accept input via socket/pipe
                else:
                    # Display image from mono camera
                    cv2.imshow('ORB-SLAM Camera Feed', frame_left)
                    
                    # TODO: Send image to ORB-SLAM via socket/pipe
                
                # Generate fake IMU data (if needed)
                if self.mode == 'stereo-inertial':
                    # Simple simulation of IMU data (accelerometer and gyroscope)
                    # In a real application, this would come from an actual IMU sensor
                    timestamp = time.time()
                    acc = np.array([0.0, 0.0, 9.81])  # Gravity in Z direction
                    gyro = np.array([0.0, 0.0, 0.0])   # No rotation
                    
                    # Add some noise
                    acc += np.random.normal(0, 0.1, 3)
                    gyro += np.random.normal(0, 0.01, 3)
                    
                    # TODO: Send IMU data to ORB-SLAM via socket/pipe
                
                # Check for key press to exit
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    break
                
                last_time = current_time
                
        finally:
            # Clean up
            cv2.destroyAllWindows()
            cap_left.release()
            if cap_right:
                cap_right.release()
            self.stop_orbslam_process()
            self.running = False
        
        return True
    
    def run_with_dataset(self, data_path, frequency=30.0):
        """
        Run ORB-SLAM with prerecorded dataset
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset
        frequency : float
            Simulated frame rate in Hz
        """
        print(f"Running with dataset: {data_path}")
        
        # Identify dataset format and load appropriate data
        images_left = []
        images_right = []
        imu_data = []
        
        # Check if it's an EuRoC format dataset
        if os.path.exists(os.path.join(data_path, "mav0")):
            # EuRoC format
            print("Detected EuRoC format dataset")
            
            # Find cam0 and cam1 directories
            cam0_dir = os.path.join(data_path, "mav0", "cam0", "data")
            cam1_dir = os.path.join(data_path, "mav0", "cam1", "data")
            imu_file = os.path.join(data_path, "mav0", "imu0", "data.csv")
            
            if os.path.exists(cam0_dir):
                images_left = sorted(glob.glob(os.path.join(cam0_dir, "*.png")))
                print("left images loaded")
                print("number of images: ", len(images_left))
            if os.path.exists(cam1_dir):
                images_right = sorted(glob.glob(os.path.join(cam1_dir, "*.png")))
                print("right images loaded")
                print("number of images: ", len(images_right))
            if os.path.exists(imu_file):
                imu_data = pd.read_csv(imu_file)
                print("imu data loaded")
                print("number of imu points: ", len(imu_data))
        else:
            # Try generic format - looking for image directories
            left_dir = os.path.join(data_path, "left")
            right_dir = os.path.join(data_path, "right")
            imu_file = os.path.join(data_path, "imu.csv")
            
            if os.path.exists(left_dir):
                images_left = sorted(glob.glob(os.path.join(left_dir, "*.png")) + 
                                     glob.glob(os.path.join(left_dir, "*.jpg")))
            if os.path.exists(right_dir):
                images_right = sorted(glob.glob(os.path.join(right_dir, "*.png")) + 
                                      glob.glob(os.path.join(right_dir, "*.jpg")))
            if os.path.exists(imu_file):
                imu_data = pd.read_csv(imu_file)
        
        # Check if we have found any images
        if not images_left:
            print("No left camera images found in the dataset")
            return False
        
        if self.mode in ['stereo', 'stereo-inertial'] and not images_right:
            print("No right camera images found in the dataset")
            return False
        
        if self.mode == 'stereo-inertial' and imu_data.empty:
            print("No IMU data found, will generate fake IMU data")
            generate_fake_imu = True
        else:
            generate_fake_imu = False
        
        # Start ORB-SLAM process (mockup for this example)
        print("Simulating ORB-SLAM process for dataset playback...")
        
        # Create trajectory file as we go
        with open(self.trajectory_file, 'w') as traj_file:
            # Timing variables
            period = 1.0 / frequency
            start_time = time.time()
            
            # Process the images
            for i in tqdm(range(len(images_left))):
                img_left = cv2.imread(images_left[i])
                
                if self.mode in ['stereo', 'stereo-inertial'] and i < len(images_right):
                    img_right = cv2.imread(images_right[i])
                    # Display combined image
                    combined = np.hstack((img_left, img_right))
                    cv2.imshow('Dataset Playback', combined)
                else:
                    cv2.imshow('Dataset Playback', img_left)
                
                # Generate fake IMU data if needed
                if self.mode == 'stereo-inertial' and generate_fake_imu:
                    timestamp = time.time() - start_time
                    acc = np.array([0.0, 0.0, 9.81]) + np.random.normal(0, 0.1, 3)
                    gyro = np.array([0.0, 0.0, 0.0]) + np.random.normal(0, 0.01, 3)
                
                # Generate fake trajectory data (normally this would come from ORB-SLAM)
                timestamp = time.time() - start_time
                # Simulate a circular path
                angle = i * 0.01
                x = 5 * np.cos(angle)
                y = 0.1 * i  # Slowly increase height
                z = 5 * np.sin(angle)
                
                # Write to trajectory file (TUM format: timestamp x y z qx qy qz qw)
                traj_file.write(f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} 0.0 0.0 0.0 1.0\n")
                
                # Check for key press to exit
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    break
                
                # Sleep to maintain target frequency
                elapsed = time.time() - start_time - timestamp
                if elapsed < period:
                    time.sleep(period - elapsed)
        
        cv2.destroyAllWindows()
        return True

def parse_trajectory_file(trajectory_file):
    """
    Parse the trajectory file output by ORB-SLAM
    
    Parameters:
    -----------
    trajectory_file : str
        Path to the trajectory file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing timestamp, x, y, z, qx, qy, qz, qw columns
    """
    try:
        # Check if it's in TUM format (timestamp x y z qx qy qz qw)
        df = pd.read_csv(trajectory_file, sep=' ', header=None)
        if df.shape[1] == 8:
            df.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
            return df
        elif df.shape[1] == 12:
            # KITTI format (r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3)
            trajectory = pd.DataFrame({
                'timestamp': np.arange(len(df)),
                'x': df[3],
                'y': df[7],
                'z': df[11],
                'qx': 0,  # Placeholder for quaternion
                'qy': 0,
                'qz': 0,
                'qw': 1
            })
            return trajectory
    except Exception as e:
        print(f"Error parsing trajectory file: {e}")
    
    print("Warning: Could not parse trajectory file. Using placeholder data.")
    return pd.DataFrame({
        'timestamp': [0],
        'x': [0],
        'y': [0],
        'z': [0],
        'qx': [0],
        'qy': [0],
        'qz': [0],
        'qw': [1]
    })

def plot_trajectory_2d(trajectory_df, output_file=None):
    """
    Plot the 2D (top-down) view of the trajectory
    
    Parameters:
    -----------
    trajectory_df : pd.DataFrame
        DataFrame containing at minimum x, y, z columns
    output_file : str, optional
        If provided, save the plot to this file
    """
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory_df['x'], trajectory_df['z'], 'b-', linewidth=1)
    plt.plot(trajectory_df['x'].iloc[0], trajectory_df['z'].iloc[0], 'go', markersize=8, label='Start')
    plt.plot(trajectory_df['x'].iloc[-1], trajectory_df['z'].iloc[-1], 'ro', markersize=8, label='End')
    
    plt.grid(True)
    plt.axis('equal')
    plt.title('ORB-SLAM Trajectory (Top-Down View)')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.legend()
    
    if output_file:
        plt.savefig(output_file)
        print(f"2D trajectory plot saved to {output_file}")
    
    plt.show()

def plot_trajectory_3d(trajectory_df, output_file=None):
    """
    Plot the 3D view of the trajectory
    
    Parameters:
    -----------
    trajectory_df : pd.DataFrame
        DataFrame containing at minimum x, y, z columns
    output_file : str, optional
        If provided, save the plot to this file
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(trajectory_df['x'], trajectory_df['z'], trajectory_df['y'], 'b-', linewidth=1)
    ax.scatter(trajectory_df['x'].iloc[0], trajectory_df['z'].iloc[0], trajectory_df['y'].iloc[0], 
               c='g', marker='o', s=100, label='Start')
    ax.scatter(trajectory_df['x'].iloc[-1], trajectory_df['z'].iloc[-1], trajectory_df['y'].iloc[-1], 
               c='r', marker='o', s=100, label='End')
    
    # Set equal aspect ratio
    max_range = np.array([
        trajectory_df['x'].max() - trajectory_df['x'].min(),
        trajectory_df['z'].max() - trajectory_df['z'].min(),
        trajectory_df['y'].max() - trajectory_df['y'].min()
    ]).max() / 2.0
    
    mid_x = (trajectory_df['x'].max() + trajectory_df['x'].min()) / 2
    mid_y = (trajectory_df['z'].max() + trajectory_df['z'].min()) / 2
    mid_z = (trajectory_df['y'].max() + trajectory_df['y'].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Y (m)')
    ax.set_title('ORB-SLAM 3D Trajectory')
    ax.legend()
    
    if output_file:
        plt.savefig(output_file)
        print(f"3D trajectory plot saved to {output_file}")
    
    plt.show()

def plot_trajectory_animation(trajectory_df, output_file=None):
    """
    Create an animated plot of the trajectory
    
    Parameters:
    -----------
    trajectory_df : pd.DataFrame
        DataFrame containing at minimum x, y, z columns
    output_file : str, optional
        If provided, save the animation to this file
    """
    from matplotlib.animation import FuncAnimation
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set equal aspect ratio
    max_range = np.array([
        trajectory_df['x'].max() - trajectory_df['x'].min(),
        trajectory_df['z'].max() - trajectory_df['z'].min(),
        trajectory_df['y'].max() - trajectory_df['y'].min()
    ]).max() / 2.0
    
    mid_x = (trajectory_df['x'].max() + trajectory_df['x'].min()) / 2
    mid_y = (trajectory_df['z'].max() + trajectory_df['z'].min()) / 2
    mid_z = (trajectory_df['y'].max() + trajectory_df['y'].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Y (m)')
    ax.set_title('ORB-SLAM 3D Trajectory Animation')
    
    line, = ax.plot([], [], [], 'b-', linewidth=1)
    point, = ax.plot([], [], [], 'ro', markersize=8)
    
    # Number of frames (use less for faster animation)
    num_frames = min(len(trajectory_df), 100)
    frame_indices = np.linspace(0, len(trajectory_df)-1, num_frames, dtype=int)
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point
    
    def animate(i):
        idx = frame_indices[i]
        x = trajectory_df['x'].iloc[:idx+1]
        y = trajectory_df['z'].iloc[:idx+1]
        z = trajectory_df['y'].iloc[:idx+1]
        
        line.set_data(x, y)
        line.set_3d_properties(z)
        
        point.set_data([x.iloc[-1]], [y.iloc[-1]])
        point.set_3d_properties([z.iloc[-1]])
        
        return line, point
    
    ani = FuncAnimation(fig, animate, frames=num_frames, init_func=init, 
                        interval=50, blit=True)
    
    if output_file:
        ani.save(output_file, writer='pillow', fps=20)
        print(f"Trajectory animation saved to {output_file}")
    
    plt.show()
    return ani

def main():
    parser = argparse.ArgumentParser(description='Run ORB-SLAM and plot the trajectory')
    parser.add_argument('--orbslam_path', type=str, required=True,
                        help='Path to the ORB-SLAM executable')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the ORB-SLAM configuration file')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save output')
    parser.add_argument('--mode', type=str, choices=['mono', 'stereo', 'stereo-inertial'], 
                        default='mono', help='ORB-SLAM mode')
    parser.add_argument('--use_camera', action='store_true',
                        help='Use live camera input instead of dataset')
    parser.add_argument('--camera_left', type=int, default=0,
                        help='Camera ID for left/mono camera')
    parser.add_argument('--camera_right', type=int, default=1,
                        help='Camera ID for right camera (stereo only)')
    parser.add_argument('--data_path', type=str,
                        help='Path to dataset (if not using camera)')
    parser.add_argument('--frequency', type=float, default=30.0,
                        help='Target/simulated frame rate in Hz')
    parser.add_argument('--skip_run', action='store_true',
                        help='Skip running ORB-SLAM and just plot the trajectory')
    parser.add_argument('--plot_animation', action='store_true',
                        help='Create animated plot of the trajectory')
    
    args = parser.parse_args()
    
    # Create ORB-SLAM runner
    runner = OrbSlamRunner(
        orbslam_path=args.orbslam_path,
        config_file=args.config_file, 
        output_dir=args.output_dir,
        mode=args.mode
    )
    
    # Run ORB-SLAM if not skipped
    if not args.skip_run:
        if args.use_camera:
            success = runner.run_with_camera(
                camera_id_left=args.camera_left,
                camera_id_right=args.camera_right,
                frequency=args.frequency
            )
        elif args.data_path:
            success = runner.run_with_dataset(
                data_path=args.data_path,
                frequency=args.frequency
            )
        else:
            print("Error: Either --use_camera or --data_path must be specified")
            return
        
        if not success:
            print("Failed to run ORB-SLAM. Check error messages above.")
            return
    
    # Path to the trajectory file
    trajectory_file = os.path.join(args.output_dir, "trajectory.txt")
    
    # Check if trajectory file exists
    if not os.path.exists(trajectory_file):
        print(f"Trajectory file not found at {trajectory_file}")
        return
    
    # Parse and plot the trajectory
    trajectory_df = parse_trajectory_file(trajectory_file)
    
    # Plot 2D trajectory
    plot_trajectory_2d(trajectory_df, os.path.join(args.output_dir, "trajectory_2d.png"))
    
    # Plot 3D trajectory
    plot_trajectory_3d(trajectory_df, os.path.join(args.output_dir, "trajectory_3d.png"))
    
    # Create animated plot if requested
    if args.plot_animation:
        ani = plot_trajectory_animation(trajectory_df, os.path.join(args.output_dir, "trajectory_animation.gif"))
    
    print("Trajectory plotting complete!")

if __name__ == "__main__":
    main()