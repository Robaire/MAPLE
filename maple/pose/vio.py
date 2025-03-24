import argparse
import yaml
import numpy as np
import cv2
import time
import threading
import queue
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable

# Define data structures
class FrontendType(Enum):
    MONO_IMU = 0
    STEREO_IMU = 1

@dataclass
class ImuMeasurement:
    timestamp: float
    linear_acceleration: np.ndarray  # [ax, ay, az]
    angular_velocity: np.ndarray     # [wx, wy, wz]

@dataclass
class CameraParams:
    camera_matrix: np.ndarray        # 3x3 intrinsic matrix
    distortion_coeffs: np.ndarray    # Distortion parameters
    resolution: Tuple[int, int]      # (width, height)

@dataclass
class StereoCalibration:
    left_cam: CameraParams
    right_cam: CameraParams
    extrinsics: np.ndarray           # 4x4 transformation matrix from left to right camera

@dataclass
class VioParams:
    frontend_type: FrontendType
    parallel_run: bool
    imu_params: Dict
    camera_params: Dict
    tracking_params: Dict
    initialization_params: Dict
    
    @classmethod
    def from_yaml_folder(cls, folder_path: str):
        """Load parameters from a folder of YAML files"""
        # Load separate yaml files
        with open(f"{folder_path}/pipeline_params.yaml", 'r') as f:
            pipeline_params = yaml.safe_load(f)
        
        with open(f"{folder_path}/imu_params.yaml", 'r') as f:
            imu_params = yaml.safe_load(f)
            
        with open(f"{folder_path}/camera_params.yaml", 'r') as f:
            camera_params = yaml.safe_load(f)
            
        with open(f"{folder_path}/tracker_params.yaml", 'r') as f:
            tracking_params = yaml.safe_load(f)
            
        with open(f"{folder_path}/initialization_params.yaml", 'r') as f:
            initialization_params = yaml.safe_load(f)
            
        # Parse frontend type
        frontend_str = pipeline_params.get("frontend_type", "stereo_imu").lower()
        frontend_type = FrontendType.STEREO_IMU if "stereo" in frontend_str else FrontendType.MONO_IMU
        
        return cls(
            frontend_type=frontend_type,
            parallel_run=pipeline_params.get("parallel_run", True),
            imu_params=imu_params,
            camera_params=camera_params,
            tracking_params=tracking_params,
            initialization_params=initialization_params
        )

@dataclass
class FrameData:
    timestamp: float
    image: np.ndarray

@dataclass
class StereoFrameData:
    timestamp: float
    left_image: np.ndarray
    right_image: np.ndarray


class DataProvider:
    """Base class for data providers"""
    def __init__(self, params: VioParams):
        self.params = params
        self.imu_callback = None
        self.left_frame_callback = None
        self.right_frame_callback = None
        self._is_running = False
        
    def register_imu_callback(self, callback: Callable[[ImuMeasurement], None]):
        self.imu_callback = callback
        
    def register_left_frame_callback(self, callback: Callable[[FrameData], None]):
        self.left_frame_callback = callback
        
    def register_right_frame_callback(self, callback: Callable[[FrameData], None]):
        self.right_frame_callback = callback
        
    def shutdown(self):
        self._is_running = False
        
    def has_data(self) -> bool:
        """Return True if there is more data to process"""
        return False
        
    def spin(self) -> bool:
        """Process one step of data. Return False when done."""
        return False


class EurocDataProvider(DataProvider):
    """Data provider for EuRoC dataset format"""
    def __init__(self, params: VioParams, dataset_path: str):
        super().__init__(params)
        self.dataset_path = dataset_path
        
        # Load data paths and timestamps
        self.imu_data = self._load_imu_data()
        self.left_cam_data = self._load_camera_data("cam0")
        
        if params.frontend_type == FrontendType.STEREO_IMU:
            self.right_cam_data = self._load_camera_data("cam1")
        
        self.current_imu_idx = 0
        self.current_cam_idx = 0
        self._is_running = True
        
    def _load_imu_data(self):
        """Load IMU data from EuRoC format"""
        imu_data = []
        
        # Load IMU csv file
        imu_csv = f"{self.dataset_path}/mav0/imu0/data.csv"
        with open(imu_csv, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                timestamp = float(parts[0]) * 1e-9  # Convert from ns to seconds
                wx, wy, wz = float(parts[1]), float(parts[2]), float(parts[3])
                ax, ay, az = float(parts[4]), float(parts[5]), float(parts[6])
                
                imu_data.append(ImuMeasurement(
                    timestamp=timestamp,
                    angular_velocity=np.array([wx, wy, wz]),
                    linear_acceleration=np.array([ax, ay, az])
                ))
                
        return imu_data
    
    def _load_camera_data(self, cam_name):
        """Load camera timestamps and image paths"""
        cam_data = []
        
        # Load camera csv file
        cam_csv = f"{self.dataset_path}/mav0/{cam_name}/data.csv"
        with open(cam_csv, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                timestamp = float(parts[0]) * 1e-9  # Convert from ns to seconds
                img_path = f"{self.dataset_path}/mav0/{cam_name}/data/{parts[1]}"
                
                cam_data.append((timestamp, img_path))
                
        return cam_data
    
    def has_data(self) -> bool:
        return (self.current_cam_idx < len(self.left_cam_data) and 
                self.current_imu_idx < len(self.imu_data))
    
    def spin(self) -> bool:
        if not self._is_running or not self.has_data():
            return False
            
        # Process camera data
        if self.current_cam_idx < len(self.left_cam_data):
            left_timestamp, left_img_path = self.left_cam_data[self.current_cam_idx]
            left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
            
            if self.params.frontend_type == FrontendType.STEREO_IMU:
                # For stereo, get the right image with the same timestamp
                right_timestamp, right_img_path = self.right_cam_data[self.current_cam_idx]
                right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
                
                # Send the frame data to the pipeline
                if self.left_frame_callback:
                    self.left_frame_callback(FrameData(left_timestamp, left_img))
                    
                if self.right_frame_callback:
                    self.right_frame_callback(FrameData(right_timestamp, right_img))
            else:
                # Mono pipeline
                if self.left_frame_callback:
                    self.left_frame_callback(FrameData(left_timestamp, left_img))
            
            self.current_cam_idx += 1
            
        # Process IMU data up to the next camera timestamp
        next_cam_timestamp = float('inf')
        if self.current_cam_idx < len(self.left_cam_data):
            next_cam_timestamp = self.left_cam_data[self.current_cam_idx][0]
            
        while (self.current_imu_idx < len(self.imu_data) and 
               self.imu_data[self.current_imu_idx].timestamp < next_cam_timestamp):
            if self.imu_callback:
                self.imu_callback(self.imu_data[self.current_imu_idx])
            self.current_imu_idx += 1
            
        return True


class CustomDataProvider(DataProvider):
    """Data provider for custom synchronized stereo+IMU data"""
    def __init__(self, params: VioParams, stereo_images, imu_data, calibration):
        """
        Initialize with pre-loaded data arrays
        
        Args:
            params: VIO parameters
            stereo_images: List of tuples (timestamp, left_img, right_img)
            imu_data: List of ImuMeasurement objects
            calibration: StereoCalibration object
        """
        super().__init__(params)
        self.stereo_images = stereo_images
        self.imu_data = imu_data
        self.calibration = calibration
        
        self.current_imu_idx = 0
        self.current_frame_idx = 0
        self._is_running = True
        
    def has_data(self) -> bool:
        return (self.current_frame_idx < len(self.stereo_images) and 
                self.current_imu_idx < len(self.imu_data))
    
    def spin(self) -> bool:
        if not self._is_running or not self.has_data():
            return False
            
        # Process stereo frame
        if self.current_frame_idx < len(self.stereo_images):
            timestamp, left_img, right_img = self.stereo_images[self.current_frame_idx]
            
            # Send frame data to pipeline
            if self.left_frame_callback:
                self.left_frame_callback(FrameData(timestamp, left_img))
                
            if self.params.frontend_type == FrontendType.STEREO_IMU and self.right_frame_callback:
                self.right_frame_callback(FrameData(timestamp, right_img))
                
            self.current_frame_idx += 1
            
        # Process IMU data up to the next frame timestamp
        next_frame_timestamp = float('inf')
        if self.current_frame_idx < len(self.stereo_images):
            next_frame_timestamp = self.stereo_images[self.current_frame_idx][0]
            
        while (self.current_imu_idx < len(self.imu_data) and 
               self.imu_data[self.current_imu_idx].timestamp < next_frame_timestamp):
            if self.imu_callback:
                self.imu_callback(self.imu_data[self.current_imu_idx])
            self.current_imu_idx += 1
            
        return True


class VioFrontend:
    """Frontend for visual feature tracking and matching"""
    def __init__(self, params: VioParams):
        self.params = params
        self.feature_detector = cv2.FastFeatureDetector_create(
            threshold=self.params.tracking_params.get("fast_threshold", 10)
        )
        self.feature_descriptor = cv2.ORB_create()
        self.max_features = self.params.tracking_params.get("max_features_per_frame", 150)
        
    def detect_features(self, frame: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect features in a frame"""
        keypoints = self.feature_detector.detect(frame, None)
        
        # Keep only the strongest features
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
        keypoints = keypoints[:self.max_features]
        
        # Compute descriptors
        keypoints, descriptors = self.feature_descriptor.compute(frame, keypoints)
        
        return keypoints, descriptors
        
    def track_features(self, prev_frame: np.ndarray, curr_frame: np.ndarray, 
                       prev_keypoints: List[cv2.KeyPoint]) -> Tuple[List[cv2.KeyPoint], np.ndarray, np.ndarray]:
        """Track features from previous frame to current frame using optical flow"""
        if not prev_keypoints:
            return [], np.array([]), np.array([])
            
        # Convert keypoints to points
        prev_points = np.array([kp.pt for kp in prev_keypoints], dtype=np.float32).reshape(-1, 1, 2)
        
        # Use Lucas-Kanade optical flow
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, prev_points, None, 
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Keep only good points
        good_new = curr_points[status == 1]
        good_old = prev_points[status == 1]
        
        # Convert points back to keypoints
        new_keypoints = [cv2.KeyPoint(x=point[0], y=point[1], _size=7) for point in good_new]
        
        return new_keypoints, good_new, good_old
        
    def stereo_match(self, left_img: np.ndarray, right_img: np.ndarray, 
                     left_keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """Find stereo correspondences for keypoints in the left image"""
        if not left_keypoints:
            return np.array([])
            
        # Convert keypoints to points
        left_points = np.array([kp.pt for kp in left_keypoints], dtype=np.float32).reshape(-1, 1, 2)
        
        # Use Lucas-Kanade optical flow for stereo matching
        right_points, status, err = cv2.calcOpticalFlowPyrLK(
            left_img, right_img, left_points, None, 
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Calculate disparities
        disparity = np.zeros(len(left_points))
        for i in range(len(left_points)):
            if status[i] == 1:
                disparity[i] = left_points[i][0][0] - right_points[i][0][0]
            else:
                disparity[i] = -1  # Invalid disparity
                
        return disparity


class ImuPreintegrator:
    """Handles IMU pre-integration between frames"""
    def __init__(self, params: Dict):
        # IMU noise parameters
        self.gyro_noise = params.get("gyroscope_noise_density", 0.000175)
        self.accel_noise = params.get("accelerometer_noise_density", 0.0025)
        self.gyro_bias_noise = params.get("gyroscope_random_walk", 2.41e-5)
        self.accel_bias_noise = params.get("accelerometer_random_walk", 3e-3)
        
        # Gravity vector
        self.gravity = np.array([0, 0, -9.81])
        
        # Current bias estimates
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        
        # Pre-integration results
        self.reset()
        
    def reset(self):
        """Reset pre-integration state"""
        self.delta_rotation = np.eye(3)  # Rotation matrix (SO3)
        self.delta_velocity = np.zeros(3)
        self.delta_position = np.zeros(3)
        self.start_time = None
        self.measurements = []
        
    def add_measurement(self, imu_measurement: ImuMeasurement):
        """Add an IMU measurement for pre-integration"""
        if self.start_time is None:
            self.start_time = imu_measurement.timestamp
            self.measurements.append(imu_measurement)
            return
            
        # Store measurement
        self.measurements.append(imu_measurement)
        
        # Get time delta
        prev_time = self.measurements[-2].timestamp
        curr_time = imu_measurement.timestamp
        dt = curr_time - prev_time
        
        if dt <= 0:
            return  # Skip invalid timestamps
            
        # Correct for bias
        gyro = imu_measurement.angular_velocity - self.gyro_bias
        accel = imu_measurement.linear_acceleration - self.accel_bias
        
        # Simple pre-integration (Euler method)
        # For production, use mid-point or RK4 integration
        
        # Update rotation
        angle_axis = gyro * dt
        delta_R = self._exp_map(angle_axis)
        self.delta_rotation = self.delta_rotation @ delta_R
        
        # Update velocity and position
        rotated_accel = self.delta_rotation @ accel
        self.delta_velocity += (rotated_accel + self.gravity) * dt
        self.delta_position += self.delta_velocity * dt + 0.5 * (rotated_accel + self.gravity) * dt * dt
        
    def get_delta_time(self) -> float:
        """Get the time duration covered by current pre-integration"""
        if not self.measurements or self.start_time is None:
            return 0.0
        return self.measurements[-1].timestamp - self.start_time
        
    def _exp_map(self, angle_axis: np.ndarray) -> np.ndarray:
        """Exponential map from so(3) to SO(3)"""
        angle = np.linalg.norm(angle_axis)
        if angle < 1e-10:
            return np.eye(3)
            
        axis = angle_axis / angle
        s = np.sin(angle)
        c = np.cos(angle)
        
        skew_axis = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        return np.eye(3) + s * skew_axis + (1 - c) * (skew_axis @ skew_axis)


class VioPipeline:
    """Base class for VIO pipelines"""
    def __init__(self, params: VioParams):
        self.params = params
        self.is_running = False
        self.shutdown_callback = None
        
        # Data queues
        self.imu_queue = queue.Queue()
        self.left_frame_queue = queue.Queue()
        
        # Visualization and results
        self.trajectory = []  # List of (timestamp, position, orientation)
        
    def register_shutdown_callback(self, callback: Callable[[], None]):
        self.shutdown_callback = callback
        
    def fill_single_imu_queue(self, imu_data: ImuMeasurement):
        """Add IMU data to the queue"""
        self.imu_queue.put(imu_data)
        
    def fill_left_frame_queue(self, frame_data: FrameData):
        """Add left camera frame to the queue"""
        self.left_frame_queue.put(frame_data)
        
    def spin(self) -> bool:
        """Process one step of the pipeline. Return False when done."""
        return False
        
    def shutdown(self):
        """Shutdown the pipeline"""
        self.is_running = False
        if self.shutdown_callback:
            self.shutdown_callback()
            
    def spin_viz(self):
        """Spin visualization in the main thread"""
        self.is_running = True
        
        while self.is_running:
            self._update_visualization()
            time.sleep(0.05)
            
    def _update_visualization(self):
        """Update visualization (to be implemented by subclasses)"""
        pass
        
    def wait_for_shutdown(self, data_done_condition: Callable[[], bool], 
                         timeout_ms: int, check_queues_empty: bool) -> bool:
        """Wait for pipeline to finish"""
        start_time = time.time()
        
        while self.is_running:
            if data_done_condition():
                if check_queues_empty:
                    if (self.imu_queue.empty() and 
                        self.left_frame_queue.empty()):
                        self.shutdown()
                        return True
                else:
                    self.shutdown()
                    return True
                    
            # Check for timeout
            if timeout_ms > 0 and (time.time() - start_time) * 1000 > timeout_ms:
                self.shutdown()
                return False
                
            time.sleep(0.05)
            
        return True


class StereoImuPipeline(VioPipeline):
    """Pipeline for Stereo+IMU VIO"""
    def __init__(self, params: VioParams):
        super().__init__(params)
        
        # Additional data queue for right frames
        self.right_frame_queue = queue.Queue()
        
        # Initialize frontend for feature tracking
        self.frontend = VioFrontend(params)
        
        # Initialize IMU preintegrator
        self.imu_preintegrator = ImuPreintegrator(params.imu_params)
        
        # State variables
        self.position = np.zeros(3)          # World position
        self.orientation = np.eye(3)         # Rotation matrix (world to body)
        self.velocity = np.zeros(3)          # Velocity in world frame
        
        self.prev_left_frame = None
        self.prev_left_keypoints = []
        self.prev_left_descriptors = None
        
        self.is_initialized = False
        self.prev_timestamp = None
        
    def fill_right_frame_queue(self, frame_data: FrameData):
        """Add right camera frame to the queue"""
        self.right_frame_queue.put(frame_data)
        
    def spin(self) -> bool:
        """Process one step of the pipeline"""
        if not self.is_running:
            self.is_running = True
            
        # Try to get synchronized stereo frames
        try:
            left_frame = self.left_frame_queue.get_nowait()
            right_frame = self.right_frame_queue.get_nowait()
            
            # Simple check for synchronization
            if abs(left_frame.timestamp - right_frame.timestamp) > 0.01:
                print(f"Warning: Stereo frames not synchronized: {left_frame.timestamp} vs {right_frame.timestamp}")
            
            # Process all IMU measurements up to this frame
            self._process_imu_up_to(left_frame.timestamp)
            
            # Process stereo frame
            self._process_stereo_frame(
                left_frame.timestamp, 
                left_frame.image, 
                right_frame.image
            )
            
            return True
            
        except queue.Empty:
            # No frames available
            return True
        
    def _process_stereo_frame(self, timestamp: float, left_img: np.ndarray, right_img: np.ndarray):
        """Process a stereo frame pair"""
        # On first frame, just detect features
        if self.prev_left_frame is None:
            keypoints, descriptors = self.frontend.detect_features(left_img)
            self.prev_left_frame = left_img
            self.prev_left_keypoints = keypoints
            self.prev_left_descriptors = descriptors
            self.prev_timestamp = timestamp
            return
            
        # Track features from previous left frame to current left frame
        curr_keypoints, curr_points, prev_points = self.frontend.track_features(
            self.prev_left_frame, left_img, self.prev_left_keypoints
        )
        
        # Find stereo correspondences
        disparity = self.frontend.stereo_match(left_img, right_img, curr_keypoints)
        
        # Count valid stereo matches
        valid_stereo = np.sum(disparity > 0)
        
        if not self.is_initialized:
            # Initialize if we have enough stereo matches
            if valid_stereo > 20:
                self._initialize(left_img, right_img, curr_keypoints, disparity)
            return
            
        # Update VIO state
        self._update_state(timestamp, left_img, right_img, curr_keypoints, disparity)
        
        # Detect new features if needed
        if len(curr_keypoints) < self.params.tracking_params.get("min_features_threshold", 50):
            new_keypoints, new_descriptors = self.frontend.detect_features(left_img)
            
            # Filter out keypoints close to existing ones
            if curr_keypoints:
                curr_points = np.array([kp.pt for kp in curr_keypoints], dtype=np.float32)
                new_points = np.array([kp.pt for kp in new_keypoints], dtype=np.float32)
                
                # Simple filtering by distance
                keep_indices = []
                for i, new_pt in enumerate(new_points):
                    if np.min(np.linalg.norm(new_pt - curr_points, axis=1)) > 10:
                        keep_indices.append(i)
                        
                new_keypoints = [new_keypoints[i] for i in keep_indices]
                if new_descriptors is not None:
                    new_descriptors = new_descriptors[keep_indices]
                    
            # Add new keypoints
            curr_keypoints.extend(new_keypoints)
            if self.prev_left_descriptors is not None and new_descriptors is not None:
                self.prev_left_descriptors = np.vstack((self.prev_left_descriptors, new_descriptors))
                
        # Update previous frame data
        self.prev_left_frame = left_img
        self.prev_left_keypoints = curr_keypoints
        self.prev_timestamp = timestamp
        
    def _process_imu_up_to(self, frame_timestamp: float):
        """Process all IMU measurements up to given timestamp"""
        # Reset preintegrator for new frame
        self.imu_preintegrator.reset()
        
        while not self.imu_queue.empty():
            imu_data = self.imu_queue.queue[0]  # Peek at the queue
            
            if imu_data.timestamp <= frame_timestamp:
                # Process this IMU measurement
                imu_data = self.imu_queue.get()
                self.imu_preintegrator.add_measurement(imu_data)
            else:
                # This IMU measurement is after the frame, stop
                break
                
    def _initialize(self, left_img: np.ndarray, right_img: np.ndarray, 
                   keypoints: List[cv2.KeyPoint], disparity: np.ndarray):
        """Initialize the VIO state from stereo data"""
        print("Initializing VIO state...")
        
        # Simple initialization: 
        # - Set initial pose to identity
        # - Zero velocity
        # - Create initial map from stereo triangulation
        
        self.position = np.zeros(3)
        self.orientation = np.eye(3)
        self.velocity = np.zeros(3)
        
        # Add first pose to trajectory
        self.trajectory.append((self.prev_timestamp, self.position.copy(), self.orientation.copy()))
        
        self.is_initialized = True
        print("VIO initialized!")
        
    def _update_state(self, timestamp: float, left_img: np.ndarray, right_img: np.ndarray,
                     keypoints: List[cv2.KeyPoint], disparity: np.ndarray):
        """Update VIO state with new measurements"""
        # For simplicity, just do IMU-based prediction
        # A full VIO solution would:
        # 1. Use IMU for state prediction
        # 2. Triangulate 3D points from stereo
        # 3. Perform bundle adjustment to refine pose
        
        dt = timestamp - self.prev_timestamp
        if dt <= 0:
            return
            
        # Get IMU preintegration results
        delta_R = self.imu_preintegrator.delta_rotation
        delta_v = self.imu_preintegrator.delta_velocity
        delta_p = self.imu_preintegrator.delta_position
        
        # Update state (simple IMU integration)
        self.orientation = self.orientation @ delta_R
        self.velocity = self.velocity + delta_v
        self.position = self.position + self.velocity * dt + delta_p
        
        # Add to trajectory
        self.trajectory.append((timestamp, self.position.copy(), self.orientation.copy()))
        
    def _update_visualization(self):
        """Update visualization of trajectory and features"""
        if not self.trajectory:
            return
            
        # Create a simple visualization of the trajectory
        # In a real app, you'd use matplotlib, OpenGL, or a similar library
        
        # For this example, just print latest state
        latest = self.trajectory[-1]
        print(f"Latest pose: t={latest[0]:.2f}, pos=[{latest[1][0]:.2f}, {latest[1][1]:.2f}, {latest[1][2]:.2f}]")


def run_vio_pipeline(params_folder: str, dataset_type: str, dataset_path: str = None, 
                    stereo_images=None, imu_data=None, calibration=None):
    """
    Run the VIO pipeline
    
    Args:
        params_folder: Path to folder containing parameter YAML files
        dataset_type: Type of dataset ('euroc' or 'custom')
        dataset_path: Path to dataset (for 'euroc' type)
        stereo_images: List of (timestamp, left_img, right_img) tuples (for 'custom' type)
        imu_data: List of ImuMeasurement objects (for 'custom' type)
        calibration: StereoCalibration object (for 'custom' type)
        
    Returns:
        Trajectory as a list of (timestamp, position, orientation) tuples
    """
    print(f"Loading parameters from {params_folder}")
    
    # Parse VIO parameters
    vio_params = VioParams.from_yaml_folder(params_folder)
    
    # Build dataset parser
    data_provider = None
    
    if dataset_type.lower() == 'euroc':
        if dataset_path is None:
            raise ValueError("Dataset path must be provided for EuRoC dataset")
        
        print(f"Using EuRoC dataset from {dataset_path}")
        data_provider = EurocDataProvider(vio_params, dataset_path)
    elif dataset_type.lower() == 'custom':
        if stereo_images is None or imu_data is None or calibration is None:
            raise ValueError("stereo_images, imu_data, and calibration must be provided for custom dataset")
        
        print("Using custom dataset")
        data_provider = CustomDataProvider(vio_params, stereo_images, imu_data, calibration)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Use 'euroc' or 'custom'.")
    
    # Create pipeline
    vio_pipeline = None
    
    if vio_params.frontend_type == FrontendType.MONO_IMU:
        raise NotImplementedError("Mono+IMU pipeline not implemented in this example")
    elif vio_params.frontend_type == FrontendType.STEREO_IMU:
        print("Creating Stereo+IMU pipeline")
        vio_pipeline = StereoImuPipeline(vio_params)
    else:
        raise ValueError(f"Unsupported frontend type: {vio_params.frontend_type}")
    
    # Register shutdown callback
    vio_pipeline.register_shutdown_callback(data_provider.shutdown)
    
    # Register callbacks for data flow
    data_provider.register_imu_callback(vio_pipeline.fill_single_imu_queue)
    data_provider.register_left_frame_callback(vio_pipeline.fill_left_frame_queue)
    
    if vio_params.frontend_type == FrontendType.STEREO_IMU:
        data_provider.register_right_frame_callback(vio_pipeline.fill_right_frame_queue)
    
    # Process data
    start_time = time.time()
    is_pipeline_successful = False
    
    print("Starting pipeline processing...")
    
    if vio_params.parallel_run:
        # Run in parallel threads
        data_thread = threading.Thread(target=lambda: data_provider.spin())
        pipeline_thread = threading.Thread(target=lambda: vio_pipeline.spin())
        
        # Start threads
        data_thread.start()
        pipeline_thread.start()
        
        # Start visualization in main thread
        viz_thread = threading.Thread(target=vio_pipeline.spin_viz)
        viz_thread.start()
        
        # Wait for data provider to finish
        data_thread.join()
        
        # Wait for pipeline to process remaining data
        shutdown_thread = threading.Thread(
            target=lambda: vio_pipeline.wait_for_shutdown(
                lambda: not data_provider.has_data(), 
                500, True
            )
        )
        shutdown_thread.start()
        shutdown_thread.join()
        
        # Stop other threads
        vio_pipeline.shutdown()
        pipeline_thread.join()
        viz_thread.join()
        
        is_pipeline_successful = True
    else:
        # Run sequentially
        vio_pipeline.is_running = True
        while data_provider.spin() and vio_pipeline.spin():
            continue
        
        vio_pipeline.shutdown()
        is_pipeline_successful = True
    
    # Output stats
    end_time = time.time()
    processing_time = (end_time - start_time) * 1000  # Convert to ms
    
    print(f"Processing finished in {processing_time:.2f} ms")
    print(f"Pipeline successful? {'Yes!' if is_pipeline_successful else 'No!'}")
    
    # Return trajectory
    return vio_pipeline.trajectory


def create_sample_params():
    """Create sample parameter files for testing"""
    import os
    
    params_dir = "sample_params"
    os.makedirs(params_dir, exist_ok=True)
    
    # Pipeline params
    pipeline_params = {
        "frontend_type": "stereo_imu",
        "parallel_run": True
    }
    
    # IMU params
    imu_params = {
        "gyroscope_noise_density": 0.000175,
        "accelerometer_noise_density": 0.0025,
        "gyroscope_random_walk": 2.41e-5,
        "accelerometer_random_walk": 3e-3,
        "imu_integration_sigma": 0.0,
        "n_gravity": [0, 0, -9.81]
    }
    
    # Camera params
    camera_params = {
        "distortion_model": "radtan",
        "distortion_coefficients": [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05],
        "image_dimension": [752, 480],
        "baseline": 0.110
    }
    
    # Tracker params
    tracker_params = {
        "fast_threshold": 10,
        "max_features_per_frame": 150,
        "min_features_threshold": 50,
        "quality_level": 0.001,
        "min_distance": 8
    }
    
    # Initialization params
    initialization_params = {
        "min_num_features": 15,
        "min_num_poses": 3,
        "min_parallax": 5.0
    }
    
    # Write files
    with open(f"{params_dir}/pipeline_params.yaml", 'w') as f:
        yaml.dump(pipeline_params, f)
        
    with open(f"{params_dir}/imu_params.yaml", 'w') as f:
        yaml.dump(imu_params, f)
        
    with open(f"{params_dir}/camera_params.yaml", 'w') as f:
        yaml.dump(camera_params, f)
        
    with open(f"{params_dir}/tracker_params.yaml", 'w') as f:
        yaml.dump(tracker_params, f)
        
    with open(f"{params_dir}/initialization_params.yaml", 'w') as f:
        yaml.dump(initialization_params, f)
    
    print(f"Sample parameter files created in {params_dir}/")
    return params_dir


def load_sample_data(num_frames=10):
    """Generate sample data for testing the VIO pipeline"""
    # Sample timestamps (1/30 second apart)
    timestamps = np.linspace(0, (num_frames-1)/30, num_frames)
    
    # Create sample stereo images (just random noise for testing)
    width, height = 640, 480
    stereo_frames = []
    
    for ts in timestamps:
        # Create random images with some structure (gradients) for feature detection
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xv, yv = np.meshgrid(x, y)
        
        left_img = np.zeros((height, width), dtype=np.uint8)
        right_img = np.zeros((height, width), dtype=np.uint8)
        
        # Add gradient patterns
        left_img = ((xv * yv * 255) + 30 * np.sin(xv * 10) * np.cos(yv * 10)).astype(np.uint8)
        
        # Right image slightly shifted
        right_img = np.zeros_like(left_img)
        right_img[:, :-5] = left_img[:, 5:]
        
        # Add some noise
        left_img = left_img + np.random.randint(0, 10, (height, width)).astype(np.uint8)
        right_img = right_img + np.random.randint(0, 10, (height, width)).astype(np.uint8)
        
        stereo_frames.append((ts, left_img, right_img))
    
    # Create IMU measurements (10x more frequent than images)
    imu_timestamps = np.linspace(0, timestamps[-1], num_frames * 10)
    
    # Generate synthetic IMU data
    # In a real scenario, this would come from actual IMU readings
    imu_data = []
    for ts in imu_timestamps:
        # Create sine wave motion pattern
        gyro = np.array([0.1 * np.sin(ts * 2), 0.05 * np.cos(ts * 3), 0.01])
        accel = np.array([0.1 * np.cos(ts), 0.1 * np.sin(ts), 9.81 + 0.05 * np.sin(ts * 5)])
        
        # Add noise
        gyro += np.random.normal(0, 0.01, 3)
        accel += np.random.normal(0, 0.1, 3)
        
        imu_data.append(ImuMeasurement(ts, accel, gyro))
    
    # Create sample calibration
    fx, fy = 458.654, 457.296
    cx, cy = 367.215, 248.375
    
    left_cam_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Same intrinsics for right camera (typical for stereo rigs)
    right_cam_matrix = left_cam_matrix.copy()
    
    # Sample distortion coefficients
    dist_coeffs = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
    
    # Create stereo extrinsics (baseline of 0.1 meters along x-axis)
    extrinsics = np.eye(4)
    extrinsics[0, 3] = 0.1  # 10cm baseline
    
    left_cam = CameraParams(left_cam_matrix, dist_coeffs, (width, height))
    right_cam = CameraParams(right_cam_matrix, dist_coeffs, (width, height))
    
    calibration = StereoCalibration(left_cam, right_cam, extrinsics)
    
    return stereo_frames, imu_data, calibration


def main():
    """Main function to demonstrate the VIO pipeline"""
    parser = argparse.ArgumentParser(description='Python Kimera-VIO Implementation')
    parser.add_argument('--params_folder', type=str, default=None,
                        help='Path to folder containing parameter YAML files')
    parser.add_argument('--dataset_type', type=str, default='custom', choices=['euroc', 'custom'],
                        help='Type of dataset to use')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to dataset (for EuRoC)')
    parser.add_argument('--create_params', action='store_true',
                        help='Create sample parameter files')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames for sample data')
    
    args = parser.parse_args()
    
    # Create sample parameters if requested
    if args.create_params:
        params_dir = create_sample_params()
        if args.params_folder is None:
            args.params_folder = params_dir
    
    # Use default parameter path if none provided
    if args.params_folder is None:
        args.params_folder = "sample_params"
        create_sample_params()
    
    # Run with EuRoC dataset
    if args.dataset_type == 'euroc':
        if args.dataset_path is None:
            print("Error: Dataset path must be provided for EuRoC dataset")
            return
        
        trajectory = run_vio_pipeline(
            args.params_folder,
            args.dataset_type,
            dataset_path=args.dataset_path
        )
    # Run with custom data
    else:
        print("Using sample data for testing")
        stereo_frames, imu_data, calibration = load_sample_data(args.num_frames)
        
        trajectory = run_vio_pipeline(
            args.params_folder,
            args.dataset_type,
            stereo_images=stereo_frames,
            imu_data=imu_data,
            calibration=calibration
        )
    
    # Print trajectory summary
    if trajectory:
        print(f"\nGenerated trajectory with {len(trajectory)} poses")
        print(f"Start position: {trajectory[0][1]}")
        print(f"End position: {trajectory[-1][1]}")
    

if __name__ == "__main__":
    main()