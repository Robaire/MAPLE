""" This module includes the StereoVisualOdometer class with hybrid depth and feature tracking for robust pose estimation """
import numpy as np
import open3d as o3d
import open3d.core as o3c
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


class StereoVisualOdometer(object):

    def __init__(self, intrinsics: np.ndarray, baseline: float, method_name="hybrid", device="cuda"):
        """ Initializes the stereo visual odometry system with specified intrinsics, baseline, method, and device.
        Args:
            intrinsics: Camera intrinsic parameters.
            baseline: The baseline distance between stereo cameras in meters.
            method_name: The name of the odometry computation method to use ('hybrid' or 'point_to_plane').
            device: The computation device ('cuda' or 'cpu').
        """
        device = "CUDA:0" if device == "cuda" else "CPU:0"
        self.device = o3c.Device(device)
        self.intrinsics = o3d.core.Tensor(intrinsics, o3d.core.Dtype.Float64)
        self.intrinsics_np = intrinsics
        self.baseline = baseline
        self.last_abs_pose = None
        self.last_frame = None
        self.criteria_list = [
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500)]
        self.setup_method(method_name)
        self.max_depth = 10.0
        self.min_depth = 0.1
        self.last_left_img = None
        self.last_right_img = None
        self.last_depth = None
        self.last_rgbd = None
        self.last_keypoints = None
        self.last_descriptors = None
        
        # Stereo matching parameters - tuned for closer regions
        self.stereo_block_size = 5
        self.stereo_num_disparities = 96
        self.stereo_min_disparity = 0
        self.stereo_uniqueness_ratio = 15
        self.stereo_speckle_window_size = 150
        self.stereo_speckle_range = 2
        self.stereo_disp12_max_diff = 1
        self.stereo_pre_filter_cap = 63
        self.stereo_mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        
        # Initialize stereo matcher
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=self.stereo_min_disparity,
            numDisparities=self.stereo_num_disparities,
            blockSize=self.stereo_block_size,
            P1=8 * 3 * self.stereo_block_size**2,
            P2=32 * 3 * self.stereo_block_size**2,
            disp12MaxDiff=self.stereo_disp12_max_diff,
            uniquenessRatio=self.stereo_uniqueness_ratio,
            speckleWindowSize=self.stereo_speckle_window_size,
            speckleRange=self.stereo_speckle_range,
            preFilterCap=self.stereo_pre_filter_cap,
            mode=self.stereo_mode
        )
        
        # Feature tracking parameters
        self.max_features = 3000
        self.feature_detector = cv2.ORB_create(self.max_features)
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_feature_matches = 50
        self.ransac_thresh = 2.0
        self.ransac_confidence = 0.99
        
        # Motion filtering parameters
        self.max_translation = 1.0  # Maximum translation in meters between frames
        self.max_rotation = 0.2     # Maximum rotation in radians between frames
        
        # ROI for focusing on important regions (will skip sky)
        self.use_roi = True
        self.roi_top_percent = 0.4  # Skip top 40% of image (likely sky)
        
        # Debug and validation mode
        self.debug_mode = False

    def setup_method(self, method_name: str) -> None:
        """ Sets up the odometry computation method based on the provided method name.
        Args:
            method_name: The name of the odometry method to use ('hybrid' or 'point_to_plane').
        """
        if method_name == "hybrid":
            self.method = o3d.t.pipelines.odometry.Method.Hybrid
        elif method_name == "point_to_plane":
            self.method = o3d.t.pipelines.odometry.Method.PointToPlane
        else:
            raise ValueError("Odometry method does not exist!")

    def compute_depth_from_stereo(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """ Computes a depth map from stereo image pair using semi-global block matching,
        with specific focus on the lower portion of the image (ground region).
        Args:
            left_img: The left stereo image as a numpy ndarray.
            right_img: The right stereo image as a numpy ndarray.
        Returns:
            Computed depth map as a numpy ndarray.
        """
        height, width = left_img.shape[:2]
        
        # Define ROI to exclude sky (if enabled)
        if self.use_roi:
            roi_top = int(height * self.roi_top_percent)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            roi_mask[roi_top:, :] = 255
        else:
            roi_mask = np.ones((height, width), dtype=np.uint8) * 255
            
        # Convert to grayscale if images are colored
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_img.copy()
            right_gray = right_img.copy()
            
        # Apply ROI masking - set sky region to uniform value
        if self.use_roi:
            sky_value = np.median(left_gray[0:roi_top, :]).astype(np.uint8)
            left_gray_roi = left_gray.copy()
            right_gray_roi = right_gray.copy()
            left_gray_roi[0:roi_top, :] = sky_value
            right_gray_roi[0:roi_top, :] = sky_value
        else:
            left_gray_roi = left_gray
            right_gray_roi = right_gray
            
        # Enhance contrast in the ROI for better feature matching
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        left_gray_enhanced = clahe.apply(left_gray_roi)
        right_gray_enhanced = clahe.apply(right_gray_roi)
        
        # Further preprocessing
        left_gray_filtered = cv2.GaussianBlur(left_gray_enhanced, (3, 3), 0)
        right_gray_filtered = cv2.GaussianBlur(right_gray_enhanced, (3, 3), 0)
            
        # Compute disparity map
        disparity = self.stereo_matcher.compute(left_gray_filtered, right_gray_filtered)
        disparity = np.float32(disparity) / 16.0  # Scale factor for SGBM output
        
        # Apply ROI masking - zero out sky regions in disparity
        if self.use_roi:
            disparity[0:roi_top, :] = 0
        
        # Basic post-processing
        # Apply morphological operations to clean up the disparity map
        kernel = np.ones((3, 3), np.uint8)
        disparity_mask = (disparity > 0).astype(np.uint8)
        disparity_mask = cv2.morphologyEx(disparity_mask, cv2.MORPH_CLOSE, kernel)
        disparity_mask = cv2.morphologyEx(disparity_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply additional noise filtering
        disparity_filtered = cv2.medianBlur(disparity, 5)
        
        # Convert disparity to depth using the formula: depth = baseline * focal_length / disparity
        depth = np.zeros_like(disparity_filtered, dtype=np.float32)
        focal_length = self.intrinsics_np[0, 0]  # Assuming fx is at index [0,0]
        
        # Only compute depth for valid disparities
        valid_mask = (disparity_filtered > 0) & np.isfinite(disparity_filtered) & (disparity_mask > 0)
        depth[valid_mask] = self.baseline * focal_length / disparity_filtered[valid_mask]
        
        # Filter depths outside the valid range
        depth[depth < self.min_depth] = 0.0
        depth[depth > self.max_depth] = 0.0
        
        # Apply distance-based weighting - give higher importance to closer objects
        weight_map = np.zeros_like(depth)
        valid_depth = (depth > 0)
        if np.any(valid_depth):
            max_valid_depth = np.max(depth[valid_depth])
            weight_map[valid_depth] = 1.0 - (depth[valid_depth] / max_valid_depth) * 0.5
            # Apply the weight map to create a weighted depth map
            depth = depth * weight_map
        
        # Final bilateral filter to smooth while preserving edges
        depth_filtered = cv2.bilateralFilter(depth, 5, 0.5, 2.0)
        
        if self.debug_mode:
            plt.figure(figsize=(16, 10))
            plt.subplot(231)
            plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) if len(left_img.shape) == 3 else left_img, cmap='gray')
            plt.title('Left Image')
            plt.subplot(232)
            plt.imshow(left_gray_enhanced, cmap='gray')
            plt.title('Enhanced Left Image')
            plt.subplot(233)
            plt.imshow(roi_mask, cmap='gray')
            plt.title('ROI Mask')
            plt.subplot(234)
            plt.imshow(disparity_filtered, cmap='jet')
            plt.title('Filtered Disparity Map')
            plt.colorbar()
            plt.subplot(235)
            plt.imshow(depth, cmap='viridis')
            plt.title('Raw Depth Map')
            plt.colorbar()
            plt.subplot(236)
            plt.imshow(depth_filtered, cmap='viridis')
            plt.title('Final Depth Map')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('depth_computation_debug.png')
            plt.close()
        
        return depth_filtered

    def detect_and_compute_features(self, image: np.ndarray):
        """ Detects and computes features for a given image using ORB,
        with focus on the lower portion of the image where more stable features exist.
        Args:
            image: Input image as numpy array
        Returns:
            Tuple of keypoints and descriptors
        """
        height, width = image.shape[:2]
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Create ROI mask to ignore sky region
        if self.use_roi:
            roi_top = int(height * self.roi_top_percent)
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[roi_top:, :] = 255
        else:
            mask = None
            
        # Enhance contrast for better feature detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # Apply preprocessing to improve feature detection
        gray_filtered = cv2.GaussianBlur(gray_enhanced, (3, 3), 0)
        
        # Detect and compute keypoints and descriptors
        keypoints = self.feature_detector.detect(gray_filtered, mask)
        keypoints, descriptors = self.feature_detector.compute(gray_filtered, keypoints)
        
        if self.debug_mode and keypoints:
            # Draw ROI and keypoints for debugging
            debug_img = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)
            if self.use_roi:
                cv2.line(debug_img, (0, roi_top), (width, roi_top), (0, 0, 255), 2)
            debug_img = cv2.drawKeypoints(debug_img, keypoints, None, color=(0, 255, 0), flags=0)
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Detected Keypoints: {len(keypoints)}')
            plt.savefig('keypoints_debug.png')
            plt.close()
            
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """ Matches features between two sets of descriptors.
        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors
        Returns:
            List of matches
        """
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        
        matches = self.feature_matcher.match(desc1, desc2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Keep only good matches
        good_matches = matches[:int(len(matches) * 0.75)]
        
        return good_matches

    def estimate_pose_from_features(self, kp1, kp2, matches, intrinsics):
        """ Estimates relative pose from matched features using RANSAC.
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of matches between kp1 and kp2
            intrinsics: Camera intrinsics matrix
        Returns:
            Tuple of (success, transformation matrix)
        """
        if len(matches) < self.min_feature_matches:
            print(f"Not enough matches: {len(matches)} < {self.min_feature_matches}")
            return False, np.eye(4)
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Estimate essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, intrinsics, 
            method=cv2.RANSAC, 
            prob=self.ransac_confidence, 
            threshold=self.ransac_thresh
        )
        
        if E is None or E.shape != (3, 3):
            print("Essential matrix estimation failed")
            return False, np.eye(4)
        
        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, intrinsics, mask=mask)
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t.ravel()
        
        # Validate the transformation
        if not self.validate_transformation(transform):
            print("Transformation rejected by validation")
            return False, np.eye(4)
        
        return True, transform

    def validate_transformation(self, transform):
        """ Validates a transformation matrix against motion constraints.
        Args:
            transform: 4x4 transformation matrix
        Returns:
            True if valid, False otherwise
        """
        # Extract translation and rotation
        translation = np.linalg.norm(transform[:3, 3])
        
        # Get rotation in axis-angle representation
        rot = Rotation.from_matrix(transform[:3, :3])
        rot_angle = np.linalg.norm(rot.as_rotvec())
        
        # Check against constraints
        if translation > self.max_translation:
            print(f"Translation too large: {translation:.3f}m > {self.max_translation}m")
            return False
            
        if rot_angle > self.max_rotation:
            print(f"Rotation too large: {rot_angle:.3f}rad > {self.max_rotation}rad")
            return False
            
        return True

    def update_last_frames(self, left_img: np.ndarray, right_img: np.ndarray) -> None:
        """ Updates the last stereo frames stored in the system.
        Args:
            left_img: The new left stereo image as a numpy ndarray.
            right_img: The new right stereo image as a numpy ndarray.
        """
        self.last_left_img = left_img.copy()
        self.last_right_img = right_img.copy()
        
        # Compute depth from stereo pair
        self.last_depth = self.compute_depth_from_stereo(left_img, right_img)
        
        # Create RGBD image for Open3D
        self.last_rgbd = o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(np.ascontiguousarray(left_img).astype(np.float32)).to(self.device),
            o3d.t.geometry.Image(np.ascontiguousarray(self.last_depth).astype(np.float32)).to(self.device))
            
        # Compute features for feature tracking
        self.last_keypoints, self.last_descriptors = self.detect_and_compute_features(left_img)

    def estimate_rel_pose(self, left_img: np.ndarray, right_img: np.ndarray, init_transform=np.eye(4)):
        """ Estimates the relative pose of the current frame with respect to the last frame
        using a hybrid approach combining stereo depth odometry and feature tracking.
        
        Args:
            left_img: The current left stereo image as a numpy ndarray.
            right_img: The current right stereo image as a numpy ndarray.
            init_transform: An initial transformation guess as a numpy ndarray. Defaults to identity.
        Returns:
            The relative transformation matrix as a numpy ndarray.
        """
        # If this is the first call, just store the stereo pair and return identity transform
        if self.last_rgbd is None:
            self.update_last_frames(left_img, right_img)
            return np.eye(4)
        
        # First, try feature-based pose estimation for robustness
        current_keypoints, current_descriptors = self.detect_and_compute_features(left_img)
        
        if self.debug_mode:
            print(f"Features detected: {len(current_keypoints)} (current) vs {len(self.last_keypoints)} (last)")
        
        matches = self.match_features(self.last_descriptors, current_descriptors)
        
        if self.debug_mode:
            print(f"Feature matches found: {len(matches)}")
            match_img = cv2.drawMatches(self.last_left_img, self.last_keypoints, 
                                       left_img, current_keypoints, 
                                       matches[:100], None, flags=2)
            plt.figure(figsize=(15, 8))
            plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Feature Matches: {len(matches)}')
            plt.savefig('matches_debug.png')
            plt.close()
        
        # Try to estimate pose from features
        feature_success, feature_transform = self.estimate_pose_from_features(
            self.last_keypoints, current_keypoints, matches, self.intrinsics_np)
        
        # Compute depth from current stereo pair
        current_depth = self.compute_depth_from_stereo(left_img, right_img)
        
        # Count valid depth pixels to check depth quality
        valid_depth_ratio = np.sum(current_depth > 0) / current_depth.size
        
        if self.debug_mode:
            print(f"Valid depth ratio: {valid_depth_ratio:.3f}")
        
        # Create current RGBD image
        current_rgbd = o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(np.ascontiguousarray(left_img).astype(np.float32)).to(self.device),
            o3d.t.geometry.Image(np.ascontiguousarray(current_depth).astype(np.float32)).to(self.device))
        
        # If we have good feature matches and poor depth, use feature-based estimate
        if feature_success and valid_depth_ratio < 0.3:
            rel_transform = feature_transform
            if self.debug_mode:
                print("Using feature-based pose estimation (poor depth)")
        else:
            # Otherwise, try RGBD odometry first
            try:
                # Use feature transform as initial guess if available
                initial_guess = feature_transform if feature_success else init_transform
                
                # Perform odometry
                result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                    self.last_rgbd, current_rgbd, self.intrinsics, o3c.Tensor(initial_guess),
                    1.0, self.max_depth, self.criteria_list, self.method)
                
                # Get transformation matrix
                rgbd_transform = result.transformation.cpu().numpy()
                
                # Validate the transformation
                if self.validate_transformation(rgbd_transform):
                    rel_transform = rgbd_transform
                    if self.debug_mode:
                        print("Using RGBD odometry estimate")
                else:
                    # Fallback to feature transform if available
                    if feature_success:
                        rel_transform = feature_transform
                        if self.debug_mode:
                            print("Using feature-based pose estimation (RGBD validation failed)")
                    else:
                        # No good estimates, use identity with small motion
                        rel_transform = np.eye(4)
                        if self.debug_mode:
                            print("Using identity transform (no good estimates)")
            except Exception as e:
                # Handle potential exceptions in RGBD odometry
                print(f"RGBD odometry failed: {e}")
                if feature_success:
                    rel_transform = feature_transform
                    if self.debug_mode:
                        print("Using feature-based pose estimation (RGBD failure)")
                else:
                    rel_transform = np.eye(4)
                    if self.debug_mode:
                        print("Using identity transform (all methods failed)")
        
        # Update last frames
        self.last_rgbd = current_rgbd.clone()
        self.last_left_img = left_img.copy()
        self.last_right_img = right_img.copy()
        self.last_depth = current_depth.copy()
        self.last_keypoints = current_keypoints
        self.last_descriptors = current_descriptors

        # Adjust for the coordinate system difference
        rel_transform[0, [1, 2, 3]] *= -1
        rel_transform[1, [0, 2, 3]] *= -1
        rel_transform[2, [0, 1, 3]] *= -1

        return rel_transform
    
    def set_debug_mode(self, enabled=True):
        """Enable or disable debug mode with visualizations.
        Args:
            enabled: Boolean to toggle debug mode
        """
        self.debug_mode = enabled
        print(f"Debug mode {'enabled' if enabled else 'disabled'}")
# class StereoVisualOdometer(object):

#     def __init__(self, intrinsics: np.ndarray, baseline: float, method_name="hybrid", device="cuda"):
#         """ Initializes the stereo visual odometry system with specified intrinsics, baseline, method, and device.
#         Args:
#             intrinsics: Camera intrinsic parameters.
#             baseline: The baseline distance between stereo cameras in meters.
#             method_name: The name of the odometry computation method to use ('hybrid' or 'point_to_plane').
#             device: The computation device ('cuda' or 'cpu').
#         """
#         device = "CUDA:0" if device == "cuda" else "CPU:0"
#         self.device = o3c.Device(device)
#         self.intrinsics = o3d.core.Tensor(intrinsics, o3d.core.Dtype.Float64)
#         self.baseline = baseline
#         self.last_abs_pose = None
#         self.last_frame = None
#         self.criteria_list = [
#             o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500),
#             o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500),
#             o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500)]
#         self.setup_method(method_name)
#         self.max_depth = 10.0
#         self.last_left_img = None
#         self.last_right_img = None
#         self.last_depth = None
#         self.last_rgbd = None
        
#         # Stereo matching parameters
#         self.stereo_block_size = 5
#         self.stereo_num_disparities = 128
#         self.stereo_min_disparity = 0
#         self.stereo_uniqueness_ratio = 10
#         self.stereo_speckle_window_size = 100
#         self.stereo_speckle_range = 32
#         self.stereo_disp12_max_diff = 1
        
#         # Initialize stereo matcher
#         self.stereo_matcher = cv2.StereoSGBM_create(
#             minDisparity=self.stereo_min_disparity,
#             numDisparities=self.stereo_num_disparities,
#             blockSize=self.stereo_block_size,
#             P1=8 * 3 * self.stereo_block_size**2,
#             P2=32 * 3 * self.stereo_block_size**2,
#             disp12MaxDiff=self.stereo_disp12_max_diff,
#             uniquenessRatio=self.stereo_uniqueness_ratio,
#             speckleWindowSize=self.stereo_speckle_window_size,
#             speckleRange=self.stereo_speckle_range,
#             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#         )

#     def setup_method(self, method_name: str) -> None:
#         """ Sets up the odometry computation method based on the provided method name.
#         Args:
#             method_name: The name of the odometry method to use ('hybrid' or 'point_to_plane').
#         """
#         if method_name == "hybrid":
#             self.method = o3d.t.pipelines.odometry.Method.Hybrid
#         elif method_name == "point_to_plane":
#             self.method = o3d.t.pipelines.odometry.Method.PointToPlane
#         else:
#             raise ValueError("Odometry method does not exist!")

#     def compute_depth_from_stereo(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
#         """ Computes a depth map from stereo image pair using semi-global block matching.
#         Args:
#             left_img: The left stereo image as a numpy ndarray.
#             right_img: The right stereo image as a numpy ndarray.
#         Returns:
#             Computed depth map as a numpy ndarray.
#         """
#         # Convert to grayscale if images are colored
#         if len(left_img.shape) == 3:
#             left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
#             right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
#         else:
#             left_gray = left_img
#             right_gray = right_img
            
#         # Compute disparity
#         disparity = self.stereo_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
#         # Filter invalid disparities
#         valid_mask = disparity > 0
        
#         # Convert disparity to depth using the formula: depth = baseline * focal_length / disparity
#         depth = np.zeros_like(disparity, dtype=np.float32)
#         focal_length = self.intrinsics.cpu().numpy()[0, 0]  # Assuming fx is at index [0,0]
#         depth[valid_mask] = self.baseline * focal_length / disparity[valid_mask]
        
#         # Set invalid depths to 0
#         depth[~valid_mask] = 0.0
#         depth[depth > self.max_depth] = 0.0
        
#         return depth

#     def update_last_frames(self, left_img: np.ndarray, right_img: np.ndarray) -> None:
#         """ Updates the last stereo frames stored in the system.
#         Args:
#             left_img: The new left stereo image as a numpy ndarray.
#             right_img: The new right stereo image as a numpy ndarray.
#         """
#         self.last_left_img = left_img.copy()
#         self.last_right_img = right_img.copy()
        
#         # Compute depth from stereo pair
#         self.last_depth = self.compute_depth_from_stereo(left_img, right_img)
        
#         # Create RGBD image for Open3D
#         self.last_rgbd = o3d.t.geometry.RGBDImage(
#             o3d.t.geometry.Image(np.ascontiguousarray(left_img).astype(np.float32)).to(self.device),
#             o3d.t.geometry.Image(np.ascontiguousarray(self.last_depth).astype(np.float32)).to(self.device))

#     def estimate_rel_pose(self, left_img: np.ndarray, right_img: np.ndarray, init_transform=np.eye(4)):
#         """ Estimates the relative pose of the current frame with respect to the last frame using stereo odometry.
#         Args:
#             left_img: The current left stereo image as a numpy ndarray.
#             right_img: The current right stereo image as a numpy ndarray.
#             init_transform: An initial transformation guess as a numpy ndarray. Defaults to the identity matrix.
#         Returns:
#             The relative transformation matrix as a numpy ndarray.
#         """
#         # If this is the first call, just store the stereo pair and return identity transform
#         if self.last_rgbd is None:
#             self.update_last_frames(left_img, right_img)
#             return np.eye(4)
        
#         # Compute depth from current stereo pair
#         current_depth = self.compute_depth_from_stereo(left_img, right_img)
        
#         # Create current RGBD image
#         current_rgbd = o3d.t.geometry.RGBDImage(
#             o3d.t.geometry.Image(np.ascontiguousarray(left_img).astype(np.float32)).to(self.device),
#             o3d.t.geometry.Image(np.ascontiguousarray(current_depth).astype(np.float32)).to(self.device))
        
#         # Perform odometry
#         rel_transform = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
#             self.last_rgbd, current_rgbd, self.intrinsics, o3c.Tensor(init_transform),
#             1.0, self.max_depth, self.criteria_list, self.method)
        
#         # Update last frames
#         self.last_rgbd = current_rgbd.clone()
#         self.last_left_img = left_img.copy()
#         self.last_right_img = right_img.copy()
#         self.last_depth = current_depth.copy()

#         # Adjust for the coordinate system difference
#         rel_transform = rel_transform.transformation.cpu().numpy()
#         rel_transform[0, [1, 2, 3]] *= -1
#         rel_transform[1, [0, 2, 3]] *= -1
#         rel_transform[2, [0, 1, 3]] *= -1

#         return rel_transform