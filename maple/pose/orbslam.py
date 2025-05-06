import numpy as np
from numpy.typing import NDArray
import orbslam3
import importlib.resources
import time
from pytransform3d.transformations import concat, invert_transform

from maple.pose.estimator import Estimator
from maple.utils import carla_to_pytransform


class OrbslamEstimator(Estimator):
    """Provides pose estimation using ORB-SLAM3."""

    def __init__(self, agent, left, right=None, mode="stereo"):
        """Create the estimator.

        Args:
            agent: The Agent instance
            left: The left camera instance (string or object)
            right: The right camera instance (string or object) (not required for mono mode)
            mode: The mode to run the estimator in ("mono", "stereo", "stereo_imu")
        """
        self.agent = agent
        self._imu_data = []
        self.mode = mode
        self.init_time = agent.get_mission_time()

        # Validate the camera configuration
        # Look through all the camera objects in the agent's sensors and save the one for the left camera
        for key in agent.sensors().keys():
            if str(key) == str(left):
                self.left = key

        if self.left is None:
            raise ValueError(f"{left} is not defined in the agent's sensors.")

        # If stereo, check that the right camera is defined
        if mode != "mono":
            if right is None:
                raise ValueError("Right camera is required for stereo modes")

            for key in agent.sensors().keys():
                if str(key) == str(right):
                    self.right = key

            if self.right is None:
                raise ValueError(f"{right} is not defined in the agent's sensors.")

        # Load the vocabulary .txt file
        with importlib.resources.path("resources", "ORBvoc.txt") as fpath:
            self.vocabulary = str(fpath)

        # Find the camera config
        with importlib.resources.path("resources", "orbslam_config.yaml") as fpath:
            self.camera_config = str(fpath)

        # Initialize the ORB-SLAM3 system
        if mode == "stereo":
            self.slam = orbslam3.system(
                self.vocabulary, self.camera_config, orbslam3.Sensor.STEREO
            )
        elif mode == "stereo_imu":
            self.slam = orbslam3.system(
                self.vocabulary, self.camera_config, orbslam3.Sensor.IMU_STEREO
            )
        elif mode == "mono":
            self.slam = orbslam3.system(
                self.vocabulary, self.camera_config, orbslam3.Sensor.MONOCULAR
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Get the position of the orbslam frame in the global frame
        self.rover_init_global = carla_to_pytransform(self.agent.get_initial_position())
        self.last_valid_pose = self.rover_init_global
        self._set_orbslam_global(self.rover_init_global)

        # Initialize the ORB-SLAM3 system
        self.slam.initialize()

    def _set_orbslam_global(self, rover_global):
        """Set the ORB-SLAM global frame.

        Args:
            rover_global: The position of the rover in the global frame when orbslam is initialized
        """
        self.rover_init_global = rover_global
        # Get the position of the camera in the rover frame
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))
        camera_global = concat(camera_rover, rover_global)
        self.camera_init_global = camera_global
        self.last_valid_pose = rover_global

    def _correct_estimate(self, estimate: NDArray) -> NDArray:
        """Corrects the estimate to be in the global frame.

        Args:
            estimate: The transformation matrix from ORB-SLAM

        Returns:
            NDArray: The rover in the global frame
        """
        # Get the position of the rover in the camera frame
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))
        rover_camera = invert_transform(camera_rover)

        # Get a rotation matrix to rotate from Z-Forward to Z-Up
        z_forward_to_z_up = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        pose_camera = np.eye(4)
        # Rotate the orientation in place to correct the axes
        pose_camera[:3, :3] = z_forward_to_z_up @ estimate[:3, :3] @ z_forward_to_z_up.T
        # Apply the transform to the position to swap the axes
        pose_camera[:3, 3] = z_forward_to_z_up @ estimate[:3, 3]

        p_r = concat(pose_camera, camera_rover)
        p_g = concat(p_r, self.rover_init_global)
        p_g = concat(rover_camera, p_g)

        return p_g

    def shutdown(self):
        """Shutdown ORB-SLAM"""
        self.slam.shutdown()

    def reset(self, rover_global=None):
        """Reinitialize ORB-SLAM with a new rover position."""
        # Restart the ORB-SLAM system
        # TODO: Check if this works if NaNs are present...

        # THIS WILL CRASH OTHER INSTANCES OF ORB-SLAM
        # self.slam.reset()

        self.slam.shutdown()

        # Doesn't seem necessary, but just in case
        while not self.slam.is_shutdown():
            time.sleep(0.1)

        self.slam.initialize()

        # Wait for ORB-SLAM to be running again
        # Also doesn't seem necessary, but just in case
        while self.slam.is_shutdown():
            time.sleep(0.1)

        # Reset the reference time to now
        self.init_time = self.agent.get_mission_time()

        # Clear any accumulated IMU data
        self._imu_data = []

        # If no rover global is provided, use the last valid pose
        if rover_global is None:
            rover_global = self.last_valid_pose

        # Set the new ORB-SLAM global frame
        self._set_orbslam_global(rover_global)

    def estimate(self, input_data: dict) -> NDArray | None:
        """If an image is present, run ORB-SLAM and return a pose estimate.

        Args:
            input_data: Input data from the environment.

        Returns:
            NDArray: A pose estimate, or None if no image is present.
        """

        # This can get called every frame, even frames with no images.
        # In those cases, we should log the IMU data and then include it in the next frame that does have an image.

        if self.mode == "mono":
            estimate = self._estimate_mono(input_data)
        else:
            estimate = self._estimate_stereo(input_data)

        if estimate is None:
            # TODO: Should we reset the ORB-SLAM system here?
            return None

        # Check for NaNs, this indicates a fatal error with orbslam
        if np.isnan(estimate).any():
            raise RuntimeError("NANs in pose estimate: \n" + str(estimate))

        # Correct the estimate to be in the global frame
        estimate = self._correct_estimate(estimate)

        # TODO: Check if the estimate is close to the last valid pose
        # Check if the estimate is close to the last valid pose
        # If it is far away, reset the origin to the last valid pose

        # Update the last valid pose
        self.last_valid_pose = estimate

        return estimate

    def _log_imu(self):
        # Extract the elements in the camera body frame
        ax_r, ay_r, az_r, gx_r, gy_r, gz_r = self.agent.get_imu_data()

        ax_o = -ay_r
        ay_o = -az_r
        az_o = ax_r

        gx_o = -gy_r
        gy_o = -gz_r
        gz_o = gx_r

        imu_orbslam = [ax_o, ay_o, az_o, gx_o, gy_o, gz_o]

        # Correct the IMU data for the camera orientation
        # TODO: Validate this works for all camera orientations
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))
        rover_camera = invert_transform(camera_rover)

        imu_orbslam[:3] = imu_orbslam[:3] @ rover_camera[:3, :3]
        imu_orbslam[3:] = imu_orbslam[3:] @ rover_camera[:3, :3]

        # Add the timestamp and append to the list
        imu_orbslam.append(self._timestamp)
        self._imu_data.append(imu_orbslam)

    def _estimate_mono(self, input_data: dict) -> NDArray | None:
        try:
            image = input_data["Grayscale"][self.left]
        except (KeyError, TypeError):
            return None

        if image is None:
            return None

        # We do have images, so run ORB-SLAM
        pose = self.slam.process_image_mono(image, self._timestamp)

        return pose

    def _estimate_stereo(self, input_data: dict) -> NDArray | None:
        # Log IMU data
        if self.mode == "stereo_imu":
            self._log_imu()

        # Check cameras are defined
        try:
            left = input_data["Grayscale"][self.left]
            right = input_data["Grayscale"][self.right]
        except (KeyError, TypeError):
            return None

        # Check for images in this frame
        if left is None or right is None:
            return None

        # We do have images, so run ORB-SLAM
        if self.mode == "stereo":
            pose = self.slam.process_image_stereo(left, right, self._timestamp)

        elif self.mode == "stereo_imu":
            pose = self.slam.process_image_stereo_imu(
                left,
                right,
                self._timestamp,
                self._imu_data,
            )
            self._imu_data = []  # Clear the IMU data after processing

        return pose

    def _get_pose(self) -> NDArray | None:
        """Get the last element of the trajectory."""
        pose = self.slam.get_current_pose()

        if pose is None:
            return None

        # Correct position and axis orientation
        return self._correct_estimate(pose)

    def _get_trajectory(self) -> list[NDArray]:
        """Get the trajectory from ORB-SLAM."""
        trajectory = self.slam.get_trajectory()

        # Correct position and axis orientation
        return [self._correct_estimate(t) for t in trajectory]

    @property
    def _timestamp(self) -> float:
        """Get the current timestamp in seconds."""
        return self.agent.get_mission_time() - self.init_time

    def _wrap(self, angles):
        """Wrap angles to -pi to pi"""
        return np.mod(angles + np.pi, 2 * np.pi) - np.pi
