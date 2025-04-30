import numpy as np
from numpy.typing import NDArray
import orbslam3
import importlib.resources

from pytransform3d.transformations import concat, invert_transform

from maple.pose.estimator import Estimator
from maple.utils import carla_to_pytransform, pytransform_to_tuple, tuple_to_pytransform


class OrbslamEstimator(Estimator):
    """Provides pose estimation using ORB-SLAM3."""

    agent: None
    left: None  # This is the carla.SensorPosition object
    right: None  # This is the carla.SensorPosition object
    slam: None  # The ORB-SLAM3 system
    init_time: float  # The time of the first frame
    _imu_data: list  # Accumulated IMU data
    camera_init_global: NDArray  # The initial camera position in the global frame

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
        self.pose_dict = {}  # Not sure where this should live
        self.frame_id = 0
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

        # Initialize the ORB-SLAM3 system
        self.slam.initialize()

        # Get the position of the orbslam frame in the global frame
        rover_global = carla_to_pytransform(agent.get_initial_position())
        self.set_orbslam_global(rover_global)

    def set_orbslam_global(self, rover_global):
        """Set the ORB-SLAM global frame.

        Args:
            rover_global: The position of the rover in the global frame when orbslam is initialized
        """
        # Get the position of the camera in the rover frame
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))
        camera_global = concat(camera_rover, rover_global)
        self.camera_init_global = camera_global

    def _correct_estimate(self, estimate: NDArray) -> NDArray:
        """Corrects the estimate to be in the global frame.

        Args:
            estimate: The transformation matrix from ORB-SLAM

        Returns:
            NDArray: The rover in the global frame
        """
        # Get the rotation of the orbslam frame in the initial camera frame
        x_o, y_o, z_o, roll_o, pitch_o, yaw_o = pytransform_to_tuple(estimate)

        # Create a transform of just the translation with the axes swapped
        camera_orbslam = np.eye(4)
        camera_orbslam[:3, 3] = [z_o, -x_o, -y_o]

        # Get the position of the orbslam frame in the global frame
        orbslam_global = self.camera_init_global
        camera_global = concat(camera_orbslam, orbslam_global)

        # Get the position of the rover in the camera frame
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))
        rover_camera = invert_transform(camera_rover)

        # Get the rover in the global frame
        rover_global = concat(rover_camera, camera_global)

        # Extract the translation and rotation
        x, y, z, _, _, _ = pytransform_to_tuple(rover_global)
        _, _, _, roll_i, pitch_i, yaw_i = pytransform_to_tuple(self.camera_init_global)

        # Apply the orbslam camera rotations to the initial rover rotation,
        # correcting the axes orientation and direction
        rover_global = tuple_to_pytransform(
            (x, y, z, roll_i + yaw_o, pitch_i - roll_o, yaw_i - pitch_o)
        )

        return rover_global

    @property
    def lost(self) -> bool:
        return self.slam.is_lost()

    def shutdown(self):
        """Shutdown ORB-SLAM"""
        self.slam.shutdown()

    def reset(self, rover_global: NDArray):
        """Reinitialize ORB-SLAM with a new rover position."""
        # Restart the ORB-SLAM system
        # TODO: Check if this works as expected
        self.slam.shutdown()
        self.slam.initialize()

        self.init_time = (
            self.agent.get_mission_time()
        )  # Reset the reference time to now
        self._imu_data = []  # Clear any accumulated IMU data
        self.pose_dict = {}  # Clear the pose dictionary
        self.frame_id = 0  # Reset the frame ID
        # TODO: Figure out how to reinitialize the position
        self.set_orbslam_global(rover_global)

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
            return None

        # Check for NaNs, this indicates a fatal error with orbslam
        if np.isnan(estimate).any():
            print(f"NANs in pose estimate: \n{estimate}")
            return None

        self.pose_dict[self.frame_id] = estimate
        self.frame_id += 1

        return self._correct_estimate(estimate)

        # TODO: Reset the orbslam frame when the map resets

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

    def get_pose_dict(self):
        return self.pose_dict

    @property
    def _timestamp(self) -> float:
        """Get the current timestamp in seconds."""
        return self.agent.get_mission_time() - self.init_time
