import numpy as np
from numpy.typing import NDArray
import orbslam3
import importlib.resources

from pytransform3d.transformations import transform_from, concat
from pytransform3d.rotations import matrix_from_euler

from maple.pose.estimator import Estimator
from maple.utils import carla_to_pytransform


class OrbslamEstimator(Estimator):
    """Provides pose estimation using ORB-SLAM3."""

    agent: None
    left: None  # This is the carla.SensorPosition object
    right: None  # This is the carla.SensorPosition object
    slam: None  # The ORB-SLAM3 system
    init_time: float  # The time of the first frame
    orbslam_global: NDArray  # The orbslam frame in the global frame
    _imu_data: list  # Accumulated IMU data

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

        # Get the position of the orbslam frame in the global frame
        self.orbslam_global = carla_to_pytransform(agent.get_initial_position())

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

        self.slam.initialize()

    def set_orbslam_global(self, rover_global):
        # Get the position of the orbslam frame in the global frame
        # camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))
        # self.orbslam_global = concat(camera_rover, rover_global)
        self.orbslam_global = rover_global

    @property
    def lost(self) -> bool:
        return self.slam.is_lost()

    def shutdown(self):
        """Shutdown ORB-SLAM"""
        self.slam.shutdown()

    def reset(self):
        """Reset ORB-SLAM"""
        # TODO: Figure out how to reinitialize the position

        self.init_time = (
            self.agent.get_mission_time()
        )  # Reset the reference time to now
        self._imu_data = []  # Clear any accumulated IMU data
        self.pose_dict = {}  # Clear the pose dictionary
        self.frame_id = 0  # Reset the frame ID
        self.slam.shutdown()
        self.slam.initialize()

    def estimate(self, input_data: dict) -> NDArray:
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

        # Correct axis orientation (z-forward to z-up)
        correction = transform_from(
            matrix_from_euler(
                [-np.pi / 2, 0, -np.pi / 2],
                2,
                1,
                0,
                False,
            ),
            [0, 0, 0],
        )
        camera_orbslam = concat(estimate, correction)

        # return camera_orbslam

        # TODO: Reset the orbslam frame when the map resets

        # Convert from the orbslam frame to the global frame
        return concat(camera_orbslam, self.orbslam_global)

    def _estimate_mono(self, input_data: dict) -> NDArray:
        try:
            image = input_data["Grayscale"][self.left]
        except (KeyError, TypeError):
            return None

        # We do have images, so run ORB-SLAM
        success = self.slam.process_image_mono(image, self._timestamp)

        if success:
            # Update the pose dictionary
            pose = self._get_pose()
            self.pose_dict[self.frame_id] = pose
            self.frame_id += 1

            return pose
        else:
            return None

    def _log_imu(self):
        # Get the IMU data and append the timestamp
        imu_data = self.agent.get_imu_data()

        # TODO: Test correcting IMU data
        # Correct axis orientation (z-up to z-forward)
        correction = matrix_from_euler(
            [-np.pi / 2, 0, -np.pi / 2],
            2,
            1,
            0,
            False,
        ).T

        imu_data[:3] = imu_data[:3] @ correction  # Acceleration
        imu_data[3:] = imu_data[3:] @ correction  # Angular velocity

        imu_data.append(self._timestamp)
        self._imu_data.append(imu_data)

    def _estimate_stereo(self, input_data: dict) -> NDArray:
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

        self.pose_dict[self.frame_id] = pose
        self.frame_id += 1
        return pose

    def _get_pose(self) -> NDArray | None:
        """Get the last element of the trajectory."""
        pose = self.slam.get_current_pose()

        if pose is None:
            return None

        # Correct axis orientation (z-forward to z-up)
        correction = transform_from(
            matrix_from_euler(
                [-np.pi / 2, 0, -np.pi / 2],
                2,
                1,
                0,
                False,
            ),
            [0, 0, 0],
        )

        camera_orbslam = concat(pose, correction)
        camera_global = concat(camera_orbslam, self.orbslam_global)
        return camera_global

    def _get_trajectory(self) -> list[NDArray]:
        """Get the trajectory from ORB-SLAM."""
        trajectory = self.slam.get_trajectory()

        # Correct axis orientation (z-forward to z-up)
        correction = transform_from(
            matrix_from_euler(
                [-np.pi / 2, 0, -np.pi / 2],
                2,
                1,
                0,
                False,
            ),
            [0, 0, 0],
        )

        cameras_orbslam = [concat(t, correction) for t in trajectory]
        cameras_global = [concat(c_o, self.orbslam_global) for c_o in cameras_orbslam]
        return cameras_global

    def get_pose_dict(self):
        return self.pose_dict

    @property
    def _timestamp(self) -> float:
        """Get the current timestamp in seconds."""
        return self.agent.get_mission_time() - self.init_time
