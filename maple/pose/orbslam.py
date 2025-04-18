import numpy as np
from numpy.typing import NDArray
import orbslam3
import importlib.resources
import tarfile

from maple.pose.estimator import Estimator
from maple.utils import carla_to_pytransform


class OrbslamEstimator(Estimator):
    """Provides pose estimation using ORB-SLAM3."""

    agent: None
    left: None  # This is the carla.SensorPosition object
    right: None  # This is the carla.SensorPosition object
    slam: None  # The ORB-SLAM3 system
    init_time: float  # The time of the first frame
    rover_global: NDArray  # The rover's initial position in the global frame
    _imu_data: list  # Accumulated IMU data

    def __init__(self, agent, left, right=None, mode="mono"):
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

        self.init_time = agent.get_mission_time()
        self.rover_global = carla_to_pytransform(agent.get_initial_position())

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

        # Try to load the vocabulary file
        try:
            with importlib.resources.path("resources", "ORBvoc.txt") as fpath:
                self.vocabulary = str(fpath)

        except FileNotFoundError:
            # Try to extract from tar if file doesn't exist
            with tarfile.open("resources/ORBvoc.txt.tar.gz", "r:gz") as tar:
                tar.extractall(path="resources")

            # Try loading again after extraction
            with importlib.resources.path("resources", "ORBvoc.txt") as fpath:
                self.vocabulary = str(fpath)

        # Find the camera config
        with importlib.resources.path("resources", "LAC_cam.yaml") as fpath:
            self.camera_config = str(fpath)

        # Initialize the ORB-SLAM3 system
        if mode == "stereo":
            self.slam = orbslam3.system(
                self.vocabulary, self.camera_config, orbslam3.Sensor.STEREO
            )
        elif mode == "stereo_imu":
            self.slam = orbslam3.system(
                self.vocabulary, self.camera_config, orbslam3.Sensor.STEREO_IMU
            )
        elif mode == "mono":
            self.slam = orbslam3.system(
                self.vocabulary, self.camera_config, orbslam3.Sensor.MONO
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.slam.initialize()

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
            return self._estimate_mono(input_data)
        else:
            return self._estimate_stereo(input_data)

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

    def _estimate_stereo(self, input_data: dict) -> NDArray:
        # Log IMU data
        if self.mode == "stereo_imu":
            # Convert IMU data to IMUPoint
            imu_data = self.agent.get_imu_data()
            imu_point = orbslam3.IMUPoint(0, 0, 0, 0, 0, 0, 0)
            imu_point.a = imu_data[0:3]
            imu_point.w = imu_data[3:6]
            imu_point.t = self._timestamp
            self._imu_data.append(imu_point)

        # Check for images
        try:
            left = input_data["Grayscale"][self.left]
            right = input_data["Grayscale"][self.right]
        except (KeyError, TypeError):
            return None

        # We do have images, so run ORB-SLAM
        # TODO: We probably want to update the ORBSLAM bindings to be a little more sensible
        if self.mode == "stereo":
            success = self.slam.process_image_stereo(left, right, self._timestamp)
        elif self.mode == "stereo_imu":
            success = self.slam.process_image_stereo_imu(
                left, right, self._timestamp, self._imu_data
            )
            self._imu_data = []  # Clear the IMU data after processing
        else:
            success = False

        if success:
            # Update the pose dictionary
            pose = self._get_pose()
            self.pose_dict[self.frame_id] = pose
            self.frame_id += 1

            return pose

        else:
            return None

    def _get_pose(self) -> NDArray:
        """Get the last element of the trajectory."""
        # TODO: Refactor the trajectory stuff

        trajectory = self._get_trajectory()
        if len(trajectory) > 0:
            return trajectory[-1]
        else:
            return self.rover_global

    def _get_trajectory(self):
        """Get the trajectory from ORB-SLAM."""
        # TODO: Refactor this

        trajectory = self.slam.get_trajectory()

        # Rotation to convert Z-forward to Z-up
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        new_traj = []
        for pose in trajectory:
            R_wc = pose[:3, :3]
            t_wc = pose[:3, 3]

            # Apply rotation
            R_new = R @ R_wc
            t_new = R @ t_wc

            T_new = np.eye(4)
            T_new[:3, :3] = R_new
            T_new[:3, 3] = t_new
            new_traj.append(T_new)
        return new_traj

    def get_pose_dict(self):
        return self.pose_dict

    @property
    def _timestamp(self) -> float:
        """Get the current timestamp in seconds."""
        return self.agent.get_mission_time() - self.init_time
