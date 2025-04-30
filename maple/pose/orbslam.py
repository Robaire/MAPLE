import numpy as np
from numpy.typing import NDArray
import orbslam3
import importlib.resources
import pytransform3d.transformations as pyt_t
import pytransform3d.rotations as pyt_r

from maple.pose.estimator import Estimator
from maple.utils import carla_to_pytransform, pytransform_to_tuple, tuple_to_pytransform


class OrbslamEstimator(Estimator):
    """Provides pose estimation using ORB-SLAM3."""

    agent: None
    left: None  # This is the carla.SensorPosition object
    right: None  # This is the carla.SensorPosition object
    slam: None  # The ORB-SLAM3 system
    init_time: float  # The time of the first frame
    rover_global: NDArray  # The rover's initial position in the global frame
    front: bool
    _imu_data: list  # Accumulated IMU data

    def __init__(self, agent, left, right=None, mode="stereo", front=True):
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
        self.front = front

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
        # Check for images
        try:
            left = input_data["Grayscale"][self.left]
            right = input_data["Grayscale"][self.right]
        except (KeyError, TypeError):
            return None

        # Log IMU data
        if self.mode == "stereo_imu":
            # Convert IMU data to IMUPoint
            imu_data = self.agent.get_imu_data()
            imu_point = orbslam3.IMUPoint(0, 0, 0, 0, 0, 0, 0)
            imu_point.a = imu_data[0:3]
            imu_point.w = imu_data[3:6]
            imu_point.t = self._timestamp

            self._imu_data.append(imu_point)
        # We do have images, so run ORB-SLAM
        # TODO: We probably want to update the ORBSLAM bindings to be a little more sensible
        if self.mode == "stereo":
            success = self.slam.process_image_stereo(left, right, self._timestamp)
        elif self.mode == "stereo_imu":
            print("timestamp: ", self._timestamp)
            print("imu data: ", self._imu_data)
            success = self.slam.process_image_stereo_imu(
                left, right, self._timestamp, self._imu_data
            )
            self._imu_data = []  # Clear the IMU data after processing
        else:
            success = None

        # print("success: ", success)
        # if not np.allclose(success, np.zeros((4, 4))):
        # print("success: ", success)
        if success is None:
            return None

        if np.isnan(success).any():
            print("NAN in estimate, SOPHOS failed")
            return None

        if success is not None:
            # Update the pose dictionary
            pose = self._get_pose()
            self.pose_dict[self.frame_id] = pose
            self.frame_id += 1

            return pose

        else:
            return None

    def _get_pose(self) -> NDArray | None:
        """Get the last element of the trajectory."""
        T_orb_estimate = self.slam.get_current_pose()

        if T_orb_estimate is None:
            return None

        ########################################################
        # # ORBSLAM   ->  ROVER
        # # X-left    ->  X-forward
        # # Y-up      ->  Y-left
        # # Z-forward ->  Z-up
        # R_corr = np.array([[0, 0, 1],
        #                    [1, 0, 0],
        #                    [0, 1, 0]])

        # T_camera_estimate = np.eye(4)
        # T_camera_estimate[:3, :3] = R_corr @ T_orb_estimate[:3, :3]
        # T_camera_estimate[:3, 3] = R_corr @ T_orb_estimate[:3, 3]

        # # Transform from the orbslam (camera) frame to the agent frame
        # T_agent_camera = carla_to_pytransform(self.agent.get_camera_position(self.left))
        # # T_camera_agent = pyt_t.invert_transform(T_agent_camera)
        # T_agent_estimate = T_agent_camera @ T_camera_estimate

        # # Transform from the agent frame to the world frame
        # T_world_agent = carla_to_pytransform(self.agent.get_initial_position())
        # T_world_estimate = T_world_agent @ T_agent_estimate
        ##########################################################

        return self._correct_estimate(T_orb_estimate)
    
    def _correct_estimate(self, estimate: NDArray) -> NDArray:
        """Corrects the estimate to be in the global frame.

        Args:
            estimate: The transformation matrix from ORB-SLAM

        Returns:
            NDArray: The rover in the global frame
        """
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))
        camera_init_global = pyt_t.concat(camera_rover, self.rover_global)

        # Get the rotation of the orbslam frame in the initial camera frame
        x_o, y_o, z_o, roll_o, pitch_o, yaw_o = pytransform_to_tuple(estimate)

        # Create a transform of just the translation with the axes swapped
        camera_orbslam = np.eye(4)
        camera_orbslam[:3, 3] = [z_o, -x_o, -y_o]

        # Get the position of the orbslam frame in the global frame
        orbslam_global = camera_init_global
        camera_global = pyt_t.concat(camera_orbslam, orbslam_global)

        # Get the position of the rover in the camera frame
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))
        rover_camera = pyt_t.invert_transform(camera_rover)

        # Get the rover in the global frame
        rover_global = pyt_t.concat(rover_camera, camera_global)

        # Extract the translation and rotation
        x, y, z, _, _, _ = pytransform_to_tuple(rover_global)
        _, _, _, roll_i, pitch_i, yaw_i = pytransform_to_tuple(camera_init_global)

        # Apply the orbslam camera rotations to the initial rover rotation,
        # correcting the axes orientation and direction
        rover_global = tuple_to_pytransform(
            (x, y, z, roll_i + yaw_o, pitch_i - roll_o, yaw_i - pitch_o)
        )

        return rover_global

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

    # INCLUDED ONLY FOR COMPATIBILITY WITH stereoslam.py, WILL BE REMOVED
    def get_current_pose(self) -> NDArray:
        """Get the current pose from ORB-SLAM."""
        return self._get_pose()

    def get_trajectory(self) -> list[NDArray]:
        """Get the trajectory from ORB-SLAM."""
        return self._get_trajectory()
