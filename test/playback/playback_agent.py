import tarfile
import toml
import pandas as pd
import os

from test.mocks.mock_carla_transform import Transform


class PlaybackAgent:
    """Mock agent class that can be used to playback data."""

    tar_file: tarfile.TarFile
    initial: dict
    frames: pd.DataFrame
    camera_frames: dict[str, pd.DataFrame] = {}
    custom_records: dict[str, pd.DataFrame] = {}

    def __init__(self, path: str):
        """Initialize the playback agent.

        Args:
            path: The path to the data file.
        """

        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")

        # Open the tar file
        self.tar_file = tarfile.open(path, "r:gz")

        # Read the initialization data
        self.initial = toml.loads(
            self.tar_file.extractfile("initial.toml").read().decode("utf-8")
        )
        try:
            [self.initial[key] for key in ["fiducials", "lander", "rover", "cameras"]]
        except KeyError:
            raise ValueError("initial.toml is missing required keys")

        # Read the frame sensor data
        self.frames = pd.read_csv(self.tar_file.extractfile("frames.csv"))

        # Read frame data for each camera
        for camera in self.initial["cameras"].keys():
            try:
                self.camera_frames[camera] = pd.read_csv(
                    self.tar_file.extractfile(f"cameras/{camera}/{camera}_frames.csv")
                )
            except (pd.errors.EmptyDataError, KeyError):
                # Some cameras may not have any frames
                pass

        # Read any custom records
        for record in self.tar_file.getnames():
            if record.startswith("custom/"):
                self.custom_records[record.split("/")[-1].split(".")[0]] = pd.read_csv(
                    self.tar_file.extractfile(record)
                )

    def __del__(self):
        try:
            self.tar_file.close()
        except AttributeError:
            # There is no tar file to close
            pass

    # Constant Functions
    def use_fiducials(self) -> bool:
        pass

    def sensors(self) -> dict:
        pass

    def get_initial_position(self) -> Transform:
        pass

    def get_initial_lander_position(self) -> Transform:
        pass

    # Frame Dependent Functions
    def get_mission_time(self) -> float:
        pass

    def get_current_power(self) -> float:
        pass

    def get_consumed_power(self) -> float:
        raise NotImplementedError("get_consumed_power not implemented")

    def get_imu_data(self) -> list:
        pass

    def get_linear_speed(self) -> float:
        pass

    def get_angular_speed(self) -> float:
        pass

    def get_front_arm_angle(self) -> float:
        raise NotImplementedError("get_front_arm_angle not implemented")

    def get_back_arm_angle(self) -> float:
        raise NotImplementedError("get_back_arm_angle not implemented")

    def get_front_drums_speed(self) -> float:
        raise NotImplementedError("get_front_drums_speed not implemented")

    def get_back_drums_speed(self) -> float:
        raise NotImplementedError("get_back_drums_speed not implemented")

    def get_radiator_cover_angle(self) -> float:
        pass

    # Camera Functions
    def get_light_state(self, camera: str) -> float:
        pass

    def get_camera_state(self, camera: str) -> bool:
        pass

    def get_camera_position(self, camera: str) -> Transform:
        pass

    def get_light_position(self, camera: str) -> Transform:
        position = self.get_camera_position(camera)
        # TODO: Add a fixed offset for the light position
        pass

    def get_transform(self) -> Transform:
        pass
