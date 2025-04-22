from .core import FrameDataReader, ImageDataReader
from .mocks import Transform


class PlaybackAgent:
    """Mock agent class that can be used to playback data."""

    _frame_data: FrameDataReader
    _image_data: ImageDataReader

    def __init__(self, data_path: str):
        """Initialize the playback agent.

        Args:
            data_path: The path to the data file.
        """
        self._frame_data = FrameDataReader(data_path)
        self._image_data = ImageDataReader(data_path)

    # Constant Functions
    def use_fiducials(self) -> bool:
        return self._frame_data.initial["fiducials"]

    def sensors(self) -> dict:
        return self._frame_data.initial["cameras"]

    def get_initial_position(self) -> Transform:
        return Transform(
            p=self._frame_data.initial["lander"][:3],
            e=self._frame_data.initial["lander"][3:],
        )

    def get_initial_lander_position(self) -> Transform:
        return Transform(
            p=self._frame_data.initial["lander"][:3],
            e=self._frame_data.initial["lander"][3:],
        )

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
