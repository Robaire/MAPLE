import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import matrix_from_euler
from test.mocks.mock_carla_transform import Transform
from pytest_mock import MockerFixture
from dataclasses import dataclass
from maple.utils import tuple_to_pytransform


class CSVParser:
    """
    Simple interface for loading data collected in a .csv and interfacing with it.
    Each value is stored in a np.array of length N. Modified from maple.utils.dataloader.py
    to be used for testing.

    Values:
        power (float): Power level of the rover
        cmd_v (float): Commanded forward velocity
        cmd_w (float): Commanded rotational velocity
        T_gt (transform): 4x4 transformation matrix of the ground truth pose
        imu_acceleration (vector): 3x1 vector of imu accelerations
        imu_gyro (vector): 3x1 vector of imu rotations
    """

    def __init__(self, data_path: str) -> None:
        """
        Initialize the dataloader with a specific dataset

        Args:
            name (str): Name of the dataset in the data folder (e.g. 003)
        """
        csv_path = Path(data_path) / "imu_data.csv"

        # Read csv into a pandas dataframe
        data = pd.read_csv(csv_path)

        # Assert data is proper lengths
        assert data.apply(len).nunique() == 1, "Invalid .csv"

        # Extract all of the information from the csv
        self.power = data["power"].to_numpy()
        self.cmd_v = data["input_v"].to_numpy()
        self.cmd_w = data["input_w"].to_numpy()
        gt_x = data["gt_x"].to_numpy()
        gt_y = data["gt_y"].to_numpy()
        gt_z = data["gt_z"].to_numpy()
        gt_roll = data["gt_roll"].to_numpy()
        gt_pitch = data["gt_pitch"].to_numpy()
        gt_yaw = data["gt_yaw"].to_numpy()
        imu_accel_x = data["imu_accel_x"].to_numpy()
        imu_accel_y = data["imu_accel_y"].to_numpy()
        imu_accel_z = data["imu_accel_z"].to_numpy()
        imu_gyro_x = data["imu_gyro_x"].to_numpy()
        imu_gyro_y = data["imu_gyro_y"].to_numpy()
        imu_gyro_z = data["imu_gyro_z"].to_numpy()

        # Store all of the ground truth poses
        self.T_gt = np.array(
            [
                tuple_to_pytransform((x, y, z, roll, pitch, yaw))
                for x, y, z, roll, pitch, yaw in zip(
                    gt_x, gt_y, gt_z, gt_roll, gt_pitch, gt_yaw
                )
            ]
        )

        # Store IMU values as vectors
        self.imu_acceleration = np.vstack((imu_accel_x, imu_accel_y, imu_accel_z)).T
        self.imu_gyro = np.vstack((imu_gyro_x, imu_gyro_y, imu_gyro_z)).T

        # Camera path lookup dictionary for later access
        self._cam_paths = {
            "back": data_path / "Back",
            "backleft": data_path / "BackLeft",
            "backright": data_path / "BackRight",
            "front": data_path / "Front",
            "frontleft": data_path / "FrontLeft",
            "frontright": data_path / "FrontRight",
            "left": data_path / "Left",
            "right": data_path / "Right",
        }

    def __len__(self):
        return self.length

    @property
    def length(self):
        return len(self.power)

    def cam(
        self, cam: str, idx: int, semantic: bool = False, path: bool = False
    ) -> Image:
        """
        Access a specific camera image at a given timestep.
        NOTE: Images are collected every other frame, this function floors even frames down.

        Args:
            cam (str): Camera to lookup, valid values are: ``Back, BackLeft,
            BackRight, Front, FrontLeft, FrontRight, Left, Right``
            idx (int): Frame to get the image from
            semantic (bool): Returns the ground truth semantic image if True
            path (bool): Returns the path to the image if True

        Returns:
            Image: PIL Image from the cam at the given idx

        Raises:
            IndexError: If idx is outside of valid range for dataset
            KeyError: If desired camera is not present in this dataset
            FileNotFoundError: If the camera data is fragmented (image missing)
        """
        cam = cam.lower()
        assert cam in self._cam_paths.keys(), "Invalid camera"

        # Valid idx check
        if idx < 0 or idx > len(self):
            raise IndexError("Invalid frame index")

        # Images are collected every other frame, group even frames with frame below, 0 is unique
        if idx == 0:
            idx = 1
        elif idx % 2 == 0:
            idx -= 1

        # Valid path check (if this camera is missing from this dataset this will fail)
        folder_path = self._cam_paths[cam]
        if not folder_path.exists():
            raise KeyError(f"{cam} camera not in dataset")

        # Open the image
        if semantic:
            img_path = folder_path / f"{idx}_sem.png"
            img = Image.open(img_path)
        else:
            img_path = folder_path / f"{idx}.png"
            img = Image.open(img_path).convert("L")

        if path:
            return img_path

        return img

    def cam_sequence(
        self,
        cam: str,
        range: tuple[int, int],
        skip_repeat: bool,
        semantic: bool = False,
    ) -> list[Image]:  # type: ignore
        """
        Access a specific camera image at a given range of timesteps.
        NOTE: Images are collected every other frame, this function floors even frames down.

        Args:
            cam (str): Camera to lookup, valid values are: ``Back, BackLeft,
            BackRight, Front, FrontLeft, FrontRight, Left, Right``
            range ((int, int)): Range of images to select from, end index is exclusive
            skip_repeat (bool): Only returns unique frames if True (see NOTE above)
            semantic (bool): Returns the ground truth semantic image if True

        Returns:
            [Image]: PIL Image array from the cam for the given range

        Raises:
            IndexError: If idx is outside of valid range for dataset
            KeyError: If desired camera is not present in this dataset
            FileNotFoundError: If the camera data is fragmented (image missing)
        """
        lower, upper = range

        # Valid idx check
        if lower < 0 or lower > len(self) or lower > upper:
            raise IndexError("Invalid lower frame index")
        if upper < 0 or upper > len(self):
            raise IndexError("Invalid upper frame index")

        # Select indicies to loop over, remove evens if we are skipping repeats
        indices = np.arange(lower, upper)
        if skip_repeat:
            indices = indices[indices % 2 != 0]

        return [self.cam(cam, i, semantic=semantic) for i in indices]

    def get_pose(self, idx: int) -> np.ndarray:
        """
        Get the ground truth pose transformation matrix at a specific index.

        Args:
            idx (int): Index of the desired pose

        Returns:
            np.ndarray: 4x4 transformation matrix representing the ground truth pose
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Invalid frame index")

        return self.T_gt[idx]


class CSVAgent:
    """Fake agent that can be used for testing with CSVParser data"""

    def get_camera_position(self, camera):
        cameras = {
            "FrontLeft": Transform(p=(0.28, 0.081, 0.131)),
            "FrontRight": Transform(p=(0.28, -0.081, 0.131)),
        }

        try:
            return cameras[camera]
        except KeyError:
            return Transform()

    # Typically the keys are objects, but for testing we use strings
    def sensors(self):
        return {
            "FrontLeft": {
                "camera_active": True,
                "light_intensity": 0,
                "width": "1280",
                "height": "720",
            },
            "FrontRight": {
                "camera_active": True,
                "light_intensity": 0,
                "width": "1280",
                "height": "720",
            },
        }

    # def get_pose(self, idx: int) -> np.ndarray:
    #     return self.data.get_pose(idx)


@dataclass
class Constants:
    """Constants for the geometric map."""

    map_size: float  # overall map width [m]
    cell_size: float  # individual cell width [m]
    cell_number: int  # number of cells [#]


def create_base_map(constants):
    """
    Creates the base geometric map that will be given to the agent for its completion.
    It is a 2D numpy matrix where each element in them represents the [x,y, height, rock flag].
    """
    ROCK_UNCOMPLETED_VALUE = np.NINF
    MAP_UNCOMPLETED_VALUE = np.NINF

    base_map = np.array(np.zeros((constants.cell_number, constants.cell_number, 4)))
    low = -constants.map_size / 2 + constants.cell_size / 2
    high = constants.map_size / 2 - constants.cell_size / 2
    values = np.arange(
        low, high + 0.05, constants.cell_size
    )  # Make sure float imprecision doesn't remove the last one
    indexes = np.arange(0, constants.cell_number, 1)

    for x_index in indexes:
        for y_index in indexes:
            base_map[x_index, y_index] = [
                values[x_index],
                values[y_index],
                MAP_UNCOMPLETED_VALUE,
                ROCK_UNCOMPLETED_VALUE,
            ]

    return base_map


class CSVGeometricMap:
    """Mock implementation of GeometricMap for testing with .csv data."""

    def __init__(self):
        """Initialize the geometric map with given constants.

        Args:
            constants: Configuration constants for the map
        """
        constants = Constants(9, 0.15, 60)
        self._map = create_base_map(constants)
        self._map_size = constants.map_size
        self._cell_size = constants.cell_size
        self._cell_number = constants.cell_number

    def _is_cell_valid(self, x_index, y_index):
        """Returns whether the index is a valid one"""
        if x_index is None or x_index < 0 or x_index >= self._cell_number:
            return False
        if y_index is None or y_index < 0 or y_index >= self._cell_number:
            return False
        return True

    def get_map_array(self):
        """Returns the geometric map. This returns the actual numpy array"""
        return self._map

    def get_map_size(self):
        """Returns the map size"""
        return self._map_size

    def get_cell_size(self):
        """Returns the cell size"""
        return self._cell_size

    def get_cell_number(self):
        """Returns the amount of cells per dimensions"""
        return self._cell_number

    def get_cell_indexes(self, x, y):
        """
        Given an x and y coordinates, returns the cell indexes that are closest to the given position.
        Returns None if the position is outside the mapping area.
        """
        cell_values = self._map[:, 0, 0]
        min_cell = self._map[0, 0, 0]
        max_cell = self._map[-1, 0, 0]
        max_distance = self.get_cell_size() / 2

        x_index = sum(cell_values < x - max_distance)
        if x_index == 0 and abs(min_cell - x) > max_distance:
            x_index = None
        if x_index == self._cell_number and abs(max_cell - x) > max_distance:
            x_index = None

        y_index = sum(cell_values < y - max_distance)
        if y_index == 0 and abs(min_cell - y) > max_distance:
            y_index = None
        if y_index == self._cell_number and abs(max_cell - y) > max_distance:
            y_index = None

        return (x_index, y_index)
