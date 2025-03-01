import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import matrix_from_euler


class DataLoader:
    """
    Simple interface for loading data collected in a .csv and interfacing with it.
    Each value is stored in a np.array of length N.

    Values:
        power (float): Power level of the rover
        cmd_v (float): Commanded forward velocity
        cmd_w (float): Commanded rotational velocity
        T_gt (transform): 4x4 transformation matrix of the ground truth pose
        imu_acceleration (vector): 3x1 vector of imu accelerations
        imu_gyro (vector): 3x1 vector of imu rotations
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the dataloader with a specific dataset

        Args:
            name (str): Name of the dataset in the data folder (e.g. 003)
        """
        # Assumes repository structure is unmodified -> MAPLE/maple/utils/dataloader.py
        repo_path = Path(__file__).parents[2]
        data_path = repo_path / "data" / name
        csv_path = data_path / "imu_data.csv"

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
        # IS THIS THE CORRECT ORDER FOR ROLL PITCH YAW?
        self.T_gt = np.array(
            [
                transform_from(
                    matrix_from_euler([yaw, pitch, roll], 2, 1, 0, extrinsic=False),
                    [x, y, z],
                )
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

    def cam(self, cam: str, idx: int, semantic: bool = False) -> Image:
        """
        Access a specific camera image at a given timestep.
        NOTE: Images are collected every other frame, this function floors even frames down.

        Args:
            cam (str): Camera to lookup, valid values are: ``Back, BackLeft,
            BackRight, Front, FrontLeft, FrontRight, Left, Right``
            idx (int): Frame to get the image from
            semantic (bool): Returns the ground truth semantic image if True

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

        return img

    def cam_sequence(
        self,
        cam: str,
        range: tuple[int, int],
        skip_repeat: bool,
        semantic: bool = False,
    ) -> list[Image]:
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

class DataLoader_AgentInterface:
    """
    A simple interface that can be called by an agent to help collect data for the DataLoader class.
    """
    def __init__(self, agent, data_path) -> None:
        """
        Initialize the interface for an agent and provide the path in which the data will be stored.
        
        Inputs:
        - agent: an agent object
        - name: the path at which the data will be stored"""
        self.agent = agent
        self.data_path = data_path

    def collect_data(self):
        """
        Collect data from the agent and store it in a list for later saving.
        The data that is collected includes:
        - timestamp
        - power
        - input_v
        - input_w
        - gt_x
        - gt_y
        - gt_z
        - gt_roll
        - gt_pitch
        - gt_yaw
        - imu_accel_x
        - imu_accel_y
        - imu_accel_z
        - imu_gyro_x
        - imu_gyro_y
        - imu_gyro_z
        - est_x
        - est_y
        - est_z
        - est_roll
        - est_pitch
        - est_yaw"""





def save_for_dataloader(data, data_path, name):
    """
    This function saves data generated by an agent in csv files that can be loaded by the DataLoader class.
    The data is saved in the following format:
    power, input_v, input_w, gt_x, gt_y, gt_z, gt_roll, gt_pitch, gt_yaw, imu_accel_x, imu_accel_y, imu_accel_z, imu_gyro_x, imu_gyro_y, imu_gyro_z

    Inputs:
    - data: a dictionary containing the following keys:
        - timestamp: a list of timestamps
        - power: a list of power values
        - input_v: a list of commanded forward velocities
        - input_w: a list of commanded rotational velocities
        - gt_x: a list of x coordinates of the ground truth pose
        - gt_y: a list of y coordinates of the ground truth pose
        - gt_z: a list of z coordinates of the ground truth pose
        - gt_roll: a list of roll angles of the ground truth pose
        - gt_pitch: a list of pitch angles of the ground truth pose
        - gt_yaw: a list of yaw angles of the ground truth pose
        - imu_accel_x: a list of x components of the imu acceleration
        - imu_accel_y: a list of y components of the imu acceleration
        - imu_accel_z: a list of z components of the imu acceleration
        - imu_gyro_x: a list of x components of the imu gyro
        - imu_gyro_y: a list of y components of the imu gyro
        - imu_gyro_z: a list of z components of the imu gyro
        - est_x: a list of x coordinates of the estimated pose
        - est_y: a list of y coordinates of the estimated pose
        - est_z: a list of z coordinates of the estimated pose
        - est_roll: a list of roll angles of the estimated pose
        - est_pitch: a list of pitch angles of the estimated pose
        - est_yaw: a list of yaw angles of the estimated pose

    - data_path: the path to the directory where the data will be saved
    - name: the name of the folder containing the dataset"""
    
    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a csv file
    df.to_csv(data_path / name / "imu_data.csv", index=False)

    print("Data saved successfully")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from debug import playback_camera, display_trajectory

    test_loader = DataLoader("004")
    i = 54
    cam = "frontleft"

    ###########################
    # REGULAR DATA LOADING TEST
    ###########################

    print(f"Length of data: {len(test_loader)}")
    print(f"power[{i}]: {test_loader.power[i]}")
    print(f"cmd_v[{i}]: {test_loader.cmd_v[i]}")
    print(f"cmd_w[{i}]: {test_loader.cmd_w[i]}")
    print(f"T_gt shape: {test_loader.T_gt.shape}, value[{i}]:")
    print(test_loader.T_gt[i])
    print(f"imu_acceleration shape: {test_loader.imu_acceleration.shape}, value[{i}]:")
    print(test_loader.imu_acceleration[i])
    print(f"imu_gyro shape: {test_loader.imu_gyro.shape}, value[{i}]:")
    print(test_loader.imu_gyro[i])

    img_sem = test_loader.cam(cam, i, semantic=True)
    plt.imshow(img_sem)
    plt.axis("off")
    plt.show()

    # Images are returned with the correct bitdepth, but matplotlib assumes colored
    img = test_loader.cam(cam, i, semantic=False)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

    #####################
    # IMAGE SEQUENCE TEST
    #####################

    playback_camera(test_loader, cam, (120, 160), semantic=False)

    ######################
    # TRAJECTORY PLOT TEST
    ######################

    display_trajectory(test_loader)
