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

    def __init__(self, 
                 name: str) -> None:
        """
        Initialize the dataloader with a specific dataset
        
        Args:
            name (str): Name of the dataset in the data folder (e.g. 003)
        """
        # Assumes repository structure is unmodified -> MAPLE/maple/utils/dataloader.py
        repo_path = Path(__file__).parents[2]
        data_path = repo_path / 'data' / name
        csv_path  = data_path / 'imu_data.csv'

        # Read csv into a pandas dataframe
        data = pd.read_csv(csv_path)

        # Assert data is proper lengths
        assert data.apply(len).nunique() == 1, "Invalid .csv"
        
        # Extract all of the information from the csv
        self.power  = data['power'].to_numpy()
        self.cmd_v  = data['input_v'].to_numpy()
        self.cmd_w  = data['input_w'].to_numpy()
        gt_x        = data['gt_x'].to_numpy()
        gt_y        = data['gt_y'].to_numpy()
        gt_z        = data['gt_z'].to_numpy()
        gt_roll     = data['gt_roll'].to_numpy()
        gt_pitch    = data['gt_pitch'].to_numpy()
        gt_yaw      = data['gt_yaw'].to_numpy()
        imu_accel_x = data['imu_accel_x'].to_numpy()
        imu_accel_y = data['imu_accel_y'].to_numpy()
        imu_accel_z = data['imu_accel_z'].to_numpy()
        imu_gyro_x  = data['imu_gyro_x'].to_numpy()
        imu_gyro_y  = data['imu_gyro_y'].to_numpy()
        imu_gyro_z  = data['imu_gyro_z'].to_numpy()

        # Store all of the ground truth poses
        self.T_gt = np.array([
            transform_from(
                matrix_from_euler([r, p, y], 0, 1, 2, False), [x, y, z]
            )
            for x, y, z, r, p, y in zip(gt_x, gt_y, gt_z, gt_roll, gt_pitch, gt_yaw)
        ])

        # Store IMU values as vectors
        self.imu_acceleration = np.vstack((imu_accel_x, imu_accel_y, imu_accel_z)).T
        self.imu_gyro         = np.vstack((imu_gyro_x, imu_gyro_y, imu_gyro_z)).T

        # Camera path lookup dictionary for later access
        self._cam_paths = {
            "back":       data_path / "Back",
            "backleft":   data_path / "BackLeft",
            "backright":  data_path / "BackRight",
            "front":      data_path / "Front",
            "frontleft":  data_path / "FrontLeft",
            "frontright": data_path / "FrontRight",
            "left":       data_path / "Left",
            "right":      data_path / "Right"
        }

    def __len__(self): return self.length

    @property
    def length(self): return len(self.power)

    def cam(self,
            cam: str,
            idx: int,
            semantic: bool = False) -> Image:
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
        if idx < 0 or idx > len(self): raise IndexError("Invalid frame index")

        # Images are collected every other frame, group even frames with frame below, 0 is unique
        if   idx == 0:     idx = 1
        elif idx % 2 == 0: idx -= 1

        # Valid path check (if this camera is missing from this dataset this will fail)
        folder_path = self._cam_paths[cam]
        if not folder_path.exists(): raise KeyError(f"{cam} camera not in dataset")

        # Open the image 
        if semantic: 
            img_path = folder_path /f"{idx}_sem.png"
            img = Image.open(img_path)
        else:        
            img_path = folder_path /f"{idx}.png"
            img = Image.open(img_path).convert("L")

        return img
    
    def cam_sequence(self,
                     cam: str,
                     range: tuple[int, int],
                     skip_repeat: bool,
                     semantic: bool = False) -> list[Image]:
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
        if lower < 0 or lower > len(self) or lower > upper: raise IndexError("Invalid lower frame index")
        if upper < 0 or upper > len(self):                  raise IndexError("Invalid upper frame index")

        # Select indicies to loop over, remove evens if we are skipping repeats
        indices = np.arange(lower, upper)
        if skip_repeat: indices = indices[indices % 2 != 0]
            
        return [self.cam(cam, i, semantic=semantic) for i in indices]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    test_loader = DataLoader('004')
    i = 54
    cam = 'frontleft'

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
    plt.axis('off')
    plt.show()
    
    # Images are returned with the correct bitdepth, but matplotlib assumes colored
    img = test_loader.cam(cam, i, semantic=False)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    #####################
    # IMAGE SEQUENCE TEST
    #####################

    img_seq = test_loader.cam_sequence(cam, (120, 160), skip_repeat=False, semantic=True)
    frames = [np.array(img) for img in img_seq]

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
    plt.show()