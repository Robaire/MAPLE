import numpy as np
from numpy.typing import NDArray
from maple.geometry import rover
from maple.surface.post_processing import PostProcessor
import pytransform3d.transformations as pytrans
import pytransform3d.rotations as pyrot
from maple.utils import carla_to_pytransform
from maple import geometry


# DO NOT IMPORT MATPLOTLIB IN CODE THE ROVER USES!!!!
# import matplotlib
# matplotlib.use("Agg")  # Set the backend before importing pyplot
# import matplotlib.pyplot as plt


class SurfaceHeight:
    def __init__(self, geometric_map):
        """
        Args:
            geometric_map: The GeometricMap object from the leaderboard
        """

        self.geometric_map = geometric_map
        self._last_height_map = None

    # DO NOT USE MATPLOTLIB IN SurfaceHeight, PUT THIS IN A VISUALIZATION FILE
    # def visualize_height_map(
    #     self, height_map: NDArray = None, save_path: str = "height_map.png"
    # ):
    #     """Visualizes the height map using a color-coded plot and saves it to a file.

    #     Args:
    #         height_map: Optional pre-generated height map. If None, uses the last generated map.
    #         save_path: Path where the plot should be saved. Defaults to 'height_map.png'
    #     """
    #     if height_map is None:
    #         height_map = self._last_height_map

    #     # Create a masked array to handle NINF values
    #     masked_height_map = np.ma.masked_where(height_map == np.NINF, height_map)

    #     plt.figure(figsize=(10, 8))
    #     im = plt.imshow(masked_height_map.T, origin="lower", cmap="terrain")
    #     plt.colorbar(im, label="Height")
    #     plt.title("Surface Height Map")
    #     plt.xlabel("X Cell Index")
    #     plt.ylabel("Y Cell Index")
    #     plt.savefig(save_path)  # Save the plot to file
    #     plt.close()  # Close the figure to free memory

    def _generate_map(self, samples: list) -> NDArray:
        """Generates a 2D array of the average surface height for each cell."""
        size = self.geometric_map.get_cell_number()
        height_map = np.full((size, size), np.NINF)
        cell_counts = np.zeros((size, size))

        # for sample in samples:
        #     x, y, z = sample
        #     cell_indexes = self.geometric_map.get_cell_indexes(x, y)
        #     # print(cell_indexes)
        #     # print('x:', x, 'y:', y, 'z:', z)

        #     if cell_indexes is not None:
        #         x_c, y_c = cell_indexes
        #         if self.geometric_map._is_cell_valid(x_c, y_c):
        #             if height_map[x_c, y_c] == np.NINF:
        #                 height_map[x_c, y_c] = 0
        #             height_map[x_c, y_c] += z
        #             cell_counts[x_c, y_c] += 1

        nonzero_cells = cell_counts > 0
        height_map[nonzero_cells] /= cell_counts[nonzero_cells]

        # Interpolate missing values with confidence levels
        post_processor = PostProcessor(height_map)

        height_map = post_processor.reject_noisy_samples_grid(samples)
        # interpolated_map, confidence = post_processor.interpolate_with_confidence()
        interpolated_map = post_processor.interpolate_and_smooth(filter_size=7)

        # Optionally, you could filter out low-confidence estimates
        # interpolated_map[confidence < 0.5] = np.NINF

        self._last_height_map = interpolated_map
        return interpolated_map

    def set_map(self, samples: list):
        """Set the heights in the geometric_map.
        Args:
            samples: A list of ground sample points
        """

        height_map = self._generate_map(samples)
        self._last_height_map = height_map

        for x, y in np.ndindex(height_map.shape):
            self.geometric_map.set_cell_height(x, y, height_map[x, y])


def sample_surface(rover_global, pitch_roll_threshold=60) -> list:
    """Generates ground samples based on the pose of the rover.

    Args:
        rover_global: The pose of the rover in the global coordinate frame
        pitch_roll_theshold: The threshold for pitch and roll angle magnitudes (in degrees), beyond
        which rover_global is considered invalid and no samples are generated.

    Returns:
        A list of four ground sample points [x, y, z] (where each wheel touches the ground)
    """
    # Check if the rover is not tilted beyond the threshold
    print("rover_global", rover_global)
    roll, pitch, yaw = pyrot.euler_from_matrix(
        rover_global[:3, :3], i=0, j=1, k=2, extrinsic=True
    )
    if np.abs(pitch) > np.deg2rad(pitch_roll_threshold) or np.abs(roll) > np.deg2rad(
        pitch_roll_threshold
    ):
        return []

    samples = []
    for wheel in rover["wheels"].values():
        # The surface point in the rover frame
        surface_rover = pytrans.transform_from(
            np.eye(3), [wheel["x"], wheel["y"], wheel["z"] - (wheel["diameter"] / 2)]
        )

        surface_global = pytrans.concat(surface_rover, rover_global)

        # Append only the x, y, z components of the surface point
        samples.append(surface_global[:3, 3].tolist())

    return samples


# def sample_surface(rover_global) -> list:
#     """Generates ground samples based on the pose of the rover.

#     Args:
#         rover_global: The pose of the rover in the global coordinate frame

#     Returns:
#         A list of four ground sample points [x, y, z] (where each wheel touches the ground)
#     """

#     samples = []
#     for wheel in rover["wheels"].values():
#         # The surface point in the rover frame
#         surface_rover = pytrans.transform_from(
#             np.eye(3), [wheel["x"], wheel["y"], wheel["z"] - (wheel["diameter"] / 2)]
#         )

#         surface_global = pytrans.concat(surface_rover, rover_global)

#         # Append only the x, y, z components of the surface point
#         samples.append(surface_global[:3, 3].tolist())

#     return samples


def sample_lander(agent):
    """
    Based on the lander's position and orientation, generate a list of ground samples from the
    estimated feet positions.

    Inputs:
    - agent: The agent object from the leaderboard

    Returns:
    - A list of a list of ground sample points [[x, y, z],...]
    """
    # At mission start we can get the position of the rover in the global coordinate frame
    # and the position of the lander in the rover's coordinate frame
    # using these we can determine the position of the lander in the global coordinate frame
    rover_global = carla_to_pytransform(agent.get_initial_position())
    lander_rover = carla_to_pytransform(agent.get_initial_lander_position())
    lander_global = pytrans.concat(lander_rover, rover_global)

    # The lander has 4 feet, we can generate ground samples for each foot
    samples = []
    tag_rotations = {"a": -45, "b": 45, "c": 135, "d": -135}
    for group, tag_group in geometry.lander["fiducials"].items():
        rotation = pyrot.matrix_from_euler(
            [np.deg2rad(tag_rotations[group]), 0, 0], 2, 1, 0, False
        )
        transl = [1.21, 0, 0]
        foot_rover = pytrans.concat(
            pytrans.transform_from(np.eye(3), transl),
            pytrans.transform_from(rotation, [0, 0, 0]),
        )
        foot_global = pytrans.concat(foot_rover, lander_global)
        samples.append(foot_global[:3, 3].tolist())
    return samples
