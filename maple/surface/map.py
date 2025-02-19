import numpy as np
from numpy.typing import NDArray
from pytransform3d.transformations import concat, transform_from
from maple.geometry import rover
from maple.surface.post_processing import PostProcessor

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

        for sample in samples:
            x, y, z = sample
            cell_indexes = self.geometric_map.get_cell_indexes(x, y)

            if cell_indexes is not None:
                x_c, y_c = cell_indexes
                if self.geometric_map._is_cell_valid(x_c, y_c):
                    if height_map[x_c, y_c] == np.NINF:
                        height_map[x_c, y_c] = 0
                    height_map[x_c, y_c] += z
                    cell_counts[x_c, y_c] += 1

        nonzero_cells = cell_counts > 0
        height_map[nonzero_cells] /= cell_counts[nonzero_cells]

        # Interpolate missing values with confidence levels
        post_processor = PostProcessor(height_map)
        interpolated_map, confidence = post_processor.interpolate_with_confidence()

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


def sample_surface(rover_global) -> list:
    """Generates ground samples based on the pose of the rover.

    Args:
        rover_global: The pose of the rover in the global coordinate frame

    Returns:
        A list of four ground sample points [x, y, z] (where each wheel touches the ground)
    """

    samples = []
    for wheel in rover["wheels"].values():
        # The surface point in the rover frame
        surface_rover = transform_from(
            np.eye(3), [wheel["x"], wheel["y"], wheel["z"] - (wheel["diameter"] / 2)]
        )

        surface_global = concat(surface_rover, rover_global)

        # Append only the x, y, z components of the surface point
        samples.append(surface_global[:3, 3].tolist())

    return samples
