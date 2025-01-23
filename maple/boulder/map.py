import numpy as np
from numpy.typing import NDArray
from pytransform3d.transformations import concat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import at top of file


class BoulderMap:
    def __init__(self, geometric_map):
        """
        Args:
            geometric_map: The GeometricMap object from the leaderboard
        """

        self.geometric_map = geometric_map

    def _generate_map(self, boulders_global: list) -> NDArray:
        """Generates a 2D array for the boulder locations in the map.
        Args:
            boulders_global: A list of transforms representing points on the surface of boulders

        Returns:
            A 2D boolean array representing the locations of boulders in the map
        """

        size = self.geometric_map.get_cell_number()
        boulder_map = np.zeros((size, size), dtype=bool)

        # TODO: Implement!
        # Add logic to generate the boulder map here
        # Hint: get_cell_indexes() might be useful here

        # To get only the x, y, z coordinates of point use
        # x, y, z = boulder[:3, 3]
        # Or for all boulders
        # boulders_xyz = [boulder[:3, 3] for boulder in boulders_global]

        """
        Notes:

        `boulders_global` is a point cloud representing a potential point on the surface
        of a boulder. However, the system generating these points isn't perfect.
        There will be extraneous outliers that are not actually boulders but 
        artifacts that were misinterpreted as boulders. In theory, real boulders
        should have been detected multiple times, therefore we should expect to
        see clusters of points where real boulders are. We need to filter the 
        point cloud to look for clusters and log these as "real boulders" so to
        speak.
        """

        return boulder_map

    def set_map(self, samples: list):
        """Set the boulder locations in the geometric_map
        Args:
            samples: A list of boulder location sample points
        """

        boulder_map = self._generate_map(samples)

        for x, y in np.ndindex(boulder_map.shape):
            self.geometric_map.set_cell_rock(x, y, boulder_map[x, y])


def rover_to_global(boulders_rover: list, rover_global: np.ndarray) -> list:  # type: ignore  # noqa: F821
    """Converts the boulder locations from the rover frame to the global frame.

    Args:
        boulders_rover: A list of transforms representing points on the surface of boulders in the rover frame

    Returns:
        A list of transforms representing points on the surface of boulders in the global frame
    """

    boulders_global = [
        concat(boulder_rover, rover_global) for boulder_rover in boulders_rover
    ]
    return boulders_global


def visualize_transforms(
    transforms: list, title: str = "Point Cloud Visualization", flatten: bool = False
):
    """Visualizes a list of transforms as a 3D point cloud.

    Args:
        transforms: List of transforms where each transform's translation represents a point
        title: Optional title for the plot
    """
    # Extract x, y, z coordinates from transforms
    points = np.array([transform[:3, 3] for transform in transforms])

    fig = plt.figure(figsize=(10, 10))

    if flatten:
        # Create 2D plot
        ax = fig.add_subplot(111)

        # Plot points
        ax.scatter(points[:, 0], points[:, 1], c="b", marker="o", alpha=0.3)

        # Labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        # Add grid lines every 0.15 meters
        ax.grid(True)
        # Calculate number of ticks needed
        x_min = np.floor(min(points[:, 0]) / 0.15) * 0.15
        x_max = np.ceil(max(points[:, 0]) / 0.15) * 0.15
        y_min = np.floor(min(points[:, 1]) / 0.15) * 0.15
        y_max = np.ceil(max(points[:, 1]) / 0.15) * 0.15
        ax.set_xticks(np.arange(x_min, x_max + 0.15, 0.15))
        ax.set_yticks(np.arange(y_min, y_max + 0.15, 0.15))
    else:
        # Create 3D plot
        ax = fig.add_subplot(111, projection="3d")

        # Plot points
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], c="b", marker="o", alpha=0.3
        )

        # Labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

    # Show plot
    plt.show()
