import numpy as np
from numpy.typing import NDArray
from pytransform3d.transformations import concat, transform_from
from maple.geometry import rover


class SurfaceHeight:
    def __init__(self, geometric_map):
        """
        Args:
            geometric_map: The GeometricMap object from the leaderboard
        """

        self.geometric_map = geometric_map

    def _generate_map(self, samples: list) -> NDArray:
        """Generates a 2D array of the average surface height for each cell."""

        # Use the functions provided by GeometricMap to determine the required size of the height map
        size = self.geometric_map.get_cell_number()
        height_map = np.zeros((size, size))

        map_size = self.geometric_map.get_map_size()
        pos_to_cell={}

        # TODO: Implement!
        # Add logic to generate the height map here
        # Hint: get_cell_indexes() might be useful here
        for x_m in range(map_size):
            for y_m in range(map_size):
                x_c,y_c= self.geometric_map.get_cell_indexes(x_m,y_m)
                height= closest_cell.get_cell_height if cell else 0
                if height:
                    if (x_c,y_c) in pos_to_cell:
                        num_of_elements=pos_to_cell[(x_c,y_c)]
                        height_map[x_c][y_c] = (height_map[x_c][y_c]*num_of_elements+height)/(num_of_elements+1)
                        pos_to_cell[(x_c,y_c)]+=1
                    else:
                        height_map[x_c][y_c] = height
                        pos_to_cell[(x_c,y_c)]=1

                
        return height_map

    def set_map(self, samples: list):
        """Set the heights in the geometric_map.
        Args:
            samples: A list of ground sample points
        """

        height_map = self._generate_map(samples)

        for x, y in np.ndindex(height_map.shape):
            self.geometric_map.set_cell_height(x, y, height_map[x, y])


def sample_surface(lander_global) -> list:
    """Generates ground samples based on the pose of the lander.

    Args:
        lander_global: The pose of the lander in the global coordinate frame

    Returns:
        A list of four ground sample points [x, y, z] (where each wheel touches the ground)
    """

    samples = []
    for wheel in rover["wheels"].values():
        # The surface point in the rover frame
        surface_lander = transform_from(
            np.eye(3), [wheel["x"], wheel["y"], wheel["z"] - (wheel["diameter"] / 2)]
        )

        surface_global = concat(surface_lander, lander_global)

        # Append only the x, y, z components of the surface point
        samples.append(surface_global[:3, 3].tolist())

    return samples
