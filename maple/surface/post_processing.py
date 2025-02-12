import numpy as np
import scipy.interpolate as interp


class PostProcessor:
    """Post-processor for the height map."""
    
    def __init__(self, height_map):
        """
        Args:
            height_map: The height map to be post-processed
        """

        self.height_map = height_map

    def interpolate_blanks(self,interpolation_method='linear'):
        """Interpolates missing values in the height map.
        
        Parameters:
        grid (2D numpy array): NxM grid with NaN indicating missing values.

        Returns:
        2D numpy array: Grid with missing values filled in."""

        if self.height_map is None:
            raise ValueError("The height map is not set.")
        grid = self.height_map

        # Find coordinates of known (nonzero) and unknown (zero) values
        known_points = np.argwhere(~np.isnan(grid))
        unknown_points = np.argwhere(np.isnan(grid))

        X = known_points[:, 1]  # x-coordinates
        Y = known_points[:, 0]  # y-coordinates
        Z = grid[Y, X]          # known height values

        # Interpolate using 'linear' method (bilinear in 2D)
        estimated_Z = interp.griddata((X, Y), Z, (unknown_points[:, 1], unknown_points[:, 0]), method=interpolation_method)

        # Fill missing values in the original grid
        grid[unknown_points[:, 0], unknown_points[:, 1]] = estimated_Z

        return grid