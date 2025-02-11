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

    def interpolate_blanks(self, interpolation_method='linear'):
        """Interpolates missing values in the height map.
        
        Parameters:
        grid (2D numpy array): NxM grid with NINF indicating missing values.

        Returns:
        2D numpy array: Grid with missing values filled in."""

        if self.height_map is None:
            raise ValueError("The height map is not set.")
        
        # Convert NINF to NaN for interpolation
        grid = self.height_map.copy()
        grid[grid == np.NINF] = np.nan

        # Find coordinates of known (non-NaN) and unknown (NaN) values
        known_points = np.argwhere(~np.isnan(grid))
        unknown_points = np.argwhere(np.isnan(grid))

        # If there are no points to interpolate, return original grid
        if len(known_points) == 0 or len(unknown_points) == 0:
            return self.height_map

        X = known_points[:, 1]  # x-coordinates
        Y = known_points[:, 0]  # y-coordinates
        Z = grid[Y, X]          # known height values

        # Interpolate using specified method
        estimated_Z = interp.griddata(
            (X, Y), Z, 
            (unknown_points[:, 1], unknown_points[:, 0]), 
            method=interpolation_method,
            fill_value=np.NINF
        )

        # Fill missing values in the grid
        result_grid = self.height_map.copy()
        result_grid[unknown_points[:, 0], unknown_points[:, 1]] = estimated_Z

        return result_grid