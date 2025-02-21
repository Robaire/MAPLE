import numpy as np
import scipy.interpolate as interp
from scipy.ndimage import convolve


class PostProcessor:
    """Post-processor for the height map with confidence-based interpolation."""
    
    def __init__(self, height_map):
        """
        Args:
            height_map: The height map to be post-processed
        """
        self.height_map = height_map
        
    def estimate_from_neighbors(self, grid, max_distance=2):
        """Estimate height using weighted average of nearby known values.
        
        Args:
            grid: The height map grid
            max_distance: Maximum distance to search for neighbors
            
        Returns:
            numpy array: Grid with estimated values
        """
        result = grid.copy()
        rows, cols = grid.shape
        
        for i in range(rows):
            for j in range(cols):
                if result[i,j] == np.NINF:
                    known_values = []
                    for d in range(1, max_distance + 1):
                        # Check cells in square pattern around point
                        for ni in range(max(0, i-d), min(rows, i+d+1)):
                            for nj in range(max(0, j-d), min(cols, j+d+1)):
                                if grid[ni,nj] != np.NINF:
                                    # Weight by inverse distance
                                    dist = np.sqrt((ni-i)**2 + (nj-j)**2)
                                    known_values.append((grid[ni,nj], 1/dist))
                    
                    if known_values:
                        values, weights = zip(*known_values)
                        result[i,j] = np.average(values, weights=weights)
        
        return result

    def estimate_with_trend(self, grid):
        """Estimate heights using overall terrain trend.
        
        Args:
            grid: The height map grid
            
        Returns:
            numpy array: Grid with trend-based estimates
        """
        valid_mask = grid != np.NINF
        if not np.any(valid_mask):
            return grid
        
        # Calculate average slope in x and y directions
        y_coords, x_coords = np.where(valid_mask)
        heights = grid[valid_mask]
        
        # Fit a plane to known points
        A = np.column_stack((x_coords, y_coords, np.ones_like(x_coords)))
        coeffs, _, _, _ = np.linalg.lstsq(A, heights, rcond=None)
        
        # Apply trend to unknown points
        result = grid.copy()
        unknown_y, unknown_x = np.where(grid == np.NINF)
        if len(unknown_x) > 0:  # Only process if there are unknown points
            result[unknown_y, unknown_x] = (
                coeffs[0] * unknown_x + coeffs[1] * unknown_y + coeffs[2]
            )
        
        return result

    def interpolate_with_confidence(self, max_distance=2):
        """Interpolate missing values using multiple methods with confidence levels.
        
        Args:
            max_distance: Maximum distance for neighbor-based interpolation
            
        Returns:
            tuple: (interpolated_grid, confidence_grid)
            - interpolated_grid: Height map with all values filled
            - confidence_grid: Confidence levels for each point (0.0 to 1.0)
        """
        if self.height_map is None:
            raise ValueError("The height map is not set.")
        
        result = self.height_map.copy()
        confidence = np.zeros_like(result)
        
        # Stage 1: Standard linear interpolation (highest confidence)
        grid = self.height_map.copy()
        grid[grid == np.NINF] = np.nan
        
        known_points = np.argwhere(~np.isnan(grid))
        unknown_points = np.argwhere(np.isnan(grid))
        
        if len(known_points) > 0 and len(unknown_points) > 0:
            X = known_points[:, 1]
            Y = known_points[:, 0]
            Z = grid[Y, X]
            
            estimated_Z = interp.griddata(
                (X, Y), Z,
                (unknown_points[:, 1], unknown_points[:, 0]),
                method='linear',
                fill_value=np.NINF
            )
            
            # Update results and confidence for interpolated points
            mask = estimated_Z != np.NINF
            result[unknown_points[mask, 0], unknown_points[mask, 1]] = estimated_Z[mask]
            confidence[unknown_points[mask, 0], unknown_points[mask, 1]] = 0.9
        
        # Stage 2: Nearest neighbor estimation (medium confidence)
        still_unknown = result == np.NINF
        if np.any(still_unknown):
            neighbor_est = self.estimate_from_neighbors(result, max_distance)
            mask = (neighbor_est != np.NINF) & still_unknown
            result[mask] = neighbor_est[mask]
            confidence[mask] = 0.6
        
        # Stage 3: Trend analysis (lowest confidence)
        still_unknown = result == np.NINF
        if np.any(still_unknown):
            trend_est = self.estimate_with_trend(result)
            mask = (trend_est != np.NINF) & still_unknown
            result[mask] = trend_est[mask]
            confidence[mask] = 0.3
        
        return result, confidence
    
    def interpolate_and_smooth(self, filter_size=3):
        """Interpolate missing values and apply a smoothing filter.
        
        Args:
            filter_size: Size of the square kernel for smoothing"""
        result, _ = self.interpolate_with_confidence()
        return self.smoothing_filter(result, filter_size)
    
    def smoothing_filter(self, zi, filter_size=3):
        """
        Apply a smoothing filter to the height data by replacing each point 
        with the average of its neighbors.

        Inputs:
        - zi: 2D array of interpolated heights
        - filter_size: Size of the filter (must be an odd integer, default: 3x3)

        Returns:
        - Smoothed 2D array of heights with the same shape as zi
        """
        if filter_size % 2 == 0:
            raise ValueError("Filter size must be an odd integer.")

        # Create a uniform averaging filter
        kernel = np.ones((filter_size, filter_size)) / (filter_size ** 2)

        # Apply the convolution
        smoothed_zi = convolve(zi, kernel, mode='nearest')

        return smoothed_zi

    def interpolate_blanks(self, interpolation_method='linear'):
        """Legacy method that uses only linear interpolation.
        Now calls interpolate_with_confidence and returns only the height map.
        """
        result, _ = self.interpolate_with_confidence()
        return result
