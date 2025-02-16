import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from pytransform3d.transformations import transform_from
from maple.surface.map import SurfaceHeight, sample_surface

class MockGeometricMap:
    def __init__(self):
        self.size = 10
        self.heights = np.full((self.size, self.size), np.NINF)
        
    def get_cell_number(self):
        return self.size
        
    def get_cell_indexes(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return int(x), int(y)
        return None
        
    def _is_cell_valid(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size
        
    def set_cell_height(self, x, y, height):
        if self._is_cell_valid(x, y):
            self.heights[x, y] = height

class TestSurfaceHeight:
    @pytest.fixture
    def mock_geometric_map(self):
        return MockGeometricMap()
    
    @pytest.fixture
    def surface_height(self, mock_geometric_map):
        return SurfaceHeight(mock_geometric_map)

    def test_generate_map(self, surface_height):
        # Test sample points
        samples = [
            [1.5, 1.5, 2.0],
            [1.5, 1.5, 2.0],  # Same point to test averaging
            [3.5, 3.5, 3.0],
        ]
        
        height_map = surface_height._generate_map(samples)
        
        assert height_map[1, 1] == 2.0  # Average of two points
        assert height_map[3, 3] == 3.0
        assert height_map[0, 0] == np.NINF  # Untouched cells should be NINF

    def test_set_map(self, surface_height, mock_geometric_map):
        samples = [
            [1.5, 1.5, 2.0],
            [3.5, 3.5, 3.0],
        ]
        
        surface_height.set_map(samples)
        
        assert mock_geometric_map.heights[1, 1] == 2.0
        assert mock_geometric_map.heights[3, 3] == 3.0
        assert mock_geometric_map.heights[0, 0] == np.NINF

    def test_interpolation_in_generate_map(self, surface_height):
        """Test that _generate_map properly interpolates missing values."""
        # Create a pattern of samples that will leave gaps
        samples = [
            [1.0, 1.0, 1.0],
            [1.0, 3.0, 2.0],
            [3.0, 1.0, 2.0],
            [3.0, 3.0, 3.0],
        ]
        
        height_map = surface_height._generate_map(samples)
        
        # Check that the sample points are preserved
        assert height_map[1, 1] == 1.0
        assert height_map[1, 3] == 2.0
        assert height_map[3, 1] == 2.0
        assert height_map[3, 3] == 3.0
        
        # Check that interpolation filled some gaps
        assert height_map[2, 2] != np.NINF
        
        # Check interpolated values are reasonable
        assert 1.0 <= height_map[2, 2] <= 3.0

    def test_visualization(self, surface_height, tmp_path):
        """Test that visualization method works and saves files."""
        samples = [
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
        ]
        
        surface_height.set_map(samples)
        
        # Test visualization
        save_path = tmp_path / 'test_height_map.png'
        surface_height.visualize_height_map(save_path=str(save_path))
        
        # Check that file was created
        assert save_path.exists()

def test_sample_surface():
    # Create a test lander pose
    lander_position = [1.0, 2.0, 3.0]
    lander_rotation = np.eye(3)  # Identity rotation matrix
    lander_global = transform_from(lander_rotation, lander_position)
    
    samples = sample_surface(lander_global)
    
    # Check that we get 4 samples (one for each wheel)
    assert len(samples) == 4
    
    # Check that each sample is a list of 3 coordinates
    for sample in samples:
        assert len(sample) == 3
        assert all(isinstance(coord, (int, float)) for coord in sample)
        
    # Check that samples are in reasonable positions relative to lander
    for sample in samples:
        # Samples should be near the lander position
        assert abs(sample[0] - lander_position[0]) < 5.0  # within 5 units
        assert abs(sample[1] - lander_position[1]) < 5.0
        # Z coordinate should be lower than lander (wheels touch ground)
        assert sample[2] < lander_position[2]
