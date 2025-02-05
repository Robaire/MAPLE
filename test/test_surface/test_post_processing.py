import numpy as np
import pytest
from maple.surface.post_processing import PostProcessor
import matplotlib.pyplot as plt

def test_interpolate_blanks_basic():
    """Test basic interpolation with a simple height map."""
    # Create a test height map with NINF values
    height_map = np.full((5, 5), np.NINF)
    # Set some known values
    height_map[0, 0] = 1.0
    height_map[0, 4] = 2.0
    height_map[4, 0] = 3.0
    height_map[4, 4] = 4.0
    
    pp = PostProcessor(height_map)
    result = pp.interpolate_blanks(interpolation_method='linear')
    
    # Check that corners maintain their values
    assert result[0, 0] == 1.0
    assert result[0, 4] == 2.0
    assert result[4, 0] == 3.0
    assert result[4, 4] == 4.0
    
    # Check that interpolated values are between the known values
    assert 1.0 <= result[2, 2] <= 4.0
    
    # Check that no NINF values remain in the interpolated region
    mask = (result[1:4, 1:4] != np.NINF)
    assert mask.all()

def test_interpolate_blanks_empty():
    """Test interpolation with an empty height map."""
    height_map = np.full((3, 3), np.NINF)
    pp = PostProcessor(height_map)
    result = pp.interpolate_blanks()
    
    # Should return the original map if no valid points exist
    assert np.array_equal(result, height_map)

def test_interpolate_blanks_full():
    """Test interpolation with a completely filled height map."""
    height_map = np.ones((3, 3))
    pp = PostProcessor(height_map)
    result = pp.interpolate_blanks()
    
    # Should return the same map if no interpolation needed
    assert np.array_equal(result, height_map)

def test_interpolate_blanks_complex():
    """Test interpolation with a more complex pattern."""
    size = 10
    height_map = np.full((size, size), np.NINF)
    
    # Create a diagonal pattern of known values
    for i in range(size):
        height_map[i, i] = i
        if i < size-1:
            height_map[i, i+1] = i + 0.5
    
    pp = PostProcessor(height_map)
    result = pp.interpolate_blanks()
    
    # Check that diagonal values remain unchanged
    for i in range(size):
        assert result[i, i] == i
        if i < size-1:
            assert result[i, i+1] == i + 0.5
    
    # Check that interpolated values are reasonable
    assert np.all((result >= 0) | (result == np.NINF))
    assert np.all((result <= size) | (result == np.NINF))

def test_interpolate_blanks_visualization(tmp_path):
    """Test interpolation with visualization (optional)."""
    size = 20
    height_map = np.full((size, size), np.NINF)
    
    # Create some random known points
    random_points = np.random.choice(size*size, 40, replace=False)
    for point in random_points:
        i, j = point // size, point % size
        height_map[i, j] = np.random.uniform(0, 10)
    
    pp = PostProcessor(height_map)
    result = pp.interpolate_blanks()
    
    # Save visualization to temporary directory
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    plt.imshow(height_map, cmap='terrain')
    plt.colorbar()
    plt.title('Original Height Map')
    
    plt.subplot(122)
    plt.imshow(result, cmap='terrain')
    plt.colorbar()
    plt.title('Interpolated Height Map')
    
    plt.savefig(tmp_path / 'interpolation_test.png')
    plt.close()
    
    # Basic checks on the result
    assert not np.all(result == np.NINF)
    assert np.all((result >= 0) | (result == np.NINF))
    assert np.all((result <= 10) | (result == np.NINF))

def test_interpolate_blanks_methods():
    """Test different interpolation methods."""
    height_map = np.full((5, 5), np.NINF)
    height_map[0, 0] = 1.0
    height_map[0, 4] = 2.0
    height_map[4, 0] = 3.0
    height_map[4, 4] = 4.0
    
    pp = PostProcessor(height_map)
    
    methods = ['linear', 'nearest', 'cubic']
    results = {}
    
    for method in methods:
        results[method] = pp.interpolate_blanks(interpolation_method=method)
        
        # Check that corners maintain their values
        assert results[method][0, 0] == 1.0
        assert results[method][0, 4] == 2.0
        assert results[method][4, 0] == 3.0
        assert results[method][4, 4] == 4.0