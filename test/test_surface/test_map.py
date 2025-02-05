# import numpy as np
# from pytest import approx
#
# from maple.surface.map import sample_surface, SurfaceMap
# from test.mocks import mock_geometric_map
#
#
# def test_sample_surface():
#     """Test that ground surface samples are generated correctly."""
#
#     # Generate samples without a lander pose
#     samples = sample_surface(np.eye(4))
#
#     assert len(samples) == 4
#     assert samples[0] == [0.222, 0.203, -0.119]
#     assert samples[1] == [0.222, -0.203, -0.119]
#     assert samples[2] == [-0.222, 0.203, -0.119]
#     assert samples[3] == [-0.222, -0.203, -0.119]
#
#
#
#     # TODO: Add tests with varied lander poses
#
#
# def test_map(mock_geometric_map):
#     """Test the surface map generation."""
#     # Create the SurfaceMap
#     surface_map = SurfaceMap(mock_geometric_map)
#
#     # TODO: Generate a bunch of surface samples
#     samples = []
#
#
#
#     # Run the map generator
#     result = surface_map._generate_map(samples)
#
#     # Check the results
#     expected = np.zeros((60, 60))
#     expected[0][0] = 1.0
#
#
#     print("Result Map:\n", result)
#     print("Expected Map:\n", expected)
#     assert result == approx(expected)
#
#
#
# def test_geo_map(mock_geometric_map):
#     """Demonstrates how to use the mock_geometric_map."""
#     gm = mock_geometric_map
#
#     assert gm.get_map_size() == 9
#     assert gm.get_cell_size() == 0.15
#     assert gm.get_cell_number() == 60
#
#     assert gm.get_cell_height(0, 0) == np.NINF
#     gm.set_cell_height(0, 0, 10)
#     assert gm.get_cell_height(0, 0) == 10











# ///// new test here ////////


import numpy as np
from pytest import approx

from maple.surface.map import sample_surface, _generate_map, set_map
from maple.surface.map import GeometricMap
from test.mocks import mock_geometric_map


def test_sample_surface():
    """Test that surface samples are generated correctly."""
    # Generate samples with an identity matrix as the lander pose
    samples = sample_surface(np.eye(4))

    # Assert the number and values of samples
    assert len(samples) == 4, "Expected 4 surface samples to be generated."
    assert samples[0] == approx([0.222, 0.203, -0.119]), "Sample 1 does not match expected values."
    assert samples[1] == approx([0.222, -0.203, -0.119]), "Sample 2 does not match expected values."
    assert samples[2] == approx([-0.222, 0.203, -0.119]), "Sample 3 does not match expected values."
    assert samples[3] == approx([-0.222, -0.203, -0.119]), "Sample 4 does not match expected values."

    # Test with a varied lander pose
    varied_pose = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, -0.2],
        [0.0, 0.0, 0.0, 1.0]
    ])
    samples_varied = sample_surface(varied_pose)

    # Verify samples are correctly transformed
    assert len(samples_varied) == 4, "Expected 4 surface samples with varied lander pose."
    assert samples_varied[0] == approx([1.222, 0.703, -0.319]), "Sample 1 with varied pose does not match expected values."
    assert samples_varied[1] == approx([1.222, 0.297, -0.319]), "Sample 2 with varied pose does not match expected values."
    assert samples_varied[2] == approx([0.778, 0.703, -0.319]), "Sample 3 with varied pose does not match expected values."
    assert samples_varied[3] == approx([0.778, 0.297, -0.319]), "Sample 4 with varied pose does not match expected values."


def test_generate_map_large_samples(mock_geometric_map):
    """Test the map generation with a large number of surface samples."""
    # Create a SurfaceMap instance
    surface_map =  GeometricMap(mock_geometric_map)

    # Generate a large number of samples across the map
    samples = []
    map_size = mock_geometric_map.get_map_size()
    cell_size = mock_geometric_map.get_cell_size()

    # Create samples for every cell in the grid
    for i in range(-30, 30):
        for j in range(-30, 30):
            x = i * cell_size
            y = j * cell_size
            z = np.random.uniform(-2.0, 2.0)  # Random height for the sample
            samples.append([x, y, z])

    # Run the map generator
    result = surface_map._generate_map(samples)

    # Expected map: compute average heights per cell
    expected = np.zeros((60, 60))  # Initialize with zeros
    cell_counts = np.zeros((60, 60))  # Track the number of samples per cell

    for sample in samples:
        x, y, z = sample
        cell_x = int((x + map_size / 2) / cell_size)
        cell_y = int((y + map_size / 2) / cell_size)
        if 0 <= cell_x < 60 and 0 <= cell_y < 60:
            expected[cell_x, cell_y] += z
            cell_counts[cell_x, cell_y] += 1

    # Compute the average height for each cell
    nonzero_cells = cell_counts > 0
    expected[nonzero_cells] /= cell_counts[nonzero_cells]
    # print("Generated Map:")
    # print(result)
    # print("Expected Map:")
    # print(expected)

    # Assert the results
    assert result.shape == expected.shape, "Resulting map dimensions do not match expected dimensions."
    assert np.allclose(result, expected,
                       equal_nan=True), "Generated map does not match the expected map for a large sample set."



def test_generate_map_varied_lander_pose(mock_geometric_map):
    """Test map generation with samples from a varied lander pose."""
    # Create a SurfaceMap instance
    surface_map = SurfaceMap(mock_geometric_map)

    # Varied lander pose
    varied_pose = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, -0.2],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Generate samples from this pose
    samples_varied_pose = sample_surface(varied_pose)
    print("Transformed Samples:")
    # Run the map generator
    result = surface_map._generate_map(samples_varied_pose)

    # Expected map: only relevant cells updated
    expected = np.full((60, 60), 0)
    for sample in samples_varied_pose:
        print(sample)
        x, y, z = sample
        cell_x = int((x + mock_geometric_map.get_map_size() / 2) / mock_geometric_map.get_cell_size())
        cell_y = int((y + mock_geometric_map.get_map_size() / 2) / mock_geometric_map.get_cell_size())
        if 0 <= cell_x < 60 and 0 <= cell_y < 60:
            expected[cell_x][cell_y] = z

    # Assert the results
    assert result.shape == expected.shape, "Resulting map dimensions do not match expected dimensions."
    assert np.allclose(result, expected, equal_nan=True), "Generated map does not match the expected map for varied lander pose."


def test_geometric_map_properties(mock_geometric_map):
    """Test properties and functionality of the mock geometric map."""
    gm = mock_geometric_map

    # Test geometric map properties
    assert gm.get_map_size() == approx(9.0), "Map size does not match the expected value."
    assert gm.get_cell_size() == approx(0.15), "Cell size does not match the expected value."
    assert gm.get_cell_number() == 60, "Cell number does not match the expected value."

    # Test setting and getting cell heights
    assert gm.get_cell_height(0, 0) == np.NINF, "Initial cell height should be -inf."
    gm.set_cell_height(0, 0, 10.0)
    assert gm.get_cell_height(0, 0) == approx(10.0), "Cell height was not set correctly."
