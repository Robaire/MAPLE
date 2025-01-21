import numpy as np
from pytest import approx

from maple.surface.map import sample_surface, SurfaceMap
from test.mocks import mock_geometric_map


def test_sample_surface():
    """Test that ground surface samples are generated correctly."""

    # Generate samples without a lander pose
    samples = sample_surface(np.eye(4))

    assert len(samples) == 4
    assert samples[0] == [0.222, 0.203, -0.119]
    assert samples[1] == [0.222, -0.203, -0.119]
    assert samples[2] == [-0.222, 0.203, -0.119]
    assert samples[3] == [-0.222, -0.203, -0.119]

    # TODO: Add tests with varied lander poses


def test_map(mock_geometric_map):
    """Test the surface map generation."""
    # Create the SurfaceMap
    surface_map = SurfaceMap(mock_geometric_map)

    # TODO: Generate a bunch of surface samples
    samples = []

    # Run the map generator
    result = surface_map._generate_map(samples)

    # Check the results
    expected = np.zeros((60, 60))
    expected[0][0] = 1.0
    assert result == approx(expected)


def test_geo_map(mock_geometric_map):
    """Demonstrates how to use the mock_geometric_map."""
    gm = mock_geometric_map

    assert gm.get_map_size() == 9
    assert gm.get_cell_size() == 0.15
    assert gm.get_cell_number() == 60

    assert gm.get_cell_height(0, 0) == np.NINF
    gm.set_cell_height(0, 0, 10)
    assert gm.get_cell_height(0, 0) == 10
