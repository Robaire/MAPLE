import numpy as np
from pytest import approx

from maple.boulder.map import BoulderMap
from test.mocks import mock_geometric_map


def test_boulder_map(mock_geometric_map):
    # Create the boulder map
    boulder_map = BoulderMap(mock_geometric_map)

    # TODO: Generate a bunch of boulder samples
    samples = []

    # Run the map generator
    result = boulder_map._generate_map(samples)

    # Check the results
    expected = np.zeros((60, 60), dtype=bool)
    expected[0][0] = True
    assert result == approx(expected)
