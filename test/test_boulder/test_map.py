import sys
from dataclasses import dataclass

import numpy as np
from pytest import approx, fixture
from maple.boulder.map import BoulderMap


@fixture
def mock_geometric_map(mocker):
    """Fixture for mocking GeometricMap"""

    # Mock carla
    mocker.patch.dict(sys.modules, {"carla": mocker.MagicMock()})

    # Import after mocking carla
    from leaderboard.agents.geometric_map import GeometricMap

    @dataclass
    class Constants:
        map_size: float  # overall map width [m]
        cell_size: float  # individual cell width [m]
        cell_number: int  # number of cells [#]

    geometric_map = GeometricMap(Constants(9, 0.15, 60))

    return geometric_map


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
