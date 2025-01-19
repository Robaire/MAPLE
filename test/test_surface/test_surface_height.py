import sys
import numpy as np
from pytest import fixture
from maple.surface.surface_height import sample_surface

from dataclasses import dataclass


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
    """Demonstrates how to use the mock_geometric_map."""
    gm = mock_geometric_map

    assert gm.get_map_size() == 9
    assert gm.get_cell_size() == 0.15
    assert gm.get_cell_number() == 60

    assert gm.get_cell_height(0, 0) == np.NINF
    gm.set_cell_height(0, 0, 10)
    assert gm.get_cell_height(0, 0) == 10
