import sys
from dataclasses import dataclass

from pytest import fixture


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
