from map import SurfaceHeight
import numpy
import matplotlib.pyplot as plt

import sys
from dataclasses import dataclass


def mock_geometric_map(mocker):
    """Fixture for mocking GeometricMap"""

    # Mock carla
    mocker.patch.dict(sys.modules, {"carla": mocker.MagicMock()})

    # Import after mocking carla
    from leaderboard.agents.geometric_map import GeometricMap

    class Constants:
        map_size: float  # overall map width [m]
        cell_size: float  # individual cell width [m]
        cell_number: int  # number of cells [#]

    geometric_map = GeometricMap(Constants(9, 0.15, 60))

    return geometric_map

    # Load the csv representing the sample data. It is in the form of [x0,y0,z0],[x1,y1,z1],...


data = numpy.loadtxt(
    "maple/surface/test_data/output_sample_list.csv", delimiter=",", dtype=float
)
# Represent the data as a list of lists: [[x0,y0,z0],[x1,y1,z1],...]
data = data.tolist()

mock_geo_map = mock_geometric_map
surfaceHeight = SurfaceHeight(mock_geo_map)
surfaceHeight._generate_map(data)
