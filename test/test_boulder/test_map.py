import numpy as np
from pytest import approx

from maple.boulder.map import BoulderMap, visualize_transforms
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


def test_visualize_transforms(flatten: bool = False):
    """
    Test the visualize_transforms function with boulder locations
    calculated from stored data

    Args:
        flatten: Whether to flatten the 3D plot to a 2D plot
    """

    # Load boulder positions from file
    transforms = np.load(
        "/home/altair_above/Lunar_Autonomy_2025/MAPLE/data/003/boulder_positions.npy"
    )
    # # Convert boulder positions to 4x4 transforms
    # transforms = [np.eye(4) for pos in boulder_positions]
    # Set translation component of each transform
    # for i, pos in enumerate(boulder_positions):
    #     transforms[i][:3, 3] = pos
    visualize_transforms(transforms, flatten=flatten)
