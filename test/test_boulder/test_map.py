import numpy as np
from pytest import approx

from maple.boulder.map import BoulderMap, visualize_transforms
from test.mocks import mock_geometric_map
from matplotlib import pyplot as plt


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


def test_visualize_transforms(positions_path: str, flatten: bool = False):
    """
    Test the visualize_transforms function with boulder locations
    calculated from stored data

    Args:
        flatten: Whether to flatten the 3D plot to a 2D plot
    """
    # Load boulder positions from file
    transforms = np.load(positions_path)
    visualize_transforms(transforms, flatten=flatten)


def test_visualize_transforms_comparison(
    positions_path: str, semantic_positions_path: str, flatten: bool = False
):
    """
    Test visualizing and comparing boulder positions from both detection methods

    Args:
        positions_path: Path to boulder positions from regular detection
        semantic_positions_path: Path to boulder positions from semantic detection
        flatten: Whether to flatten the 3D plot to a 2D plot
    """
    # Load both sets of boulder positions
    transforms = np.load(positions_path)
    transforms_semantic = np.load(semantic_positions_path)

    # Extract points from transforms
    points = np.array([transform[:3, 3] for transform in transforms])
    points_semantic = np.array([transform[:3, 3] for transform in transforms_semantic])

    # Create figure
    fig = plt.figure(figsize=(10, 10))

    if flatten:
        # 2D plot
        ax = fig.add_subplot(111)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c="blue",
            marker="o",
            alpha=0.3,
            label="Regular Detection",
        )
        ax.scatter(
            points_semantic[:, 0],
            points_semantic[:, 1],
            c="red",
            marker="o",
            alpha=0.3,
            label="Semantic Detection",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    else:
        # 3D plot
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c="blue",
            marker="o",
            alpha=0.3,
            label="Regular Detection",
        )
        ax.scatter(
            points_semantic[:, 0],
            points_semantic[:, 1],
            points_semantic[:, 2],
            c="red",
            marker="o",
            alpha=0.3,
            label="Semantic Detection",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    ax.set_title("Boulder Detection Comparison")
    ax.legend()
    plt.show()
