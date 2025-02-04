import numpy as np
from pytest import approx

from maple.boulder.map import BoulderMap
from test.mocks import mock_geometric_map
from matplotlib import pyplot as plt
from test.data_parser import CSVGeometricMap


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


def test_plot_transforms(positions_path: str, flatten: bool = False):
    """
    Test the plot_transforms function with boulder locations
    calculated from stored data

    Args:
        positions_path: Path to boulder positions from regular detection
        flatten: Whether to flatten the 3D plot to a 2D plot
    """
    # Load boulder positions from file
    transforms = np.load(positions_path)
    _plot_transforms(transforms, flatten=flatten)


def test_plot_transforms_comparison(
    positions_path: str, semantic_positions_path: str, flatten: bool = False
):
    """
    Test visualizing and comparing boulder positions from both detection methods

    Args:
        positions_path: Path to boulder positions from regular detection
        semantic_positions_path: Path to boulder positions from semantic detection
        flatten: Whether to flatten the 3D plot to 2D
    """
    # Load both sets of boulder positions
    transforms = np.load(positions_path)
    transforms_semantic = np.load(semantic_positions_path)

    # Create visualization and display it
    fig, ax = _plot_transforms_comparison(
        transforms, transforms_semantic, flatten=flatten
    )
    plt.show()


def test_plot_boulder_map(positions_path: str):
    """Plots a boulder map with optional transform overlay."""
    transforms = np.load(positions_path)
    boulder_map = BoulderMap(CSVGeometricMap())
    bool_map = boulder_map._generate_map(transforms)

    # Create the visualization with both map and scatter plot
    fig, ax = _plot_boulder_map(bool_map, show=False)

    # Add scatter plot overlay
    _plot_transforms(
        transforms,
        flatten=True,
        show=False,
        fig=fig,
        ax=ax,
        color="blue",
        label="Boulder Positions",
    )

    plt.show()


def test_plot_boulder_map_comparison(positions_path: str, semantic_positions_path: str):
    """Plots a boulder map comparison."""
    transforms = np.load(positions_path)
    transforms_semantic = np.load(semantic_positions_path)
    boulder_map = BoulderMap(CSVGeometricMap())
    bool_map = boulder_map._generate_map(transforms)
    bool_map_semantic = boulder_map._generate_map(transforms_semantic)
    _plot_boulder_map_comparison(bool_map, bool_map_semantic)


def _plot_transforms(
    transforms: list,
    title: str = "Point Cloud Visualization",
    flatten: bool = False,
    show: bool = True,
    color: str = "blue",
    label: str = None,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> tuple:
    """Plots a list of transforms as a 3D point cloud.

    Args:
        transforms: List of transforms where each transform's translation represents a point
        title: Optional title for the plot
        flatten: Whether to create a 2D plot instead of 3D
        show: Whether to display the plot immediately
        color: Color for the scatter points
        label: Label for the scatter points in the legend
        fig: Existing figure to plot on (optional)
        ax: Existing axis to plot on (optional)

    Returns:
        tuple: (figure, axis) matplotlib objects
    """
    # Extract x, y, z coordinates from transforms
    points = np.array([transform[:3, 3] for transform in transforms])

    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 10))
        if flatten:
            ax = fig.add_subplot(111)
            # Add grid lines every 0.15 meters
            ax.grid(True)
            # Calculate number of ticks needed
            x_min = np.floor(min(points[:, 0]) / 0.15) * 0.15
            x_max = np.ceil(max(points[:, 0]) / 0.15) * 0.15
            y_min = np.floor(min(points[:, 1]) / 0.15) * 0.15
            y_max = np.ceil(max(points[:, 1]) / 0.15) * 0.15
            # Set major ticks every 1 unit
            ax.set_xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 1, 1))
            ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + 1, 1))
            # Set minor ticks every 0.15 units
            ax.set_xticks(np.arange(x_min, x_max + 0.15, 0.15), minor=True)
            ax.set_yticks(np.arange(y_min, y_max + 0.15, 0.15), minor=True)
            # Enable grid for both major and minor ticks
            ax.grid(True, which="major", alpha=0.5)
            ax.grid(True, which="minor", alpha=0.2)
        else:
            ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if not flatten:
            ax.set_zlabel("Z")
        ax.set_title(title)

    if flatten:
        ax.scatter(
            points[:, 0], points[:, 1], c=color, marker="o", alpha=0.3, label=label
        )
    else:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=color,
            marker="o",
            alpha=0.3,
            label=label,
        )

    if show:
        plt.show()

    return fig, ax


def _plot_transforms_comparison(
    transforms: np.ndarray,
    transforms_semantic: np.ndarray,
    flatten: bool = False,
    show: bool = False,
) -> tuple:
    """Creates a comparison visualization of two sets of boulder transforms.

    Args:
        transforms: Array of transforms from regular detection
        transforms_semantic: Array of transforms from semantic detection
        flatten: Whether to create a 2D plot instead of 3D
        show: Whether to display the plot immediately

    Returns:
        tuple: (figure, axis) matplotlib objects
    """
    # Create base plot with first set of transforms
    fig, ax = _plot_transforms(
        transforms,
        title="Boulder Detection Comparison",
        flatten=flatten,
        show=False,
        color="blue",
        label="Regular Detection",
    )

    # Add semantic transforms using the same visualization function
    _, ax = _plot_transforms(
        transforms_semantic,
        flatten=flatten,
        show=False,
        color="red",
        label="Semantic Detection",
        fig=fig,
        ax=ax,
    )

    ax.legend()

    if show:
        plt.show()

    return fig, ax


def _plot_boulder_map(
    boulder_map: np.ndarray,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    show: bool = True,
    color: str = "red",
    label: str = None,
) -> tuple:
    """Plots a boulder map.

    Args:
        boulder_map: 2D numpy array representing the boulder map
        fig: Existing figure to plot on (optional)
        ax: Existing axis to plot on (optional)
        show: Whether to display the plot immediately
        color: Color for the boulder map
        label: Label for the boulder map in the legend

    Returns:
        tuple: (figure, axis) matplotlib objects
    """
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

    # Calculate extent based on map dimensions and cell size (0.15m)
    cell_size = 0.15
    map_size = boulder_map.shape[0] * cell_size
    extent = [
        -map_size / 2,
        map_size / 2,  # x bounds
        -map_size / 2,
        map_size / 2,  # y bounds
    ]
    im = ax.imshow(
        boulder_map.T,
        cmap=plt.matplotlib.colors.ListedColormap(["grey", color]),
        alpha=0.5,
        extent=extent,
        origin="lower",
    )
    plt.colorbar(im, ax=ax, label=label)

    # Add grid lines
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # Set major ticks every 1 unit
    ax.set_xticks(np.arange(-map_size / 2, map_size / 2 + cell_size))
    ax.set_yticks(np.arange(-map_size / 2, map_size / 2 + cell_size))
    # Set minor ticks every 0.15 units
    ax.set_xticks(
        np.arange(-map_size / 2, map_size / 2 + cell_size, cell_size), minor=True
    )
    ax.set_yticks(
        np.arange(-map_size / 2, map_size / 2 + cell_size, cell_size), minor=True
    )
    # Enable grid for both major and minor ticks
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.2)

    if show:
        plt.show()

    return fig, ax


def _plot_boulder_map_comparison(
    boulder_map: np.ndarray, boulder_map_semantic: np.ndarray
):
    """Plots a boulder map comparison."""
    fig, ax = _plot_boulder_map(
        boulder_map, show=False, color="blue", label="Regular Detection"
    )
    _, ax = _plot_boulder_map(
        boulder_map_semantic,
        show=False,
        fig=fig,
        ax=ax,
        color="red",
        label="Semantic Detection",
    )
    plt.show()
