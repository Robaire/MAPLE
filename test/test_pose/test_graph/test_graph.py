import numpy as np
from pytest import approx, raises
from pytransform3d.transformations import transform_from

from maple.pose import PoseGraph


def test_sequential():
    """Test creating a sequential graph"""
    graph = PoseGraph()

    # Test adding nodes
    graph.add_node(np.eye(4))  # 0
    graph.add_node(np.eye(4))  # 1
    graph.add_node(np.eye(4))  # 2

    assert graph.last_node == 2
    assert len(graph.nodes.values()) == 3

    # Test updating nodes
    graph.update_pose(0, np.zeros((4, 4)))
    assert graph.get_pose(0) == approx(np.zeros((4, 4)))
    assert graph.get_pose(1) == approx(np.eye(4))

    # Test attachments
    graph.attach("a", 1)
    graph.attach("a", 1, node_key=0)

    graph.attach_list("b", [5, 5, 5])
    graph.attach_list("b", [5, 5], node_key=1)

    assert graph.get_attachment("a") == [1, 1]
    assert graph.get_attachment("b") == [5, 5, 5, 5, 5]

    graph.attach("c", transform_from(np.eye(3), [1, 0, 0]), node_key=2)
    assert graph.solve_attachment("c")[0] == approx(
        transform_from(np.eye(3), [1, 0, 0])
    )


def test_non_sequential():
    """Test creating a non-sequential graph"""

    graph = PoseGraph(sequential=False)

    graph.add_node(np.eye(4), 0)
    graph.add_node(np.eye(4), 1)
    graph.add_node(np.eye(4), 5)

    assert graph.last_node == 5
    assert len(graph.nodes.values()) == 3

    with raises(ValueError):
        graph.add_node(np.eye(4))
