import numpy as np
from pytest import approx, raises
from pytransform3d.transformations import transform_from

from maple.pose.graph import _Node


def test_history():
    """Test that history is updated properly"""

    pose = np.eye(4)
    node = _Node(pose)

    assert node.get_pose() == approx(pose)

    pose[1, 1] = 2
    node.update_pose(pose)
    assert node.get_pose() == approx(pose)

    pose[1, 1] = 3
    node.update_pose(pose)
    assert node.get_pose() == approx(pose)

    assert len(node.history) == 3


def test_attachment():
    """Test that attachments are correctly inserted and retrieved"""

    node = _Node(np.eye(4))

    node["a"] += [1]
    node["a"] += [2]
    node["a"] += [3]
    node["b"] += [None]
    node["c"] += [1, 2, 3]

    assert node["a"] == [1, 2, 3]
    assert node["b"] == [None]
    assert node["c"] == [1, 2, 3]
    assert node["d"] == []


def test_solve():
    """Test that pose attachments are correctly inserted and calculated"""

    pose = transform_from(np.eye(3), [11, 7, 5])

    node = _Node(pose)

    node["a"] += [np.eye(4)]

    assert node.solve_attachment("a")[0] == approx(pose)


def test_solve_history():
    """Test that pose attachments are correctly calculated with the pose history"""

    pose = transform_from(np.eye(3), [11, 7, 5])
    node = _Node(pose)
    node["a"] += [np.eye(4)]

    pose = transform_from(np.eye(3), [4, 2, 1])
    node.update_pose(pose)

    assert node.solve_attachment("a")[0] == approx(pose)
