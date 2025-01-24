import json

import numpy as np
from PIL import Image
from pytest import approx, fixture, raises
from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import transform_from

from maple.pose import ApriltagEstimator
from test.mocks import mock_agent, Transform


@fixture
def input_data():
    front_left = np.array(
        Image.open("test/test_pose/test_apriltag/front_left_99.png").convert("L"),
        dtype=np.uint8,
    )

    front_right = np.array(
        Image.open("test/test_pose/test_apriltag/front_right_99.png").convert("L"),
        dtype=np.uint8,
    )

    rng = np.random.default_rng()
    random_image = rng.integers(0, 255, (720, 1280), dtype=np.uint8)

    return {
        "Grayscale": {
            "FrontLeft": front_left,
            "FrontRight": front_right,
            "BackLeft": random_image,  # For testing when no tags are present
            "BackRight": None,  # For testing when no image is present
        }
    }


def gt_transform(file_path):
    """Helper function for loading a ground-truth global_to_rover transform."""

    with open(file_path) as file:
        ground_truth = json.load(file)

    translation = [ground_truth["gt_x"], ground_truth["gt_y"], ground_truth["gt_z"]]
    euler = [ground_truth["gt_yaw"], ground_truth["gt_pitch"], ground_truth["gt_roll"]]
    rotation = matrix_from_euler(euler, 2, 1, 0, True)

    return transform_from(rotation, translation)


def test_lander_pose(mock_agent):
    """Test that the lander pose is correctly determined in global coordinates."""

    # No transform
    estimator = ApriltagEstimator(mock_agent)
    expected = np.eye(4, 4)
    assert estimator.lander_global == approx(expected)

    # Translation
    mock_agent.get_initial_position.return_value = Transform(p=(10, 3, 0))
    mock_agent.get_initial_lander_position.return_value = Transform(p=(-5, -3.5, 0))

    estimator = ApriltagEstimator(mock_agent)
    expected = np.eye(4, 4)
    expected[0][3] = 5.0
    expected[1][3] = -0.5
    assert estimator.lander_global == approx(expected)

    # Rotation
    mock_agent.get_initial_position.return_value = Transform(
        p=(10, 3, 0), e=(0, 0, np.pi)
    )
    mock_agent.get_initial_lander_position.return_value = Transform(
        p=(-5, -3.5, 0), e=(0, 0, np.pi / 2)
    )
    estimator = ApriltagEstimator(mock_agent)
    expected = np.array(
        [
            [0.0, 1.0, 0.0, 15.0],
            [-1.0, 0.0, 0.0, 6.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert estimator.lander_global == approx(expected)


def test_no_fiducials(mock_agent):
    """Test that an exception is raised if the agent does not have fiducials enabled."""
    mock_agent.use_fiducials.return_value = False

    with raises(ValueError):
        ApriltagEstimator(mock_agent)


def test_no_active(mock_agent):
    """Test that None is returned when no cameras are active."""

    estimator = ApriltagEstimator(mock_agent)
    assert estimator.estimate({}) is None


def test_estimate_single(mock_agent, input_data):
    """Test detections in a single image"""

    # Remove one of the images since we are only testing FrontLeft
    input_data["Grayscale"]["FrontRight"] = None

    # Check for detections in the randomly generated image
    estimator = ApriltagEstimator(mock_agent)
    estimates = estimator._estimate_image(
        "BackLeft", input_data["Grayscale"]["BackLeft"]
    )
    assert len(estimates) == 0

    # Check for detections in a single image from the simulator
    estimates = estimator._estimate_image(
        "FrontLeft", input_data["Grayscale"]["FrontLeft"]
    )
    assert len(estimates) == 4

    # These pose estimates should all be very similar to one another
    assert estimates[0] == approx(estimates[1])

    # Check the accuracy of the pose estimation
    rover_global = gt_transform("test/test_pose/test_apriltag/ground_truth_99.json")
    average = estimator(input_data)
    assert average == approx(rover_global)


def test_estimate_multi(mock_agent):
    """Test averaging detections from multiple images"""

    rng = np.random.default_rng()
    random_image = rng.integers(0, 255, (1024, 1024), dtype=np.uint8)

    # The actual keys in "Grayscale" are the sensor objects themselves
    # For testing it doesn't matter what the key is
    input_data = {
        "Grayscale": {
            "FrontLeft": None,
            "FrontRight": random_image,
        }
    }

    estimator = ApriltagEstimator(mock_agent)
    estimate = estimator(input_data)
    assert estimate is None

    input_data["Grayscale"]["FrontLeft"] = random_image  # TODO: Use an actual image
    estimate = estimator(input_data)
    expected = np.eye(4, 4)  # TODO: Figure out expected
    assert estimate == approx(expected)


def test_estimate_cameras(mock_agent):
    """Test averaging detections from multiple cameras."""

    input_data = {
        "Grayscale": {
            "FrontLeft": None,
            "FrontRight": None,
        }
    }

    estimator = ApriltagEstimator(mock_agent)
    estimate = estimator(input_data)
    expected = np.eye(4, 4)
    assert estimate == approx(expected)
