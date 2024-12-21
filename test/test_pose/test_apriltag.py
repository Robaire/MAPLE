import sys

import numpy as np
from pytest import approx, fixture, raises

from maple.pose.apriltag import Estimator
from test.mock_carla_transform import Transform


@fixture()
def mock_agent(mocker):
    # Mock carla
    mocker.patch.dict(sys.modules, {"carla": mocker.MagicMock()})

    # Mock the agent
    AutonomousAgent = mocker.patch(
        "leaderboard.autoagents.autonomous_agent.AutonomousAgent",
    )
    agent = AutonomousAgent.return_value

    # Patch needed functions
    agent.get_initial_position.return_value = Transform()
    agent.get_initial_lander_position.return_value = Transform()
    agent.get_camera_position.return_value = Transform()

    return agent


def test_lander_pose(mock_agent):
    """Test that the lander pose is correctly determined in global coordinates."""

    # No transform
    estimator = Estimator(mock_agent)
    expected = np.eye(4, 4)
    assert estimator.global_to_lander == approx(expected)

    # Translation
    mock_agent.get_initial_position.return_value = Transform(p=(10, 3, 0))
    mock_agent.get_initial_lander_position.return_value = Transform(p=(-5, -3.5, 0))

    estimator = Estimator(mock_agent)
    expected = np.eye(4, 4)
    expected[0][3] = 5.0
    expected[1][3] = -0.5
    assert estimator.global_to_lander == approx(expected)

    # TODO: Rotation
    # TODO: Composite translation + rotation


def test_no_fiducials(mock_agent):
    """Test that an exception is raised if the agent does not have fiducials enabled."""
    mock_agent.use_fiducials.return_value = False

    with raises(ValueError):
        _ = Estimator(mock_agent)


def test_no_active(mock_agent):
    """Test that None is returned when no cameras are active."""

    estimator = Estimator(mock_agent)
    assert estimator.estimate({}) is None


def test_estimate_image(mock_agent):
    """Test detections in a single image"""

    rng = np.random.default_rng()
    image = rng.integers(0, 255, (1024, 1024), dtype=np.uint8)

    estimator = Estimator(mock_agent)
    estimates = estimator._estimate_image(None, image)

    assert len(estimates) == 0

    # TODO: Test a detection with an actual image


def test_estimate(mock_agent):
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

    estimator = Estimator(mock_agent)
    estimate = estimator.estimate(input_data)
    assert estimate is None

    input_data["Grayscale"]["FrontLeft"] = random_image  # TODO: Use an actual image
    estimate = estimator.estimate(input_data)
    expected = np.eye(4, 4)  # TODO: Figure out expected
    assert estimate == approx(expected)


def test_estimate_cameras(mock_agent):
    """Test averaging detections from multiple cameras."""

    def get_camera_position(camera):
        cameras = {
            "FrontLeft": Transform(p=(0.28, 0.081, 0.131)),
            "FrontRight": Transform(p=(0.28, -0.081, 0.131)),
        }

        try:
            return cameras[camera]
        except KeyError:
            return Transform()

    mock_agent.get_camera_position = get_camera_position

    input_data = {
        "Grayscale": {
            "FrontLeft": None,
            "FrontRight": None,
        }
    }

    estimator = Estimator(mock_agent)
    estimate = estimator.estimate(input_data)
    expected = np.eye(4, 4)
    assert estimate == approx(expected)
