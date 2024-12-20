import sys

import numpy as np
from pytest import approx, fixture

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

    # Patch the position functions
    agent.get_initial_position.return_value = Transform.empty()
    agent.get_initial_lander_position.return_value = Transform.empty()

    return agent


def test_lander_pose(mock_agent):
    """Test that the lander pose is correctly determined in global coordinates."""

    # TODO: Add non-trivial test cases

    # Create the estimator
    estimator = Estimator(mock_agent)
    expected = np.eye(4, 4)
    assert estimator.lander_transform == approx(expected)


def test_estimate(mock_agent):
    """Create a noisy image and check for detections."""

    # Create the estimator
    estimator = Estimator(mock_agent)

    rng = np.random.default_rng()
    image = rng.integers(0, 255, (1024, 1024), dtype=np.uint8)

    assert not estimator.estimate(image)
