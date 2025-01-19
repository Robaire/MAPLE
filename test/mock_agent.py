import sys

from pytest import fixture

from test.mock_carla_transform import Transform


@fixture
def mock_agent(mocker):
    """Fixture for generating mock agents."""

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

    def get_camera_position(camera):
        cameras = {
            "FrontLeft": Transform(p=(0.28, 0.081, 0.131)),
            "FrontRight": Transform(p=(0.28, -0.081, 0.131)),
        }

        try:
            return cameras[camera]
        except KeyError:
            return Transform()

    agent.get_camera_position = get_camera_position

    return agent
