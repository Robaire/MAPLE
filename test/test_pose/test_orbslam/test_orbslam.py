from pytest import fixture, raises
from lac_data import PlaybackAgent

from maple.pose import OrbslamEstimator
from maple.utils import pytransform_to_tuple


@fixture
def agent():
    return PlaybackAgent("test/test_pose/test_orbslam/straight-line-1min.lac")


def _test_mode_error(agent):
    with raises(ValueError):
        estimator = OrbslamEstimator(agent, "FrontLeft", "FrontRight", mode="")


def _test_mono_init(agent):
    # Check if the camera is present
    # with raises(ValueError):
    #     estimator = OrbslamEstimator(agent, "Front", mode="mono")

    # Check initialization
    estimator = OrbslamEstimator(agent, "FrontLeft", mode="mono")
    assert estimator is not None
    estimator.shutdown()


def _test_stereo_init(agent):
    # Check if right camera is not provided
    with raises(ValueError):
        estimator = OrbslamEstimator(agent, "FrontLeft", mode="stereo")

    # Check if the right camera is not present
    # with raises(ValueError):
    #     estimator = OrbslamEstimator(agent, "FrontLeft", "Front", mode="stereo")

    # Check initialization
    estimator = OrbslamEstimator(agent, "FrontLeft", "FrontRight", mode="stereo")
    assert estimator is not None
    estimator.shutdown()

    # Check initialization
    estimator = OrbslamEstimator(agent, "FrontLeft", "FrontRight", mode="stereo_imu")
    assert estimator is not None
    estimator.shutdown()


def test_stereo(agent):
    estimator = OrbslamEstimator(agent, "FrontLeft", "FrontRight", mode="stereo")
    assert estimator is not None

    frame = 90
    agent.set_frame(frame)
    while not agent.at_end():
        input_data = agent.input_data()

        estimate = estimator.estimate(input_data)
        # assert estimate is not None
        if estimate is not None:
            print(f"Frame {frame}: {pytransform_to_tuple(estimate)}")
        else:
            print(f"Frame {frame}: None")
            pass

        frame = agent.step_frame()


def _test_stereo_imu(agent):
    # Check initialization
    estimator = OrbslamEstimator(agent, "FrontLeft", "FrontRight", mode="stereo_imu")
    assert estimator is not None

    frame = 90
    agent.set_frame(frame)
    while not agent.at_end():
        input_data = agent.input_data()

        estimate = estimator.estimate(input_data)
        # assert estimate is not None
        if estimate is not None:
            print(f"Frame {frame}: {pytransform_to_tuple(estimate)}")
        else:
            print(f"Frame {frame}: None")

        frame = agent.step_frame()
