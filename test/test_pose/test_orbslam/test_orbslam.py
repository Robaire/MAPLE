from maple.pose import OrbslamEstimator
from test.mocks import mock_agent



def test_modes(mock_agent):
    # for mode in ["mono", "stereo", "stereo_imu"]:
    for mode in ["stereo_imu"]:
        estimator = OrbslamEstimator(mock_agent, "FrontLeft", "FrontRight", mode=mode)
        assert estimator is not None
