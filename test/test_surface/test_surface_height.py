import sys
import numpy as np
from pytest import fixture
from maple.surface.surface_height import sample_surface


@fixture
def mock_geometric_map(mocker):
    """Fixture for mocking GeometricMap"""

    # Mock carla
    mocker.patch.dict(sys.modules, {"carla": mocker.MagicMock()})

    # TODO: Mock GeometricMap


def test_sample_surface():
    """Test that ground surface samples are generated correctly."""

    # Generate samples without a lander pose
    samples = sample_surface(np.eye(4))

    assert len(samples) == 4
    assert samples[0] == [0.222, 0.203, -0.119]
    assert samples[1] == [0.222, -0.203, -0.119]
    assert samples[2] == [-0.222, 0.203, -0.119]
    assert samples[3] == [-0.222, -0.203, -0.119]

    # TODO: Add tests with varied lander poses
