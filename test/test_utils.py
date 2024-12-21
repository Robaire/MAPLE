import numpy as np
from pytest import approx

from maple import utils
from test.mock_carla_transform import Transform


def test_camera_params():
    """Test that camera parameters are properly calculated given the image dimensions."""

    params = utils.camera_parameters()
    assert params == approx((1748.05, 1748.05, 1224, 1024), abs=0.01)

    image = np.zeros((2048, 2448))
    params = utils.camera_parameters(image.shape)
    assert params == approx((1748.05, 1748.05, 1224, 1024), abs=0.01)

    image = np.zeros((1024, 1224))
    params = utils.camera_parameters(image.shape)
    assert params == approx((874.03, 874.03, 612, 512), abs=0.01)


def test_carla_to_pytransform():
    """Test that carla transforms are properly converted to pytransform3d"""

    carla_transform = Transform()
    py_transform = utils.carla_to_pytransform(carla_transform)
    expected = np.eye(4, 4)
    assert py_transform == approx(expected)

    carla_transform = Transform([10, 2, 0.1], [0.05, 0.15, 0.8])
    py_transform = utils.carla_to_pytransform(carla_transform)
    expected = np.array(
        [
            [0.6888834, -0.7112560, 0.1398373, 10.0],
            [0.7093009, 0.7011938, 0.0722456, 2.0],
            [-0.1494381, 0.0494180, 0.9875354, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert py_transform == approx(expected)
