import numpy as np
from pytest import approx

from maple import utils
from test.mock_carla_transform import Transform

from maple.utils import pytransform_to_carla

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

    # Empty transform
    carla_transform = Transform()
    py_transform = utils.carla_to_pytransform(carla_transform)
    expected = np.eye(4, 4)
    assert py_transform == approx(expected)

    # Single axis rotation
    carla_transform = Transform([10, 0, 0], [0.05, 0, 0])
    py_transform = utils.carla_to_pytransform(carla_transform)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 0.9987503, -0.0499792, 0.0],
            [0.0, 0.0499792, 0.9987503, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert py_transform == approx(expected)

    # Multi axis rotation
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


def test_pytransform_to_carla():
    """Test that pytransform3d transforms are properly converted to carla like tuple"""

    # TODO: Kinda copied this from the above code and made changes so specific new example would be nice

    # Empty transform
    carla_transform = Transform()
    py_transform = utils.carla_to_pytransform(carla_transform)
    expected = np.eye(4, 4)
    assert py_transform == approx(expected)

    # Single axis rotation
    carla_transform = Transform([10, 0, 0], [0.05, 0, 0])
    py_transform = utils.carla_to_pytransform(carla_transform)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 0.9987503, -0.0499792, 0.0],
            [0.0, 0.0499792, 0.9987503, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert py_transform == approx(expected)

    # Multi axis rotation
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

    carla_transform = pytransform_to_carla(py_transform)
    x = carla_transform.location.x
    y = carla_transform.location.y
    z = carla_transform.location.z
    roll = carla_transform.rotation.roll
    pitch = carla_transform.rotation.pitch
    yaw = carla_transform.rotation.yaw

    epsilon = .00001

    assert abs(x - carla_transform.location.x) <= epsilon
    assert abs(y - carla_transform.location.y) <= epsilon
    assert abs(z - carla_transform.location.z) <= epsilon

    assert abs(roll - carla_transform.rotation.roll) <= epsilon
    assert abs(pitch - carla_transform.rotation.pitch) <= epsilon
    assert abs(yaw - carla_transform.rotation.yaw) <= epsilon

