import math

from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import transform_from


def camera_parameters(shape: tuple = None) -> tuple[float, float, float, float]:
    """Calculate the camera parameters.
    Args:
        shape: The shape of the input image
    Returns:
        Camera parameters [fx, fy, cx, cy]
    """

    if shape:
        # Numpy arrays are HEIGHT x WIDTH!
        height = shape[0]
        width = shape[1]

    else:
        # If no dimensions are given assume the max image dimensions
        height = 2048
        width = 2448

    fov = math.radians(70)  # 70 deg HFOV
    focal_length = width / (2 * math.tan(fov / 2))

    return (focal_length, focal_length, width / 2.0, height / 2.0)


def carla_to_pytransform(transform):
    """Convert a carla transform to a pytransform."""

    # Extract translation
    translation = [transform.location.x, transform.location.y, transform.location.z]

    ## For XYZ convention
    # euler = [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw]
    # rotation = matrix_from_euler(euler, 0, 1, 2, False)

    # For ZYX convention
    euler = [transform.rotation.yaw, transform.rotation.pitch, transform.rotation.roll]
    rotation = matrix_from_euler(euler, 2, 1, 0, False)

    # Create 4x4 transformation matrix
    return transform_from(rotation, translation)
