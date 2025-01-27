import math

from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import euler_from_matrix

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

# NOTE: Try not to use the below functions, lets keep it to carla and pytransforms
def tuple_to_pytransform(tuple_transform):
    """Try not to use this function for actual code, this is currently a test function to make sure the pytransform_to_carla runs correctly"""
    x, y, z, roll, pitch, yaw = tuple_transform

    translation = [x, y, z]

    euler = [yaw, pitch, roll]
    rotation = matrix_from_euler(euler, 2, 1, 0, False)

    return transform_from(rotation, translation)

def pytransform_to_tuple(transform):
    """Converts a pytransform to x, y, z, yaw, pitch, roll"""
    R = transform[:3, :3]

    x, y, z = transform[:3, 3]

    yaw, pitch ,roll = euler_from_matrix(R, 2, 1, 0, False)

    return (x, y, z, roll, pitch, yaw)
