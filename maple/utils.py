import math
import numpy as np

from numpy.typing import NDArray
from pytransform3d.rotations import euler_from_matrix, matrix_from_euler
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


def tuple_to_pytransform(elements) -> NDArray:
    """Converts a tuple of translation and rotation to a pytransform.

    Args:
        elements: A tuple of [x, y, z, roll, pitch, yaw] in meters and radians respectively
    Returns:
        A transformation matrix
    """

    x, y, z, roll, pitch, yaw = elements
    rotation = matrix_from_euler([yaw, pitch, roll], 2, 1, 0, False)

    return transform_from(rotation, [x, y, z])


def pytransform_to_tuple(transform) -> tuple:
    """Converts a pytransform to a tuple of the principle elements.

    Args:
        transform: A transformation matrix
    Returns:
        A tuple of [x, y, z, roll, pitch, yaw] in meters and radians respectively
    """

    x, y, z = transform[:3, 3]
    # TODO: Check that calls to this function are getting rotation elements back in the correct order...
    yaw, pitch, roll = euler_from_matrix(transform[:3, :3], 2, 1, 0, False)

    return (x, y, z, roll, pitch, yaw)


def pose_error(ground_truth, estimate) -> tuple:
    """Calcuate the error between two poses.

    Args:
        ground_truth: The ground truth transform
        estimate: The estimated transform

    Returns:
        A tuple of error for each pose element [x, y, z, roll, pitch, yaw]
    """

    # Translation error
    x, y, z = ground_truth[:3, 3] - estimate[:3, 3]

    # Rotation error
    gt_ypr = euler_from_matrix(ground_truth[:3, :3], 2, 1, 0, False)
    est_ypr = euler_from_matrix(estimate[:3, :3], 2, 1, 0, False)
    yaw, pitch, roll = ((gt_ypr - est_ypr) + np.pi) % (2 * np.pi) - np.pi

    return (x, y, z, roll, pitch, yaw)


def calculate_pose_errors(gt_xyz, gt_rpy, est_pose_matrix) -> dict:
    # TODO: This function should just take a ground truth transformation matrix, which is what the agent gives us anyways
    # carla_to_pytransform(agent.get_transform()) -> NDArray
    # Recommending using pose_error() instead
    """Calculate errors between ground truth pose parameters and estimated 4x4 pose matrix.
    Recommend using pose_error() instead.

    Args:
        gt_xyz (np.ndarray): Ground truth [x, y, z] position (shape: [3,])
        gt_rpy (np.ndarray): Ground truth [roll, pitch, yaw] in radians (shape: [3,])
        est_pose_matrix (np.ndarray): Estimated 4x4 pose matrix

    Returns:
        dict: Dictionary containing position and angular errors
    """

    rotation = matrix_from_euler(gt_rpy[::-1], 2, 1, 0, False)
    transform = transform_from(rotation, gt_xyz)
    x, y, z, roll, pitch, yaw = pose_error(transform, est_pose_matrix)

    errors = {
        "x_error": x,
        "y_error": y,
        "z_error": z,
        "roll_error": np.degrees(roll),
        "pitch_error": np.degrees(pitch),
        "yaw_error": np.degrees(yaw),
    }

    return errors
