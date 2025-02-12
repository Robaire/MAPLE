import math

from numpy.typing import NDArray
from pytransform3d.rotations import euler_from_matrix, matrix_from_euler
from pytransform3d.transformations import transform_from
import numpy as np
from scipy.spatial.transform import Rotation as Rot


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


def calculate_pose_errors(gt_xyz, gt_rpy, est_pose_matrix):
    """
    Calculate errors between ground truth pose parameters and estimated 4x4 pose matrix.
    
    Args:
        gt_xyz (np.ndarray): Ground truth [x, y, z] position (shape: [3,])
        gt_rpy (np.ndarray): Ground truth [roll, pitch, yaw] in radians (shape: [3,])
        est_pose_matrix (np.ndarray): Estimated 4x4 pose matrix
        
    Returns:
        dict: Dictionary containing position and angular errors
    """
    # Extract position from estimated pose matrix
    est_xyz = est_pose_matrix[:3, 3]
    
    # Extract rotation matrix from estimated pose matrix
    est_rot_matrix = est_pose_matrix[:3, :3]
    
    # Convert estimated rotation matrix to euler angles (roll, pitch, yaw)
    est_rpy = Rot.from_matrix(est_rot_matrix).as_euler('xyz')
    
    # Calculate position errors
    position_errors = gt_xyz - est_xyz
    
    # Calculate angular errors (handling wrap-around)
    angular_errors = np.zeros(3)
    for i in range(3):
        error = gt_rpy[i] - est_rpy[i]
        # Normalize to [-pi, pi]
        angular_errors[i] = (error + np.pi) % (2 * np.pi) - np.pi
    
    errors = {
        'x_error': position_errors[0],
        'y_error': position_errors[1],
        'z_error': position_errors[2],
        'roll_error': np.degrees(angular_errors[0]),  # Convert to degrees
        'pitch_error': np.degrees(angular_errors[1]),
        'yaw_error': np.degrees(angular_errors[2])
    }
    
    return errors
