import math
import numpy as np

from numpy.typing import NDArray
from pytransform3d.rotations import euler_from_matrix, matrix_from_euler
from pytransform3d.transformations import transform_from
import xml.etree.ElementTree as ET


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
    focal_length_x = width / (2 * math.tan(fov / 2))
    focal_length_y = height / (2 * math.tan(fov / 2))

    return (focal_length_x, focal_length_y, width / 2.0, height / 2.0)


def carla_to_pytransform(transform):
    """Convert a carla transform to a pytransform."""

    # Extract translation
    translation = [transform.location.x, transform.location.y, transform.location.z]

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

def euler_to_rotmat(roll, pitch, yaw):
    """
    Convert euler angles (in degrees) to rotation matrix.
    Using extrinsic rotations XYZ (roll, pitch, yaw).
    """
    # Convert to radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    
    # Roll (X-axis rotation)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (Y-axis rotation)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (Z-axis rotation)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R

def create_pose_matrix(xyz, rpy):
    """
    Create 4x4 homogeneous transformation matrix from position and euler angles.
    
    Args:
        xyz: [x, y, z] position
        rpy: [roll, pitch, yaw] in degrees
    Returns:
        4x4 homogeneous transformation matrix
    """
    # Create 4x4 identity matrix
    pose = np.eye(4)
    
    # Insert rotation matrix (3x3)
    pose[:3, :3] = euler_to_rotmat(rpy[0], rpy[1], rpy[2])
    
    # Insert translation vector (3x1)
    pose[:3, 3] = xyz
    
    return pose

def transform_to_world_frame(local_pose, agent_pose):
    """
    Transform a pose from local (agent) frame to world frame
    
    Args:
        local_pose: 4x4 transformation matrix in local frame
        agent_pose: 4x4 transformation matrix of agent in world frame
    Returns:
        4x4 transformation matrix in world frame
    """
    return agent_pose @ local_pose

def create_boulder_pose(x, y):
    """
    Create a 4x4 transformation matrix for a boulder position
    
    Args:
        x, y: position in local frame
    Returns:
        4x4 transformation matrix
    """
    pose = np.eye(4)
    pose[0, 3] = x
    pose[1, 3] = y
    return pose

def extract_rock_locations(xml_file):
    """
    Parses an XML file and extracts x, y, and z coordinates of rocks.
    
    Args:
        xml_file (str): Path to the XML file.
    
    Returns:
        list: A list of tuples containing (x, y, z) coordinates of rocks.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    rock_positions = []
    
    for rock in root.findall(".//rocks/rock"):
        x = float(rock.get("x", 0))
        y = float(rock.get("y", 0))
        z = float(rock.get("z", 0))
        rock_positions.append((x, y, z))
    
    return rock_positions

