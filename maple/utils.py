import math

from pytransform3d.rotations import matrix_from_euler, euler_from_matrix
from pytransform3d.transformations import transform_from
import numpy as np

def compute_blob_mean_and_covariance(binary_image):

    # Create a grid of pixel coordinates.
    y, x = np.indices(binary_image.shape)

    # Threshold the binary image to isolate the blob.
    blob_pixels = (binary_image > 0).astype(int)

    # Compute the mean of pixel coordinates.
    mean_x, mean_y = np.mean(x[blob_pixels == 1]), np.mean(y[blob_pixels == 1])
    mean = np.array([mean_x, mean_y])

    # Stack pixel coordinates to compute covariance using Scipy's cov function.
    pixel_coordinates = np.vstack((x[blob_pixels == 1], y[blob_pixels == 1]))

    # Compute the covariance matrix using Scipy's cov function.
    covariance_matrix = np.cov(pixel_coordinates)

    return mean, covariance_matrix

def plotErrorEllipse(ax,x,y,covariance,color=None,stdMultiplier=1,showMean=True,idText=None,marker='.'):

    covariance = np.asarray(covariance)

    (lambdas,eigenvectors) = np.linalg.eig(covariance)
    
    t = np.linspace(0,np.pi*2,30)
    
    lambda1 = lambdas[0]
    lambda2 = lambdas[1]
    
    scaledEigenvalue1 = stdMultiplier*np.sqrt(lambda1)*np.cos(t)
    scaledEigenvalue2 = stdMultiplier*np.sqrt(lambda2)*np.sin(t)
    
    scaledEigenvalues = np.vstack((scaledEigenvalue1,scaledEigenvalue2))
    
    ellipseBorderCoords = eigenvectors @ scaledEigenvalues
   
    ellipseBorderCoords_x = x+ellipseBorderCoords[0,:]
    ellipseBorderCoords_y = y+ellipseBorderCoords[1,:]
        
    if (color is not None):
        p = ax.plot(ellipseBorderCoords_x,ellipseBorderCoords_y,color=color)
    else:
        p = ax.plot(ellipseBorderCoords_x,ellipseBorderCoords_y)

    if (showMean):
        ax.plot(x,y,marker,color=p[0].get_color())

    if (idText is not None):
        ax.text(x,y,idText,bbox=dict(boxstyle='square, pad=-0.1',facecolor='white', alpha=0.5, edgecolor='none'),fontsize=8)

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


def tuple_to_pytransform(tuple_transform):
    """
    Unsure if this function will be used
    """
    x, y, z, roll, pitch, yaw = tuple_transform

    translation = [x, y, z]

    euler = [yaw, pitch, roll]
    rotation = matrix_from_euler(euler, 2, 1, 0, False)

    return transform_from(rotation, translation)


def pytransform_to_tuple(transform):
    """Converts a pytransform to x, y, z, yaw, pitch, roll"""
    R = transform[:3, :3]

    x, y, z = transform[:3, 3]

    yaw, pitch, roll = euler_from_matrix(R, 2, 1, 0, False)

    return (x, y, z, roll, pitch, yaw)
