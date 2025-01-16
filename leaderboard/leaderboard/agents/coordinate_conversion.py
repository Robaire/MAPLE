import carla
import numpy as np

from math import radians, degrees, cos, asin, acos

class Vector(object):
    def __init__(self, carla_vector):
        self.x = carla_vector.x
        self.y = -carla_vector.y
        self.z = carla_vector.z

    def __str__(self):
        return f"Vector3D(x={round(self.x, 6)}, y={round(self.y, 6)}, z={round(self.z, 6)})"

class Location(object):
    def __init__(self, carla_location):
        self.x = carla_location.x
        self.y = -carla_location.y
        self.z = carla_location.z

    def __str__(self):
        return f"Location(x={round(self.x, 6)}, y={round(self.y, 6)}, z={round(self.z, 6)})"

class Rotation(object):
    def __init__(self, carla_rotation):
        self.roll = radians(carla_rotation.roll)
        self.pitch = -radians(carla_rotation.pitch)
        self.yaw = -radians(carla_rotation.yaw)

    def __str__(self):
        return f"Rotation(roll={round(self.roll, 6)}, pitch={round(self.pitch, 6)}, yaw={round(self.yaw, 6)})"

class Transform(object):
    def __init__(self, transform):
        self.location = Location(transform.location)
        self.rotation = Rotation(transform.rotation)

    def transform(self, x, y, z):
        lh_location = carla.Location(x, -y, z)
        lh_transform = toLHCStransform(self)
        new_location = lh_transform.transform(lh_location)
        return toRHCSlocation(new_location)

    def __str__(self):
        return f"Transform({self.location}, {self.rotation})"

def toLHCSvector(vector):
    return carla.Vector3D(vector.x, -vector.y, vector.z)

def toLHCSlocation(location):
    return carla.Location(location.x, -location.y, location.z)

def toLHCSrotation(rotation):
    return carla.Rotation(roll=degrees(rotation.roll), pitch=-degrees(rotation.pitch), yaw=-degrees(rotation.yaw))

def toLHCStransform(transform):
    return carla.Transform(toLHCSlocation(transform.location), toLHCSrotation(transform.rotation))

def toRHCSvector(vector):
    return Vector(vector)

def toRHCSlocation(location):
    return Location(location)

def toRHCSrotation(rotation):
    return Rotation(rotation)

def toRHCStransform(transform):
    return Transform(transform)


def get_lander_transform(rover_transform, lander_transform):
    """Gets the transform of the lander with respect to the rover"""

    # Use the CARLA API to get the location
    lander_location = rover_transform.inverse_transform(lander_transform.location)

    # Use the matrixes tp get the rotation
    rover_matrix = np.array(rover_transform.get_inverse_matrix())[:-1,:-1]
    lander_matrix = np.array(lander_transform.get_matrix())[:-1,:-1]
    total_matrix = np.matmul(rover_matrix, lander_matrix)

    # Note that due to the nature of the sin() function, pitch will always be between -90 and 90
    pitch = asin(np.clip(total_matrix[2,0], -1, 1))

    # Use 2 elements of the matrix to differentiate between the 4 angle quadrants
    yaw1 = acos(np.clip(total_matrix[0,0] / cos(pitch), -1, 1))
    yaw2 = asin(np.clip(total_matrix[1,0] / cos(pitch), -1, 1))
    yaw = degrees(yaw1 * np.sign(yaw2))

    # Use 2 elements of the matrix to differentiate between the 4 angle quadrants
    roll1 = asin(np.clip(total_matrix[2,1] / -cos(pitch), -1, 1))
    roll2 = acos(np.clip(total_matrix[2,2] / cos(pitch), -1, 1))
    roll = degrees(roll1 * np.sign(roll2))

    pitch = degrees(pitch)

    transform = carla.Transform(lander_location, carla.Rotation(roll=roll, pitch=pitch, yaw=yaw))
    return toRHCStransform(transform)
