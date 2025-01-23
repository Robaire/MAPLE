#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Alternative: do derivative of rot matrix and express in terms of derivatives of roll, pitch, yaw

import numpy as np

from leaderboard.agents.coordinate_conversion import toRHCStransform

# Uses the right handed coordinate system given to the agent, with counterclockwise rotations
# alpha around z axis (yaw), beta around y axis (pitch), gamma around x axis (roll)
def rot_matrix(alpha, beta, gamma):

    row1 = [np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)]
    row2 = [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)]
    row3 = [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]

    matrix = np.array([row1, row2, row3])
    return matrix


class IMU():

    def __init__(self, vehicle):

        self.dt = 0.05
        self._vehicle = vehicle

        self.loc_curr = None
        self.loc_prev = None
        self.loc_prev_prev = None
        self.rotmat_curr = None
        self.rotmat_prev = None

        self.gravity = np.array([0., 0., 1.6220])  # m/s^2

    def get_acceleration(self):

        acc = (self.loc_curr + self.loc_prev_prev - 2.*self.loc_prev)/self.dt**2.

        acc += self.gravity

        acc = self.rotmat_curr.T@acc

        return acc
    
    def get_angular_velocity(self):

        rotmat_der = (self.rotmat_curr - self.rotmat_prev)/self.dt
        angvel_mat = rotmat_der@self.rotmat_curr.T

        ang_vel = np.array([ angvel_mat[2,1], angvel_mat[0,2], angvel_mat[1,0] ])

        return ang_vel

    def get_data(self):

        transform = toRHCStransform(self._vehicle.get_transform())
        loc = transform.location
        rot = transform.rotation
        self.loc_curr = np.array([loc.x, loc.y, loc.z])
        self.rotmat_curr = rot_matrix(rot.yaw, rot.pitch, rot.roll)

        # Acceleration requires 2 previous frames, angular velocity one previous
        if self.rotmat_prev is None:
            acc, angvel = self.gravity, np.zeros(3)
        elif self.loc_prev_prev is None:
            acc, angvel = self.gravity, self.get_angular_velocity()
        else:
            acc = self.get_acceleration()
            angvel = self.get_angular_velocity()

        self.loc_prev_prev = self.loc_prev
        self.loc_prev = self.loc_curr
        self.rotmat_prev = self.rotmat_curr

        return np.array([acc[0], acc[1], acc[2], angvel[0], angvel[1], angvel[2]], dtype=np.float64)

