import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as pyt_pu
import pytransform3d.rotations as pyt_r
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import itertools
import math

def rotate_pose_in_place(pose_matrix, roll_deg=0, pitch_deg=0, yaw_deg=0):
    """
    Apply a local RPY rotation on the rotation part of the pose, keeping translation fixed.
    """
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    Rx = np.array([
        [1, 0, 0], 
        [0, np.cos(roll), -np.sin(roll)], 
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0], 
        [np.sin(yaw), np.cos(yaw), 0], 
        [0, 0, 1]
    ])

    # Compose rotation in local frame
    delta_R = Rz @ Ry @ Rx

    R_old = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]

    # Apply in local frame (right multiplication)
    R_new = R_old @ delta_R

    new_pose = np.eye(4)
    new_pose[:3, :3] = R_new
    new_pose[:3, 3] = t
    return new_pose

def average_poses(T1, T2):
    """
    Average between two 4x4 transformation matrices.
    """
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]

    # Interpolate halfway (just average)
    t_mid = (t1 + t2) / 2

    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()

    # Interpolate halfway (t=0.5)
    key_times = [0, 1]
    rotations = R.from_quat([q1, q2])
    slerp = Slerp(key_times, rotations)
    R_mid = slerp(0.5).as_matrix()

    T_mid = np.eye(4)
    T_mid[:3, :3] = R_mid
    T_mid[:3, 3] = t_mid

    return T_mid

DIRECTION = False

TRAJECTORIES = 1
ERRORS = 1

gt_traj = np.load('gt.npy')
os_front_traj = np.load('os_front.npy')
os_back_traj = np.load('os_back.npy')
T_init_world = np.load('init.npy')
T_camera_rover = np.load("cam.npy")
surface = np.load('surface.npy')
avg_traj = np.array([average_poses(T1, T2) for T1, T2 in zip(os_front_traj, os_back_traj)])

if TRAJECTORIES:

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = -4
    y = 0
    r = 4
    ax.set_xlim((x - r, x + r))
    ax.set_ylim((y - r, y + r))
    ax.set_zlim((0, 2))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=40, azim=5)

    gt_plot = pyt_pu.Trajectory(gt_traj, n_frames=2, s=0.1, show_direction=DIRECTION, color='black')
    gt_plot.add_trajectory(ax)

    os_plot = pyt_pu.Trajectory(os_front_traj, n_frames=2, s=0.1, show_direction=DIRECTION, color='purple')
    os_plot.add_trajectory(ax)

    os_plot = pyt_pu.Trajectory(os_back_traj, n_frames=2, s=0.1, show_direction=DIRECTION, color='magenta')
    os_plot.add_trajectory(ax)

    avg_plot = pyt_pu.Trajectory(avg_traj, n_frames=2, s=0.1, show_direction=DIRECTION, color='cyan')
    avg_plot.add_trajectory(ax)

    ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], c='red', alpha=0.2)

    plt.show()

elif ERRORS:

    gt_t = gt_traj[:, :3, 3]
    front_t = os_front_traj[:, :3, 3]
    back_t = os_back_traj[:, :3, 3]
    avg_t = avg_traj[:, :3, 3]

    front_err = front_t - gt_t
    back_err = back_t - gt_t
    avg_err = avg_t - gt_t

    x = np.arange(len(gt_t), step=1)

    def euclidean_error(err):
        return np.linalg.norm(err, axis=1)

    front_euc = euclidean_error(front_err)
    back_euc = euclidean_error(back_err)
    avg_euc = euclidean_error(avg_err)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    # X error
    axs[0].plot(x, front_err[:, 0], label='Front')
    axs[0].plot(x, back_err[:, 0], label='Back')
    axs[0].plot(x, avg_err[:, 0], label='Avg')
    axs[0].set_title('X Error')
    axs[0].legend()
    axs[0].grid()

    # Y error
    axs[1].plot(x, front_err[:, 1], label='Front')
    axs[1].plot(x, back_err[:, 1], label='Back')
    axs[1].plot(x, avg_err[:, 1], label='Avg')
    axs[1].set_title('Y Error')
    axs[1].legend()
    axs[1].grid()

    # Z error
    axs[2].plot(x, front_err[:, 2], label='Front')
    axs[2].plot(x, back_err[:, 2], label='Back')
    axs[2].plot(x, avg_err[:, 2], label='Avg')
    axs[2].set_title('Z Error')
    axs[2].legend()
    axs[2].grid()

    # Euclidean error
    axs[3].plot(x, front_euc, label='Front')
    axs[3].plot(x, back_euc, label='Back')
    axs[3].plot(x, avg_euc, label='Avg')
    axs[3].set_title('Euclidean Error')
    axs[3].legend()
    axs[3].grid()

    plt.tight_layout()
    plt.show()