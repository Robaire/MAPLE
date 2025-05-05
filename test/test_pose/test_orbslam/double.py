import numpy as np
import matplotlib.pyplot as plt
from lac_data import PlaybackAgent

from maple.pose import DoubleSlamEstimator
from maple.utils import pytransform_to_tuple, carla_to_pytransform


if __name__ == "__main__":
    agent = PlaybackAgent("test/test_pose/test_orbslam/beaver_6.lac")
    orb = DoubleSlamEstimator(agent)

    pose_gt = []
    pose_orb = []
    pose_err = []

    # Skip to 90
    frame = 100
    agent.set_frame(frame)

    # Set the orbslam global
    orb._set_orbslam_global(carla_to_pytransform(agent.get_transform()))

    while not agent.at_end():
        # Skip odd frames since there is no image data
        if frame % 2 != 0:
            frame = agent.step_frame()
            continue

        print(f"Frame: {int(frame)}")

        input_data = agent.input_data()

        gt = pytransform_to_tuple(carla_to_pytransform(agent.get_transform()))

        pose_gt.append((frame, gt))

        try:
            estimate = orb.estimate(input_data)
        except Exception as e:
            raise e
            print(f"Frame: {int(frame)}")
            print(f"Error: {e}")
            estimate = None

        if estimate is not None:
            oe = pytransform_to_tuple(estimate)
            pose_orb.append((frame, oe))
            pose_err.append((frame, np.array(oe) - np.array(gt)))

        frame = agent.step_frame()
        if frame > 3000:
            break

    # Plot the results
    gt_f = [pose[0] for pose in pose_gt]
    gt_x = [pose[1][0] for pose in pose_gt]
    gt_y = [pose[1][1] for pose in pose_gt]
    gt_z = [pose[1][2] for pose in pose_gt]
    gt_roll = [pose[1][3] for pose in pose_gt]
    gt_pitch = [pose[1][4] for pose in pose_gt]
    gt_yaw = [pose[1][5] for pose in pose_gt]

    orb_f = [pose[0] for pose in pose_orb]
    orb_x = [pose[1][0] for pose in pose_orb]
    orb_y = [pose[1][1] for pose in pose_orb]
    orb_z = [pose[1][2] for pose in pose_orb]
    orb_roll = [pose[1][3] for pose in pose_orb]
    orb_pitch = [pose[1][4] for pose in pose_orb]
    orb_yaw = [pose[1][5] for pose in pose_orb]

    err_f = [pose[0] for pose in pose_err]
    err_x = [pose[1][0] for pose in pose_err]
    err_y = [pose[1][1] for pose in pose_err]
    err_z = [pose[1][2] for pose in pose_err]
    err_roll = [pose[1][3] for pose in pose_err]
    err_pitch = [pose[1][4] for pose in pose_err]
    err_yaw = [pose[1][5] for pose in pose_err]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # XY plot
    ax1.plot(gt_x, gt_y, label="Ground Truth")
    ax1.plot(orb_x, orb_y, label="DoubleSlam")
    ax1.legend()
    ax1.set_title("XY Trajectory")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_aspect("equal")

    # Z plot
    ax2.plot(gt_f, gt_z, label="Ground Truth")
    ax2.plot(orb_f, orb_z, label="DoubleSlam")
    ax2.legend()
    ax2.set_title("Z Position")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Z Position (m)")

    # Orientation plot
    ax3.plot(gt_f, gt_roll, label="GT Roll")
    ax3.plot(orb_f, orb_roll, label="DORB Roll")
    ax3.plot(gt_f, gt_pitch, label="GT Pitch")
    ax3.plot(orb_f, orb_pitch, label="DORB Pitch")
    ax3.plot(gt_f, gt_yaw, label="GT Yaw")
    ax3.plot(orb_f, orb_yaw, label="DORB Yaw")
    ax3.legend()
    ax3.set_title("Orientation")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Angle (rad)")
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(err_f, err_x, label="X")
    ax1.plot(err_f, err_y, label="Y")
    ax1.plot(err_f, err_z, label="Z")
    ax1.legend()
    ax1.set_title("Translation Error")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Error (m)")

    ax2.plot(err_f, np.rad2deg(err_roll), label="Roll")
    ax2.plot(err_f, np.rad2deg(err_pitch), label="Pitch")
    ax2.plot(err_f, np.rad2deg(err_yaw), label="Yaw")
    ax2.legend()
    ax2.set_title("Rotation Error")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Error (deg)")
    plt.tight_layout()
    plt.show()
