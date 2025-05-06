from lac_data import PlaybackAgent
from maple.pose import OrbslamEstimator
from maple.utils import pytransform_to_tuple, carla_to_pytransform
import numpy as np
import time

if __name__ == "__main__":
    agent = PlaybackAgent("test/test_pose/test_orbslam/beaver_6.lac")
    # agent = PlaybackAgent("test/test_pose/test_orbslam/straight-line-1min.lac")
    orb_front = OrbslamEstimator(agent, "FrontLeft", "FrontRight", mode="stereo")
    orb_rear = OrbslamEstimator(agent, "BackLeft", "BackRight", mode="stereo")

    pose_gt = []
    pose_orb_front = []
    pose_orb_rear = []

    # Skip to 90
    frame = 100
    agent.set_frame(frame)

    # Set the orbslam global
    orb_front._set_orbslam_global(carla_to_pytransform(agent.get_transform()))
    orb_rear._set_orbslam_global(carla_to_pytransform(agent.get_transform()))

    while not agent.at_end():
        # Skip frames without image data
        if frame % 2 != 0:
            frame = agent.step_frame()
            continue

        print(f"Frame: {int(frame)}")

        input_data = agent.input_data()

        pose_gt.append(
            (frame, pytransform_to_tuple(carla_to_pytransform(agent.get_transform())))
        )

        try:
            estimate_front = orb_front.estimate(input_data)
        except Exception as e:
            print(f"Error: {e}")
            estimate_front = None

        try:
            estimate_rear = orb_rear.estimate(input_data)
        except Exception as e:
            print(f"Error: {e}")
            estimate_rear = None

        if estimate_front is not None:
            pose_orb_front.append((frame, pytransform_to_tuple(estimate_front)))

        if estimate_rear is not None:
            pose_orb_rear.append((frame, pytransform_to_tuple(estimate_rear)))

        frame = agent.step_frame()
        if frame > 2000:
            break

    # Plot the results
    import matplotlib.pyplot as plt

    gt_f = [pose[0] for pose in pose_gt]
    gt_x = [pose[1][0] for pose in pose_gt]
    gt_y = [pose[1][1] for pose in pose_gt]
    gt_z = [pose[1][2] for pose in pose_gt]
    gt_roll = [pose[1][3] for pose in pose_gt]
    gt_pitch = [pose[1][4] for pose in pose_gt]
    gt_yaw = [pose[1][5] for pose in pose_gt]

    gt_f_markers = [f for f in gt_f if f % 100 == 0]
    gt_x_markers = [x[1] for x in zip(gt_f, gt_x) if x[0] % 100 == 0]
    gt_y_markers = [x[1] for x in zip(gt_f, gt_y) if x[0] % 100 == 0]

    orb_front_f = [pose[0] for pose in pose_orb_front]
    orb_front_x = [pose[1][0] for pose in pose_orb_front]
    orb_front_y = [pose[1][1] for pose in pose_orb_front]
    orb_front_z = [pose[1][2] for pose in pose_orb_front]
    orb_front_roll = [pose[1][3] for pose in pose_orb_front]
    orb_front_pitch = [pose[1][4] for pose in pose_orb_front]
    orb_front_yaw = [pose[1][5] for pose in pose_orb_front]

    orb_rear_f = [pose[0] for pose in pose_orb_rear]
    orb_rear_x = [pose[1][0] for pose in pose_orb_rear]
    orb_rear_y = [pose[1][1] for pose in pose_orb_rear]
    orb_rear_z = [pose[1][2] for pose in pose_orb_rear]
    orb_rear_roll = [pose[1][3] for pose in pose_orb_rear]
    orb_rear_pitch = [pose[1][4] for pose in pose_orb_rear]
    orb_rear_yaw = [pose[1][5] for pose in pose_orb_rear]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # XY plot
    ax1.plot(gt_x, gt_y, label="Ground Truth")
    ax1.plot(orb_front_x, orb_front_y, label="Front ORB-SLAM")
    ax1.plot(orb_rear_x, orb_rear_y, label="Rear ORB-SLAM")
    scatter = ax1.scatter(gt_x_markers, gt_y_markers, color="black", marker="x")
    # Add frame number labels
    for i, (x, y, frame) in enumerate(zip(gt_x_markers, gt_y_markers, gt_f_markers)):
        ax1.annotate(
            f"{int(frame)}",
            (x, y),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=8,
        )
    ax1.legend()
    ax1.set_title("XY Trajectory")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.set_aspect("equal")

    # Z plot
    ax2.plot(gt_f, gt_z, label="Ground Truth")
    ax2.plot(orb_front_f, orb_front_z, label="Front ORB-SLAM")
    ax2.plot(orb_rear_f, orb_rear_z, label="Rear ORB-SLAM")
    ax2.legend()
    ax2.set_title("Z Position")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Z Position")

    # Orientation plot
    ax3.plot(gt_f, gt_roll, label="GT Roll")
    ax3.plot(orb_front_f, orb_front_roll, label="Front ORB Roll")
    ax3.plot(orb_rear_f, orb_rear_roll, label="Rear ORB Roll")
    ax3.plot(gt_f, gt_pitch, label="GT Pitch")
    ax3.plot(orb_front_f, orb_front_pitch, label="Front ORB Pitch")
    ax3.plot(orb_rear_f, orb_rear_pitch, label="Rear ORB Pitch")
    ax3.plot(gt_f, gt_yaw, label="GT Yaw")
    ax3.plot(orb_front_f, orb_front_yaw, label="Front ORB Yaw")
    ax3.plot(orb_rear_f, orb_rear_yaw, label="Rear ORB Yaw")
    ax3.legend()
    ax3.set_title("Orientation")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Angle (rad)")

    plt.tight_layout()
    plt.show()
