from lac_data import PlaybackAgent
from maple.pose import OrbslamEstimator
from maple.utils import pytransform_to_tuple, carla_to_pytransform


if __name__ == "__main__":
    agent = PlaybackAgent("test/test_pose/test_orbslam/straight-line-1min.lac")
    # agent = PlaybackAgent("test/test_pose/test_orbslam/example.lac")
    estimator = OrbslamEstimator(agent, "FrontLeft", "FrontRight", mode="stereo_imu")
    assert estimator is not None

    pose_gt = []
    pose_orb = []
    traj_rt = []
    lost = 0

    # Skip to 90
    frame = 90
    agent.set_frame(frame)

    # Set the orbslam global
    estimator.set_orbslam_global(carla_to_pytransform(agent.get_transform()))

    while not agent.at_end():
        input_data = agent.input_data()

        pose_gt.append(
            pytransform_to_tuple(carla_to_pytransform(agent.get_transform()))
        )

        estimate = estimator.estimate(input_data)
        if estimator.lost:  # TODO: This never returns true
            lost += 1

        if estimate is not None:
            # Check if the estimate is reset to origin

            if pytransform_to_tuple(estimate)[:3] == (0, 0, 0):
                print(f"Frame {frame}: Reset to origin")
                continue

        # assert estimate is not None
        if estimate is not None:
            pose_orb.append(pytransform_to_tuple(estimate))
            # traj_rt.append(pytransform_to_tuple(estimator._get_pose()))
            print(f"Frame {frame}: {pytransform_to_tuple(estimate)}")
        else:
            # print(f"Frame {frame}: None")
            pass

        frame = agent.step_frame()

    print(f"Lost: {lost}")

    traj = estimator._get_trajectory()
    traj = [pytransform_to_tuple(t) for t in traj]

    # Plot the results
    import matplotlib.pyplot as plt

    gt_x = [pose[0] for pose in pose_gt]
    gt_y = [pose[1] for pose in pose_gt]
    gt_z = [pose[2] for pose in pose_gt]

    orb_x = [pose[0] for pose in pose_orb]
    orb_y = [pose[1] for pose in pose_orb]
    orb_z = [pose[2] for pose in pose_orb]

    traj_x = [pose[0] for pose in traj]
    traj_y = [pose[1] for pose in traj]
    traj_z = [pose[2] for pose in traj]

    traj_rt_x = [pose[0] for pose in traj_rt]
    traj_rt_y = [pose[1] for pose in traj_rt]
    traj_rt_z = [pose[2] for pose in traj_rt]

    fig, ax = plt.subplots()
    ax.plot(gt_x, gt_y, label="Ground Truth")
    ax.plot(orb_x, orb_y, label="Stereo ORB-SLAM")
    ax.plot(traj_x, traj_y, label="Stereo ORB-SLAM Trajectory")
    # ax.plot(traj_rt_x, traj_rt_y, label="Stereo ORB-SLAM Trajectory (RT)")
    plt.legend()
    plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(gt_x, gt_z, label="Ground Truth")
    # ax.plot(orb_x, orb_z, label="Stereo ORB-SLAM")
    # ax.plot(traj_x, traj_z, label="Stereo ORB-SLAM Trajectory")
    # ax.plot(traj_rt_x, traj_rt_z, label="Stereo ORB-SLAM Trajectory (RT)")
    # plt.legend()
    # plt.show()
