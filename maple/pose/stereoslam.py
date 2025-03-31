import cv2
import numpy as np
import orbslam3


class SimpleStereoSLAM:
    def __init__(self, vocab_path, settings_path):
        self.slam = orbslam3.system(vocab_path, settings_path, orbslam3.Sensor.STEREO)
        self.slam.initialize()
        self.frame_id = 0
        self.pose_dict = {}  # frame_id : 4x4 pose matrix

    def transform_trajectory(self, trajectory):
        # Rotation to convert Z-forward to Z-up
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        new_traj = []
        for pose in trajectory:
            R_wc = pose[:3, :3]
            t_wc = pose[:3, 3]

            # Apply rotation
            R_new = R @ R_wc
            t_new = R @ t_wc

            T_new = np.eye(4)
            T_new[:3, :3] = R_new
            T_new[:3, 3] = t_new
            new_traj.append(T_new)
        return new_traj

    def process_frame(self, left_image_path, right_image_path, timestamp):
        left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
        if left_img is None or right_img is None:
            print(f"Could not read images {left_image_path}, {right_image_path}")
            return False

        success = self.slam.process_image_stereo(left_img, right_img, timestamp)
        if success:
            trajectory = self.slam.get_trajectory()
            if len(trajectory) > 0:
                self.pose_dict[self.frame_id] = trajectory[-1]  # latest pose
        self.frame_id += 1
        return success

    def get_current_pose(self):
        trajectory = self.slam.get_trajectory()
        if len(trajectory) > 0:
            return trajectory[-1]
        else:
            return None

    def shutdown(self):
        self.slam.shutdown()

    def get_trajectory(self):
        traj = self.slam.get_trajectory()
        return self.transform_trajectory(traj)

    def get_pose_dict(self):
        return self.pose_dict
