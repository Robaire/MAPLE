import carla
import numpy as np
from enum import Enum
import pytransform3d.transformations as pyt_t
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from maple.utils import carla_to_pytransform
from maple.pose import OrbslamEstimator
from maple.boulder.detector import BoulderDetector
from maple.surface.map import SurfaceHeight
import maple.surface.map as surface

def get_entry_point():
    return "Agent"


class State(Enum):
    WARMUP = 0
    MAP = 1

# Hardcoded in orbslam config. Don't change.
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# DEBUG: print options
np.set_printoptions(precision=3, floatmode='fixed')


class Agent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        # Agent constants
        self.WARMUP_LENGTH = 50         # Initial waiting period to raise arms
        self.BOULDER_FREQ = 20          # How often to map boulders
        self.SURFACE_FREQ = 10          # How often to map the surface from wheel points
        self.LARGE_BOULDER_AREA = 100   # Boulder area cutoff to be determined large
        self.PITCH_ROLL_THRESHOLD = 60  # Pitch/Roll cutoff (degrees) to not sample surface points from a pose

        # State machine setup
        self.state = State.WARMUP
        self.frame = 1

        # SLAM setup
        self.orbslam_front = OrbslamEstimator(
            self,
            carla.SensorPosition.FrontLeft,
            carla.SensorPosition.FrontRight,
            mode="stereo"
        )
        self.orbslam_back = OrbslamEstimator(
            self,
            carla.SensorPosition.BackLeft,
            carla.SensorPosition.BackRight,
            mode="stereo"
        )

        # Boulder detector setup
        # TODO: Why does creating this slow us down significantly, even w/o calling it
        # self.detector_front = BoulderDetector(
        #     self, 
        #     carla.SensorPosition.FrontLeft, 
        #     carla.SensorPosition.FrontRight
        # )

        # Surface height sampling setup
        self.surface_samples = surface.sample_lander(self)

        # DEBUG: data saving
        self.gt_traj = []
        self.os_front_traj = []
        self.os_back_traj = []

        # Sanity checks
        assert self.BOULDER_FREQ % 2 == 0
        assert self.SURFACE_FREQ % 2 == 0

    def use_fiducials(self):
        return False

    def sensors(self):
        """
        Turn on all lights, activate stereo cameras
        """

        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": False,
                "light_intensity": 1.0,
                "width": f"{IMAGE_WIDTH}",
                "height": f"{IMAGE_HEIGHT}",
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{IMAGE_WIDTH}",
                "height": f"{IMAGE_HEIGHT}",
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{IMAGE_WIDTH}",
                "height": f"{IMAGE_HEIGHT}",
            },
            carla.SensorPosition.Left: {
                "camera_active": False,
                "light_intensity": 1.0,
                "width": f"{IMAGE_WIDTH}",
                "height": f"{IMAGE_HEIGHT}",
            },
            carla.SensorPosition.Right: {
                "camera_active": False,
                "light_intensity": 1.0,
                "width": f"{IMAGE_WIDTH}",
                "height": f"{IMAGE_HEIGHT}",
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{IMAGE_WIDTH}",
                "height": f"{IMAGE_HEIGHT}",
            },
            carla.SensorPosition.BackRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{IMAGE_WIDTH}",
                "height": f"{IMAGE_HEIGHT}",
            },
            carla.SensorPosition.Back: {
                "camera_active": False,
                "light_intensity": 1.0,
                "width": f"{IMAGE_WIDTH}",
                "height": f"{IMAGE_HEIGHT}",
            },
        }
        return sensors

    def run_step(self, input_data):
        """Execute one step of navigation"""

        match self.state:

            case State.WARMUP:

                # Lift the arms
                self.set_front_arm_angle(np.radians(90))
                self.set_back_arm_angle(np.radians(90))

                # Stay in place during warmup
                lin_vel = 0.0
                ang_vel = 0.0

                # Proceed to mapping after warmup has ended
                if self.frame >= self.WARMUP_LENGTH: 
                    self.state = State.MAP

            case State.MAP:

                # Get image to make sure we can do VIO
                img_front_left = input_data["Grayscale"][carla.SensorPosition.FrontLeft]

                # VIO with orbslam on frames with images
                if img_front_left is not None:
                    orbslam_front_pose = self.orbslam_front.estimate(input_data)
                    orbslam_back_pose = self.orbslam_back.estimate(input_data)

                    # Average between the two poses
                    pose_estimate = average_poses(orbslam_front_pose, orbslam_back_pose)

                    # DEBUG: log pose
                    formatter = {'float_kind': lambda x: f"{x: .3f}"}
                    print("Orbslam estimate:\n" + np.array2string(pose_estimate, formatter=formatter))
                    print("GT pose:\n" + np.array2string(carla_to_pytransform(self.get_transform()), formatter=formatter))

                    # DEBUG: store trajectory
                    self.os_front_traj.append(orbslam_front_pose)
                    self.os_back_traj.append(orbslam_back_pose)
                    self.gt_traj.append(carla_to_pytransform(self.get_transform()))


                # DEBUG: constant velocity forwards, change directions at frame 1000
                lin_vel = 0.1
                ang_vel = 0.0
                if self.frame >= 1000: ang_vel = -0.03

                # DEBUG: end mission and save some results
                if self.frame >= 3000:
                    self.mission_complete()

                # Map boulders at its own frequency
                if self.frame % self.BOULDER_FREQ == 0:
                    # detections, ground_points = self.detector_front.map(input_data)

                    # TODO: Include this when using navigator
                    # large_boulders_detections = self.detector_front.get_large_boulders(self.LARGE_BOULDER_AREA)
                    ...

                # Map the surface at its own frequency
                if self.frame % self.SURFACE_FREQ == 0:
                    self.surface_samples.extend(surface.sample_surface(pose_estimate, self.PITCH_ROLL_THRESHOLD))


        # DEBUG: logging
        print(f"Frame: {self.frame}")

        self.frame += 1
        return carla.VehicleVelocityControl(lin_vel, ang_vel)

    def finalize(self):
        """Finalize the surface and boulder maps"""
        # DEBUG: save results
        np.save("init", carla_to_pytransform(self.get_initial_position()))
        np.save("cam", carla_to_pytransform(self.get_camera_position(carla.SensorPosition.FrontLeft)))
        np.save("gt", np.array(self.gt_traj))
        np.save("os_front", np.array(self.os_front_traj))
        np.save("os_back", np.array(self.os_back_traj))
        np.save("surface", np.array(self.surface_samples))
        return


def average_poses(T1, T2):
    """
    Average between two 4x4 transformation matrices.
    """
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]

    # Interpolate halfway for translation (just average)
    t_mid = (t1 + t2) / 2

    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()

    # Interpolate halfway for rotation
    key_times = [0, 1]
    rotations = R.from_quat([q1, q2])
    slerp = Slerp(key_times, rotations)
    R_mid = slerp(0.5).as_matrix()

    T_mid = np.eye(4)
    T_mid[:3, :3] = R_mid
    T_mid[:3, 3] = t_mid

    return T_mid