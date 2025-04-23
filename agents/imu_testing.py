import carla
from math import radians
from datetime import datetime

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from maple.data import Recorder
from maple.pose import InertialApriltagEstimator
from maple.utils import pytransform_to_tuple


def get_entry_point():
    return "ExampleRecorderAgent"


class ExampleRecorderAgent(AutonomousAgent):
    """
    Example agent that records data from the simulation
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self._width = 1280
        self._height = 720

        self.recorder = Recorder(
            self, f"/recorder/{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.tar.gz", 1
        )
        self.recorder.description(
            "Straight line, 0.3 m/s, 1 minute, images every second"
        )
        self.frame = 1
        self.estimator = InertialApriltagEstimator(self)

    def use_fiducials(self):
        return False

    def sensors(self):
        """
        Define which sensors are going to be active at the start.
        The specifications and positions of the sensors are predefined and cannot be changed.
        """
        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": False,
                "light_intensity": 0.0,
                "width": str(self._width),
                "height": str(self._height),
                "use_semantic": False,
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": False,
                "light_intensity": 1.0,
                "width": str(self._width),
                "height": str(self._height),
                "use_semantic": False,
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": False,
                "light_intensity": 1.0,
                "width": str(self._width),
                "height": str(self._height),
                "use_semantic": False,
            },
            carla.SensorPosition.Left: {
                "camera_active": False,
                "light_intensity": 0,
                "width": str(self._width),
                "height": str(self._height),
                "use_semantic": False,
            },
            carla.SensorPosition.Right: {
                "camera_active": False,
                "light_intensity": 0,
                "width": str(self._width),
                "height": str(self._height),
                "use_semantic": False,
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": False,
                "light_intensity": 0,
                "width": str(self._width),
                "height": str(self._height),
                "use_semantic": False,
            },
            carla.SensorPosition.BackRight: {
                "camera_active": False,
                "light_intensity": 0,
                "width": str(self._width),
                "height": str(self._height),
                "use_semantic": False,
            },
            carla.SensorPosition.Back: {
                "camera_active": False,
                "light_intensity": 0,
                "width": str(self._width),
                "height": str(self._height),
                "use_semantic": False,
            },
        }
        return sensors

    def run_step(self, input_data):
        """
        Run the agent
        """
        control = [0,0]
        current_time = self.get_mission_time()
        if current_time > 0 and current_time < 10:
            #stage = 1
            control = [0.3, 0]
        elif current_time >= 10 and current_time < 20:
            #stage = 2
            control = [-0.3, 0.4]
        elif current_time >= 20 and current_time < 30:
            #stage = 3
            control = [0, -1.]
        elif current_time >= 30 and current_time < 40:
            #stage = 4
            control = [-1,0]
        elif current_time >= 40 and current_time < 50:
            #stage = 5
            control = [1,0]
        elif current_time >= 50 and current_time < 60:
            #stage = 6
            control = [0.5, 0.5]
        # Record data
        self.recorder(self.frame, input_data)

        # Run the agent
        if self.frame == 1:
            self.estimator.prev_state = self.get_initial_position()
            estimate = self.get_inital_position()
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))
        else:
            estimate, is_april_tag_estimate = self.estimator(input_data)
        estimate_tuple = pytransform_to_tuple(estimate)
        self._recorder.record_custom("imu_estimate", {"x": estimate_tuple[0], "y": estimate_tuple[1], "z": estimate_tuple[2], "roll": estimate_tuple[3], "pitch": estimate_tuple[4], "yaw": estimate_tuple[5]})
        if self.get_mission_time() > 10. and self.get_mission_time()

        # Run for 1 minute
        if self.get_mission_time() > 60 * 1:
            self.mission_complete()

        # # Record custom data
        # self._recorder.record_custom(
        #     "control", {"frame": self.frame, "linear": linear, "angular": angular}
        # )

        self.frame += 1

        # Drive in a straight line
        return carla.VehicleVelocityControl(control[0], control[1])

    def finalize(self):
        # Stop recording
        self.recorder.stop()

        # Finalize the agent
