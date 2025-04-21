import carla
from math import radians
from datetime import datetime

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from maple.data import Recorder


def get_entry_point():
    return "ExampleRecorderAgent"


class ExampleRecorderAgent(AutonomousAgent):
    """
    Example agent that records data from the simulation
    """

    _width = 1280
    _height = 720

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self._recorder = Recorder(
            self, f"/recorder/{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.tar.gz"
        )
        self.frame = 1

        self._width = 1280
        self._height = 720

    def use_fiducials(self):
        return False

    def sensors(self):
        """
        Define which sensors are going to be active at the start.
        The specifications and positions of the sensors are predefined and cannot be changed.
        """
        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": str(self._width),
                "height": str(self._height),
                "use_semantic": True,
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": True,
                "light_intensity": 0,
                "width": str(self._width),
                "height": str(self._height),
                "use_semantic": False,
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
                "light_intensity": 0,
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

        # Record data
        self._recorder(self.frame, input_data)

        # Run the agent
        if self.frame == 1:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        # Run for 10 seconds
        if self.get_mission_time() > 10:
            self.mission_complete()

        linear = 0.5
        angular = 0.0

        # Record custom data
        self._recorder.record_custom(
            "control", {"frame": self.frame, "linear": linear, "angular": angular}
        )

        self.frame += 1

        # Drive in a straight line
        return carla.VehicleVelocityControl(linear, angular)

    def finalize(self):
        # Stop recording
        self._recorder.stop()

        # Finalize the agent
