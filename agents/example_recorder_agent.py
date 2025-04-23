import carla
from math import radians

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac_data import Recorder


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

        self.recorder = Recorder(self)
        self.recorder.description(
            "Straight line, 0.3 m/s, 1 minute, images every frame"
        )
        self.frame = 1

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
                "camera_active": True,
                "light_intensity": 1.0,
                "width": str(self._width),
                "height": str(self._height),
                "use_semantic": False,
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
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

        # Record data
        self.recorder(self.frame, input_data)

        # Run the agent
        if self.frame == 1:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        # Run for 1 minute
        if self.get_mission_time() > 60 * 1:
            self.mission_complete()

        linear = 0.3
        angular = 0.0

        self.frame += 1

        # Drive in a straight line
        return carla.VehicleVelocityControl(linear, angular)

    def finalize(self):
        # Stop recording
        self.recorder.stop()

        # Finalize the agent
