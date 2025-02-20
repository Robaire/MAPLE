import carla
import numpy as np

from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from maple.surface.map import sample_surface
from maple.utils import carla_to_pytransform


def get_entry_point():
    return "BaselineAgent"


class BaselineAgent(AutonomousAgent):
    """
    The MIT Agent
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self.mission_complete()

    def use_fiducials(self):
        return False

    def sensors(self):
        """
        Enable the front and rear stereo cameras and associated lights.
        """

        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Left: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Right: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.BackRight: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Back: {
                "camera_active": False,
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
        }
        return sensors

    def run_step(self, input_data):
        """Execute one step of navigation"""

        return carla.VehicleVelocityControl(0.0, 0.0)

    def finalize(self):
        # Use the initial position since we never move
        rover_global = carla_to_pytransform(self.get_initial_position())

        # Average the Z-height of the surface samples
        z_height = np.mean([sample[2] for sample in sample_surface(rover_global)])

        # Update the geometric map
        geometric_map = self.get_geometric_map()
        for x, y in np.ndindex(geometric_map.get_map_array().shape[:2]):
            # Set the entire surface to the average
            geometric_map.set_cell_height(x, y, z_height)

            # Set every cell as having a boulder
            geometric_map.set_cell_rock(x, y, True)
