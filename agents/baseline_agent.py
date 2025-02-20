import carla
import numpy as np
from pytransform3d.transformations import concat

from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from maple.boulder import BoulderDetector
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

        # Frame counter
        self.frame = 0

        # Boulder Detectors
        self.front_detector = BoulderDetector(self, "FrontLeft", "FrontRight")
        self.rear_detector = BoulderDetector(self, "BackLeft", "BackRight")

        # Data Collection
        self.boulders_global = []
        self.surface_global = []

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
                "camera_active": True,
                "light_intensity": 1,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
                "light_intensity": 1,
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
                "camera_active": True,
                "light_intensity": 1,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.BackRight: {
                "camera_active": True,
                "light_intensity": 1,
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

        # Pre-increment the frame
        self.frame += 1

        ## Beginning Setup ##
        if self.frame == 1:
            # Move the arms out of the way
            self.set_front_arm_angle(np.deg2rad(45))
            self.set_back_arm_angle(np.deg2rad(45))

        # Wait a few seconds for the arms to move into position
        if self.get_mission_time() < 5:
            return carla.VehicleVelocityControl(0.0, 0.0)

        ## Data Collection ##
        if self.frame % 10 == 0:
            # Use the initial position since we never move
            rover_global = carla_to_pytransform(self.get_initial_position())

            # Get boulder detections
            boulders_rover = []
            boulders_rover.extend(self.front_detector(input_data))
            boulders_rover.extend(self.rear_detector(input_data))

            # Convert the boulders to the global frame
            self.boulders_global.extend(
                [concat(b_r, rover_global) for b_r in boulders_rover]
            )

            # Get surface height samples
            self.surface_global.extend(sample_surface(rover_global))

            # We're done!
            self.mission_complete()

        return carla.VehicleVelocityControl(0.0, 0.0)

    def finalize(self):
        # Get the geometric map
        geometric_map = self.get_geometric_map()

        # TODO: Calculate the final boulder map

        # Average the Z-height of the surface samples
        z_height = np.mean([s_g[3, 2] for s_g in self.surface_global])

        # Update the geometric map
        for x, y in np.ndindex(geometric_map.shape[:2]):
            # Set the entire surface to the average
            geometric_map.set_cell_height(x, y, z_height)

            # Set every cell as having a boulder
            geometric_map.set_cell_rock(x, y, True)
