import carla
import numpy as np
from pytransform3d.transformations import concat

from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from maple.boulder import BoulderDetector
from maple.boulder.map import BoulderMap
from maple.navigation import Navigator
from maple.pose import InertialApriltagEstimator, ApriltagEstimator
from maple.surface.map import sample_surface, SurfaceHeight
from maple.utils import carla_to_pytransform


def get_entry_point():
    return "MITAgent"


class MITAgent(AutonomousAgent):
    """
    The MIT Agent
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        # Frame counter
        self.frame = 0

        # Data Collection Frequency
        self.sample_frequency = 1  # [Hz]
        self.last_sample_time = 0

        # Localization System
        # self.estimator = InertialApriltagEstimator(self)
        self.estimator = ApriltagEstimator(self)

        # Boulder Detectors
        self.front_detector = BoulderDetector(self, "FrontLeft", "FrontRight")
        self.rear_detector = BoulderDetector(self, "BackLeft", "BackRight")

        # Boulder Mapper
        self.boulder_mapper = BoulderMap(self.get_geometric_map())

        # Navigation
        self.navigator = Navigator(self)
        self.linear_velocity = 0
        self.angular_velocity = 0

        # Data Collection
        self.boulders_global = []
        self.boulders_global_large = []
        self.surface_global = []

    def use_fiducials(self):
        return True

    def sensors(self):
        """
        Enable the front and rear stereo cameras, side cameras, and associated lights.
        Do not enable front and rear stand alone cameras or lights
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
                "light_intensity": 0,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Left: {
                "camera_active": True,
                "light_intensity": 1,
                "width": "2448",
                "height": "2048",
            },
            carla.SensorPosition.Right: {
                "camera_active": True,
                "light_intensity": 1,
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

        # Pre-increment the frame
        self.frame += 1

        ## Mission Complete Check ##
        # Check if there are conditions under which we would stop the simulation
        if self.get_mission_time() > 60 * 5:  # Stop after 5 minutes
            self.mission_complete()

        ## Beginning Setup ##
        if self.frame == 1:
            # Move the arms out of the way
            self.set_front_arm_angle(np.deg2rad(60))
            self.set_back_arm_angle(np.deg2rad(60))

        # Wait a few seconds for the arms to move into position
        if self.get_mission_time() < 3:
            return carla.VehicleVelocityControl(0.0, 0.0)

        ## Localization ##
        rover_global = self.estimator(input_data)
        if rover_global is None:
            # TODO: Figure out what to do when we don't have a pose estimate
            # There is no reason to attempt data collection if we don't have a pose
            # Perhaps we should utilize the last control input?
            # Alternatively, Navigator may want to handle a None input and make a decision internally
            return carla.VehicleVelocityControl(
                self.linear_velocity, self.angular_velocity
            )

        ## Data Collection ##
        # Rather than use the frame number to determine the sample frequency we use the mission time
        # because we may not have a pose estimate on the exact frame we are trying to sample on
        if (
            self.last_sample_time + (1 / self.sample_frequency)
            >= self.get_mission_time() - 0.01
            and self.frame % 2 == 0
        ):
            self.last_sample_time = self.get_mission_time()

            # Get boulder detections
            boulders_rover = []
            boulders_rover.extend(self.front_detector(input_data))
            boulders_rover.extend(self.rear_detector(input_data))

            # Convert the boulders to the global frame
            self.boulders_global.extend(
                [concat(b_r, rover_global) for b_r in boulders_rover]
            )

            # TODO: If the navigation is using interim boulder map or surface mapping data it can be processes here
            # Although really this should be processed inside of the Navigator class
            # Maybe it should take a reference to the global boulder list and surface map lists?
            # self.navigator.update_boulders(self.boulders_global)

            # Theoretically, this code should identify large boulders via cluster mapping - Allison

            # TODO: ADJUST min_area TO MINIMUM SIZE OF PROBLEMATIC BOULDERS
            min_area = 30
            # Get boulder detections
            boulders_rover_large = []
            boulders_rover_large.extend(
                self.front_detector.get_large_boulders(min_area=min_area)
            )
            boulders_rover_large.extend(
                self.rear_detector.get_large_boulders(min_area=min_area)
            )
            # Convert the boulders to the global frame
            self.boulders_global_large.extend(
                [concat(b_r, rover_global) for b_r in boulders_rover_large]
            )
            # Transforms to all large boulder detections and all large boulders
            boulders_global_large_clustered = self.boulder_mapper.generate_clusters(
                self.boulders_global_large
            )

            # End large boulder detection and clustering

        # Get surface height samples
        self.surface_global.extend(sample_surface(rover_global))

        ## Navigation ##
        self.linear_velocity, self.angular_velocity = self.navigator(rover_global)

        if self.get_mission_time() > 10:
            self.mission_complete()

        return carla.VehicleVelocityControl(self.linear_velocity, self.angular_velocity)

    def finalize(self):
        geometric_map = self.get_geometric_map()

        # Calculate the final boulder map
        # TODO: Apply clustering methods
        for boulder in self.boulders_global:
            x, y = boulder[:2, 3]
            geometric_map.set_rock(x, y, True)

        # As a baseline assign the initial height to every cell
        #rover_global = carla_to_pytransform(self.get_initial_position())
        #z_height = np.mean([sample[2] for sample in sample_surface(rover_global)])
        surfaceHeight = SurfaceHeight(geometric_map)
        
        # Generate the actual map with the sample list
        surfaceHeight.set_map(self.surface_global)

        # # Update the geometric map
        # geometric_map = self.get_geometric_map()
        # for x, y in np.ndindex(geometric_map.get_map_array().shape[:2]):
        #     # Set the entire surface to the average
        #     geometric_map.set_cell_height(x, y, z_height)

        # Calculate the final surface height map (using boulders too!)
        # self.surface_global
