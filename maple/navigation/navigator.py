from maple.utils import pytransform_to_tuple, carla_to_pytransform
from maple.navigation.drive_control import DriveController, angle_helper
from maple.navigation.constants import lander_size
from maple.navigation.static_path_planning import generate_lawnmower, generate_spiral
from maple.navigation.state.dynamic import DynamicPath
from maple.navigation.state.static import StaticPath
from pytransform3d.transformations import concat
import math

from enum import Enum


# This is a small class to keep track of the current state of the rover
class State(Enum):
    STATIC_PATH = 0
    DYNAMIC_PATH = 1
    # The below are unused, but will probaly be used
    PRE_PRE_CHARGE = 2
    PRE_CHARGE = 3
    POST_CHARGE = 4


class Navigator:
    """Provides the goal linear and angular velocity for the rover"""

    """
    This code will cycle through states to find the next location to travel to
    """

    def __init__(self, agent):
        """Create the navigator.

        Args:
            agent: The Agent instance
        """

        self.agent = agent
        # This is the start location for the rover
        self.rover_initial_position = carla_to_pytransform(agent.get_initial_position())

        # This is the start location for the lander
        lander_rover = carla_to_pytransform(agent.get_initial_lander_position())
        self.lander_initial_position = concat(lander_rover, self.rover_initial_position)

        # IMPORTANT NOTE:
        # These are the obstacles to avoid
        # It is a list of tuples with x, y, and radius
        # This will soon be a parameter
        self.lander_x, self.lander_y, _, _, _, _ = pytransform_to_tuple(
            self.lander_initial_position
        )
        self.lander_obstacle = (self.lander_x, self.lander_y, lander_size)
        self.obstacles = [self.lander_obstacle]

        # This is the state of the machine to be used to keep track of state we are in and where to switch
        self.state = State.STATIC_PATH

        # This is the goal location we are currently trying to get to, make sure to update it
        self.goal_loc = None
        self.distance_threshold = 1.0  # 1 meter threshold for updating goal

        # This is the drive controller for getting the linear and angular velocity
        self.drive_control = DriveController()

        ##### spiral path #####
        # Generate the static path points
        static_path_way_points = generate_spiral(self.lander_x, self.lander_y)

        # Initialize the static path as an object
        self.static_path = StaticPath(static_path_way_points)

        # Initialize the dynamic path as an object
        self.dynamic_path = None

        ##### spiral path #####

        # ##### lawnmower path #####
        # # This is the point we are trying to get to using the rrt along with a path to that point
        # self.dynamic_path = None
        # self.rrt_goal_loc = None # IMPORTANT NOTE: This is different than self.goal_loc because this is the goal location along the rrt path to get to self.goal_loc

        # # This is the global path, DO NOT CHANGE IT!!
        # # self.global_path = generate_spiral(self.lander_x, self.lander_y)
        # self.global_path = generate_lawnmower(self.lander_x, self.lander_y)

        # # print("global path: ", self.global_path)
        # self.global_path_index_tracker = 0
        # ##### lawnmower path #####

    def state_machine(self, rover_position):
        """
        This function acts as the state machine for the rover, while also setting the goal location
        """
        # Add this distance check at the beginning of the method
        if self.goal_loc is not None:
            rover_x, rover_y = rover_position
            goal_x, goal_y = self.goal_loc
            distance_to_goal = math.sqrt(
                (rover_x - goal_x) ** 2 + (rover_y - goal_y) ** 2
            )

            # If we're within threshold distance, force an update of the goal
            if distance_to_goal < self.distance_threshold:
                print(f"Within {self.distance_threshold}m of goal, updating...")
                if self.state == State.STATIC_PATH:
                    # Force getting a new point by setting goal_loc to None
                    self.goal_loc = None
                elif self.state == State.DYNAMIC_PATH:
                    # Force getting a new point by setting goal_loc to None
                    self.goal_loc = None

        if self.state == State.STATIC_PATH:
            # Find the next point in the static path
            possible_next_goal_location = self.static_path.traverse(
                rover_position, self.obstacles
            )

            # TODO: Later this will be used to change flower petals
            # For now if it is None regenerate the path
            if possible_next_goal_location is None:
                self.static_path = StaticPath(
                    generate_spiral(self.lander_x, self.lander_y)
                )
                possible_next_goal_location = self.static_path.traverse(
                    rover_position, self.obstacles
                )

            # NOTE: If possible next goal location is None then there are no possible locations to go to with that static plan
            if possible_next_goal_location is None:
                print("ERROR 0000: No possible next goal location")
                return

            # If there will be a collision switch to dynamic path
            if not self.static_path.is_path_collision_free(self.obstacles):
                self.state = State.DYNAMIC_PATH
                self.dynamic_path = DynamicPath(
                    [rover_position, possible_next_goal_location], self.obstacles
                )
                self.state_machine(rover_position)

        elif self.state == State.DYNAMIC_PATH:
            # Find the next point in the dynamic path
            self.goal_loc = self.dynamic_path.traverse(rover_position)

            # If there are no points switch to static path
            if self.goal_loc is None:
                self.state = State.STATIC_PATH
                self.state_machine(rover_position)

        elif self.state == State.PRE_PRE_CHARGE:
            pass

        elif self.state == State.PRE_CHARGE:
            pass

        elif self.state == State.POST_CHARGE:
            pass

    def get_goal_loc(self):
        # Returns the goal location which is either from the dynamic or static path
        return self.goal_loc

    def get_all_goal_locations(self):
        # Returns the static path that we want to follow
        return self.static_path

    def add_large_boulder_detection(self, detections):
        """
        Add a list of large boulder detections in the form of [(x1, y1, r1), (x2, y2, r2), (x3, y3, r3),...]

        Args:
            detections: The detections to add to the places to get around

        Returns:
            None
        """

        # Make sure we have the right data type!!
        assert isinstance(detections, list) or isinstance(detections, tuple)
        # assert isinstance(detections[0], list) or isinstance(detections[0], tuple)
        assert len(detections[0]) == 3

        # NOTE: We may have to add functionality to remove obstacles if this list gets too large
        self.obstacles.extend(detections)

    def get_obstacle_locations(self):
        return self.obstacles

    def __call__(self, pytransform_position):
        """Equivalent to calling `get_lin_vel_ang_vel`."""
        return self.get_lin_vel_ang_vel(pytransform_position)

    def get_lin_vel_ang_vel(self, pytransform_position, attempt=0):
        """
        Takes the position and returns the linear and angular goal velocity.
        Uses an iterative approach with fallback strategies to prevent recursion issues.

        Args:
            pytransform_position: Current position of the rover
            obstacles: List of obstacles to avoid
            attempt: Internal counter to prevent infinite loops

        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """

        # Calculate the next goal location
        rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(
            pytransform_position
        )
        self.state_machine((rover_x, rover_y))  # Change the state

        # NOTE: Static/Dynamic path planning will be done here, specialized battery is else where
        if self.state == State.STATIC_PATH or self.state == State.DYNAMIC_PATH:
            # Prevent infinite loops
            if attempt >= 5:
                print("WARNING: Maximum attempts reached, using emergency fallback")
                # Emergency fallback: turn in place then try to move slightly
                return (0.1, 0.5)

            try:
                # Extract the position information
                rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(
                    pytransform_position
                )

                # Get our current goal location
                goal_x, goal_y = self.goal_loc

                # Success!
                return self.drive_control.get_lin_vel_ang_vel_drive_control(
                    rover_x, rover_y, rover_yaw, goal_x, goal_y
                )

            except Exception as e:
                print(f"Navigation error: {e}")

                # Try one of several fallback strategies
                if attempt == 0:
                    # First fallback: try with just the lander as an obstacle
                    print("Trying fallback with just lander obstacle")
                    self.obstacles = [self.lander_obstacle]
                    return self.get_lin_vel_ang_vel(pytransform_position, attempt + 1)
                elif attempt == 1:
                    # Second fallback: try with a new goal location
                    print("Trying fallback with new goal location")
                    rover_x, rover_y, _, _, _, _ = pytransform_to_tuple(
                        pytransform_position
                    )
                    self.goal_loc = self.static_path.get_next_goal_location()
                    self.dynamic_path = None
                    return self.get_lin_vel_ang_vel(pytransform_position, attempt + 1)
                elif attempt == 2:
                    # Third fallback: try with a simpler path planning approach
                    print("Trying emergency direct path")
                    rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(
                        pytransform_position
                    )
                    # Simple emergency path - move away from current location
                    emergency_x = rover_x + 2.0
                    emergency_y = rover_y + 2.0
                    emergency_ang = angle_helper(
                        rover_x, rover_y, rover_yaw, emergency_x, emergency_y
                    )
                    return (0.2, emergency_ang)  # Slow speed for safety
                else:
                    # Last resort: rotate in place to find a clear path
                    print("CRITICAL: Using last resort movement")
                    return (0.0, 0.5)  # Rotate in place at a moderate speed

    def get_rrt_waypoints(self):
        """
        Return the full list of waypoints from the current RRT path, if it exists.
        This can be useful for visualization or debugging.

        Returns:
            A list of (x, y) tuples representing the current planned path,
            or an empty list if there is no valid path.
        """
        if self.dynamic_path is None:
            return []

        # Depending on your RRTPath implementation, you might do:
        #   return self.dynamic_path.get_full_path()
        # or
        #   return self.dynamic_path.nodes
        #
        # If `RRTPath` has no built-in method, you can store the path in a variable
        # in `RRTPath` after construction or after planning. Here, let's assume
        # there's a `get_full_path()` method for demonstration:

        return self.dynamic_path.get_full_path()
