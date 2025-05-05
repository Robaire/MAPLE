from maple.utils import pytransform_to_tuple, carla_to_pytransform
from maple.navigation.drive_control import DriveController, angle_helper
from maple.navigation.constants import lander_size
from maple.navigation.static_path_planning import (
    generate_spiral,
)
from maple.navigation.state.dynamic import DynamicPath
from maple.navigation.state.static import StaticPath
from pytransform3d.transformations import concat
from maple.navigation.state.path import is_collision
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

        # Going to add in multiple osbtacles around the lander with less size so that if we get too close we can still run rrt while just ignoring the outer most obstacle
        for new_size in range(4*lander_size):
            self.obstacles.append((self.lander_x, self.lander_y, new_size / 4))

        # This is the state of the machine to be used to keep track of state we are in and where to switch
        self.state = State.STATIC_PATH

        # This is the goal location we are currently trying to get to, make sure to update it
        self.goal_loc = None
        self.distance_threshold = 2.0  # 1 meter threshold for updating goal

        # This is the drive controller for getting the linear and angular velocity
        self.drive_control = DriveController()

        ##### spiral path #####
        # Generate the static path points
        # static_path_way_points = [
        #     (-12, -12), (-12, -10), (-12, -8), (-12, -6), (-12, -4), (-12, -2), (-12, 0), (-12, 2), (-12, 4), (-12, 6), (-12, 8), (-12, 10), (-12, 12),
        #     (-10, -12), (-10, -10), (-10, -8), (-10, -6), (-10, -4), (-10, -2), (-10, 0), (-10, 2), (-10, 4), (-10, 6), (-10, 8), (-10, 10), (-10, 12),
        #     (-8, -12), (-8, -10), (-8, -8), (-8, -6), (-8, -4), (-8, -2), (-8, 0), (-8, 2), (-8, 4), (-8, 6), (-8, 8), (-8, 10), (-8, 12),
        #     (-6, -12), (-6, -10), (-6, -8), (-6, -6), (-6, -4), (-6, -2), (-6, 0), (-6, 2), (-6, 4), (-6, 6), (-6, 8), (-6, 10), (-6, 12),
        #     (-4, -12), (-4, -10), (-4, -8), (-4, -6), (-4, -4), (-4, -2), (-4, 0), (-4, 2), (-4, 4), (-4, 6), (-4, 8), (-4, 10), (-4, 12),
        #     (-2, -12), (-2, -10), (-2, -8), (-2, -6), (-2, -4), (-2, 4), (-2, 6), (-2, 8), (-2, 10), (-2, 12),
        #     (0, -12), (0, -10), (0, -8), (0, -6), (0, -4), (0, 4), (0, 6), (0, 8), (0, 10), (0, 12),
        #     (2, -12), (2, -10), (2, -8), (2, -6), (2, -4), (2, 4), (2, 6), (2, 8), (2, 10), (2, 12),
        #     (4, -12), (4, -10), (4, -8), (4, -6), (4, -4), (4, -2), (4, 0), (4, 2), (4, 4), (4, 6), (4, 8), (4, 10), (4, 12),
        #     (6, -12), (6, -10), (6, -8), (6, -6), (6, -4), (6, -2), (6, 0), (6, 2), (6, 4), (6, 6), (6, 8), (6, 10), (6, 12),
        #     (8, -12), (8, -10), (8, -8), (8, -6), (8, -4), (8, -2), (8, 0), (8, 2), (8, 4), (8, 6), (8, 8), (8, 10), (8, 12),
        #     (10, -12), (10, -10), (10, -8), (10, -6), (10, -4), (10, -2), (10, 0), (10, 2), (10, 4), (10, 6), (10, 8), (10, 10), (10, 12),
        #     (12, -12), (12, -10), (12, -8), (12, -6), (12, -4), (12, -2), (12, 0), (12, 2), (12, 4), (12, 6), (12, 8), (12, 10), (12, 12)
        # ]

        # static_path_way_points = [
        #     (-10, -10), (-10, -5), (-10, 0), (-10, 5), (-10, 10),
        #     (-5, -10), (-5, -5), (-5, 0), (-5, 5), (-5, 10),
        #     (0, -10), (0, -5), (0, 5), (0, 10),
        #     (5, -10), (5, -5), (5, 0), (5, 5), (5, 10),
        #     (10, -10), (10, -5), (10, 0), (10, 5), (10, 10)
        # ]

        # static_path_way_points = [
        #     (-12, -12), (-12, -9), (-12, -6), (-12, -3), (-12, 0), (-12, 3), (-12, 6), (-12, 9), (-12, 12),
        #     (-9, -12), (-9, -9), (-9, -6), (-9, -3), (-9, 0), (-9, 3), (-9, 6), (-9, 9), (-9, 12),
        #     (-6, -12), (-6, -9), (-6, -6), (-6, -3), (-6, 0), (-6, 3), (-6, 6), (-6, 9), (-6, 12),
        #     (-3, -12), (-3, -9), (-3, -6), (-3, -3), (-3, 3), (-3, 6), (-3, 9), (-3, 12),
        #     (0, -12), (0, -9), (0, -6), (0, 6), (0, 9), (0, 12),
        #     (3, -12), (3, -9), (3, -6), (3, -3), (3, 3), (3, 6), (3, 9), (3, 12),
        #     (6, -12), (6, -9), (6, -6), (6, -3), (6, 0), (6, 3), (6, 6), (6, 9), (6, 12),
        #     (9, -12), (9, -9), (9, -6), (9, -3), (9, 0), (9, 3), (9, 6), (9, 9), (9, 12),
        #     (12, -12), (12, -9), (12, -6), (12, -3), (12, 0), (12, 3), (12, 6), (12, 9), (12, 12)
        # ]

        # static_path_way_points = [
        #     (-12, -12, 0.5),
        #     (-12, -9, 0.5),
        #     (-12, -6, 0.5),
        #     (-12, -3, 0.5),
        #     (-12, 0, 0.5),
        #     (-12, 3, 0.5),
        #     (-12, 6, 0.5),
        #     (-12, 9, 0.5),
        #     (-12, 12, 0.5),
        #     (-9, -12, 0.5),
        #     (-9, -9, 0.75),
        #     (-9, -6, 0.75),
        #     (-9, -3, 0.75),
        #     (-9, 0, 0.75),
        #     (-9, 3, 0.75),
        #     (-9, 6, 0.75),
        #     (-9, 9, 0.75),
        #     (-9, 12, 0.5),
        #     (-6, -12, 0.5),
        #     (-6, -9, 0.75),
        #     (-6, -6, 1),
        #     (-6, -3, 1),
        #     (-6, 0, 1),
        #     (-6, 3, 1),
        #     (-6, 6, 1),
        #     (-6, 9, 0.75),
        #     (-6, 12, 0.5),
        #     (-3, -12, 0.5),
        #     (-3, -9, 0.75),
        #     (-3, -6, 1),
        #     (-3, -3, 0.75),
        #     (-3, 3, 0.75),
        #     (-3, 6, 1),
        #     (-3, 9, 0.75),
        #     (-3, 12, 0.5),
        #     (0, -12, 0.5),
        #     (0, -9, 0.75),
        #     (0, -6, 1),
        #     (0, 6, 1),
        #     (0, 9, 0.75),
        #     (0, 12, 0.5),
        #     (3, -12, 0.5),
        #     (3, -9, 0.75),
        #     (3, -6, 1),
        #     (3, -3, 0.75),
        #     (3, 3, 0.75),
        #     (3, 6, 1),
        #     (3, 9, 0.75),
        #     (3, 12, 0.5),
        #     (6, -12, 0.5),
        #     (6, -9, 0.75),
        #     (6, -6, 1),
        #     (6, -3, 1),
        #     (6, 0, 1),
        #     (6, 3, 1),
        #     (6, 6, 1),
        #     (6, 9, 0.75),
        #     (6, 12, 0.5),
        #     (9, -12, 0.5),
        #     (9, -9, 0.75),
        #     (9, -6, 0.75),
        #     (9, -3, 0.75),
        #     (9, 0, 0.75),
        #     (9, 3, 0.75),
        #     (9, 6, 0.75),
        #     (9, 9, 0.75),
        #     (9, 12, 0.5),
        #     (12, -12, 0.5),
        #     (12, -9, 0.5),
        #     (12, -6, 0.5),
        #     (12, -3, 0.5),
        #     (12, 0, 0.5),
        #     (12, 3, 0.5),
        #     (12, 6, 0.5),
        #     (12, 9, 0.5),
        #     (12, 12, 0.5),
        # ]


        static_path_way_points = [
            (-12, -12, 0.5),
            (-12, -9, 0.5),
            (-12, -6, 0.5),
            (-12, -3, 0.5),
            (-12, 0, 0.5),
            (-12, 3, 0.5),
            (-12, 6, 0.5),
            (-12, 9, 0.5),
            (-12, 12, 0.5),
            (-9, -12, 0.5),
            (-9, -9, 0.75),
            (-9, -6, 0.75),
            (-9, -3, 0.75),
            (-9, 0, 0.75),
            (-9, 3, 0.75),
            (-9, 6, 0.75),
            (-9, 9, 0.75),
            (-9, 12, 0.5),
            (-6, -12, 0.5),
            (-6, -9, 0.75),
            (-6, -6, 1),
            (-6, -3, 1),
            (-6, 0, 1),
            (-6, 3, 1),
            (-6, 6, 1),
            (-6, 9, 0.75),
            (-6, 12, 0.5),
            (-3, -12, 0.5),
            (-3, -9, 0.75),
            (-3, -6, 1),
            (-3, 6, 1),
            (-3, 9, 0.75),
            (-3, 12, 0.5),
            (0, -12, 0.5),
            (0, -9, 0.75),
            (0, -6, 1),
            (0, 6, 1),
            (0, 9, 0.75),
            (0, 12, 0.5),
            (3, -12, 0.5),
            (3, -9, 0.75),
            (3, -6, 1),
            (3, 6, 1),
            (3, 9, 0.75),
            (3, 12, 0.5),
            (6, -12, 0.5),
            (6, -9, 0.75),
            (6, -6, 1),
            (6, -3, 1),
            (6, 0, 1),
            (6, 3, 1),
            (6, 6, 1),
            (6, 9, 0.75),
            (6, 12, 0.5),
            (9, -12, 0.5),
            (9, -9, 0.75),
            (9, -6, 0.75),
            (9, -3, 0.75),
            (9, 0, 0.75),
            (9, 3, 0.75),
            (9, 6, 0.75),
            (9, 9, 0.75),
            (9, 12, 0.5),
            (12, -12, 0.5),
            (12, -9, 0.5),
            (12, -6, 0.5),
            (12, -3, 0.5),
            (12, 0, 0.5),
            (12, 3, 0.5),
            (12, 6, 0.5),
            (12, 9, 0.5),
            (12, 12, 0.5),
        ]

        # static_path_way_points = generate_flower_rays(self.lander_x, self.lander_y)
        # static_path_way_points = [(-10, -5), (0, -2)]

        # Initialize the static path as an object
        self.static_path = StaticPath(static_path_way_points)

        # Initialize the dynamic path as an object
        self.dynamic_path = None

    def state_machine(self, rover_position, estimate, input_data):
        rover_x, rover_y = rover_position

        if self.goal_loc is None:
            # Pick initial closest goal
            if self.state == State.STATIC_PATH:
                all_goals = self.static_path.get_full_path()
                if not all_goals:
                    # print("ERROR: No goals available")
                    return
                self.goal_loc = min(
                    all_goals,
                    key=lambda goal: (rover_x - goal[0]) ** 2
                    + (rover_y - goal[1]) ** 2,
                )
                # print(f"Picked initial goal: {self.goal_loc}")

        if self.goal_loc is not None:
            goal_x, goal_y = self.goal_loc[0], self.goal_loc[1]
            distance_to_goal = math.sqrt(
                (rover_x - goal_x) ** 2 + (rover_y - goal_y) ** 2
            )

            if distance_to_goal < self.distance_threshold:
                # We've reached the current goal, update to a new one
                # print(f"[Navigator] Reached goal {self.goal_loc}, picking next closest goal...")
                self.goal_loc = self.update_goal_location(
                    rover_position, estimate, input_data
                )
                # If no more goals, it will remain None and we'll handle that case next tick

        # Check if collision with static path
        if self.state == State.STATIC_PATH:
            # possible_next_goal_location = self.static_path.traverse(
            #     rover_position, self.obstacles
            # )
            # print
            possible_next_goal_location = self.goal_loc

            if possible_next_goal_location is None:
                # print("No possible next static path goal")
                return

            # TODO: Turning of collision checking for now since it's not working...
            if is_collision(rover_position, self.goal_loc, self.obstacles):
                # print("[Navigator] Collision detected! We'd switch to Dynamic Path here...")
                self.state = State.DYNAMIC_PATH

                self.dynamic_path = DynamicPath(
                    [rover_position, self.goal_loc],
                    self.obstacles,
                )

                self.goal_loc = self.update_goal_location(
                    rover_position, estimate, input_data
                )

                # self.goal_loc = self.dynamic_path.path.pop(0)
                # print(f'setting the new goal loc for the start {self.goal_loc} with the total path as {self.dynamic_path.path}')

        elif self.state == State.DYNAMIC_PATH:
            # print("using dynamic state!")
            # print(self.dynamic_path.path)
            # print("goal loc: ", self.goal_loc)
            # print("last of dynamic path: ", self.dynamic_path.path[-1])
            # Since we are popping from the list we can check once it is empty to transition back to the static path

            dynamic_goal_x, dynamic_goal_y = self.dynamic_path.dynamic_goal

            # print(dynamic_goal_x, dynamic_goal_y)

            distance_to_dynamic_goal = math.sqrt(
                (rover_x - dynamic_goal_x) ** 2 + (rover_y - dynamic_goal_y) ** 2
            )

            if distance_to_dynamic_goal < self.distance_threshold:
                # We've reached the current goal, update to a new one
                # print(f"[Navigator] Reached goal {self.goal_loc}, picking next closest goal...")
                self.goal_loc = self.update_goal_location(
                    rover_position, estimate, input_data
                )

            # TODO: This isn't actually right, need to point to whatever dynamic path is on rn
            # if self.dynamic_path.path[-1] == self.goal_loc:
            # print("the last one is it so switching")
            # self.state = State.STATIC_PATH
            # if self.goal_loc is None:
            #     # # If no goal, pick the closest static goal again
            #     # all_goals = self.static_path.get_full_path()
            #     # if all_goals:
            #     #     self.goal_loc = min(
            #     #         all_goals,
            #     #         key=lambda goal: (rover_x - goal[0])**2 + (rover_y - goal[1])**2
            #     #     )
            #     #     # print(f"[Navigator] Re-picked goal {self.goal_loc} for dynamic path.")
            #     # else:
            #     #     # print("[Navigator] No goals left to pick for dynamic path.")
            #     #     return
            #     self.goal_loc = self.update_goal_location(rover_position, estimate, input_data)

            # self.state = State.DYNAMIC_PATH
            # self.dynamic_path = DynamicPath(
            #     [rover_position, self.goal_loc],
            #     self.static_path,
            #     self.obstacles,
            # )

            # # We should be checking and removing points in the state machine
            # self.goal_loc = None
            # return

        # elif self.state == State.DYNAMIC_PATH:
        #     self.goal_loc = self.dynamic_path.traverse(rover_position)

        #     if self.goal_loc is None:
        #         # Dynamic path finished
        #         # print("[Navigator] Dynamic path finished. Switching back to Static Path.")
        #         self.state = State.STATIC_PATH
        #         self.goal_loc = None  # Will pick new closest static goal next tick

    def update_goal_location(self, rover_position, estimate, input_data):
        """
        Updates the goal location after reaching the current goal.
        Uses the Path.find_closest_goal method to select the next goal.

        Args:
            rover_position: (x, y) tuple of the current rover position

        Returns:
            The new goal location or None if no more goals available
        """
        # First make sure the reached goal is removed from paths
        if self.goal_loc is not None:
            if (
                self.state == State.STATIC_PATH
                and self.goal_loc in self.static_path.path
            ):
                # print(f"[Navigator] Removing reached goal {self.goal_loc} from static path")
                self.static_path.path.remove(self.goal_loc)
            elif (
                self.state == State.DYNAMIC_PATH
                and self.dynamic_path
                and self.goal_loc in self.dynamic_path.path
            ):
                # # print(f"[Navigator] Removing reached goal {self.goal_loc} from dynamic path")
                self.dynamic_path.path.remove(self.goal_loc)

        # Now find the next closest goal using the existing method
        if self.state == State.STATIC_PATH:
            # # print("running static??")
            new_goal = self.static_path.find_closest_goal(
                rover_position, estimate, input_data, self.agent, pop_if_found=False
            )
            # if new_goal:
            #     # print(f"[Navigator] Selected new static path goal: {new_goal}")
            # else:
            # print("[Navigator] No more static goals available")
            return new_goal

        elif self.state == State.DYNAMIC_PATH:
            # Assuming dynamic_path also has find_closest_goal method
            new_goal = self.dynamic_path.traverse(rover_position, self.obstacles)
            # print("the current goal is: ", new_goal)
            if new_goal is not None:
                print(f"[Navigator] Selected new dynamic path goal: {new_goal}")
            else:
                # print("[Navigator] No more dynamic goals available")
                self.state = State.STATIC_PATH
            return new_goal

        return None

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

    def __call__(self, pytransform_position, input_data):
        """Equivalent to calling `get_lin_vel_ang_vel`."""
        return self.get_lin_vel_ang_vel(pytransform_position, input_data)

    def get_lin_vel_ang_vel(self, pytransform_position, input_data, attempt=0):
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

        # print("Calling state machine")

        self.state_machine(
            (rover_x, rover_y), pytransform_position, input_data
        )  # Change the state

        # NOTE: Static/Dynamic path planning will be done here, specialized battery is else where
        if self.state == State.STATIC_PATH or self.state == State.DYNAMIC_PATH:
            # Prevent infinite loops
            if attempt >= 5:
                # print("WARNING: Maximum attempts reached, using emergency fallback")
                # Emergency fallback: turn in place then try to move slightly
                return (0.1, 0.5)

            try:
                # Extract the position information
                rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(
                    pytransform_position
                )

                # Add this check to handle None goal_loc
                if self.goal_loc is None:
                    # If no goal is set, pick a fallback goal or return a default motion
                    # print("WARNING: No goal location set, selecting fallback goal")
                    # self.goal_loc = self.static_path.find_closest_goal((rover_x, rover_y), pop_if_found=False)
                    self.goal_loc = self.static_path.find_closest_goal(
                        (rover_x, rover_y),
                        pytransform_position,
                        input_data,
                        self.agent,
                        pop_if_found=False,
                    )
                    # print(f'updating with the point {self.goal_loc}')

                    # If still no goal, use an emergency motion
                    if self.goal_loc is None:
                        # print("CRITICAL: No goals available, using emergency motion")
                        return (0.0, 0.5)  # Rotate in place to find a path

                # Get our current goal location
                goal_x, goal_y = self.goal_loc[0], self.goal_loc[1]

                # Success!
                return self.drive_control.get_lin_vel_ang_vel_drive_control(
                    rover_x, rover_y, rover_yaw, goal_x, goal_y
                )

            except Exception as e:
                # print(f"Navigation error: {e}")

                # Try one of several fallback strategies
                if attempt == 0:
                    # First fallback: try with just the lander as an obstacle
                    # print("Trying fallback with just lander obstacle")
                    self.obstacles = [self.lander_obstacle]
                    return self.get_lin_vel_ang_vel(pytransform_position, attempt + 1)
                elif attempt == 1:
                    # Second fallback: try with a new goal location
                    # print("Trying fallback with new goal location")
                    rover_x, rover_y, _, _, _, _ = pytransform_to_tuple(
                        pytransform_position
                    )
                    self.goal_loc = self.static_path.get_next_goal_location()
                    # print(f'here 1010 {self.goal_loc}')
                    self.dynamic_path = None
                    return self.get_lin_vel_ang_vel(pytransform_position, attempt + 1)
                elif attempt == 2:
                    # Third fallback: try with a simpler path planning approach
                    # print("Trying emergency direct path")
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
                    # print("CRITICAL: Using last resort movement")
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
