from maple.utils import pytransform_to_tuple, carla_to_pytransform
from maple.navigation.drive_control import DriveController, angle_helper
from maple.navigation.constants import lander_size
from maple.navigation.static_path_planning import (
    generate_spiral,
)
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
        self.distance_threshold = 2.0  # 1 meter threshold for updating goal

        # This is the drive controller for getting the linear and angular velocity
        self.drive_control = DriveController()

        ##### spiral path #####
        # Generate the static path points
        static_path_way_points = [
            (-12, -12), (-12, -10), (-12, -8), (-12, -6), (-12, -4), (-12, -2), (-12, 0), (-12, 2), (-12, 4), (-12, 6), (-12, 8), (-12, 10), (-12, 12),
            (-10, -12), (-10, -10), (-10, -8), (-10, -6), (-10, -4), (-10, -2), (-10, 0), (-10, 2), (-10, 4), (-10, 6), (-10, 8), (-10, 10), (-10, 12),
            (-8, -12), (-8, -10), (-8, -8), (-8, -6), (-8, -4), (-8, -2), (-8, 0), (-8, 2), (-8, 4), (-8, 6), (-8, 8), (-8, 10), (-8, 12),
            (-6, -12), (-6, -10), (-6, -8), (-6, -6), (-6, -4), (-6, -2), (-6, 0), (-6, 2), (-6, 4), (-6, 6), (-6, 8), (-6, 10), (-6, 12),
            (-4, -12), (-4, -10), (-4, -8), (-4, -6), (-4, -4), (-4, -2), (-4, 0), (-4, 2), (-4, 4), (-4, 6), (-4, 8), (-4, 10), (-4, 12),
            (-2, -12), (-2, -10), (-2, -8), (-2, -6), (-2, -4), (-2, 4), (-2, 6), (-2, 8), (-2, 10), (-2, 12),
            (0, -12), (0, -10), (0, -8), (0, -6), (0, -4), (0, 4), (0, 6), (0, 8), (0, 10), (0, 12),
            (2, -12), (2, -10), (2, -8), (2, -6), (2, -4), (2, 4), (2, 6), (2, 8), (2, 10), (2, 12),
            (4, -12), (4, -10), (4, -8), (4, -6), (4, -4), (4, -2), (4, 0), (4, 2), (4, 4), (4, 6), (4, 8), (4, 10), (4, 12),
            (6, -12), (6, -10), (6, -8), (6, -6), (6, -4), (6, -2), (6, 0), (6, 2), (6, 4), (6, 6), (6, 8), (6, 10), (6, 12),
            (8, -12), (8, -10), (8, -8), (8, -6), (8, -4), (8, -2), (8, 0), (8, 2), (8, 4), (8, 6), (8, 8), (8, 10), (8, 12),
            (10, -12), (10, -10), (10, -8), (10, -6), (10, -4), (10, -2), (10, 0), (10, 2), (10, 4), (10, 6), (10, 8), (10, 10), (10, 12),
            (12, -12), (12, -10), (12, -8), (12, -6), (12, -4), (12, -2), (12, 0), (12, 2), (12, 4), (12, 6), (12, 8), (12, 10), (12, 12)
        ]

        # static_path_way_points = [
        #     (-10, -10), (-10, -5), (-10, 0), (-10, 5), (-10, 10),
        #     (-5, -10), (-5, -5), (-5, 0), (-5, 5), (-5, 10),
        #     (0, -10), (0, -5), (0, 5), (0, 10),
        #     (5, -10), (5, -5), (5, 0), (5, 5), (5, 10),
        #     (10, -10), (10, -5), (10, 0), (10, 5), (10, 10)
        # ]


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
                    print("ERROR: No goals available")
                    return
                self.goal_loc = min(
                    all_goals,
                    key=lambda goal: (rover_x - goal[0])**2 + (rover_y - goal[1])**2
                )
                print(f"Picked initial goal: {self.goal_loc}")

        if self.goal_loc is not None:
            goal_x, goal_y = self.goal_loc
            distance_to_goal = math.sqrt(
                (rover_x - goal_x) ** 2 + (rover_y - goal_y) ** 2
            )

            if distance_to_goal < self.distance_threshold:
                # ✅ Only now remove the goal!
                print(f"[Navigator] Reached goal {self.goal_loc}, picking next closest goal...")
                if self.state == State.STATIC_PATH:
                    if self.goal_loc in self.static_path.path:
                        self.static_path.path.remove(self.goal_loc)
                elif self.state == State.DYNAMIC_PATH:
                    if self.goal_loc in self.dynamic_path.path:
                        self.dynamic_path.path.remove(self.goal_loc)

                self.goal_loc = None  # Will trigger picking a new goal next tick

        # Check if collision with static path
        if self.state == State.STATIC_PATH:
            possible_next_goal_location = self.static_path.traverse(
                rover_position, self.obstacles
            )

            if possible_next_goal_location is None:
                print("No possible next static path goal")
                return

            if not self.static_path.is_path_collision_free(self.obstacles):
                print("[Navigator] Collision detected! Switching to Dynamic Path.")

                if self.goal_loc is None:
                    # If no goal, pick the closest static goal again
                    all_goals = self.static_path.get_full_path()
                    if all_goals:
                        self.goal_loc = min(
                            all_goals,
                            key=lambda goal: (rover_x - goal[0])**2 + (rover_y - goal[1])**2
                        )
                        print(f"[Navigator] Re-picked goal {self.goal_loc} for dynamic path.")
                    else:
                        print("[Navigator] No goals left to pick for dynamic path.")
                        return

                self.state = State.DYNAMIC_PATH
                self.dynamic_path = DynamicPath(
                    [rover_position, self.goal_loc],
                    self.static_path,
                    self.obstacles,
                )
                return


        elif self.state == State.DYNAMIC_PATH:
            self.goal_loc = self.dynamic_path.traverse(rover_position)

            if self.goal_loc is None:
                # Dynamic path finished
                print("[Navigator] Dynamic path finished. Switching back to Static Path.")
                self.state = State.STATIC_PATH
                self.goal_loc = None  # Will pick new closest static goal next tick

    # TODO: This is supposd to pick waypoints based on orb features in that camera region but it's not working rn
    # def state_machine(self, rover_position, estimate, input_data):
    #     """
    #     Updated state machine that uses ORB feature quality to select goals.
    #     """
    #     rover_x, rover_y = rover_position

    #     # 1. Check if we've reached the current goal
    #     if self.goal_loc is not None:
    #         goal_x, goal_y = self.goal_loc
    #         distance_to_goal = math.sqrt((rover_x - goal_x)**2 + (rover_y - goal_y)**2)

    #         if distance_to_goal < self.distance_threshold:
    #             print(f"[Navigator] Reached goal {self.goal_loc}")
                
    #             # Remove the reached goal from the appropriate path
    #             if self.state == State.STATIC_PATH:
    #                 if self.goal_loc in self.static_path.path:
    #                     self.static_path.path.remove(self.goal_loc)
    #             elif self.state == State.DYNAMIC_PATH:
    #                 if self.dynamic_path and self.goal_loc in self.dynamic_path.path:
    #                     self.dynamic_path.path.remove(self.goal_loc)
                        
    #             # Clear current goal to trigger selection of a new one
    #             self.goal_loc = None

    #     # 2. Select a new goal if needed
    #     if self.goal_loc is None:
    #         if self.state == State.STATIC_PATH:
    #             # Find nearby goals to consider
    #             nearby_goals = self.static_path.find_nearby_goals(rover_position)
                
    #             if nearby_goals:
    #                 # Use ORB feature quality to select the best goal
    #                 selected_goal = self.static_path.pick_goal(
    #                     estimate, nearby_goals, input_data, self.agent.orb
    #                 )

    #                 self.goal_loc = selected_goal
                    
    #                 if selected_goal is not None:
    #                     print(f"[Navigator] Selected new goal {selected_goal} based on ORB features")
    #                     self.goal_loc = selected_goal
    #                 else:
    #                     # Fallback: pick closest goal
    #                     all_goals = self.static_path.get_full_path()
    #                     if all_goals:
    #                         self.goal_loc = min(
    #                             all_goals, 
    #                             key=lambda goal: (rover_x - goal[0])**2 + (rover_y - goal[1])**2
    #                         )
    #                         print(f"[Navigator] Fallback to closest goal: {self.goal_loc}")
    #             else:
    #                 # No nearby goals, pick the closest from all goals
    #                 all_goals = self.static_path.get_full_path()
    #                 if all_goals:
    #                     self.goal_loc = min(
    #                         all_goals, 
    #                         key=lambda goal: (rover_x - goal[0])**2 + (rover_y - goal[1])**2
    #                     )
    #                     print(f"[Navigator] No nearby goals, using closest: {self.goal_loc}")
    #                 else:
    #                     print("[Navigator] ERROR: No goals left!")
            
    #         elif self.state == State.DYNAMIC_PATH:
    #             # For dynamic path, we follow the path planner's suggestion
    #             self.goal_loc = self.dynamic_path.traverse(rover_position)
                
    #             if self.goal_loc is None:
    #                 # Dynamic path is complete, switch back to static
    #                 print("[Navigator] Dynamic path complete. Switching to Static Path.")
    #                 self.state = State.STATIC_PATH
    #                 # Will select a static goal on next iteration

    #     # 3. Check for collisions and switch to dynamic path if needed
    #     if self.state == State.STATIC_PATH:
    #         if not self.static_path.is_path_collision_free(self.obstacles):
    #             print("[Navigator] Collision detected! Switching to Dynamic Path.")
                
    #             # Create dynamic path from current position to current goal
    #             start_point = rover_position
    #             end_point = self.goal_loc if self.goal_loc else self.static_path.find_closest_goal(rover_position, pop_if_found=False)
                
    #             if end_point:
    #                 self.state = State.DYNAMIC_PATH
    #                 self.dynamic_path = DynamicPath(
    #                     [start_point, end_point],
    #                     self.static_path,
    #                     self.obstacles
    #                 )
    #                 # Will select dynamic goal on next iteration
    #                 self.goal_loc = None
    #             else:
    #                 print("[Navigator] WARNING: No valid goal for dynamic path!")


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
        
        self.state_machine((rover_x, rover_y), pytransform_position, input_data)  # Change the state

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

                # Add this check to handle None goal_loc
                if self.goal_loc is None:
                    # If no goal is set, pick a fallback goal or return a default motion
                    print("WARNING: No goal location set, selecting fallback goal")
                    self.goal_loc = self.static_path.find_closest_goal((rover_x, rover_y), pop_if_found=False)
                    
                    # If still no goal, use an emergency motion
                    if self.goal_loc is None:
                        print("CRITICAL: No goals available, using emergency motion")
                        return (0.0, 0.5)  # Rotate in place to find a path
                
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
