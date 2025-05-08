from maple.utils import pytransform_to_tuple, carla_to_pytransform
from maple.navigation.drive_control import DriveController, angle_helper
from maple.navigation.constants import lander_size
from maple.navigation.static_path_planning import (
    generate_spiral,
)
from pytransform3d.transformations import concat
from maple.navigation.state.path import is_collision
from maple.navigation.utils import get_distance_between_points
from maple.navigation.constants import radius_from_goal_location
from maple.navigation.state.static import StaticPath
import math


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
        # self.lander_obstacle = (self.lander_x, self.lander_y, lander_size+10)
        self.lander_obstacle = (self.lander_x, self.lander_y, lander_size)

        self.obstacles = [self.lander_obstacle]

        # Going to add in multiple osbtacles around the lander with less size so that if we get too close we can still run rrt while just ignoring the outer most obstacle
        for new_size in range(4 * lander_size):
            self.obstacles.append((self.lander_x, self.lander_y, new_size / 4))

        # This is the goal location we are currently trying to get to, make sure to update it
        self.goal_loc = None
        self.distance_threshold = 2.0  # 1 meter threshold for updating goal

        self.frames_since_last_obstacle_reset = 0

        # This is the drive controller for getting the linear and angular velocity
        self.drive_control = DriveController()

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
        #     # (-9, -6, 1),
        #     (-9, -3, 0.75),
        #     (-9, 0, 1),
        #     (-9, 3, 0.75),
        #     (-9, 6, 1),
        #     (-9, 9, 0.75),
        #     (-9, 12, 0.5),
        #     (-6, -12, 0.5),
        #     (-6, -9, 1),
        #     (-6, -6, 1),
        #     (-6, -3, 1),
        #     (-6, 0, 0.75),
        #     (-6, 3, 1),
        #     (-6, 6, 1),
        #     (-6, 9, 1),
        #     (-6, 12, 0.5),
        #     (-3, -12, 0.5),
        #     (-3, -9, 0.75),
        #     (-3, -6, 1),
        #     (-3, 6, 1),
        #     (-3, 9, 0.75),
        #     (-3, 12, 0.5),
        #     (0, -12, 0.5),
        #     (0, -9, 1),
        #     (0, -6, 0.75),
        #     (0, 6, 0.75),
        #     (0, 9, 1),
        #     (0, 12, 0.5),
        #     (3, -12, 0.5),
        #     (3, -9, 0.75),
        #     (3, -6, 1),
        #     (3, 6, 1),
        #     (3, 9, 0.75),
        #     (3, 12, 0.5),
        #     (6, -12, 0.5),
        #     (6, -9, 1),
        #     (6, -6, 1),
        #     (6, -3, 1),
        #     (6, 0, 0.75),
        #     (6, 3, 1),
        #     (6, 6, 1),
        #     (6, 9, 0.75),
        #     (6, 12, 0.5),
        #     (9, -12, 0.5),
        #     (9, -9, 0.75),
        #     (9, -6, 1),
        #     (9, -3, 0.75),
        #     (9, 0, 1),
        #     (9, 3, 0.75),
        #     (9, 6, 1),
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
            (0, -6, 0.75),
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

    def change_state(self):
        """
        This function will be called and handle the state changing, it will change the state within its code

        Returns: None
        """
        raise NotImplementedError

    # def get_goal_location(self, rover_position, estimate, input_data):
    #     """
    #     Updates the goal location after reaching the current goal.
    #     Uses the Path.find_closest_goal method to select the next goal.

    #     Args:
    #         rover_position: (x, y) tuple of the current rover position

    #     Returns:
    #         The new goal location or None if no more goals available
    #     """

    #     # IMPORTANT TODO: Once more states are added in change this code to handle that correctlydef find

    #     # Check if we are close enough to the current goal loc to set a new one
    #     if (
    #         self.goal_loc is None
    #         or get_distance_between_points(*rover_position, *self.goal_loc)
    #         < radius_from_goal_location
    #     ):
    #         print(f"[Navigator] Removing reached goal {self.goal_loc} from static path")

    #         # Pick a new goal location based off of the features in that direction while elimating ones across the lander
    #         new_goal_with_weight = self.static_path.find_closest_goal(
    #             rover_position,
    #             estimate,
    #             input_data,
    #             self.agent,
    #             pop_if_found=True,
    #             obstacles=self.obstacles,
    #         )

    #         # The function above extracts the goal with the corresponding weight
    #         goal_x, goal_y, goal_w = new_goal_with_weight

    #         new_goal = (goal_x, goal_y)

    #         return new_goal

    #     # rover_position is (x, y)
    #     nearby_obstacles = []
    #     for obs in self.obstacles:
    #         obs_x, obs_y, _ = obs  # assuming each obstacle is (x, y, size)
    #         distance = (
    #             (obs_x - rover_position[0]) ** 2 + (obs_y - rover_position[1]) ** 2
    #         ) ** 0.5
    #         if distance <= 1.75:
    #             nearby_obstacles.append(obs)

    #     # Now check collision only with nearby obstacles
    #     if is_collision(
    #         rover_position, (self.goal_loc[0], self.goal_loc[1]), nearby_obstacles
    #     ):
    #         # if is_collision(rover_position, (self.goal_loc[0], self.goal_loc[1]), self.obstacles):
    #         print("picking new direction because of an obstacle!")
    #         # TODO: for soem reason it gets rid of this when we don't want it to so I'm re-adding it, need to pass in the real weight of the point?
    #         self.static_path.path.append((self.goal_loc[0], self.goal_loc[1], 0.85))
    #         # Pick a new goal location based off of the features in that direction while elimating ones across the lander
    #         new_goal_with_weight = self.static_path.find_closest_goal(
    #             rover_position,
    #             estimate,
    #             input_data,
    #             self.agent,
    #             pop_if_found=False,
    #             obstacles=self.obstacles,
    #         )

    #         # The function above extracts the goal with the corresponding weight
    #         goal_x, goal_y, goal_w = new_goal_with_weight

    #         new_goal = (goal_x, goal_y)

    #         print("resetting obstacles")

    #         if self.frames_since_last_obstacle_reset % 10 == 0:
    #             self.obstacles = [self.lander_obstacle]

    #         print("obstacles now: ", self.obstacles)

    #         self.frames_since_last_obstacle_reset += 1

    #         return new_goal

    #     return self.goal_loc


    def get_goal_location(self, rover_position, estimate, input_data):
        """
        Updates the goal location after reaching the current goal.
        Uses the Path.find_closest_goal method to select the next goal.

        Args:
            rover_position: (x, y) tuple of the current rover position

        Returns:
            The new goal location or None if no more goals available
        """

        # IMPORTANT TODO: Once more states are added in change this code to handle that correctlydef find

        # Check if we are close enough to the current goal loc to set a new one
        if (
            self.goal_loc is None
            or get_distance_between_points(*rover_position, *self.goal_loc)
            < radius_from_goal_location
        ):
            print(f"[Navigator] Removing reached goal {self.goal_loc} from static path")

            # Pick a new goal location based off of the features in that direction while elimating ones across the lander
            new_goal_with_weight = self.static_path.find_closest_goal(
                rover_position,
                estimate,
                input_data,
                self.agent,
                pop_if_found=True,
                obstacles=self.obstacles,
            )

            # self.obstacles = [self.lander_obstacle]

            # The function above extracts the goal with the corresponding weight
            goal_x, goal_y, goal_w = new_goal_with_weight

            new_goal = (goal_x, goal_y)

            return new_goal
        

        # Now check collision only with nearby obstacles
        if is_collision(rover_position, (self.goal_loc[0], self.goal_loc[1]), self.obstacles): #and self.frame%200==0:

            print("picking new direction because of an obstacle!")
            # TODO: for soem reason it gets rid of this when we don't want it to so I'm re-adding it, need to pass in the real weight of the point?
            self.static_path.path.append((self.goal_loc[0], self.goal_loc[1], 0.75))
            # Pick a new goal location based off of the features in that direction while elimating ones across the lander
            new_goal_with_weight = self.static_path.find_closest_goal(
                rover_position,
                estimate,
                input_data,
                self.agent,
                pop_if_found=False,
                obstacles=self.obstacles,
            )

            # The function above extracts the goal with the corresponding weight
            goal_x, goal_y, goal_w = new_goal_with_weight

            new_goal = (goal_x, goal_y)

            # print("resetting obstacles")

            # if self.frames_since_last_obstacle_reset % 15 == 0:
            #     # self.obstacles = [self.lander_obstacle]
            #     new_goal = 

            # print("obstacles now: ", self.obstacles)

            # self.frames_since_last_obstacle_reset += 1

            return new_goal

        return self.goal_loc

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

        # Rather than assert this, filter out any detections that don't have 3 elements
        # assert len(detections[0]) == 3
        detections = [detection for detection in detections if len(detection) == 3]

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

        # Extracting useful information
        rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(
            pytransform_position
        )

        # Call the state change function once we add more states

        # Get the goal location to go towards
        self.goal_loc = self.get_goal_location(
            (rover_x, rover_y), pytransform_position, input_data
        )
        if self.goal_loc is None:
            print("We ran out of static points to navigate to!!!")
        goal_x, goal_y = self.goal_loc

        # Success!
        return self.drive_control.get_lin_vel_ang_vel_drive_control(
            rover_x, rover_y, rover_yaw, goal_x, goal_y
        )
