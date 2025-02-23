from math import atan2
import math
import numpy as np

import numpy as np

from maple.navigation.rrt_path import RRTPath
from maple.utils import pytransform_to_tuple, carla_to_pytransform
from pytransform3d.transformations import concat

class Navigator:
    """Provides the goal linear and angular velocity for the rover"""

    """
    This code uses a global pre planed path that is made at the start and should never be changed (compile_time_path)
    Will traveling if there are sections that are goal points that cant be reached they are removed in this path (real_time_path)
    To get from point to point a rrt path will be used (rrt_path)
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
        self.lander_x, self.lander_y, _, _, _, _ = pytransform_to_tuple(self.lander_initial_position)
        self.lander_obstacle = (self.lander_x, self.lander_y, 3)
        self.obstacles = [self.lander_obstacle]

        # This is how far from our current rover position along the path that we want to be the point our rover is trying to go to
        self.radius_from_goal_location = .5

        # This is the speed we are set to travel at (.48m/s is max linear and 4.13rad/s is max angular)
        self.goal_speed = .4
        self.goal_hard_turn_speed = .3

        # This is the location we are trying to get to on navigationr
        self.goal_loc = (0, 0) # IMPORTANT NOTE: This is for testing purpoese, will need to change

        # This is the point we are trying to get to using the rrt along with a path to that point
        self.rrt_path = None
        self.rrt_goal_loc = None # IMPORTANT NOTE: This is different than self.goal_loc because this is the goal location along the rrt path to get to self.goal_loc

        # This is the global path, DO NOT CHANGE IT!!
        self.global_path = generate_spiral(self.lander_x, self.lander_y)
        self.global_path_index_tracker = 0

    def get_next_goal_location(self, rover_x, rover_y):
        # NOTE: This function just loops through the global path

        # Update the index in a loop to allways have a point
        self.global_path_index_tracker = (self.global_path_index_tracker + 1) % len(self.global_path)

        return self.global_path[self.global_path_index_tracker]

    def get_goal_loc(self):
        return self.goal_loc

    def __call__(self, pytransform_position):
        """Equivalent to calling `get_lin_vel_ang_vel`."""
        return self.get_lin_vel_ang_vel(pytransform_position)

    def get_lin_vel_ang_vel(self, pytransform_position, obstacles = None):
        """
        Takes the position and returns the linear and angular goal velocity
        """

        # Update the obstacles, removing old ones, but keeping the lander
        if obstacles is not None:
            self.obstacles = obstacles
            self.obstacles.append(self.lander_obstacle)

        # Get the goal speed
        current_goal_speed = self.goal_speed

        # Extract the position information
        rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(
            pytransform_position
        )

        # Check if there will be a collision on the path, if so get rid of this one
        if self.rrt_path is not None and not self.rrt_path.is_path_collision_free(self.obstacles):
            self.rrt_path = None

        # Check if we have an rrt path and make one if we dont have one
        if self.rrt_path is None:
            self.rrt_path = RRTPath([(rover_x, rover_y), self.goal_loc], self.obstacles)

        # Check if it is possible to reach our goal location, if not pick a new one and rerun
        if not self.rrt_path.is_possible_to_reach(*self.goal_loc, self.obstacles):
            self.goal_loc = self.get_next_goal_location(rover_x, rover_y)
            self.rrt_path = None
            return self.get_lin_vel_ang_vel(pytransform_position)

        # Get the next path along the rrt path
        self.rrt_goal_loc = self.rrt_path.traverse((rover_x, rover_y), self.radius_from_goal_location)

        # Catch the case where there is no goal location (as in we made it there)
        if self.rrt_goal_loc is None:
            self.goal_loc = self.get_next_goal_location(rover_x, rover_y)
            self.rrt_path = None
            return self.get_lin_vel_ang_vel(pytransform_position)

        # Follow the rrt path
        rrt_goal_x, rrt_goal_y = self.rrt_goal_loc

        current_goal_ang = angle_helper(rover_x, rover_y, rover_yaw, rrt_goal_x, rrt_goal_y)
            
        # Check if we need to do a tight turn then override goal speed
        if abs(current_goal_ang) > .1:
            current_goal_speed = self.goal_hard_turn_speed

        print(f"the rover position is {rover_x} and {rover_y}")
        print(f"the new goal location is {self.goal_loc}")
        print(f'the goal location along the rrt path is {self.rrt_goal_loc}')
        print(f"the goal ang is {current_goal_ang}")

        # TODO: Figure out a better speed
        return (current_goal_speed, current_goal_ang)
    
def angle_helper(start_x, start_y, yaw, end_x, end_y):
    """Given a a start location and yaw this will return the desired turning angle to point towards end

    Args:
        start_x (float): _description_
        start_y (float): _description_
        yaw (float): _description_
        end_x (float): _description_
        end_y (float): _description_

    Returns:
        The goal angle for turning
    """

    # Do trig to find the angle between the goal location and rover location
    angle_of_triangle = atan2((end_y - start_y), (end_x - start_x))
    # Get the exact value outside of pi/2 and -pi/2

    # Calculate goal angular velocity
    goal_ang = angle_of_triangle - yaw

    # Normalize the angle to be within [-pi, pi]
    while goal_ang > np.pi:
        goal_ang -= 2 * np.pi
    while goal_ang < -np.pi:
        goal_ang += 2 * np.pi

    # NOTE: Test code
    # print(f'taking in start of {(start_x, start_y)} and the robot yaw is {yaw}. The goal location is {(end_x, end_y)}. The current goal angle turn is {goal_ang}')

    return goal_ang

def generate_spiral(x0, y0, initial_radius=4.0, num_points=400, spiral_rate=0.1, frequency=8):
    """
    Generates a list of (x, y) points forming a spiral around (x0, y0).

    :param x0: X-coordinate of the center
    :param y0: Y-coordinate of the center
    :param initial_radius: Starting radius from the center
    :param num_points: Number of points in the spiral
    :param spiral_rate: Controls how quickly the spiral expands
    :param frequency: Controls how closely spaced the points are
    :return: List of (x, y) points forming the spiral
    """
    points = []
    for i in range(num_points):
        theta = -i / frequency  # Angle in radians
        r = initial_radius + spiral_rate * theta  # Radius grows over time
        x = x0 + r * np.cos(theta)
        y = y0 + r * np.sin(theta)

        points.append((x, y))
    
    return points