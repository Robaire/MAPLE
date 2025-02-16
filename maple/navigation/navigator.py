from math import atan2

import numpy as np

from maple.navigation.path import Path
from maple.utils import pytransform_to_tuple


class Navigator:
    """Provides the goal linear and angular velocity for the rover"""

    def __init__(self, agent):
        """Create the navigator.

        Args:
            agent: The Agent instance
        """

        self.agent = agent

        # This is the start location for the rover
        self.rover_initial_position = agent.get_initial_position()

        # This is the start location for the lander
        self.lander_initial_position = agent.get_initial_lander_position()

        # This will generate a list of goal points for our path to go through, see the function for fine tuning variables
        basic_spiral = self.generate_spiral(
            self.lander_initial_position.location.x,
            self.lander_initial_position.location.y,
        )

        print(
            f"the basic spiral is {basic_spiral} while the intial lander position is {self.lander_initial_position.location.x}, {self.lander_initial_position.location.y}"
        )

        # I am thinking for starter we go towards the lander then spiral around it
        self.path = Path(basic_spiral)
        # IMPORTANT NOTE: This is a test to see how navigation does trying to go directly to lander
        # self.path = Path([(self.lander_initial_position.location.x, -self.lander_initial_position.location.y), (0, 0)])

        # This is how far from our current rover position along the path that we want to be the point our rover is trying to go to
        self.optimal_distance = 0.0001

        # This is the speed we are set to travel at (.48m/s is max linear and 4.13rad/s is mac angular)
        self.goal_speed = 0.15

        # This is the point we are currently trying to get to
        self.goal_loc = self.path.traverse(self.path.get_start(), self.optimal_distance)

    def generate_spiral(
        self, x0, y0, initial_radius=0.1, num_points=4, spiral_rate=0.0, frequency=10
    ):
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
            theta = i / frequency  # Angle in radians
            r = initial_radius + spiral_rate * theta  # Radius grows over time
            x = x0 + r * np.cos(theta)
            y = y0 + r * np.sin(theta)
            # IMPORTANT NOTE: Switch the y axis
            points.append((x, -y))

        return points

    def __call__(self, pytransform_position):
        """Equivalent to calling `get_lin_vel_ang_vel`."""
        return self.get_lin_vel_ang_vel(pytransform_position)

    def get_lin_vel_ang_vel(self, pytransform_position):
        """
        Takes the position and returns the linear and angular goal velocity
        """

        # Check if we are getting any location estimate, if not then turn right
        if pytransform_position is None:
            return (self.goal_speed, 5)

        # Extract the position information
        rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(
            pytransform_position
        )

        # Do trig to find the angle between the goal location and rover location
        goal_x, goal_y = self.goal_loc
        angle_of_triangle = atan2((goal_x - rover_x), (goal_y - rover_y))

        goal_ang = angle_of_triangle - rover_yaw

        # Move the goal point along the path
        self.goal_loc = self.path.traverse((rover_x, rover_y), self.optimal_distance)

        # print(f"the rover position is {rover_x} and {rover_y}")
        # print(f"the new goal location is {self.goal_loc}")
        # print(f"the goal ang is {goal_ang}")

        # TODO: Figure out a better speed
        return (self.goal_speed, goal_ang)
