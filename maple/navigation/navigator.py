from maple.utils import pytransform_to_tuple
from math import atan2
import numpy as np

from maple.navigation.path import Path

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
        basic_spiral = self.generate_spiral(self.lander_initial_position.location.x, self.lander_initial_position.location.y)

        print(f'the basic spiral is {basic_spiral} while the intial lander position is {self.lander_initial_position.location.x}, {self.lander_initial_position.location.y}')

        # I am thinking for starter we go towards the lander then spiral around it
        self.path = Path(basic_spiral)
        # IMPORTANT NOTE: This is a test to see how navigation does trying to go directly to lander
        # self.path = Path([(self.lander_initial_position.location.x, self.lander_initial_position.location.y), (0, 0)])

        # This is how far from our current rover position along the path that we want to be the point our rover is trying to go to
        self.optimal_distance = .0001

        # This is the speed we are set to travel at (.48m/s is max linear and 4.13rad/s is max angular)
        self.goal_speed = .15

        # This is the point we are currently trying to get to
        self.goal_loc = self.path.traverse(self.path.get_start(), self.optimal_distance)

    def generate_spiral(self, x0, y0, initial_radius=4, num_points=1000, spiral_rate=0, frequency=10):
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
            # IMPORTANT NOTE: Switch the y axis
            points.append((x, y))
        
        return points

    def __call__(self, pytransform_position):
        """Equivalent to calling `get_lin_vel_ang_vel`."""
        return self.get_lin_vel_ang_vel(pytransform_position)

    def get_lin_vel_ang_vel(self, pytransform_position):
        """
        Takes the position and returns the linear and angular goal velocity
        """

        # Extract the position information
        rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(pytransform_position)

        goal_x, goal_y = self.goal_loc

        goal_ang = angle_helper(rover_x, rover_y, rover_yaw, goal_x, goal_y)

        # Move the goal point along the path
        self.goal_loc = self.path.traverse((rover_x, rover_y), self.optimal_distance)

        print(f'the rover position is {rover_x} and {rover_y}')
        print(f'the new goal location is {self.goal_loc}')
        print(f'the goal ang is {goal_ang}')

        # TODO: Figure out a better speed
        return (self.goal_speed, goal_ang)
    
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
    angle_of_triangle = atan2((end_x - start_x), (end_y - start_y))
    # Get the exact value outside of pi/2 and -pi/2
    angle_of_triangle = angle_of_triangle if (end_x - start_x) > 0 else 10 # turn right at 10

    # Calculate goal angular velocity
    goal_ang = angle_of_triangle - yaw

    # Normalize the angle to be within [-pi, pi]
    while goal_ang > np.pi:
        goal_ang -= 2 * np.pi
    while goal_ang < -np.pi:
        goal_ang += 2 * np.pi

    # TODO: Make this check for facing directions better
    # Distance squared between the points
    # distance_sq = (start_x-end_x)**2 + (start_y-end_y)**2

    # Predicted location using the current angle


    # # If the robot is facing the wrong direction (more than 90 degrees off), force it to turn in place
    # if abs(goal_ang) > np.pi / 8:  # More than 90 degrees off
    #     # TODO: Come up with better turn functinality
    #     return 10  # Override normal motion to just turn

    return goal_ang