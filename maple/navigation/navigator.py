from maple.utils import pytransform_to_tuple
from math import atan2

from maple.navigation.path import Path

class Navigator:
    """Provides the goal linear and angular velocity for the rover"""

    def __init__(self, agent):
        """Create the navigator.

        Args:
            agent: The Agent instance
        """

        self.agent = agent
        
        # This is test code
        self.path = Path(*(0, 0), *(100, 0))
        # This is test code

        # This is how far from our current rover position along the path that we want to be the point our rover is trying to go to
        self.optimal_distance = 1

        # This is the point we are currently trying to get to
        self.goal_loc = self.path.traverse(self.path.get_start(), self.optimal_distance)

    def __call__(self, pytransform_position):
        """Equivalent to calling `get_lin_vel_ang_vel`."""
        return self.get_lin_vel_ang_vel(pytransform_position)

    def get_lin_vel_ang_vel(self, pytransform_position):
        """
        Takes the position and returns the linear and angular goal velocity
        """

        # Extract the position information
        rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(pytransform_position)

        # Do trig to find the angle between the goal location and rover location
        goal_x, goal_y = self.goal_loc
        angle_of_triangle = atan2((goal_x - rover_x), (goal_y - rover_y))

        goal_ang = angle_of_triangle - rover_yaw

        # Move the goal point along the path
        self.goal_loc = self.path.traverse((rover_x, rover_y), self.optimal_distance)

        print(f'the new goal location is {self.goal_loc}')
        print(f'the goal ang is {goal_ang}')

        # TODO: Figure out a better speed
        return (100, goal_ang)
    