# This is an abstraction of all position estimators so that we can call this outside of dev to get a gauranteed position estimate

from maple.utils import pytransform_to_tuple
from math import atan2

class Navigator:
    """Provides position estimate using other python files"""

    def __init__(self, agent):
        """Create the estimator.

        Args:
            agent: The Agent instance
        """

        self.agent = agent

    def __call__(self, pytransform_position):
        """Equivalent to calling `get_lin_vel_ang_vel`."""
        return self.get_lin_vel_ang_vel(pytransform_position)

    def get_lin_vel_ang_vel(self, pytransform_position):
        """
        Takes the position and returns the linear and angular goal velocity
        """

        # This is test code to navigate to the position (0, 0)
        goal_x, goal_y = 0, 100

        # Extract the position information
        rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(pytransform_position)

        # Do trig to find the angle between the goal location and rover location
        angle_of_triangle = atan2((goal_x - rover_x), (goal_y - rover_y))

        goal_ang = angle_of_triangle - rover_yaw

        print(f'the goal ang is {goal_ang}')

        return (100, goal_ang)
    