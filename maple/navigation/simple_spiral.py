
import math
from maple.utils import pytransform_to_tuple, carla_to_pytransform

class Navigation:

    def __init__(self, agent):
        self.agent = agent

        self.lander_initial_position_pytransform = carla_to_pytransform(agent.get_initial_lander_position())
        self.lander_x, self.lander_y, self.lander_z, self.lander_roll, self.lander_pitch, self.lander_yaw = pytransform_to_tuple(self.lander_initial_position_pytransform)

    def dumb_spiral(self, estimate):
        """This is a simple function to only use the april tags and return the goal linear and angular velocity"""

        rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(estimate)

        # Calculate the distance and angle to the lander
        dx = self.lander_x - rover_x
        dy = self.lander_y - rover_y
        distance_to_lander = math.sqrt(dx**2 + dy**2)
        angle_to_lander = math.atan2(dy, dx)
        
        # Calculate the angular difference (error)
        angle_error = angle_to_lander - rover_yaw

        # Normalize angle_error to the range [-pi, pi]
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

        # Spiral parameters
        linear_velocity_base = 10  # Base linear velocity (m/s)
        angular_velocity_decay = 1.0 / (distance_to_lander + 1)  # Decays as distance increases

        # Calculate the goal velocities
        goal_linear_velocity = linear_velocity_base
        goal_angular_velocity = angle_error * angular_velocity_decay  # Proportional control with decay

        return goal_linear_velocity, goal_angular_velocity