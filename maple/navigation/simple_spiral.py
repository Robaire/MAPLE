
import math

def april_tag_input_only(rover_x, rover_y, rover_theta, lander_x, lander_y):
    """This is a simple function to only use the april tags and return the goal linear and angular velocity"""

    # Calculate the distance and angle to the lander
    dx = lander_x - rover_x
    dy = lander_y - rover_y
    distance_to_lander = math.sqrt(dx**2 + dy**2)
    angle_to_lander = math.atan2(dy, dx)
    
    # Calculate the angular difference (error)
    angle_error = angle_to_lander - rover_theta

    # Normalize angle_error to the range [-pi, pi]
    angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

    # Spiral parameters
    linear_velocity_base = 10  # Base linear velocity (m/s)
    angular_velocity_decay = 1.0 / (distance_to_lander + 1)  # Decays as distance increases

    # Calculate the goal velocities
    goal_linear_velocity = linear_velocity_base
    goal_angular_velocity = angle_error * angular_velocity_decay  # Proportional control with decay

    return goal_linear_velocity, goal_angular_velocity