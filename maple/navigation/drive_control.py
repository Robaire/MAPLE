from maple.navigation.constants import goal_hard_turn_speed, goal_speed

from math import atan2
import numpy as np

def get_lin_vel_ang_vel_drive_control(rover_x, rover_y, rover_yaw, goal_x, goal_y):
    """
    Get the linear and angular velocity to drive the rover to the goal location
    """

    # Get the goal speed
    current_goal_speed = goal_speed

    # TODO: Add Dami code here
    # Follow the path with simple formula
    current_goal_ang = angle_helper(rover_x, rover_y, rover_yaw, goal_x, goal_y)
    
    # Check if we need to do a tight turn then override goal speed
    if abs(current_goal_ang) > 0.1:
        current_goal_speed = goal_hard_turn_speed
    
    # Success!
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

    return goal_ang