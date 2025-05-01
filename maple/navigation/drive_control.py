from maple.navigation.constants import DT
from maple.navigation import constants

from math import atan2
import numpy as np


class DriveController:
    def __init__(self):
        self.linear_pid = PIDController(
            kp=0.01, ki=0.01, kd=0.1, setpoint=constants.goal_speed
        )
        self.angular_pid = PIDController(
            kp=0.01, ki=0.01, kd=0.1, setpoint=0
        )  # 0 is considered towards the goal location in this code

        self.prev_distance_to_goal = 0

    def reset(self):
        """
        Function to reset all value for derivative and stuff so we dont use old information for new goal locations
        """
        self.linear_pid = PIDController(
            kp=0.01, ki=0.01, kd=0.1, setpoint=constants.goal_speed
        )
        self.angular_pid = PIDController(
            kp=0.01, ki=0.01, kd=0.1, setpoint=0
        )  # 0 is considered towards the goal location in this code

        self.prev_distance_to_goal = 0

    def get_lin_vel_ang_vel_drive_control_straight(self, rover_x, rover_y, rover_yaw):
        """
        Have the robot drive straight, will pick a fake goal point that is in a line
        """

        # Pick a point in a stright line away
        delta_x = 100
        goal_x, goal_y = rover_x + delta_x, rover_y + delta_x * rover_yaw

        # Call the function with the fake goal location
        return self.get_lin_vel_ang_vel_drive_control(
            rover_x, rover_y, rover_yaw, goal_x, goal_y
        )

    def get_lin_vel_ang_vel_drive_control(
        self, rover_x, rover_y, rover_yaw, goal_x, goal_y
    ):
        """
        Get the linear and angular velocity to drive the rover to the goal location
        """

        # TODO: Reset the memory information when going to a "new" goal location so part measurements dont effect

        # Calculate the angle helper
        goal_ang = angle_helper(rover_x, rover_y, rover_yaw, goal_x, goal_y)

        # Negate the goal_ang to tell how far off we are from the measurement, where zero is towards the goal location
        measured_off_ang = -goal_ang

        # Calculate distance to the goal
        distance_to_goal = np.sqrt((goal_x - rover_x) ** 2 + (goal_y - rover_y) ** 2)

        # Was warned not to trust the velocity function so calcualting velocity from position estiamtes
        measured_velocity = (self.prev_distance_to_goal - distance_to_goal) / DT

        # Update PID controllers
        linear_velocity = self.linear_pid.update(measured_velocity, DT)
        angular_velocity = self.angular_pid.update(measured_off_ang, DT)

        # Check if we need to do a tight turn then override goal speed
        if abs(goal_ang) > 0.1:
            linear_velocity = constants.goal_hard_turn_speed

        # print(f"the rover position is {rover_x} and {rover_y}")
        # print(f"the new goal location is {(goal_x, goal_y)}")
        # print(f"the goal ang is {goal_ang}")

        return linear_velocity, angular_velocity


class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0

    def update(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


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
