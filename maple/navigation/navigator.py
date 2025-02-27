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
        self.goal_speed = .3
        self.goal_hard_turn_speed = .3

        # This is the location we are trying to get to on navigationr
        self.goal_loc = (0, 0) # IMPORTANT NOTE: This is for testing purpoese, will need to change

        ##### spiral path #####
        # This is the point we are trying to get to using the rrt along with a path to that point
        self.rrt_path = None
        self.rrt_goal_loc = None # IMPORTANT NOTE: This is different than self.goal_loc because this is the goal location along the rrt path to get to self.goal_loc

        # This is the global path, DO NOT CHANGE IT!!
        self.global_path = generate_spiral(self.lander_x, self.lander_y)
        self.global_path_index_tracker = 0
        ##### spiral path #####

        # ##### lawnmower path #####
        # # This is the point we are trying to get to using the rrt along with a path to that point
        # self.rrt_path = None
        # self.rrt_goal_loc = None # IMPORTANT NOTE: This is different than self.goal_loc because this is the goal location along the rrt path to get to self.goal_loc

        # # This is the global path, DO NOT CHANGE IT!!
        # # self.global_path = generate_spiral(self.lander_x, self.lander_y)
        # self.global_path = generate_lawnmower(self.lander_x, self.lander_y)

        # # print("global path: ", self.global_path)
        # self.global_path_index_tracker = 0
        # ##### lawnmower path #####


    def get_all_goal_locations(self):
        return self.global_path

    def add_large_boulder_detection(self, detections):
        self.obstacles.extend(detections)

    def get_obstacle_locations(self):
        return self.obstacles

    def get_next_goal_location(self, rover_x, rover_y):
        # NOTE: This function just loops through the global path

        # Update the index in a loop to allways have a point
        self.global_path_index_tracker = (self.global_path_index_tracker + 1) % len(self.global_path)

        # Goal loc
        goal_loc = self.global_path[self.global_path_index_tracker]

        # # Loop until we find a point we can make it to
        # while not self.rrt_path.is_possible_to_reach(*goal_loc, self.obstacles):
        #     print(f'the index is {self.global_path_index_tracker} the len is {len(self.global_path)}')
        #     # Update the index in a loop to allways have a point
        #     self.global_path_index_tracker = (self.global_path_index_tracker + 1) % len(self.global_path)
        #     print(f'the goal loc is {goal_loc}')
        #     # Goal loc
        #     goal_loc = self.global_path[self.global_path_index_tracker]
        #     print(f'the obstacles are {self.obstacles}')

        return goal_loc
    
    def get_goal_loc(self):
        return self.goal_loc

    def __call__(self, pytransform_position):
        """Equivalent to calling `get_lin_vel_ang_vel`."""
        return self.get_lin_vel_ang_vel(pytransform_position)
    
    def get_lin_vel_ang_vel(self, pytransform_position, obstacles=None, attempt=0):
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

        # Prevent infinite loops
        if attempt >= 5:
            print("WARNING: Maximum attempts reached, using emergency fallback")
            # Emergency fallback: turn in place then try to move slightly
            return (0.1, 0.5)
            
        # Update obstacles
        if obstacles is not None:
            self.obstacles = [o for o in obstacles]
            if self.lander_obstacle not in self.obstacles:
                self.obstacles.append(self.lander_obstacle)
        
        # Get the goal speed
        current_goal_speed = self.goal_speed
        
        try:
            # Extract the position information
            rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(pytransform_position)
            
            # Check if there will be a collision on the path, if so get rid of this one
            if self.rrt_path is not None and not self.rrt_path.is_path_collision_free(self.obstacles):
                self.rrt_path = None
            
            # Check if we have an rrt path and make one if we don't have one
            if self.rrt_path is None:
                self.rrt_path = RRTPath([(rover_x, rover_y), self.goal_loc], self.obstacles)
            
            # Check if it is possible to reach our goal location
            if not self.rrt_path.is_possible_to_reach(*self.goal_loc, self.obstacles):
                # Try a new goal location without recursion
                self.goal_loc = self.get_next_goal_location(rover_x, rover_y)
                self.rrt_path = None
                
                # Instead of recursion, increment attempt counter and try again
                return self.get_lin_vel_ang_vel(pytransform_position, obstacles, attempt + 1)
            
            # Get the next path along the rrt path
            self.rrt_goal_loc = self.rrt_path.traverse((rover_x, rover_y), self.radius_from_goal_location)
            
            # If no goal location (we made it there), pick new one
            if self.rrt_goal_loc is None:
                self.goal_loc = self.get_next_goal_location(rover_x, rover_y)
                self.rrt_path = None
                
                # Instead of recursion, increment attempt counter and try again
                return self.get_lin_vel_ang_vel(pytransform_position, obstacles, attempt + 1)
            
            # Follow the rrt path
            rrt_goal_x, rrt_goal_y = self.rrt_goal_loc
            current_goal_ang = angle_helper(rover_x, rover_y, rover_yaw, rrt_goal_x, rrt_goal_y)
            
            # Check if we need to do a tight turn then override goal speed
            if abs(current_goal_ang) > 0.1:
                current_goal_speed = self.goal_hard_turn_speed
            
            print(f"the rover position is {rover_x} and {rover_y}")
            print(f"the new goal location is {self.goal_loc}")
            print(f'the goal location along the rrt path is {self.rrt_goal_loc}')
            print(f"the goal ang is {current_goal_ang}")
            
            # Success!
            return (current_goal_speed, current_goal_ang)
            
        except Exception as e:
            print(f"Navigation error: {e}")
            
            # Try one of several fallback strategies
            if attempt == 0:
                # First fallback: try with just the lander as an obstacle
                print("Trying fallback with just lander obstacle")
                self.obstacles = [self.lander_obstacle]
                return self.get_lin_vel_ang_vel(pytransform_position, self.obstacles, attempt + 1)
            elif attempt == 1:
                # Second fallback: try with a new goal location
                print("Trying fallback with new goal location")
                rover_x, rover_y, _, _, _, _ = pytransform_to_tuple(pytransform_position)
                self.goal_loc = self.get_next_goal_location(rover_x, rover_y)
                self.rrt_path = None
                return self.get_lin_vel_ang_vel(pytransform_position, self.obstacles, attempt + 1)
            elif attempt == 2:
                # Third fallback: try with a simpler path planning approach
                print("Trying emergency direct path")
                rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(pytransform_position)
                # Simple emergency path - move away from current location
                emergency_x = rover_x + 2.0
                emergency_y = rover_y + 2.0
                emergency_ang = angle_helper(rover_x, rover_y, rover_yaw, emergency_x, emergency_y)
                return (0.2, emergency_ang)  # Slow speed for safety
            else:
                # Last resort: rotate in place to find a clear path
                print("CRITICAL: Using last resort movement")
                return (0.0, 0.5)  # Rotate in place at a moderate speed
            

    def get_rrt_waypoints(self):
        """
        Return the full list of waypoints from the current RRT path, if it exists.
        This can be useful for visualization or debugging.
        
        Returns:
            A list of (x, y) tuples representing the current planned path,
            or an empty list if there is no valid path.
        """
        if self.rrt_path is None:
            return []
        
        # Depending on your RRTPath implementation, you might do:
        #   return self.rrt_path.get_full_path()
        # or
        #   return self.rrt_path.nodes
        #
        # If `RRTPath` has no built-in method, you can store the path in a variable
        # in `RRTPath` after construction or after planning. Here, let's assume
        # there's a `get_full_path()` method for demonstration:
        
        return self.rrt_path.get_full_path()

    
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
        theta = i / frequency  # Angle in radians
        r = initial_radius + spiral_rate * theta  # Radius grows over time
        x = x0 + r * np.cos(theta)
        y = y0 + r * np.sin(theta)

        points.append((x, y))
    
    return points

import numpy as np

def rotate_point(x, y, angle_degrees):
    """
    Rotate point (x, y) around the origin (0, 0) by angle_degrees in a standard
    mathematical positive rotation (counter-clockwise).
    """
    angle = np.radians(angle_degrees)
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    return x_rot, y_rot

def generate_lawnmower(x0, y0, width=9.0, height=9.0, spacing=2.0):
    """
    Generate a lawnmower (back-and-forth) path that covers a rectangular region
    of size width x height, centered at (x0, y0). Spacing determines the distance
    between successive "passes".
    """
    # Start from the top-left corner in local coordinates (-width/2, +height/2)
    # and move row by row downwards.
    half_w = width / 2.0
    half_h = height / 2.0

    points = []
    # Determine how many rows (back-and-forth lines) we'll have
    num_rows = int(np.ceil(height / spacing))

    for row in range(num_rows + 1):
        # Current y in local coordinates (top to bottom)
        y_local = half_h - row * spacing

        # If the row is even, move left-to-right; if odd, move right-to-left
        if row % 2 == 0:
            # left to right
            x_line = np.linspace(-half_w, half_w, num=10)  # e.g. 10 points per row
        else:
            # right to left
            x_line = np.linspace(half_w, -half_w, num=10)

        for x_local in x_line:
            # Shift back to global coordinates
            x_global = x0 + x_local
            y_global = y0 + y_local
            points.append((x_global, y_global))

    return points

def generate_multi_angle_lawnmower(x0, y0, angles, width=9.0, height=9.0, spacing=2.0):
    """
    Generate a combined coverage path by doing a lawnmower sweep at multiple angles.
    'angles' is a list of angles (in degrees) by which we'll rotate the region
    around (x0, y0).
    """
    all_points = []

    for angle in angles:
        # 1. Generate a standard lawnmower pattern around (0,0) to keep it simple
        #    and interpret that pattern in local coordinates.
        lawnmower_local = generate_lawnmower(0, 0, width=width, height=height, spacing=spacing)

        # 2. Rotate each point by 'angle' around origin, then shift to (x0, y0).
        #    Because we generated at (0,0) "center", the rotation is straightforward.
        rotated_path = []
        for (lx, ly) in lawnmower_local:
            rx, ry = rotate_point(lx, ly, angle)
            # Shift by the global center (x0, y0)
            rx_global = x0 + rx
            ry_global = y0 + ry
            rotated_path.append((rx_global, ry_global))

        # 3. Append to our big list
        all_points.extend(rotated_path)

    return all_points

# Example usage:
# Cover 9x9 region, center at (0,0), with passes at 0°, 90°, and 45°.
waypoints = generate_multi_angle_lawnmower(
    x0=0.0, 
    y0=0.0, 
    angles=[0, 45, 90], 
    width=9.0, 
    height=9.0, 
    spacing=2.0
)

