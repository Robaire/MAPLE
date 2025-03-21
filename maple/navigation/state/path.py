from maple.navigation.state.constants import radius_from_goal_location
from math import hypot


# This is the path class which will be used to represent a path and have helper functions for navigation to be able to use it
class Path:
    """This is the parent class to be able to extend"""

    def __init__(self, target_locations):
        """This initializes a path

        Args:
            target_locations (tuple): Provide a list of locations like this [(0, 0), (1, 2), (3, 3), (6, 1)] and a path will be generated to go from one to the next
        """

        # Initialize the path with the points to go through
        # IMPORTANT TODO: Make sure we never run out of target locations
        self.path = target_locations

        # Save the start location
        self.start = target_locations[0]

        # This is the current checkpoint so that we are always progressing on the path
        self.current_check_point_index = 0

        # Save the end location
        self.end = target_locations[-1]

    def get_full_path(self):
        return self.path

    def is_path_collision_free(self, obstacles):
        """Check if the current path is free of collisions with given obstacles."""
        if not self.path or len(self.path) < 2:
            return False
            
        for i in range(len(self.path) - 1):
            if is_collision(self.path[i], self.path[i+1], obstacles):
                return False
        return True

    def __call__(self):
        return None

    def get_start(self):
        """
        This function returns the start location
        """

        return self.start

    def get_end(self):
        return self.end

    def get_distance_between_points(self, x1, y1, x2, y2):
        return hypot(x1 - x2, y1 - y2)

    def traverse(self, rover_position, obstacles = []):
        """
        This function takes the rover position and radius from goal location to be considered at that location
        """

        # Handle no path and longer index correctly
        if self.path is None or self.current_check_point_index >= len(self.path):
            return None

        # Increment the goal check point until we are not considered there or in an obstacle
        while self.get_distance_between_points(*rover_position, *self.path[self.current_check_point_index]) < radius_from_goal_location or not is_possible_to_reach(*self.path[self.current_check_point_index], obstacles):
            self.current_check_point_index += 1

            if self.current_check_point_index >= len(self.path):
                return None

        return self.path[self.current_check_point_index]

def is_possible_to_reach(x, y, obstacles):
    """Check if x, y is possible to reach
    ie not in an obstacle"""
        
    for (ox, oy, r) in obstacles:
        if hypot(x - ox, y - oy) <= r:
            return False
    return True

def is_collision(p1, p2, obstacles) -> bool:
    """
    Check if the line segment from p1 to p2 intersects any circular obstacles.
    Each obstacle is defined as a tuple (ox, oy, radius).
    """
    for (ox, oy, r) in obstacles:
        # Vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # If p1 and p2 are the same point, check that point only.
        if dx == 0 and dy == 0:
            if hypot(p1[0] - ox, p1[1] - oy) <= r:
                return True
            continue

        # Parameter t for the projection of the circle center onto the line p1->p2.
        t = ((ox - p1[0]) * dx + (oy - p1[1]) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # Clamp t to the [0, 1] segment
        closest_x = p1[0] + t * dx
        closest_y = p1[1] + t * dy
        if hypot(closest_x - ox, closest_y - oy) <= r:
            return True
    return False