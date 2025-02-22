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

    def traverse(self, rover_position, radius_from_goal_location = .5):
        """
        This function takes the rover position and radius from goal location to be considered at that location
        """

        # Increment the goal check point until we are not considered there
        while self.get_distance_between_points(*rover_position, *self.path[self.current_check_point_index]) < radius_from_goal_location:
            self.current_check_point_index += 1

            # IMPORTANT NOTE: This repeats the path if we run out of target points, this should not be our goal, our path should get updated!!!
            if self.current_check_point_index >= len(self.path):
                self.current_check_point_index = 0

        return self.path[self.current_check_point_index]
    
    def is_collision(self, p1, p2, obstacles) -> bool:
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
                if self.get_distance_between_points(*p1, ox, oy) <= r:
                    return True
                continue

            # Parameter t for the projection of the circle center onto the line p1->p2.
            t = ((ox - p1[0]) * dx + (oy - p1[1]) * dy) / (dx * dx + dy * dy)
            t = max(0, min(1, t))  # Clamp t to the [0, 1] segment
            closest_x = p1[0] + t * dx
            closest_y = p1[1] + t * dy
            if self.get_distance_between_points(closest_x, closest_y, ox, oy) <= r:
                return True
        return False
    
    def is_possible_to_reach(self, goal_loc_x, goal_loc_y, obstacles=None):
        """Given the obstacles as (x, y, size) this will return if it is possible to reach the goal location

        Args:
            goal_loc_x (_type_): _description_
            goal_loc_y (_type_): _description_
            obstacles (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        if obstacles is None:
            return True
        
        # Check if it is blocked by an obstacle
        for obstacle_x, obstacle_y, obstacle_size in obstacles:
            if self.get_distance_between_points(goal_loc_x, goal_loc_y, obstacle_x, obstacle_y) <= obstacle_size:
                return False
            
        return True
    
    def remove_current_goal_location(self):
        """This function removes the current goal location from the path and sets the goal location to the next one
        """

        del self.path[self.current_check_point_index]
