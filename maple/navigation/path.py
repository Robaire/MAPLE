from math import hypot


# This is the path class which will be used to represent a path and have helper functions for navigation to be able to use it
class Path:
    """This class is designed to store the whole goal path which can be used later"""

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
