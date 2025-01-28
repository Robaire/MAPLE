
from math import hypot

class Path:
    """This class is designed to store the whole goal path which can be used later"""
    # IMPORTANT NOTE: The straight line assumption is not permanent, need to build off of this later (I plan on doing that after RRT)

    def __init__(self, x1, y1, x2, y2, delta=.01):
        """Create the path. Currently a simple line from (x1, y1) to (x2, y2)
        """

        # This is how much change we allow on our estimate, this will be used along the x axis and is positive if going in pos x and eng otherwise
        self.delta = delta if x2 > x1 else -delta

        # This is code to make sure we are not going straight up which will cause issues with slope
        if x1 == x2:
            x1 += .0001

        self.start = (x1, y1)
        self.end = (x2, y2)

        # This is the slope (more complex paths need a better formula)
        self.m = (y2-y1)/(x2-x1)

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
        return hypot(x1-x2, y1-y2)
    
    def get_next_point(self, loc_on_path):
        """
        This function uses the delta established in the init and a locaiton on the path to return the next point on the path
        """

        # NOTE: This is only because it is a line, will have to update this as the path becomes more complex
        new_x = loc_on_path[0] + self.delta
        new_y = loc_on_path[1] + self.m*self.delta

        return (new_x, new_y)

    def traverse(self, start_loc_on_path, distance):
        """
        This function takes the location on a path and a goal distance to travel and will return the next point that is at least this distance away
        """

        # This counts the distance so we know how far we have gone
        distance_count = 0

        # This is the current position on the path we are
        current_loc_on_path = start_loc_on_path

        # Set up the while loop to repeat until we have traveled down the path long enough
        while distance_count < distance:

            # Get the next possible point on path
            new_current_loc_on_path = self.get_next_point(current_loc_on_path)

            # Add to the distance count so we can check if the goal distance has been met
            distance_count += self.get_distance_between_points(*current_loc_on_path, *new_current_loc_on_path)

            current_loc_on_path = new_current_loc_on_path

        return current_loc_on_path
