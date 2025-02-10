
from math import hypot
from shapely.geometry import LineString, Point

# This is the path class which will be used to represent a path and have helper functions for navigation to be able to use it
class Path:
    """This class is designed to store the whole goal path which can be used later"""
    # IMPORTANT NOTE: The straight line assumption is not permanent, need to build off of this later (I plan on doing that after RRT)

    def __init__(self, target_locations):
        """ This initializes a path

        Args:
            target_locations (tuple): Provide a list of locations like this [(0, 0), (1, 2), (3, 3), (6, 1)] and a path will be generated to go from one to the next
        """

        # Initialize the path with the points to go through
        self.path = LineString(target_locations)

        # Save the start location
        self.start = target_locations[0]

        # This is the current checkpoint so that we are always progressing on the path
        self.current_check_point = self.start

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
        return hypot(x1-x2, y1-y2)
    
    def get_next_point(self, current_point, step_size=.01):
        """ This function uses the delta established in the init and a locaiton on the path to return the next point on the path

        Args:
            current_point (list): a list with our current x and y location
            step_size (float, optional): _description_. Defaults to .01.

        Returns:
            list: The next (x, y) point to try and go towards
        """

        # Convert current_point to the point object to be used in the shapely code
        current_point = Point(current_point)

        # Find the nearest point on the path to the given current point
        # IMPORTANT NOTE: This project only matters for the first few before the points on the path are fed through
        nearest_distance = self.path.project(current_point)  # Distance along the path
        
        # Calculate the next distance along the path
        next_distance = nearest_distance + step_size

        # Ensure we don't exceed the path length
        if next_distance > self.path.length:
            next_distance = self.path.length  # Stay at the endpoint

        # Get the next point
        next_point = self.path.interpolate(next_distance)
        
        # NOTE: This is to convert it out of the Point import, we can code this ourselves later for better efficiency
        return (next_point.x, next_point.y)
    
    def traverse(self, rover_position, distance):
        """
        This function takes the location on a path and a goal distance to travel and will return the next point that is at least this distance away
        """

        # This counts the distance so we know how far we have gone, it is initialized to our last check point on the path to our current position
        distance_count = self.get_distance_between_points(*self.current_check_point, *rover_position)

        # Set up the while loop to repeat until we have traveled down the path long enough
        while distance_count < distance:

            # Get the next possible point on path
            current_loc_on_path = self.get_next_point(self.current_check_point)

            # Add to the distance count so we can check if the goal distance has been met
            distance_holder = self.get_distance_between_points(*self.current_check_point, *current_loc_on_path)
            distance_count += distance_holder

            # NOTE: If distance holder doesnt change then we are too close to the end of the path to take another step so break the traverse
            if distance_holder == 0:
                # IMPORTANT TODO: Maybe remove this during actual comp
                print(f'ERROR: Ran out of pathing')
                exit()
                break

            self.current_check_point = current_loc_on_path

        return self.current_check_point
