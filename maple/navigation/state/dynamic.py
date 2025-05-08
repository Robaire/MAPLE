from typing import List, Tuple

from maple.navigation.state.path import Path
from maple.navigation.PythonRobotics.PathPlanning.RRTStar.rrt_star import RRTStar
from maple.navigation.constants import radius_from_goal_location
from maple.navigation.utils import is_collision, is_possible_to_reach, is_path_collision, is_in_obstacle, get_distance_between_points

from math import hypot

class DynamicPath(Path):
    """ This is the random tree search path to get from point A to point B when the straight path has collisions
    """

    def __init__(self, target_locations, obstacles=None):
        """ Only have 2 locations for the target locations, the start location and the end locations
        """
        assert len(target_locations) == 2
        super().__init__(target_locations)

        start_loc, goal_loc = target_locations
        start_x, start_y = start_loc

        self.goal_loc = goal_loc
        
        if obstacles is None:
            obstacles = []
            
        self.reset_path(start_loc, obstacles)

    def remvoe_old_points(self, agent_position):
        # Perform the pop operation on every point in the path that we are within the radius
        while len(self.path) > 0 and get_distance_between_points(agent_position[0], agent_position[1], self.path[0][0], self.path[0][1]) < radius:
            self.path.pop(0)

    def get_goal_location(self, agent_position, radius=radius_from_goal_location):
        """ Get the goal location for the path, will pop all locations we are considered to have reached based off of radius

        Args:
            agent_position (_type_): x y position of the agent

        Returns:
            _type_: x y position of the goal or None if there is no goal
        """

        self.remvoe_old_points(agent_position)

        # check if the path is empty, return None if so
        if len(self.path) == 0:
            return None

        # Return the first point in the path if it is not empty
        return self.path[0]


    def is_path_cpmplete(self):
        """ Check if the path is complete
        """
        return len(self.path) == 0

    def is_path_collision_free(self, agent_position, obstacles):
        """ Checks if the path is collision free

        Args:
            agent_position (_type_): x y position of the agent
            obstacles (_type_): a list of all the osbtacles

        Returns:
            _type_: boolean
        """
        # Checking if the path is collision free, adding in the agent position at the start
        return is_path_collision([agent_position] + self.path, obstacles)
    
    def reset_path(self, agent_position, obstacles):
        """ Resets the path, best to use if the current path is not valid

        Args:
            agent_position (_type_): x y position of the agent

        Returns:
            _type_: None
        """

        # Set a retry count with different parameters if initial path fails
        max_retries = 3
        for i in range(max_retries):
            # Increase step size and max iterations with each retry
            step_size = 0.5 + (i * 0.5)  # 0.5, 1.0, 1.5
            max_iter = 1000 + (i * 500)   # 1000, 1500, 2000
            
            # Try to find a path
            path = calculate(agent_position, self.goal_loc, obstacles, 
                        step_size=step_size, max_iter=max_iter)
            
            if path is not None:
                self.path = path
                print(f"Dynamic path found on attempt {i+1} with {len(path)} points")
                return
                
        # If all attempts fail, create a straight-line path as last resort
        print("WARNING: Dynamic failed to find path, using emergency straight-line path")
        self.path = [agent_position, self.goal_loc]
        

# IMPORTANT NOTE: This controls the limits for our search
def calculate(start, goal, obstacles, limits=[-9, 9], step_size=0.5, max_iter=1000) -> List[Tuple[float | int, float | int]]:
    """
    Run a basic dynamic algorithm to find a collision-free path from start to goal.
    
    Parameters:
        start (tuple): Starting point (x, y).
        goal (tuple): Goal point (x, y).
        obstacles (list): List of obstacles (ox, oy, radius).
        x_limits (tuple): (min_x, max_x) for random sampling.
        y_limits (tuple): (min_y, max_y) for random sampling.
        step_size (float): Incremental step size.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        list: The collision-free path as a list of (x, y) points if found, else None.
    """

    # Check if we are in an obstacle then ignore it
    new_obstacles = []
    for obstacle in obstacles:
        if is_in_obstacle(start[0], start[1], obstacle):
            continue
        new_obstacles.append(obstacle)
    obstacles = new_obstacles

    try:
        # Set Initial parameters
        print(f'the information for the rrt star is {start=} {goal=} {limits=} {obstacles=}')
        rrt_star = RRTStar(
            start=start,
            goal=goal,
            rand_area=limits,
            obstacle_list=obstacles,
            expand_dis=1,
            robot_radius=0.1,
            max_iter=max_iter,
            step_size=step_size,)
        path = rrt_star.planning()
        return path
    except Exception as e:
        print(f'Loc 102: the exception that occured is {e}')
                    
    # IMPORTANTE TODO: Make sure we have a path somehow
    return None