import math
from typing import List

from maple.navigation.state.path import Path
from maple.navigation.state.static import is_collision
from maple.navigation.PythonRobotics.PathPlanning.RRTStar.rrt_star import RRTStar

class DynamicPath(Path):
    """ This is the random tree search path to get from point A to point B when the straight path has collisions
    """

    def __init__(self, target_locations, obstacles=None):
        """ Only have 2 locations for the target locations, the start location and the end locations
        """
        assert len(target_locations) == 2
        super().__init__(target_locations)
        
        if obstacles is None:
            obstacles = []
            
        # Set a retry count with different parameters if initial path fails
        max_retries = 3
        for i in range(max_retries):
            # Increase step size and max iterations with each retry
            step_size = 0.5 + (i * 0.5)  # 0.5, 1.0, 1.5
            max_iter = 1000 + (i * 500)   # 1000, 1500, 2000
            
            # Try to find a path
            path = calculate(target_locations[0], target_locations[1], obstacles, 
                        step_size=step_size, max_iter=max_iter)
            
            if path is not None:
                self.path = path
                print(f"Dynamic path found on attempt {i+1} with {len(path)} points")
                return
                
        # If all attempts fail, create a straight-line path as last resort
        print("WARNING: Dynamic failed to find path, using emergency straight-line path")
        self.path = [target_locations[0], target_locations[1]]

# IMPORTANT NOTE: This controls the limits for our search
def calculate(start, goal, obstacles, limits=[-9, 9], step_size=0.5, max_iter=1000) -> List[(float | int, float | int)]:
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

    # Lets try a straight line broskey
    print(f'the obstacles are {obstacles}')
    if not is_collision(start, goal, obstacles):
        return [start, goal]

    try:
        # Set Initial parameters
        print(f'the information for the rrt star is {start=} {goal=} {limits=} {obstacles=}')
        rrt_star = RRTStar(
            start=start,
            goal=goal,
            rand_area=limits,
            obstacle_list=obstacles,
            expand_dis=1,
            robot_radius=0.3)
        path = rrt_star.planning()
        return path
    except Exception as e:
        print(f'Loc 102: the exception that occured is {e}')
                    
    # IMPORTANTE TODO: Make sure we have a path somehow
    return None
