from typing import List, Tuple

from maple.navigation.state.path import Path
from maple.navigation.state.static import StaticPath
from maple.navigation.state.path import is_collision, is_possible_to_reach
from maple.navigation.PythonRobotics.PathPlanning.RRTStar.rrt_star import RRTStar
from maple.navigation.constants import radius_from_goal_location

from math import hypot

import signal


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Function timed out")


def run_with_timeout(func, args, timeout):
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args)
    except TimeoutException as e:
        result = str(e)
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)
    return result


class DynamicPath(Path):
    """This is the random tree search path to get from point A to point B when the straight path has collisions"""

    def __init__(self, target_locations, obstacles=None):
        """Only have 2 locations for the target locations, the start location and the end locations"""
        assert len(target_locations) == 2
        super().__init__(target_locations)

        start_loc, goal_loc = target_locations
        start_x, start_y = start_loc

        if obstacles is None:
            obstacles = []

        # Set a retry count with different parameters if initial path fails
        max_retries = 3
        for i in range(max_retries):
            # Increase step size and max iterations with each retry
            step_size = 0.5 + (i * 0.5)  # 0.5, 1.0, 1.5
            max_iter = 100 + (i * 50)  # 100, 150, 200

            # Try to find a path
            path = calculate(
                start_loc,
                goal_loc,
                obstacles,
                step_size=step_size,
                max_iter=max_iter,
            )

            if path is not None:
                # IMPORTANT TODO: figure out why this is flipped
                self.path = path[::-1]
                print(f"Dynamic path found on attempt {i+1} with {path} points")
                return

        # If all attempts fail, create a straight-line path as last resort
        print(
            "WARNING: Dynamic failed to find path, using emergency straight-line path"
        )
        self.path = [target_locations[0], target_locations[1]]


# IMPORTANT NOTE: This controls the limits for our search
def calculate(
    start,
    goal,
    obstacles,
    limits=[-10, 10],
    step_size=0.5,
    max_iter=1000,
) -> List[Tuple[float | int, float | int]]:
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

    # # Lets try a straight line just in case broskey
    # if not is_collision(start, goal, obstacles):
    #     return [start, goal]

    # If we are in an obstacle just ignore it
    start_x, start_y = start
    new_obstacles = []
    for ox, oy, r in obstacles:
        if hypot(start_x - ox, start_y - oy) > r:
            new_obstacles.append((ox, oy, r))
    obstacles = new_obstacles

    try:
        # Set Initial parameters
        print(
            f"the information for the rrt star is {start=} {goal=} {limits=} {obstacles=}"
        )

        print(f"the goal is {goal}")

        rrt_star = RRTStar(
            start=start,
            goal=goal,
            rand_area=limits,
            obstacle_list=obstacles,
            expand_dis=step_size,
            robot_radius=0.01,
            max_iter=max_iter,
        )

        # NOTE: This is an option to run with a timeout or not, the number is in seconds
        # path = run_with_timeout(lambda _: rrt_star.planning(), None, 10) # DO the path planning with a timelimit of 10 seconds
        path = rrt_star.planning()

        print(f"foud a path of {path}")

        return path
    except Exception as e:
        print(
            f"Loc 102: the exception that occured is {e} going to go towards the lander"
        )
        # Something went wrong with nav, go further along the static path

    # IMPORTANTE TODO: Make sure we have a path somehow
    return None
