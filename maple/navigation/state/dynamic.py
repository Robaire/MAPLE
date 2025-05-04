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
            max_iter = 1000 + (i * 500)  # 1000, 1500, 2000

            # # IMPORANT TODO: Clean up this code to get out from an obstacle
            # # Lets check if we are within an "obstacle" then get out of it by picking a point just outside of obstacle
            # if not is_possible_to_reach(start_x, start_y, obstacles):
            #     print('WARNING: We are "inside" an obstacle, trying to get out of it')
            #     # NOTE: One of these should return since is_possible_to_reach should have returned False
            #     for ox, oy, r in obstacles:
            #         dist_to_center = hypot(start_x - ox, start_y - oy)
            #         if dist_to_center <= r:
            #             dx = start_x - ox
            #             dy = start_y - oy
            #             if dist_to_center == 0:
            #                 # If the start is exactly at the center, pick any direction (e.g., x+ direction)
            #                 dx, dy = 1, 0
            #                 dist_to_center = 1  # Avoid division by zero
            #             # Normalize the direction and move to r + buffer_distance
            #             # IMPORTANT NOTE: This is a quick fix to pick a goal point outside of the obstacle to be able to move out from it, make this better
            #             # NOTE: Adding 1 just to make sure it is outside the obstacle, will look into
            #             scale_numerator = r + radius_from_goal_location + 1
            #             scale = scale_numerator / dist_to_center
            #             new_x = ox + dx * scale
            #             new_y = oy + dy * scale

            #             # This is a loop to eventually pick a point outside, it is stupid, I have to clean this up
            #             while not is_possible_to_reach(new_x, new_y, obstacles):
            #                 scale_numerator += radius_from_goal_location
            #                 scale = scale_numerator / dist_to_center
            #                 new_x = ox + dx * scale
            #                 new_y = oy + dy * scale

            #             self.path = [start_loc, (new_x, new_y)]

            #             return

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
                print(f"Dynamic path found on attempt {i+1} with {len(path)} points")
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
    limits=[-9, 9],
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

    try:
        # Set Initial parameters
        print(
            f"the information for the rrt star is {start=} {goal=} {limits=} {obstacles=}"
        )

        print(f'the goal is {goal}')

        rrt_star = RRTStar(
            start=start,
            goal=goal,
            rand_area=limits,
            obstacle_list=obstacles,
            expand_dis=1,
            robot_radius=0.3,
        )

        print(f'the rrt star path is {rrt_star}')

        # TODO: FIX THIS BS WHEN YOU HAVE TIME LUKE
        # path = run_with_timeout(lambda _: rrt_star.planning(), None, 1)
        path = rrt_star.planning()

        print(f'the path is {path}')

        return path
    except Exception as e:
        print(
            f"Loc 102: the exception that occured is {e} going to go towards the lander"
        )
        # Something went wrong with nav, go further along the static path

    # IMPORTANTE TODO: Make sure we have a path somehow
    return None
