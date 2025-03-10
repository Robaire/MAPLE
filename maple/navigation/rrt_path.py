import math
import random
from typing import List

from maple.navigation.path import Path

from maple.navigation.PythonRobotics.PathPlanning.RRTStar.rrt_star import RRTStar

# A simple node class to represent points in the tree for the damn rrt
class Node:
    def __init__(self, point, parent=None):
        self.point = point  # (x, y)
        self.parent = parent  # reference to the parent Node

class RRTPath(Path):
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
            path = rrt(target_locations[0], target_locations[1], obstacles, 
                        step_size=step_size, max_iter=max_iter)
            
            if path is not None:
                self.path = path
                print(f"RRT path found on attempt {i+1} with {len(path)} points")
                return
                
        # If all attempts fail, create a straight-line path as last resort
        print("WARNING: RRT failed to find path, using emergency straight-line path")
        self.path = [target_locations[0], target_locations[1]]

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
        
    def is_possible_to_reach(self, x, y, obstacles):
        """Check if the goal (x,y) is reachable from the last point in the path."""
        if not self.path:
            return False
            
        # Check direct path from last point to goal
        return not is_collision(self.path[-1], (x, y), obstacles)
        
    def traverse(self, current_position, radius_from_goal):
        """Find the next point along the path that is within the radius."""
        if not self.path:
            return None
            
        # Find the closest point on the path to current position
        closest_idx = 0
        min_dist = float('inf')
        for i, point in enumerate(self.path):
            dist = distance(current_position, point)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        # Look ahead on the path to find a point that's within the desired radius
        for i in range(closest_idx + 1, len(self.path)):
            if distance(current_position, self.path[i]) > radius_from_goal:
                return self.path[i]
                
        # If we're at the end of the path, return None to indicate completion
        return None
    
def distance(p1, p2):
    """Return Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def nearest_node(tree, random_point):
    """Find the node in the tree that is closest to the random point."""
    nearest = tree[0]
    min_dist = distance(nearest.point, random_point)
    for node in tree:
        d = distance(node.point, random_point)
        if d < min_dist:
            min_dist = d
            nearest = node
    return nearest

def steer(from_point, to_point, step_size):
    """
    Move from 'from_point' toward 'to_point' by a fixed step size.
    If the distance is less than step_size, return to_point.
    """
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    d = math.hypot(dx, dy)
    if d <= step_size:
        return to_point
    theta = math.atan2(dy, dx)
    new_x = from_point[0] + step_size * math.cos(theta)
    new_y = from_point[1] + step_size * math.sin(theta)
    return (new_x, new_y)

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
            if math.hypot(p1[0] - ox, p1[1] - oy) <= r:
                return True
            continue

        # Parameter t for the projection of the circle center onto the line p1->p2.
        t = ((ox - p1[0]) * dx + (oy - p1[1]) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # Clamp t to the [0, 1] segment
        closest_x = p1[0] + t * dx
        closest_y = p1[1] + t * dy
        if math.hypot(closest_x - ox, closest_y - oy) <= r:
            return True
    return False

def construct_path(goal_node):
    """Reconstruct the path from the start node to the goal node."""
    path = []
    node = goal_node
    while node is not None:
        path.append(node.point)
        node = node.parent
    path.reverse()
    return path

# IMPORTANT NOTE: This controls the limits of our search
def rrt(start, goal, obstacles, limits=[-9, 9], step_size=0.5, max_iter=1000)-> List[Node] or None:
    """
    Run a basic RRT algorithm to find a collision-free path from start to goal.
    
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

    try:
        # Set Initial parameters
        rrt_star = RRTStar(
            start=start,
            goal=goal,
            rand_area=limits,
            obstacle_list=obstacles,
            expand_dis=1,
            robot_radius=0.8)
        path = rrt_star.planning()
        return path
    except Exception as e:
        print(f'Loc 102: the exception that occured is {e}')
                    
    # IMPORTANTE TODO: Make sure we have a path somehow
    return None
