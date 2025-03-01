import math
import random
from typing import List

# A simple node class to represent points in the tree.
class Node:
    def __init__(self, point, parent=None):
        self.point = point  # (x, y)
        self.parent = parent  # reference to the parent Node

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

def rrt(start, goal, obstacles, x_limits, y_limits, step_size=0.5, max_iter=1000)-> List[Node] or None:
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
    tree = [Node(start)]
    
    for _ in range(max_iter):
        # Sample a random point in the given space.
        rand_point = (random.uniform(x_limits[0], x_limits[1]),
                      random.uniform(y_limits[0], y_limits[1]))
        
        # Find the nearest node in the current tree.
        nearest = nearest_node(tree, rand_point)
        new_point = steer(nearest.point, rand_point, step_size)
        
        # Check if the edge from nearest.point to new_point is free of obstacles.
        if not is_collision(nearest.point, new_point, obstacles):
            new_node = Node(new_point, nearest)
            tree.append(new_node)
            
            # If the new point is close enough to the goal, try to connect directly.
            if distance(new_node.point, goal) <= step_size:
                if not is_collision(new_node.point, goal, obstacles):
                    goal_node = Node(goal, new_node)
                    tree.append(goal_node)
                    return construct_path(goal_node)
                    
    return None

