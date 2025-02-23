import math
import random
from typing import List

from maple.navigation.path import Path

# A simple node class to represent points in the tree for the damn rrt
class Node:
    def __init__(self, point, parent=None):
        self.point = point  # (x, y)
        self.parent = parent  # reference to the parent Node

class RRTPath(Path):
    """ This is the random tree search path to get from point A to point B when the straight path has collisions

    Args:
        Path (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, target_locations, obstacles=None):
        """ Only have 2 locations for the target locations, the start location and the end locations

        Args:
            target_locations (_type_): _description_
        """

        assert len(target_locations) == 2

        self.path = self.rrt(target_locations[0], target_locations[1], obstacles)

        super().__init__(target_locations)

    def construct_path(self, goal_node):
        """Reconstruct the path from the start node to the goal node."""
        path = []
        node = goal_node
        while node is not None:
            path.append(node.point)
            node = node.parent
        path.reverse()
        return path
    
    def nearest_node(self, tree, random_point):
        """Find the node in the tree that is closest to the random point."""
        nearest = tree[0]
        min_dist = self.distance(*nearest.point, *random_point)
        for node in tree:
            d = self.distance(*node.point, *random_point)
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
    
    def rrt(self, start, goal, obstacles, x_limits, y_limits, step_size=0.5, max_iter=1000)-> List[Node] or None:
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
            list: The collision-free path as a list of (x, y) points if found, else None. (Also sets the path for later usage)
        """
        tree = [Node(start)]
        
        for _ in range(max_iter):
            # Sample a random point in the given space.
            rand_point = (random.uniform(x_limits[0], x_limits[1]),
                        random.uniform(y_limits[0], y_limits[1]))
            
            # Find the nearest node in the current tree.
            nearest = self.nearest_node(tree, rand_point)
            new_point = self.steer(nearest.point, rand_point, step_size)
            
            # Check if the edge from nearest.point to new_point is free of obstacles.
            if not self.is_collision(nearest.point, new_point, obstacles):
                new_node = Node(new_point, nearest)
                tree.append(new_node)
                
                # If the new point is close enough to the goal, try to connect directly.
                if self.distance(new_node.point, goal) <= step_size:
                    if not self.is_collision(new_node.point, goal, obstacles):
                        goal_node = Node(goal, new_node)
                        tree.append(goal_node)
                        return self.construct_path(goal_node)
        
        # IMPORTANT TODO: Handle this better if rrt doesnt work
        print(f'couldnt find path')
        return None



