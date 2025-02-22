import unittest
import random
from maple.navigation.path import Path
from maple.navigation.local_path import LocalPath
from maple.navigation.rrt_path import RRTPath, Node

class TestPath(unittest.TestCase):
    def setUp(self):
        # Create a simple path with three checkpoints.
        self.target_locations = [(0, 0), (1, 1), (2, 2)]
        self.path = Path(self.target_locations)
    
    def test_get_start(self):
        self.assertEqual(self.path.get_start(), (0, 0))
    
    def test_get_end(self):
        self.assertEqual(self.path.get_end(), (2, 2))
    
    def test_get_distance_between_points(self):
        # Distance between (0,0) and (3,4) should be 5.
        self.assertAlmostEqual(self.path.get_distance_between_points(0, 0, 3, 4), 5)
    
    def test_traverse(self):
        # With rover at (0,0), the first target (0,0) is already "reached"
        # so traverse should advance to the next checkpoint (1,1).
        rover_position = (0, 0)
        next_checkpoint = self.path.traverse(rover_position, radius_from_goal_location=0.5)
        self.assertEqual(next_checkpoint, (1, 1))
    
    def test_is_collision(self):
        # Check collision: line from (0,0) to (2,0) with an obstacle at (1,0) radius 0.5.
        collision = self.path.is_collision((0, 0), (2, 0), obstacles=[(1, 0, 0.5)])
        self.assertTrue(collision)
        # Check no collision: obstacle not in the path.
        no_collision = self.path.is_collision((0, 0), (2, 0), obstacles=[(1, 1, 0.5)])
        self.assertFalse(no_collision)
    
    def test_is_possible_to_reach(self):
        # Without obstacles, any point is reachable.
        self.assertTrue(self.path.is_possible_to_reach(1, 1))
        # With an obstacle at (1,1) that covers the point, it should return False.
        self.assertFalse(self.path.is_possible_to_reach(1, 1, obstacles=[(1, 1, 1)]))
        # With an obstacle far away, the point is reachable.
        self.assertTrue(self.path.is_possible_to_reach(1, 1, obstacles=[(5, 5, 1)]))
    
    def test_remove_current_goal_location(self):
        original_length = len(self.path.path)
        # Set the current checkpoint index to 1 (point (1,1)).
        self.path.current_check_point_index = 1
        self.path.remove_current_goal_location()
        # The path should have one less element.
        self.assertEqual(len(self.path.path), original_length - 1)
        # The removed point should no longer be in the path.
        self.assertNotIn((1, 1), self.path.path)

class TestLocalPath(unittest.TestCase):
    def setUp(self):
        # Create a LocalPath with three checkpoints.
        self.target_locations = [(0, 0), (1, 1), (2, 2)]
        self.local_path = LocalPath(self.target_locations)
    
    def test_traverse_no_obstacles(self):
        # When obstacles is None, LocalPath.traverse should behave like Path.traverse.
        rover_position = (0, 0)
        next_checkpoint = self.local_path.traverse(rover_position, obstacles=None, radius_from_goal_location=0.5)
        self.assertEqual(next_checkpoint, (1, 1))
    
    # Testing obstacle behavior in LocalPath.traverse can be tricky because
    # the loop is based on reachability of the rover's position. For now, we test
    # the no-obstacle case to ensure it calls the parent traverse correctly.

class TestRRTPath(unittest.TestCase):
    def setUp(self):
        # RRTPath requires exactly two target locations: start and goal.
        self.start = (0, 0)
        self.goal = (5, 5)
        self.rrt_path = RRTPath([self.start, self.goal])
        # For reproducibility in the random sampling.
        random.seed(42)
    
    def test_construct_path(self):
        # Create a simple chain of nodes: (0,0) -> (1,1) -> (2,2)
        node1 = Node((0, 0))
        node2 = Node((1, 1), parent=node1)
        node3 = Node((2, 2), parent=node2)
        constructed = self.rrt_path.construct_path(node3)
        self.assertEqual(constructed, [(0, 0), (1, 1), (2, 2)])
    
    def test_nearest_node(self):
        # Build a small tree of nodes.
        nodes = [Node((0, 0)), Node((5, 5)), Node((1, 1))]
        # For the point (1.2, 1.2), the nearest should be (1,1).
        nearest = self.rrt_path.nearest_node(nodes, (1.2, 1.2))
        self.assertEqual(nearest.point, (1, 1))
    
    def test_steer(self):
        # Test moving from (0,0) towards (2,0) with a step of 1 should give (1,0).
        new_point = RRTPath.steer((0, 0), (2, 0), 1)
        self.assertAlmostEqual(new_point[0], 1)
        self.assertAlmostEqual(new_point[1], 0)
        # If the step size is larger than the distance, it should return the target.
        new_point = RRTPath.steer((0, 0), (1, 0), 2)
        self.assertEqual(new_point, (1, 0))
    
    def test_rrt(self):
        # Run the RRT algorithm with no obstacles in a large sampling region.
        obstacles = []
        x_limits = (-10, 10)
        y_limits = (-10, 10)
        result_path = self.rrt_path.rrt(self.start, self.goal, obstacles, x_limits, y_limits,
                                        step_size=1, max_iter=5000)
        # We expect a valid path from start to goal.
        self.assertIsNotNone(result_path)
        self.assertEqual(result_path[0], self.start)
        self.assertEqual(result_path[-1], self.goal)
    
    def test_is_collision_in_rrt(self):
        # Verify that the collision detection inherited from Path works correctly.
        collision = self.rrt_path.is_collision((0, 0), (2, 0), obstacles=[(1, 0, 0.5)])
        self.assertTrue(collision)
        no_collision = self.rrt_path.is_collision((0, 0), (2, 0), obstacles=[(1, 1, 0.5)])
        self.assertFalse(no_collision)

if __name__ == '__main__':
    unittest.main()
