from maple.navigation.constants import radius_from_goal_location
from math import hypot
import carla
import numpy as np


# This is the path class which will be used to represent a path and have helper functions for navigation to be able to use it
class Path:
    """This is the parent class to be able to extend"""

    def __init__(self, target_locations):
        """This initializes a path

        Args:
            target_locations (tuple): Provide a list of locations like this [(0, 0), (1, 2), (3, 3), (6, 1)] and a path will be generated to go from one to the next
        """

        # Initialize the path with the points to go through
        # IMPORTANT TODO: Make sure we never run out of target locations
        self.path = target_locations

        print("number of target goal locations: ", len(target_locations))

        # Save the start location
        # self.start = target_locations[0]

        # This is the current checkpoint so that we are always progressing on the path
        self.current_check_point_index = 0

        # Save the end location
        # self.end = target_locations[-1]

        self.nearby_goals = None

    def get_full_path(self):
        return self.path

    def is_path_collision_free(self, obstacles):
        """Check if the current path is free of collisions with given obstacles."""
        if not self.path or len(self.path) < 2:
            return False

        for i in range(len(self.path) - 1):
            if is_collision(self.path[i], self.path[i + 1], obstacles):
                return False
        return True

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
        return hypot(x1 - x2, y1 - y2)
    
    # def update_and_pick_closest_goal(self, rover_position, distance_threshold=2.0):
    #     """
    #     Remove the current reached goal, then pick the closest next one.
    #     """

    #     if not self.path or len(self.path) == 0:
    #         return None

    #     x_rover, y_rover = rover_position

    #     # 1. First, find the single goal we just reached (within threshold)
    #     reached_goals = [
    #         (x, y) for (x, y) in self.path
    #         if self.get_distance_between_points(x, y, x_rover, y_rover) <= distance_threshold
    #     ]

    #     # 2. If found any goals within threshold, remove them
    #     for goal in reached_goals:
    #         print(f"Removing reached goal: {goal}")
    #         self.path.remove(goal)

    #     # 3. If path is now empty, return None
    #     if len(self.path) == 0:
    #         return None

    #     # 4. Pick the closest goal remaining
    #     closest_goal = min(
    #         self.path,
    #         key=lambda goal: self.get_distance_between_points(goal[0], goal[1], x_rover, y_rover)
    #     )

    #     return closest_goal

    def find_closest_goal(self, rover_position, pop_if_found=True):
        """
        Find the closest goal to the rover position.
        If pop_if_found=True, also remove it from the list.
        """
        if not self.path:
            return None

        x_rover, y_rover = rover_position

        # Find closest
        closest_goal = min(
            self.path,
            key=lambda goal: (goal[0] - x_rover) ** 2 + (goal[1] - y_rover) ** 2
        )

        if pop_if_found:
            print(f"[Path] Removing reached goal {closest_goal}")
            self.path.remove(closest_goal)

        return closest_goal


    # def pick_goal(self, estimate, nearby_goals, input_data, orb):
    #     # 1. Detect features and calculate scores for each view
    #     camera_views = {
    #         "FrontLeft": input_data["Grayscale"][carla.SensorPosition.FrontLeft],
    #         "BackLeft": input_data["Grayscale"][carla.SensorPosition.BackLeft],
    #         "Left": input_data["Grayscale"][carla.SensorPosition.Left],
    #         "Right": input_data["Grayscale"][carla.SensorPosition.Right],
    #     }

    #     camera_scores = {}
    #     for name, img in camera_views.items():
    #         if img is not None:
    #             keypoints = orb.detect(img, None)
    #             if keypoints:
    #                 responses = [kp.response for kp in keypoints]
    #                 avg_response = np.mean(responses)
    #                 camera_scores[name] = avg_response
    #             else:
    #                 camera_scores[name] = 0
    #         else:
    #             camera_scores[name] = 0

    #     print("Camera scores:", camera_scores)

    #     # 2. Sort cameras by score (best first)
    #     sorted_cameras = sorted(camera_scores.items(), key=lambda x: x[1], reverse=True)
    #     sorted_camera_names = [cam[0] for cam in sorted_cameras]
    #     print(f"Sorted cameras: {sorted_camera_names}")

    #     # 3. For each camera in sorted order, try to find a goal
    #     pose_inv = np.linalg.inv(estimate)  # world -> robot

    #     for best_camera in sorted_camera_names:
    #         print(f"Trying camera: {best_camera}")

    #         goal_scores = []
    #         for goal in nearby_goals:
    #             goal_world = np.array([goal[0], goal[1], 0, 1])  # (x, y, z=0, 1)
    #             goal_robot = pose_inv @ goal_world
    #             x_r, y_r, z_r, _ = goal_robot

    #             # Direction check based on camera
    #             direction_match = False
    #             if best_camera == "FrontLeft" and y_r > 0:
    #                 direction_match = True
    #             elif best_camera == "BackLeft" and y_r < 0:
    #                 direction_match = True
    #             elif best_camera == "Left" and x_r > 0:
    #                 direction_match = True
    #             elif best_camera == "Right" and x_r < 0:
    #                 direction_match = True

    #             if direction_match:
    #                 distance = np.linalg.norm([x_r, y_r])
    #                 goal_scores.append((distance, goal))  # Prefer closer goals

    #         if goal_scores:
    #             # If we found any matching goals for this camera
    #             goal_scores.sort()  # Sort by distance
    #             best_goal = goal_scores[0][1]
    #             print(f"Selected goal: {best_goal} using camera {best_camera}")
    #             return best_goal

    #     # If no goal found in any camera direction
    #     print("No suitable goal found in any camera direction.")
    #     return None

    # First, let's modify the Path class to include a better pick_goal method
    def pick_goal(self, estimate, nearby_goals, input_data, orb_detector):
        """
        Select the best goal based on ORB feature quality in different camera views.
        
        Args:
            estimate: Current pose estimate (world -> rover transform)
            nearby_goals: List of candidate goals to choose from
            input_data: Dictionary containing camera images
            orb_detector: ORB detector instance
            
        Returns:
            Selected (x, y) goal or None if no suitable goal found
        """
        if not nearby_goals:
            print("[Path] No nearby goals available to pick from")
            return None
            
        # 1. Detect features and calculate scores for each camera view
        camera_views = {
            "FrontLeft": input_data["Grayscale"].get(carla.SensorPosition.FrontLeft),
            "FrontRight": input_data["Grayscale"].get(carla.SensorPosition.FrontRight),
            "BackLeft": input_data["Grayscale"].get(carla.SensorPosition.BackLeft),
            "BackRight": input_data["Grayscale"].get(carla.SensorPosition.BackRight),
            "Left": input_data["Grayscale"].get(carla.SensorPosition.Left),
            "Right": input_data["Grayscale"].get(carla.SensorPosition.Right),
        }
        
        camera_scores = {}
        for name, img in camera_views.items():
            if img is not None:
                keypoints = orb_detector.detect(img, None)
                if keypoints:
                    # Calculate feature quality score based on:
                    # 1. Number of keypoints (more is better)
                    # 2. Average response strength (higher is better)
                    responses = [kp.response for kp in keypoints]
                    avg_response = np.mean(responses)
                    num_keypoints = len(keypoints)
                    # Combined score: normalize both factors
                    camera_scores[name] = (avg_response * num_keypoints) / 1000
                    print(f"[Path] Camera {name}: {num_keypoints} keypoints, avg response={avg_response:.4f}, score={camera_scores[name]:.2f}")
                else:
                    camera_scores[name] = 0
                    print(f"[Path] Camera {name}: No keypoints detected")
            else:
                camera_scores[name] = 0
        
        # 2. Sort cameras by score (best first)
        sorted_cameras = sorted(camera_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_camera_names = [cam[0] for cam in sorted_cameras]
        print(f"[Path] Camera ranking: {sorted_camera_names}")
        
        # 3. Transform goals from world to robot frame
        pose_inv = np.linalg.inv(estimate)  # world -> robot
        
        # Store goals with their scores
        goal_scores = []
        
        for goal in nearby_goals:
            goal_world = np.array([goal[0], goal[1], 0, 1])  # (x, y, z=0, 1)
            goal_robot = pose_inv @ goal_world
            x_r, y_r, z_r, _ = goal_robot
            
            # Calculate angle to goal in robot frame
            angle = np.arctan2(y_r, x_r) * 180 / np.pi
            
            # Determine which camera this goal aligns with
            camera_match = None
            if -45 <= angle <= 45:  # Front
                camera_match = "FrontLeft"
            elif 45 < angle <= 135:  # Left
                camera_match = "Left"
            elif -135 <= angle < -45:  # Right
                camera_match = "Right"
            elif angle > 135 or angle < -135:  # Back
                camera_match = "BackLeft"
                
            if camera_match:
                # Get camera's score
                camera_score = camera_scores.get(camera_match, 0)
                
                # Calculate distance (closer is better)
                distance = np.sqrt(x_r**2 + y_r**2)
                
                # Score based on both camera quality and distance
                # We want high camera score and low distance
                goal_score = camera_score / (distance + 0.1)  # Add 0.1 to avoid division by zero
                
                goal_scores.append((goal_score, goal, camera_match))
                print(f"[Path] Goal {goal} aligns with {camera_match}, score={goal_score:.2f}")
        
        if goal_scores:
            # Sort by score and pick the best
            goal_scores.sort(reverse=True)  # Higher score is better
            best_goal = goal_scores[0][1]
            best_camera = goal_scores[0][2]
            print(f"[Path] Selected goal {best_goal} using camera {best_camera}")
            return best_goal
        
        # Fallback - if no goal with a good camera match was found
        print("[Path] No suitable goal found with good camera view")
        return None
    def find_nearby_goals(self, current_position):
        """
        Given the current_position (x, y), find all waypoints within 7m.
        If none are found, increase search radius to 10m.
        Returns a list of nearby waypoints.
        """
        x_current, y_current = current_position

        # First try with 7 meters
        nearby = [
            (x, y) for (x, y) in self.path
            if self.get_distance_between_points(x_current, y_current, x, y) <= 3.0
        ]

        if not nearby:
            # If none found, expand radius to 10 meters
            nearby = [
                (x, y) for (x, y) in self.path
                if self.get_distance_between_points(x_current, y_current, x, y) <= 5.0
            ]

        if not nearby:
            # If none found, expand radius to 10 meters
            nearby = [
                (x, y) for (x, y) in self.path
                if self.get_distance_between_points(x_current, y_current, x, y) <= 7.0
            ]

        self.nearby_goals = nearby

        return nearby

    def get_next_goal_location(self):
        """
        This function doesnt check for any obstacles and returns the next goal location or None if there is None
        Try not to use this function, it's main purpose is to satisfy previous code
        """

        # Get the next point
        self.current_check_point_index += 1

        # Check if there is a point and return None if there isnt
        if self.path is None or self.current_check_point_index >= len(self.path):
            return None

        # Return the next point
        return self.path[self.current_check_point_index]

    def forced_traverse(self, rover_position, obstacles=[]):
        """
        Same as traverse but force skips a goal point at the start
        """

        self.current_check_point_index += 1

        return self.traverse(rover_position, obstacles)

    def traverse(self, rover_position, obstacles=[]):
        """
        This function takes the rover position and radius from goal location to be considered at that location
        """

        # Handle no path and longer index correctly
        if self.path is None or self.current_check_point_index >= len(self.path):
            return None

        # Increment the goal check point until we are not considered there or in an obstacle
        while self.get_distance_between_points(
            *rover_position, *self.path[self.current_check_point_index]
        ) < radius_from_goal_location or not is_possible_to_reach(
            *self.path[self.current_check_point_index], obstacles
        ):
            self.current_check_point_index += 1

            if self.current_check_point_index >= len(self.path):
                return None

        return self.path[self.current_check_point_index]


def is_possible_to_reach(x, y, obstacles):
    """Check if x, y is possible to reach
    ie not in an obstacle"""

    # print(f'the obstacles are {obstacles}')
    for ox, oy, r in obstacles:
        if hypot(x - ox, y - oy) <= r:
            return False
    return True


def is_collision(p1, p2, obstacles) -> bool:
    """
    Check if the line segment from p1 to p2 intersects any circular obstacles.
    Each obstacle is defined as a tuple (ox, oy, radius).
    """
    for ox, oy, r in obstacles:
        # Vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # If p1 and p2 are the same point, check that point only.
        if dx == 0 and dy == 0:
            if hypot(p1[0] - ox, p1[1] - oy) <= r:
                return True
            continue

        # Parameter t for the projection of the circle center onto the line p1->p2.
        t = ((ox - p1[0]) * dx + (oy - p1[1]) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # Clamp t to the [0, 1] segment
        closest_x = p1[0] + t * dx
        closest_y = p1[1] + t * dy
        if hypot(closest_x - ox, closest_y - oy) <= r:
            return True
    return False
