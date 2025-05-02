from maple.navigation.constants import radius_from_goal_location
from math import hypot
import carla
import numpy as np
from maple.utils import carla_to_pytransform



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
    

# This one does entropy stuff
    # def find_closest_goal(self, rover_position, estimate, input_data, pop_if_found=True):
    #     """
    #     Find the closest goal to the rover position.
    #     If pop_if_found=True, also remove it from the list.
    #     """
    #     if not self.path:
    #         return None
        
    #     camera_views = {
    #             "FrontLeft": input_data["Grayscale"].get(carla.SensorPosition.FrontLeft),
    #             "BackLeft": input_data["Grayscale"].get(carla.SensorPosition.BackLeft),
    #             "Left": input_data["Grayscale"].get(carla.SensorPosition.Left),
    #             "Right": input_data["Grayscale"].get(carla.SensorPosition.Right),
    #         }
        
    #     camera_scores = {}
    #     for name, img in camera_views.items():
    #             if img is not None:
    #                 entropy, gradient = is_risky_area(img)
    #                 print("camera ", name, ' entropy ', entropy, ' gradient ', gradient)
    #                 # camera_scores[name] = is_risky_area(img)
    #             else:
    #                 print("camera not found")
    #                 # camera_scores[name] = 0

    #     # sorted_cameras = sorted(camera_scores.items(), key=lambda x: x[1], reverse=True)
    #     # sorted_camera_names = [cam[0] for cam in sorted_cameras]
    #     # print(f"[Path] Camera ranking: {sorted_camera_names}")

    #     x_rover, y_rover = rover_position

    #     # print("finding closest goal using find_closest_goal function in path.py")

    #     # # Find closest
    #     # closest_goal = min(
    #     #     self.path,
    #     #     key=lambda goal: (goal[0] - x_rover) ** 2 + (goal[1] - y_rover) ** 2
    #     # )


    #     # Get forward direction from the estimate
    #     forward_vec = estimate[:2, 0]  # +X direction (slice x and y)
    #     left_vec = estimate[:2, 1]
    #     forward_vec = forward_vec / np.linalg.norm(forward_vec)  # Normalize

    #     print(f"Forward vector: {forward_vec}")

    #     # Find goals that are in the forward direction
    #     forward_goals = []
    #     for goal in self.path:
    #         goal_vec = np.array([goal[0] - x_rover, goal[1] - y_rover])
    #         if np.linalg.norm(goal_vec) == 0:
    #             continue  # Skip if already at the goal
    #         goal_vec_norm = goal_vec / np.linalg.norm(goal_vec)
    #         dot = np.dot(forward_vec, goal_vec_norm)
            
    #         if dot > 0.5:  # Tunable threshold: 1.0 is perfectly forward, 0.5 is 60 degrees cone
    #             forward_goals.append(goal)

    #     if not forward_goals:
    #         print("[Path] No forward goals found! Falling back to closest goal.")
    #         forward_goals = self.path  # If nothing ahead, fall back to any goal

    #     print("forward goals: ", forward_goals)

    #     # Find the closest among forward goals
    #     closest_goal = min(
    #         forward_goals,
    #         key=lambda goal: (goal[0] - x_rover) ** 2 + (goal[1] - y_rover) ** 2
    #     )

    #     if pop_if_found:
    #         print(f"[Path] Removing reached goal {closest_goal}")
    #         self.path.remove(closest_goal)

    #     return closest_goal

# # This one picks forwrad goal
#     def find_closest_goal(self, rover_position, estimate, input_data, agent, pop_if_found=True):
#         """
#         Find the closest goal to the rover position that is in the forward direction.
#         Uses camera transform to reliably determine the forward direction.
#         Includes robust handling of edge cases and fallbacks.
#         """
#         if not self.path:
#             return None
        
#         x_rover, y_rover = rover_position
        
#         # Get the camera transform relative to the rover
#         try:
#             camera_rover = carla_to_pytransform(
#                 agent.get_camera_position(carla.SensorPosition.FrontLeft)
#             )
            
#             # Extract the forward direction from the camera transform
#             forward_vec = camera_rover[:2, 0]  # Extract x and y components
            
#             # Normalize the forward vector
#             norm = np.linalg.norm(forward_vec)
#             if norm > 0:
#                 forward_vec = forward_vec / norm
#             else:
#                 raise ValueError("Camera forward vector has zero norm")
                
#             print(f"Using camera transform for forward direction: {forward_vec}")
#         except Exception as e:
#             # Fallback if camera transform is not available or invalid
#             print(f"Error using camera transform: {e}")
#             print("Falling back to estimate matrix for forward direction")
            
#             # Try using the estimate matrix as fallback
#             forward_vec = estimate[:2, 0]  # Use first column by default
#             norm = np.linalg.norm(forward_vec)
#             if norm > 0:
#                 forward_vec = forward_vec / norm
#             else:
#                 # Last resort fallback
#                 print("Warning: Both camera transform and estimate failed, using default forward")
#                 forward_vec = np.array([1.0, 0.0])  # Default to positive X
        
#         print(f"Rover position: {rover_position}")
#         print(f"Forward vector: {forward_vec}")
        
#         # Configuration parameters - can be adjusted as needed
#         forward_threshold = 0.5  # Cos of angle (0.5 = 60 degrees)
#         max_distance = float('inf')  # Option to limit max distance
        
#         # Find goals that are in the forward direction
#         forward_goals = []
#         goal_details = []  # For sorting and debugging
        
#         for goal in self.path:
#             goal_vec = np.array([goal[0] - x_rover, goal[1] - y_rover])
#             distance = np.linalg.norm(goal_vec)
            
#             if distance < 0.001:
#                 continue  # Skip if already at this goal
                
#             if distance > max_distance:
#                 continue  # Skip if beyond max distance
                
#             goal_vec_norm = goal_vec / np.linalg.norm(goal_vec)
#             dot = np.dot(forward_vec, goal_vec_norm)
            
#             # Include this goal info for debugging
#             goal_details.append({
#                 'goal': goal,
#                 'distance': distance,
#                 'dot': dot,
#                 'is_forward': dot > forward_threshold
#             })
            
#             if dot > forward_threshold:
#                 forward_goals.append(goal)
        
#         # Sort and print goal details for debugging
#         goal_details.sort(key=lambda x: x['distance'])
#         print("\nAll goals (sorted by distance):")
#         for detail in goal_details:
#             print(f"  Goal {detail['goal']}: distance={detail['distance']:.2f}, " +
#                 f"dot={detail['dot']:.3f}, considered_forward={detail['is_forward']}")
        
#         print(f"Forward goals: {forward_goals}")
        
#         if not forward_goals:
#             print("[Path] No forward goals found! Applying fallback strategy.")
            
#             # Fallback strategy options:
#             # 1. Use the closest goal overall
#             # closest_goal = min(self.path, key=lambda goal: 
#             #     (goal[0] - x_rover) ** 2 + (goal[1] - y_rover) ** 2)
            
#             # 2. Use a wider angle for forward (e.g., 90 degrees = 0.0 threshold)
#             wider_threshold = 0.0  # Cos(90°) = 0.0
#             wider_forward_goals = [goal for goal in self.path if 
#                                 np.dot(forward_vec, (np.array(goal) - rover_position) / 
#                                     np.linalg.norm(np.array(goal) - rover_position)) > wider_threshold]
            
#             if wider_forward_goals:
#                 print(f"Using wider angle ({wider_threshold}) found {len(wider_forward_goals)} goals")
#                 closest_goal = min(wider_forward_goals, key=lambda goal: 
#                     (goal[0] - x_rover) ** 2 + (goal[1] - y_rover) ** 2)
#             else:
#                 # Last resort: just the closest
#                 print("Using closest goal regardless of direction")
#                 closest_goal = min(self.path, key=lambda goal: 
#                     (goal[0] - x_rover) ** 2 + (goal[1] - y_rover) ** 2)
#         else:
#             # Find the closest among forward goals
#             closest_goal = min(
#                 forward_goals,
#                 key=lambda goal: (goal[0] - x_rover) ** 2 + (goal[1] - y_rover) ** 2
#             )
        
#         print(f"[Path] Selected goal: {closest_goal}")
        
#         if pop_if_found and closest_goal in self.path:
#             print(f"[Path] Removing reached goal {closest_goal}")
#             self.path.remove(closest_goal)
        
#         return closest_goal
    def find_closest_goal(self, rover_position, estimate, input_data, agent, pop_if_found=True):
        """
        Find the best goal prioritizing direction:
        1. Prefer forward goals.
        2. If none, pick closer of left or right goals.
        3. If none, pick back goal.
        4. If none, fallback to any closest goal.
        Only directions with camera entropy > 4 are considered.
        """
        if not self.path:
            return None

        x_rover, y_rover = rover_position

        # 1. Compute camera entropies
        camera_views = {
            "FrontLeft": input_data["Grayscale"].get(carla.SensorPosition.FrontLeft),
            "BackLeft": input_data["Grayscale"].get(carla.SensorPosition.BackLeft),
            "Left": input_data["Grayscale"].get(carla.SensorPosition.Left),
            "Right": input_data["Grayscale"].get(carla.SensorPosition.Right),
        }

        camera_entropies = {}
        for name, img in camera_views.items():
            if img is not None:
                entropy, gradient = is_risky_area(img)
                print(f"[Camera] {name}: Entropy = {entropy:.2f}, Gradient = {gradient:.2f}")
                camera_entropies[name] = entropy
            else:
                print(f"[Camera] {name} not found")
                camera_entropies[name] = 0.0  # Missing cameras = low entropy

        # 2. Compute rover direction vectors
        try:
            camera_rover = carla_to_pytransform(
                agent.get_camera_position(carla.SensorPosition.FrontLeft)
            )
            base_forward = camera_rover[:2, 0]
            norm = np.linalg.norm(base_forward)
            if norm > 0:
                base_forward = base_forward / norm
            else:
                raise ValueError("Camera forward vector has zero norm")
            print(f"Using camera transform for forward direction: {base_forward}")
        except Exception as e:
            print(f"Error using camera transform: {e}")
            print("Falling back to estimate matrix for forward direction")
            base_forward = estimate[:2, 0]
            norm = np.linalg.norm(base_forward)
            if norm > 0:
                base_forward = base_forward / norm
            else:
                print("Warning: Using default forward")
                base_forward = np.array([1.0, 0.0])

        forward_vec = base_forward
        left_vec = np.array([-base_forward[1], base_forward[0]])   # 90° CCW
        right_vec = np.array([base_forward[1], -base_forward[0]])  # 90° CW
        back_vec = -base_forward                                   # 180°

        direction_vectors = {
            'forward': forward_vec,
            'left': left_vec,
            'right': right_vec,
            'back': back_vec,
        }

        forward_threshold = 0.5  # Cosine threshold for acceptable alignment

        # 3. Map cameras to directions
        camera_to_direction = {
            "FrontLeft": 'forward',
            "Left": 'left',
            "Right": 'right',
            "BackLeft": 'back',
        }

        # 4. Determine valid directions based on entropy
        valid_directions = {
            camera_to_direction[cam]: (entropy > 4)
            for cam, entropy in camera_entropies.items()
            if cam in camera_to_direction
        }

        print(f"[Filter] Valid directions (entropy > 4): {valid_directions}")

        # 5. Collect candidate goals per direction
        candidates = {dir_name: [] for dir_name in direction_vectors.keys()}

        for goal in self.path:
            goal_vec = np.array([goal[0] - x_rover, goal[1] - y_rover])
            distance = np.linalg.norm(goal_vec)

            if distance < 0.001:
                continue

            goal_vec_norm = goal_vec / distance

            for dir_name, dir_vec in direction_vectors.items():
                if not valid_directions.get(dir_name, False):
                    continue  # Skip invalid directions

                dot = np.dot(dir_vec, goal_vec_norm)
                if dot > forward_threshold:
                    candidates[dir_name].append((goal, distance))
                    break  # Match to only one direction

        # 6. Select goal based on priority: forward > (left/right) > back
        best_goal = None

        if candidates['forward']:
            best_goal = min(candidates['forward'], key=lambda x: x[1])[0]
            print(f"[Result] Selected best FORWARD goal: {best_goal}")
        elif candidates['left'] or candidates['right']:
            left_goal = min(candidates['left'], key=lambda x: x[1])[0] if candidates['left'] else None
            right_goal = min(candidates['right'], key=lambda x: x[1])[0] if candidates['right'] else None

            if left_goal and right_goal:
                left_distance = np.linalg.norm(np.array([left_goal[0] - x_rover, left_goal[1] - y_rover]))
                right_distance = np.linalg.norm(np.array([right_goal[0] - x_rover, right_goal[1] - y_rover]))
                best_goal = left_goal if left_distance <= right_distance else right_goal
            else:
                best_goal = left_goal or right_goal
            print(f"[Result] Selected best LEFT/RIGHT goal: {best_goal}")
        elif candidates['back']:
            best_goal = min(candidates['back'], key=lambda x: x[1])[0]
            print(f"[Result] Selected best BACK goal: {best_goal}")
        else:
            # fallback: pick closest goal overall
            print("[Fallback] No good directional goals. Using closest overall.")
            best_goal = min(
                self.path,
                key=lambda goal: (goal[0] - x_rover) ** 2 + (goal[1] - y_rover) ** 2
            )
            print(f"[Fallback Result] Closest goal overall: {best_goal}")

        # 7. Remove the goal if needed
        if pop_if_found and best_goal in self.path:
            print(f"[Path] Removing reached goal {best_goal}")
            self.path.remove(best_goal)

        return best_goal

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

import numpy as np
import cv2

def compute_image_entropy(image_gray: np.ndarray) -> float:
    """Compute the entropy of a grayscale image."""
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    hist = hist.ravel()
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Avoid log(0)
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def compute_gradient_energy(image_gray: np.ndarray) -> float:
    """Compute the sum of squared gradients (edge energy) of a grayscale image."""
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_sq = grad_x**2 + grad_y**2
    energy = np.sum(grad_mag_sq)
    return energy

def is_risky_area(image_gray: np.ndarray, entropy_threshold=4.0, gradient_energy_threshold=1e6) -> bool:
    """Determine if an area is risky (featureless) based on entropy and gradient energy."""
    entropy = compute_image_entropy(image_gray)
    gradient_energy = compute_gradient_energy(image_gray)
    
    print(f"[Monitoring] Entropy: {entropy:.2f}, Gradient Energy: {gradient_energy:.2e}")

    # if entropy < entropy_threshold or gradient_energy < gradient_energy_threshold:
    #     return True  # Risky zone detected
    # return False

    return entropy, gradient_energy
