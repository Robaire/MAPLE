from maple.navigation.constants import radius_from_goal_location
from math import hypot
import carla
import numpy as np
from maple.utils import carla_to_pytransform
from maple.navigation.utils import get_distance_between_points
import numpy as np
import cv2
from maple.navigation.utils import is_collision

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

        # self.global_goals = [(7, 7), (-7, -7), (7, -7), (-7, 7)]

        self.last_global_goal_frame = 0  # Frame counter
        self.global_goal_interval = 2500  # Every 500 frames

        print("number of target goal locations: ", len(target_locations))

        # This is the current checkpoint so that we are always progressing on the path
        self.current_check_point_index = 0

        self.nearby_goals = None

        self.goal = (0, 0)
        self.dynamic_goal = (0, 0)

    def get_full_path(self):
        return self.path

    def __call__(self):
        return None

    def get_start(self):
        """
        This function returns the start location
        """

        return self.start

    def get_end(self):
        return self.end

    def find_closest_goal(
        self,
        rover_position,
        estimate,
        input_data,
        agent,
        pop_if_found=True,
        obstacles=[],
    ):
        """
        Find the best goal prioritizing direction with highest weight:
        1. Determine valid directions based on camera entropy thresholds
        2. Categorize goals by direction (forward, left, right, back)
        3. Select the goal with the highest weight, prioritizing forward > left/right > back if tie
        4. Fall back to closest goal overall if no valid directions
        """

        print(f"[DEBUG] find_closest_goal called with pop_if_found={pop_if_found}")
        print(f"[DEBUG] Current path has {len(self.path)} waypoints")
    
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
                print(
                    f"[Camera] {name}: Entropy = {entropy:.2f}, Gradient = {gradient:.2f}"
                )
                camera_entropies[name] = entropy
            else:
                print(f"[Camera] {name} not found")
                camera_entropies[name] = 0.0

        # 2. Rover heading vectors
        forward_vec = estimate[:2, 0]
        norm = np.linalg.norm(forward_vec)
        forward_vec = forward_vec / norm if norm > 0 else np.array([1.0, 0.0])

        left_vec = estimate[:2, 1]
        norm = np.linalg.norm(left_vec)
        left_vec = left_vec / norm if norm > 0 else np.array([0.0, 1.0])

        right_vec = -left_vec
        back_vec = -forward_vec

        direction_vectors = {
            "forward": forward_vec,
            "left": left_vec,
            "right": right_vec,
            "back": back_vec,
        }

        forward_threshold = 0.707

        camera_to_direction = {
            "FrontLeft": "forward",
            "Left": "left",
            "Right": "right",
            "BackLeft": "back",
        }

        valid_directions = {
            camera_to_direction[cam]: (entropy > 3.25)
            for cam, entropy in camera_entropies.items()
            if cam in camera_to_direction
        }

        print(f"[Filter] Valid directions (entropy > 4): {valid_directions}")

        candidates = {dir_name: [] for dir_name in direction_vectors.keys()}

        for goal in self.path:
            if len(goal) >= 3:
                x, y, w = goal[0], goal[1], goal[2]
            else:
                x, y = goal[0], goal[1]
                w = 1.0

            goal_vec = np.array([x - x_rover, y - y_rover])
            distance = np.linalg.norm(goal_vec)
            if distance < 0.001:
                continue

            goal_vec_norm = goal_vec / distance

            for dir_name, dir_vec in direction_vectors.items():
                dot = np.dot(dir_vec, goal_vec_norm)
                if dot > forward_threshold:
                    candidates[dir_name].append((goal, distance, w))
                    break

        valid_dir_has_goals = any(
            valid_directions.get(dir_name, False) and candidates[dir_name]
            for dir_name in direction_vectors
        )

        # Go through the cannidates and eliminate any that go through the lander
        for dir_name in candidates:
            for goal_distance_w in candidates[dir_name]:
                goal, distance, w = goal_distance_w

                # Extract the goal information
                if len(goal) == 2:
                    goal_x, goal_y = goal
                else:
                    goal_x, goal_y, _ = goal

                if is_collision(rover_position, (goal_x, goal_y), obstacles):
                    candidates[dir_name].remove(goal_distance_w)

        best_goal = None

        if valid_dir_has_goals:
            closest_goals_by_dir = {}
            for dir_name, goals in candidates.items():
                if not valid_directions.get(dir_name, False) or not goals:
                    continue
                closest_goal = min(goals, key=lambda g: g[1])
                closest_goals_by_dir[dir_name] = closest_goal

            # Print the selected goals and weights
            print("[Candidates]")
            for dir_name in ["forward", "left", "right", "back"]:
                if dir_name in closest_goals_by_dir:
                    g, d, w = closest_goals_by_dir[dir_name]
                    print(
                        f"  {dir_name.upper()}: Goal {g} | Distance {d:.2f} | Weight {w:.2f}"
                    )
                else:
                    print(f"  {dir_name.upper()}: No goal")

            # Find the highest weight(s)
            highest_weight = max(
                (w for (g, d, w) in closest_goals_by_dir.values()), default=None
            )

            if highest_weight is not None:
                # Filter goals with highest weight
                top_goals = {
                    dir_name: (g, d, w)
                    for dir_name, (g, d, w) in closest_goals_by_dir.items()
                    if w == highest_weight
                }

                # Prioritize forward > left/right > back
                for preferred_dir in ["forward", "left", "right", "back"]:
                    if preferred_dir in top_goals:
                        best_goal = top_goals[preferred_dir][0]
                        print(
                            f"[Result] Selected {preferred_dir.upper()} goal: {best_goal}"
                        )
                        break
        else:
            print("[Fallback] No good directional goals. Using closest overall.")
            best_goal = min(
                self.path,
                key=lambda goal: (goal[0] - x_rover) ** 2 + (goal[1] - y_rover) ** 2,
            )
            print(f"[Fallback Result] Closest goal overall: {best_goal}")

        if pop_if_found and best_goal in self.path:
            print(f"[Path] Removing reached goal {best_goal}")
            self.path.remove(best_goal)

        self.goal = best_goal
        return best_goal


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
                    print(
                        f"[Path] Camera {name}: {num_keypoints} keypoints, avg response={avg_response:.4f}, score={camera_scores[name]:.2f}"
                    )
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
                goal_score = camera_score / (
                    distance + 0.1
                )  # Add 0.1 to avoid division by zero

                goal_scores.append((goal_score, goal, camera_match))
                print(
                    f"[Path] Goal {goal} aligns with {camera_match}, score={goal_score:.2f}"
                )

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
            (x, y)
            for (x, y) in self.path
            if get_distance_between_points(x_current, y_current, x, y) <= 3.0
        ]

        if not nearby:
            # If none found, expand radius to 10 meters
            nearby = [
                (x, y)
                for (x, y) in self.path
                if get_distance_between_points(x_current, y_current, x, y) <= 5.0
            ]

        if not nearby:
            # If none found, expand radius to 10 meters
            nearby = [
                (x, y)
                for (x, y) in self.path
                if get_distance_between_points(x_current, y_current, x, y) <= 7.0
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
    
    def is_path_collision_free(self, agent_position, obstacles):
        """Check if the current path is free of collisions with given obstacles."""

        for i in range(len(self.path) - 1):
            if is_collision(agent_position, self.goal, obstacles):
                print("obstacles: ", obstacles)
                print(
                    "collisison detected between ", agent_position, " and ", self.goal
                )
                return False
        return True

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


def is_risky_area(
    image_gray: np.ndarray, entropy_threshold=4.0, gradient_energy_threshold=1e6
) -> bool:
    """Determine if an area is risky (featureless) based on entropy and gradient energy."""
    entropy = compute_image_entropy(image_gray)
    gradient_energy = compute_gradient_energy(image_gray)

    print(
        f"[Monitoring] Entropy: {entropy:.2f}, Gradient Energy: {gradient_energy:.2e}"
    )

    # if entropy < entropy_threshold or gradient_energy < gradient_energy_threshold:
    #     return True  # Risky zone detected
    # return False

    return entropy, gradient_energy
