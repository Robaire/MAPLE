from math import hypot


def get_distance_between_points(x1, y1, x2, y2):
    return hypot(x1 - x2, y1 - y2)



def is_collision(agent_position, goal_position, obstacles) -> bool:
    """
    Check if the straight line from agent_position to goal_position
    intersects any circular obstacles.

    agent_position: (x, y)
    goal_position: (x, y)
    obstacles: list of (ox, oy, radius)

    Returns True if collision detected, else False.
    """
    for ox, oy, r in obstacles:
        # Vector from agent to goal
        dx = goal_position[0] - agent_position[0]
        dy = goal_position[1] - agent_position[1]

        # If agent and goal are the same point, just check collision at that point
        if dx == 0 and dy == 0:
            if hypot(agent_position[0] - ox, agent_position[1] - oy) <= r:
                return True
            continue

        # Project center of obstacle onto the line
        t = ((ox - agent_position[0]) * dx + (oy - agent_position[1]) * dy) / (
            dx * dx + dy * dy
        )
        t = max(0, min(1, t))  # Clamp t to the [0, 1] segment only

        closest_x = agent_position[0] + t * dx
        closest_y = agent_position[1] + t * dy

        # If the closest point is within the obstacle radius, it's a collision
        if hypot(closest_x - ox, closest_y - oy) <= r:
            return True

    return False

def is_possible_to_reach(x, y, obstacles):
    """Check if x, y is possible to reach
    ie not in an obstacle"""

    for ox, oy, r in obstacles:
        if hypot(x - ox, y - oy) <= r:
            return False
    return True

def is_in_obstacle(x, y, obstacle):
    """Check if x, y is in an obstacle
    ie not possible to reach"""

    ox, oy, r = obstacle
    if hypot(x - ox, y - oy) <= r:
        return True
    return False

def is_path_collision(path, obstacles) -> bool:
    """
    Check if any segment in the path collides with circular obstacles.

    path: list of (x, y) tuples representing waypoints
    obstacles: list of (ox, oy, radius) tuples

    Returns True if any segment collides with an obstacle, else False.
    """
    for i in range(len(path) - 1):
        if is_collision(path[i], path[i + 1], obstacles):
            return True
    return False