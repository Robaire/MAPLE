import numpy as np

def generate_spiral(x0, y0, initial_radius=4.0, num_points=300, spiral_rate=0.3, frequency=12):
    """
    Generates a list of (x, y) points forming a spiral around (x0, y0).

    :param x0: X-coordinate of the center
    :param y0: Y-coordinate of the center
    :param initial_radius: Starting radius from the center
    :param num_points: Number of points in the spiral
    :param spiral_rate: Controls how quickly the spiral expands
    :param frequency: Controls how closely spaced the points are
    :return: List of (x, y) points forming the spiral
    """
    points = []
    for i in range(num_points):
        theta = i / frequency  # Angle in radians
        r = initial_radius + spiral_rate * theta  # Radius grows over time
        x = x0 + r * np.cos(theta)
        y = y0 + r * np.sin(theta)

        points.append((x, y))
    
    return points

def rotate_point(x, y, angle_degrees):
    """
    Rotate point (x, y) around the origin (0, 0) by angle_degrees in a standard
    mathematical positive rotation (counter-clockwise).
    """
    angle = np.radians(angle_degrees)
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    return x_rot, y_rot

def generate_lawnmower(x0, y0, width=20.0, height=20.0, spacing=2.0):
    """
    Generate a lawnmower (back-and-forth) path that covers a rectangular region
    of size width x height, centered at (x0, y0). Spacing determines the distance
    between successive "passes".
    """
    # Start from the top-left corner in local coordinates (-width/2, +height/2)
    # and move row by row downwards.
    half_w = width / 2.0
    half_h = height / 2.0

    points = []
    # Determine how many rows (back-and-forth lines) we'll have
    num_rows = int(np.ceil(height / spacing))

    for row in range(num_rows + 1):
        # Current y in local coordinates (top to bottom)
        y_local = half_h - row * spacing

        # If the row is even, move left-to-right; if odd, move right-to-left
        if row % 2 == 0:
            # left to right
            x_line = np.linspace(-half_w, half_w, num=10)  # e.g. 10 points per row
        else:
            # right to left
            x_line = np.linspace(half_w, -half_w, num=10)

        for x_local in x_line:
            # Shift back to global coordinates
            x_global = x0 + x_local
            y_global = y0 + y_local
            points.append((x_global, y_global))

    return points

def generate_flower_rays(x0, y0, min_radius=4.0, radius_step=2.0, num_petals=6, max_radius=12.0):
    """
    Generate points along 6 rays emanating from the center.
    Points start at min_radius and increase by radius_step.
    
    Parameters:
    - x0, y0: Center coordinates
    - min_radius: Starting radius in meters
    - radius_step: Increase in radius between points (meters)
    - num_petals: Number of rays/petals (default: 6)
    - max_radius: Maximum radius to reach
    
    Returns a list of (x,y) coordinates defining the path.
    """
    points = []
    
    # Calculate angles for each petal/ray
    petal_angles = [2 * np.pi * i / num_petals for i in range(num_petals)]
    
    # Calculate number of points per ray
    points_per_ray = int((max_radius - min_radius) / radius_step) + 1
    
    # Generate points - going ray by ray
    for angle in petal_angles:
        for i in range(points_per_ray):
            radius = min_radius + (i * radius_step)
            if radius <= max_radius:
                x = x0 + radius * np.cos(angle)
                y = y0 + radius * np.sin(angle)
                points.append((x, y))
    
    return points

def generate_flower(x0, y0, min_radius=2.0, max_radius=10.0, num_petals=6, points_per_petal=10, num_circles=5):
    """
    Generate a flower pattern with specified number of petals.
    The path starts at min_radius from center (x0, y0) and expands outward in each petal direction.
    
    Parameters:
    - x0, y0: Center coordinates
    - min_radius: Minimum radius (inner boundary) of the flower
    - max_radius: Maximum radius (outer boundary) of the flower
    - num_petals: Number of petals (directions)
    - points_per_petal: Number of points along each petal ray
    - num_circles: Number of concentric circles to create
    
    Returns a list of (x,y) coordinates defining the path.
    """
    points = []
    
    # Start at the minimum radius on the first petal
    initial_angle = 0
    start_x = x0 + min_radius * np.cos(initial_angle)
    start_y = y0 + min_radius * np.sin(initial_angle)
    points.append((start_x, start_y))
    
    # For each concentric circle
    for circle in range(num_circles):
        # Calculate current radius for this circle
        current_radius = min_radius + ((max_radius - min_radius) * circle / (num_circles - 1)) if num_circles > 1 else min_radius
        
        # For each petal direction
        for petal in range(num_petals):
            angle = 2 * np.pi * petal / num_petals
            
            # Points along the ray from min_radius to max_radius
            for i in range(points_per_petal):
                # Calculate distance from center
                r = current_radius
                x = x0 + r * np.cos(angle)
                y = y0 + r * np.sin(angle)
                points.append((x, y))
            
            # If we're not at the last circle, move to the starting point of the next circle
            if circle < num_circles - 1:
                next_radius = min_radius + ((max_radius - min_radius) * (circle + 1) / (num_circles - 1))
                next_x = x0 + next_radius * np.cos(angle)
                next_y = y0 + next_radius * np.sin(angle)
                points.append((next_x, next_y))
    
    return points

def generate_multi_angle_lawnmower(x0, y0, angles, width=9.0, height=9.0, spacing=2.0):
    """
    Generate a combined coverage path by doing a lawnmower sweep at multiple angles.
    'angles' is a list of angles (in degrees) by which we'll rotate the region
    around (x0, y0).
    """
    all_points = []

    for angle in angles:
        # 1. Generate a standard lawnmower pattern around (0,0) to keep it simple
        #    and interpret that pattern in local coordinates.
        lawnmower_local = generate_lawnmower(0, 0, width=width, height=height, spacing=spacing)

        # 2. Rotate each point by 'angle' around origin, then shift to (x0, y0).
        #    Because we generated at (0,0) "center", the rotation is straightforward.
        rotated_path = []
        for (lx, ly) in lawnmower_local:
            rx, ry = rotate_point(lx, ly, angle)
            # Shift by the global center (x0, y0)
            rx_global = x0 + rx
            ry_global = y0 + ry
            rotated_path.append((rx_global, ry_global))

        # 3. Append to our big list
        all_points.extend(rotated_path)

    return all_points

# Example usage:
# Cover 9x9 region, center at (0,0), with passes at 0°, 90°, and 45°.
waypoints = generate_multi_angle_lawnmower(
    x0=0.0, 
    y0=0.0, 
    angles=[0, 45, 90], 
    width=9.0, 
    height=9.0, 
    spacing=2.0
)

