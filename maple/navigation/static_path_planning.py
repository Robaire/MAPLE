import numpy as np

def generate_spiral(x0, y0, initial_radius=4.0, num_points=400, spiral_rate=0.1, frequency=8):
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

def generate_lawnmower(x0, y0, width=9.0, height=9.0, spacing=2.0):
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

