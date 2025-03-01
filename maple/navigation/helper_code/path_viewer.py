# NOTE: This is a util to help view different paths before implementing

import numpy as np
import matplotlib.pyplot as plt

def generate_spiral(x0, y0, initial_radius=4.0, num_points=1000, spiral_rate=0, frequency=4):
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

import math
# Example usage:
spiral_points = generate_spiral(0, 0, initial_radius=4.0, num_points=4, spiral_rate=0, frequency=2/math.pi)

# Extract x and y coordinates for plotting
x_vals, y_vals = zip(*spiral_points)

# Plot the spiral
plt.figure(figsize=(6,6))
plt.plot(x_vals, y_vals, marker='o', linestyle='-', markersize=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Spiral Visualization")
plt.grid()
plt.show()
