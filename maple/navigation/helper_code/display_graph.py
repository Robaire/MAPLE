import matplotlib.pyplot as plt
import math

def display_path(path_points, obstacles_x_y_size) -> None:
    """taking a list of (x, y) points this will display a connected graph

    Args:
        points (_type_): _description_
    """

    # Separate the list of points into x and y coordinates.
    x_vals = [p[0] for p in path_points]
    y_vals = [p[1] for p in path_points]

    # Plot the points with a line connecting them and markers for each point.
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='Path')

    # Optionally, plot just the individual points as a scatter plot.
    plt.scatter(x_vals, y_vals, color='r')

    x_vals = [p[0] for p in obstacles_x_y_size]
    y_vals = [p[1] for p in obstacles_x_y_size]

    # Plot unconnected red dots with a set marker size.
    # The 's' parameter sets the size (area) of the markers in points^2.

    # Check if it is x, y or x, y, obstacl_size
    if len(obstacles_x_y_size[0]) == 2:
        plt.scatter(x_vals, y_vals, color='red', s=100)
    else:
        for x, y, obstacle_size in obstacles_x_y_size:

            # This is stupid way to make a circle around the point
            num_points = 100
            for theta in range(1, num_points+1):
                new_x = obstacle_size * math.cos(2*math.pi / num_points * theta) + x
                new_y = obstacle_size * math.sin(2*math.pi / num_points * theta) + y
                plt.scatter(new_x, new_y, color='red', s=10)  # Adjust 's' for larger or smaller dots

    # Add title and labels to the graph.
    plt.title('Graph of (x, y) Points, Unconnected Red Dots with Set Radius Size')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Display grid lines for better readability.
    plt.grid(True)

    # Add a legend.
    plt.legend()

    # Show the plot.
    plt.show()


if __name__ == '__main__':
    # Example list of (x, y) points
    path_points = [(0, 1), (1, 3), (2, 2), (3, 5), (4, 4)]
    obstacle_points = [(1, 1), (2, 3), (4, 2)]

    display_path(path_points, obstacle_points)