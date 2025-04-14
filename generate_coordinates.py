import random

def generate_points_with_error(x, y, error_range, count):
    """
    Generate random points within a specified error range around x, y coordinates.
    
    Args:
        x (float): The x-coordinate of the center point
        y (float): The y-coordinate of the center point
        error_range (float): The maximum distance from the center in any direction
        count (int): Number of points to generate
    
    Returns:
        None: Prints the points to console
    """

    error_range_x, error_range_y = error_range
    for i in range(count):
        # Generate random offsets within error_range for both x and y
        x_error = random.uniform(-error_range_x, error_range_x)
        y_error = random.uniform(-error_range_y, error_range_y)
        
        # Add the error to the original coordinates
        new_x = x + x_error
        new_y = y + y_error
        
        # Print the result
        print(f"{new_x}, {new_y}")

# Example usage
if __name__ == "__main__":
    # Get user input
    x = float(input("Enter x coordinate: "))
    y = float(input("Enter y coordinate: "))
    error_range_x = float(input("Enter x error range: "))
    error_range_y = float(input("Enter y error range: "))
    count = int(input("Enter number of points to generate: "))
    
    # Generate and print points
    generate_points_with_error(x, y, (error_range_x, error_range_y), count)