from typing import Tuple, List, Dict
import numpy as np
import random
import copy

# This is the command to run the custom environment with the custom configuration
# python3 train.py --env Custom --config config/custom.yml

background_color = [255, 128, 64]

class Custom:
    """
    This is a class so you can design your own custom Q environment
    """

    def __init__(self, grid_size: int = 16, pixels_per_square_side: int = 25, n: int = 9):
        """Initializes the function

        Args:
            grid_size (int, optional): This is how many squares there are in one side, it will make it into a square. Defaults to 5.
            pixels_per_square_side (int, optional): This is how many pixels are in each square for rendering purposes. Defaults to 75.
            n (int, optional): This is the action space and observation space values. Defaults to 9.
        """
        self.n = n
        self.action_space = self.ActionSpace(n)
        self.observation_space = self.ObservationSpace(n)

        self.grid_size = grid_size
        self.pixels_per_grid_side: int = grid_size * pixels_per_square_side
        self.render_grid = np.zeros((self.pixels_per_grid_side, self.pixels_per_grid_side, 3), dtype=np.uint8)  # RGB grid
        self.position = (0, 0)  # Initial position of the red square

    def step(self, action) -> Tuple[int, float, bool, bool, dict]:
        """

        Args:
            action (_type_): _description_

        Returns new_state, reward, terminated, truncated, info
        """
        return (0, 0, False, False, None)

    def reset(self, seed = None) -> Tuple[int, dict]:
        """
        Reset the state of the object to its initial state.
        Override this method in subclasses if needed.

        returns state, info
        """
        # print(f"Resetting {self.name} to its initial state.")
        return (0, None)

    def render(self) -> np.array:
        """
        Render the component, if applicable.

        Args:
            mode (str): The rendering mode (e.g., "human" or "rgb_array").

        Returns:
            None or an array, depending on the rendering mode.
        """

        # Run this every turn to reset the graphics to the background color and create a new object so that we dont change the image before it
        self.generate_graphics(self.pixels_per_grid_side)

        # Add squares
        self.draw_square(0, 0, 25, [255, 255, 0])

        return self.render_grid
    
    def generate_graphics(self, size: int):
        """This should be ran once each turn to reset all the graphics array

        Args:
            size (int): This is the sides of a square (in pixels, I suggest 256)

        Returns:
            _type_: img type file arrays with correct types to be rendered
        """

        # Create the innermost ndarray (shape: (3,), type: numpy.uint8)
        inner_most_array = np.array(background_color, dtype=np.uint8)

        # Create the middle ndarray (shape: (size, 3), type: numpy.uint8)
        middle_array = np.tile(inner_most_array, (size, 1))

        # Create the outermost ndarray (shape: (size, size, 3), type: numpy.uint8)
        outer_array = np.tile(copy.deepcopy(middle_array[:, np.newaxis, :]), (1, size, 1))

        self.render_grid = outer_array
    
    def draw_square(self, pixel_x: int, pixel_y: int, size: int, square_color):

        # Convert the square color to a numpy array
        fixed_color = np.array(square_color, dtype=np.uint8)

        # Draw squares on the graphics array
        # for x, y in square_positions:
        for x in range(pixel_x, pixel_x+size):
            for y in range(pixel_y+size):
                if 0 <= x < self.pixels_per_grid_side and 0 <= y < self.pixels_per_grid_side:
                    self.render_grid[y:y+size, x:x+size] = fixed_color
        
    class ObservationSpace:
        """
        Custom observation space for the environment.
        """
        def __init__(self, n: int):
            self.n: int= n

    class ActionSpace:
        """
        Custom action space for the environment.
        """
        def __init__(self, n: int):
            self.n: int = n

        def sample(self):
            """
            Sample a random action from the action space.
            """
            return random.randint(0, self.n - 1)

if __name__ == '__main__':
    # Create the innermost ndarray (shape: (3,), type: numpy.uint8)
    inner_most_array = np.array([255, 128, 64], dtype=np.uint8)

    # Create the middle ndarray (shape: (256, 3), type: numpy.ndarray)
    middle_array = np.array([inner_most_array for _ in range(256)], dtype=object)

    # Create the outermost ndarray (shape: (256, 256, 3), type: numpy.ndarray)
    outer_array = np.array([middle_array for _ in range(256)], dtype=object)

    # Print shapes and types
    print(f"Outer array shape: {outer_array.shape}, type: {type(outer_array)}")
    print(f"Middle array shape: {middle_array.shape}, type: {type(middle_array)}")
    print(f"Inner most array shape: {inner_most_array.shape}, type: {type(inner_most_array)}")
    print(f"Inner most array dtype: {type(inner_most_array[0])}")