"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
import random
from .map_generator import generate_ground

style.use("ggplot")


class Tetris:
    robot_position = None
    
    # This is the height of the square we are exploring
    n = 256
    vertical_height = 5
    # TODO: Remove the below 2 setters and see if runs, should be set in the reset
    grid_heights = generate_ground(n, n, vertical_height)
    grid_explored_tracker = np.zeros((n, n), dtype=bool)
    # TODO: Change this to include height or something
    # IMPORTANT DECISION: Decide if the robot ever goes air born
    robot_position = (0, 0)
    robot_orientation = 0
    robot_velocity = (0, 0)

    # This is to keep track of time and end the sim when it is over
    time_count = 0
    time_max = 100

    # These are the possible des_vel_and_des_angs the bot can go
    des_vel_and_des_angs = ((0, 1), (0, -1), (1, 0), (-1, 0))

    explored_color = (255, 255, 255)
    robot_color = (255, 0, 0)

    def __init__(self, height=20, width=10, block_size=20):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
        self.text_color = (200, 20, 220)
        self.reset()

    ##### THIS FUNCTION IS IMPORTANT TO OPTIMIZE #####    
    def get_next_position(self, position, orientation, velocity, desired_lin_v, desired_ang_v, dt=1):
        
        # Extract current position and velocity
        x, y = position
        vx, vy = velocity

        # Update position with terrain slope considerations
        # gradient_x = (height_map[int(y), int(x) + 1] - height_map[int(y), int(x) - 1]) / 2
        # gradient_y = (height_map[int(y) + 1, int(x)] - height_map[int(y) - 1, int(x)]) / 2
        
        # g = 9.8  # Gravitational acceleration
        # terrain_effect_x = -g * gradient_x
        # terrain_effect_y = -g * gradient_y

        # Update acceleration with terrain effects
        # ax = current_acceleration[0] + terrain_effect_x
        # ay = current_acceleration[1] + terrain_effect_y

        # Update the position
        x += vx * dt
        y += vy * dt

        # Update the velocity
        vx += np.sin(orientation) * desired_lin_v * dt
        vy += np.cos(orientation) * desired_lin_v * dt

        # Update orientation based on angular velocity
        new_orientation = orientation + desired_ang_v * dt

        new_velocity = (vx, vy)
        # IMPORTANT TODO: Decide if we will allow this to be a non int
        new_position = (int(x), int(y))

        # Ensure the position stays within the height map bounds
        # x = max(0, min(x, height_map.shape[1] - 1))
        # y = max(0, min(y, height_map.shape[0] - 1))

        return new_position, new_orientation, new_velocity

    ##### THIS FUNCTION IS IMPORTANT TO OPTIMIZE #####    
    def get_updated_grid_explorer_tracker(self, grid, position):
        """This function returns a copy of the grid representation (which is a numpy double array of boolean values) with the new position added in for explored

        Args:
            grid (_type_): This is assumed to be a square
            position (_type_): This is a tuple of (x, y)
        """

        grid_copy = grid.copy()

        # TODO: Currently using a vision radius to estimate the new known squares but we should incorpoarte heights and if cameras are on, that way we can see from atop a hill
        vision_radius = 5

        pos_x, pos_y = position

        # Make a square visible
        for x in range(pos_x - vision_radius, pos_x + vision_radius + 1):
            for y in range(pos_y - vision_radius, pos_y + vision_radius + 1):
                if (self.is_on((x, y), len(grid_copy))):
                    grid_copy[(x, y)] = True

        return grid_copy

    def get_grid_explorer_tracker_score(self, grid) -> int:
        """Takes in a numpy double array of boolean values representing explored or not and returns a score

        Args:
            grid (_type_): _description_
        """
        
        # Currently add up all the explored squares for a square
        return sum((1 if p else 0) for row in grid for p in row)

    def reset(self):
        self.score = 0
        self.gameover = False

        # TODO: shouldn't change maps every time (its better to learn skills for specific map completely then move on), come up with a way to change maps every so often
        # self.grid_heights = generate_ground(self.n, self.n, self.vertical_height)
        self.grid_explored_tracker = np.zeros((self.n, self.n), dtype=bool)
        self.robot_position = (0, 0)
        self.robot_orientation = 0
        self.robot_velocity = (0, 0)

        self.time_count = 0

        return self.get_state_properties(self.robot_position)

    def rotate(self, piece):
        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array

    ##### THIS FUNCTION IS IMPORTANT TO OPTIMIZE #####    
    # Use this function for a quick representation of the state
    def get_state_properties(self, possible_next_position):

        next_grid = self.get_updated_grid_explorer_tracker(self.grid_explored_tracker, possible_next_position)
        next_score: int = self.get_grid_explorer_tracker_score(next_grid)

        # TODO: Somewhere, maybe not here, add a constraint that the bot should always try to move

        return torch.FloatTensor([possible_next_position[0], possible_next_position[1], next_score])
    
    # Checks if this position would fall on an n by n square
    def is_on(self, position, n):
        return (0 <= position[0] < n) and (0 <= position[1] < n)

    # IMPORTANT NOTE: Make sure this returns at least one state that is on the grid (otherwise the simulation may crash/end early)
    def get_next_states(self):
        states = {}
        print(f'the function is getting called')
        for des_vel, des_ang in self.des_vel_and_des_angs:
            # TODO: Decide if we want to use non integer values for position
            possible_next_position, _, _ = self.get_next_position(self.robot_position, self.robot_orientation, self.robot_velocity, des_vel, des_ang)
            # print(f'the information after is {possible_next_position} with values {possible_next_position[0]} and {possible_next_position[1]} of type {type(possible_next_position[0])}')
            print(f'the posible positions are {possible_next_position}')
            if (self.is_on(possible_next_position, self.n)):
                # Set the key to action and the value to a representation of the state
                states[(des_vel, des_ang)] = self.get_state_properties(possible_next_position)

        return states

    def step(self, action, render=True, video=None):
        if render:
            self.render(video)

        # update the time
        self.time_count += 1

        # Currently the only action is des_vel and des_ang
        des_vel, des_ang = action

        # Get the next position
        # self.robot_position = (self.robot_position[0] + direction[0], self.robot_position[1] + direction[1])
        self.robot_position, self.robot_orientation, self.robot_velocity = self.get_next_position(self.robot_position, self.robot_orientation, self.robot_velocity, des_vel, des_ang)

        # Get the next grid explored tracker
        self.grid_explored_tracker = self.get_updated_grid_explorer_tracker(self.grid_explored_tracker, self.robot_position)

        # Get the next score
        self.score = self.get_grid_explorer_tracker_score(self.grid_explored_tracker)

        # Get if the game is over
        self.gameover = self.is_gameover(self.grid_explored_tracker)

        return self.score, self.gameover
    
    def is_gameover(self, grid) -> bool:
        """Return True if all the grid has been explored, False otherwise

        Args:
            grid (_type_): _description_
        """

        # Check if the time has been maxed out
        if self.time_count >= self.time_max:
            return True
        
        # Check if everything has been explored
        for row in grid:
            for elem in row:
                # Return False if there is an unexplored square
                if not elem:
                    return False
                
        # Return True as everything has been explored
        return True
    
    def get_height_map_color(self, array, position):
        """This function takes in an array and position and returns a corresponding color that should be in that square

        Args:
            array (_type_): _description_
            position (_type_): _description_
        """

        # This gets the height of the array at the position
        height = array[position]

        if (height < 1):
            return (0, 0, 255)
        elif (height < 2):
            return (34, 139, 34)
        elif (height < 3):
            return (120, 180, 140)
        elif (height < 4):
            return (139, 69, 19)
        else :
            return (255, 69, 0)

    def render(self, video=None):
        if not self.gameover:
            # img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
            img = [(self.explored_color if p else self.get_height_map_color(self.grid_heights, (index1, index2))) for index1, row in enumerate(self.grid_explored_tracker) for index2, p in enumerate(row)]
        else:
            img = [self.explored_color for row in self.grid_explored_tracker for p in row]

        # Resize the grid type stuff
        img = np.array(img).reshape((self.n, self.n, 3)).astype(np.uint8)

        # Add the moving object
        img[self.robot_position] = self.robot_color

        # Resize the image
        img = Image.fromarray(img, "RGB")
        img = img.resize((self.width * self.block_size, self.height * self.block_size), 0)
        img = np.array(img)

        # Draw additional UI
        img = np.concatenate((img, self.extra_board), axis=1)

        cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.score),
                    (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        if video:
            video.write(img)

        cv2.imshow("Exploration For Robot", img)
        cv2.waitKey(1)
