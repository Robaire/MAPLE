"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
import random

style.use("ggplot")


class Tetris:
    robot_position = None
    
    # This is the height of the square we are exploring
    n = 256
    grid_heights = np.zeros((n, n))
    grid_explored_tracker = np.zeros((n, n), dtype=bool)
    robot_position = (0, 0)

    # These are the possible directions the bot can go
    directions = ((0, 1), (0, -1), (1, 0), (-1, 0))

    unexplored_color = (0, 0, 0)
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

    def get_updated_grid_explorer_tracker(self, grid, position):
        """This function returns a copy of the grid representation (which is a numpy double array of boolean values) with the new position added in for explored

        Args:
            grid (_type_): _description_
            position (_type_): _description_
        """

        grid_copy = grid.copy()

        # update the new explored positions in the grid copy
        grid_copy[position] = True

        return grid_copy

    def get_grid_explorer_tracker_score(self, grid) -> int:
        """Takes in a numpy double array of boolean values representing explored or not and returns a score

        Args:
            grid (_type_): _description_
        """
        
        # Currently add up all the explored squares for a square
        return sum((1 if p else 0) for row in grid for p in row)

    def reset(self):
        # self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        # self.tetrominoes = 0
        # self.cleared_lines = 0
        # self.bag = list(range(len(self.pieces)))
        # random.shuffle(self.bag)
        # self.ind = self.bag.pop()
        # self.piece = [row[:] for row in self.pieces[self.ind]]
        # self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False

        self.grid_heights = np.zeros((self.n, self.n))
        self.grid_explored_tracker = np.zeros((self.n, self.n), dtype=bool)
        self.robot_position = (0, 0)

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

    # Use this function for a quick representation of the state
    def get_state_properties(self, possible_next_position):
        # lines_cleared, board = self.check_cleared_rows(board)
        # holes = self.get_holes(board)
        # bumpiness, height = self.get_bumpiness_and_height(board)

        next_grid = self.get_updated_grid_explorer_tracker(self.grid_explored_tracker, possible_next_position)
        next_score: int = self.get_grid_explorer_tracker_score(next_grid)

        # TODO: Add a time constraint to increase explorartion speed
        # TODO: Somewhere, maybe not here, add a constraint that the bot should always try to move

        return torch.FloatTensor([possible_next_position[0], possible_next_position[1], next_score])
    
    # Checks if this position would fall on an n by n square
    def is_on(self, position, n):
        return (0 <= position[0] < n) and (0 <= position[1] < n)

    def get_next_states(self):
        states = {}

        for direction in self.directions:
            possible_next_position = (self.robot_position[0] + direction[0], self.robot_position[1] + direction[1])
            if (self.is_on(possible_next_position, self.n)):
                # Set the key to action and the value to a representation of the state
                states[direction] = self.get_state_properties(possible_next_position)

        # piece_id = self.ind
        # curr_piece = [row[:] for row in self.piece]
        # if piece_id == 0:  # O piece
        #     num_rotations = 1
        # elif piece_id == 2 or piece_id == 3 or piece_id == 4:
        #     num_rotations = 2
        # else:
        #     num_rotations = 4

        # for i in range(num_rotations):
        #     valid_xs = self.width - len(curr_piece[0])
        #     for x in range(valid_xs + 1):
        #         piece = [row[:] for row in curr_piece]
        #         pos = {"x": x, "y": 0}
        #         while not self.check_collision(piece, pos):
        #             pos["y"] += 1
        #         self.truncate(piece, pos)
        #         board = self.store(piece, pos)
        #         states[(x, i)] = self.get_state_properties(board)
        #     curr_piece = self.rotate(curr_piece)
        return states

    def step(self, action, render=True, video=None):
        # x, num_rotations = action
        # self.current_pos = {"x": x, "y": 0}
        # for _ in range(num_rotations):
        #     self.piece = self.rotate(self.piece)

        # while not self.check_collision(self.piece, self.current_pos):
        #     self.current_pos["y"] += 1
        #     if render:
        #         self.render(video)

        if render:
            self.render(video)

        # overflow = self.truncate(self.piece, self.current_pos)
        # if overflow:
        #     self.gameover = True

        # self.board = self.store(self.piece, self.current_pos)

        # lines_cleared, self.board = self.check_cleared_rows(self.board)
        # score = 1 + (lines_cleared ** 2) * self.width
        # self.score += score
        # self.tetrominoes += 1
        # self.cleared_lines += lines_cleared
        # if not self.gameover:
        #     self.new_piece()
        # if self.gameover:
        #     self.score -= 2

        # Currently the only action is direction
        direction = action

        # Get the next position
        self.robot_position = (self.robot_position[0] + direction[0], self.robot_position[1] + direction[1])

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

        for row in grid:
            for elem in row:
                # Return False if there is an unexplored square
                if not elem:
                    return False
                
        # Return True as everything has been explored
        return True

    def render(self, video=None):
        if not self.gameover:
            # img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
            img = [(self.explored_color if p else self.unexplored_color) for row in self.grid_explored_tracker for p in row]
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
