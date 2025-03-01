import pandas as pd
import pytransform3d.rotations as pr
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pytransform3d.transformations as pt
from maple.utils import carla_to_pytransform
import ast

"""A file for performing a quick data analysis. Should be rolled into a proper data analysis class at some point."""
class DataLoader():
    def __init__(self):
        data = pd.read_csv('data_output.csv')
        self.gt_data = data["Actual"]
        # Process all matrices
        self.gt_data = [self.string_to_numpy(matrix_str) for matrix_str in self.gt_data]
        self.est_data = data["Estimated"]
        self.est_data = [self.string_to_numpy(matrix_str) for matrix_str in self.est_data]
        self.times = data["Time"]

    def string_to_numpy(self,matrix_str):
        """Convert a string representation of a 4x4 transformation matrix to a NumPy array."""
        return np.fromstring(matrix_str.replace("[", "").replace("]", "").replace("\n", " "), sep=" ").reshape(4, 4)

    def plot_animated_transform(self):
        """Using pytransform and matplotlib, plot an animation that shows the ground truth and estimated trajectories."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        gt_transforms = self.gt_data
        est_transforms = self.est_data
        ani = FuncAnimation(fig, self.update, frames=len(gt_transforms), fargs=(ax, gt_transforms, est_transforms), interval=50)
        ani.save('traj_comparison.gif')
        plt.show()

        # Update function for animation
    def update(self, frame, ax, gt_transforms, est_transforms):
        ax.clear()
        ax.set_xlim([-7, 7])
        ax.set_ylim([-7, 7])
        ax.set_zlim([-2, 5])
        
        pr.plot_basis(ax)  # Draw world coordinate frame
        pt.plot_transform(ax, gt_transforms[frame])  # Draw current transform
        pt.plot_transform(ax, est_transforms[frame])



if __name__ == '__main__':
    dl = DataLoader()
    dl.plot_animated_transform()