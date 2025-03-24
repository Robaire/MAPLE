import pandas as pd
import pytransform3d.rotations as pr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pytransform3d.transformations as pt
from maple.utils import carla_to_pytransform

"""A file for performing a quick data analysis. Should be rolled into a proper data analysis class at some point."""
class DataLoader():
    def __init__(self):
        data = pd.read_csv('data_output.csv')
        self.gt_data = data["Actual"]
        self.est_data = data["Estimated"]
        self.times = data["Time"]

    def plot_animated_transform(self):
        """Using pytransform and matplotlib, plot an animation that shows the ground truth and estimated trajectories."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        gt_transforms = [carla_to_pytransform(x) for x in self.gt_data]
        est_transforms = [carla_to_pytransform(x) for x in self.est_data]
        ani = FuncAnimation(fig, self.update, frames=len(gt_transforms), fargs=(ax, gt_transforms, est_transforms), interval=100)
        plt.show()

        # Update function for animation
    def update(self, frame, ax, gt_transforms, est_transforms):
        ax.clear()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        
        pr.plot_basis(ax)  # Draw world coordinate frame
        pt.plot_transform(ax, gt_transforms[frame])  # Draw current transform
        pt.plot_transform(ax, est_transforms[frame])



if __name__ == '__main__':
    dl = DataLoader()
    dl.plot_animated_transform()