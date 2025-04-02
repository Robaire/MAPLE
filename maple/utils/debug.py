import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pytransform3d.trajectories import pqs_from_transforms, plot_trajectory
from dataloader import DataLoader

def playback_camera(dataloader: DataLoader,
                    cam: str,
                    range: tuple[int, int],
                    semantic: bool = False) -> None:
    """
    Animate and display the specified camera for the specified range on the
    passed dataloader

    Args:
        dataloader (DataLoader): Loaded dataset to playback
        cam (str): Camera to lookup, valid values are: ``Back, BackLeft, 
        BackRight, Front, FrontLeft, FrontRight, Left, Right``
        range ((int, int)): Range of images to select from, end index is exclusive
        semantic (bool): Returns the ground truth semantic image if True
    """
    img_seq = dataloader.cam_sequence(cam, range, skip_repeat=True, semantic=semantic)
    frames = [np.array(img) for img in img_seq]

    fig, ax = plt.subplots()
    if semantic: im = ax.imshow(frames[0])
    else:        im = ax.imshow(frames[0], cmap='gray')

    def update(frame):
        im.set_array(frame)
        return [im]

    anim = FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
    plt.show()

def display_trajectory(dataloader: DataLoader,
                       range: tuple[int, int] = None) -> None:
    """
    Plots the 3D trajectory of the rover over a given range or the entire dataset.

    Args:
        dataloader (DataLoader): Loaded dataset to playback
        range ((int, int)): Range of data to select from, end index is exclusive.
        If not passed, will assume entire dataset
    """
    if range is not None:
        start = range[0]
        end   = range[1]
    else:
        start = 0
        end   = -1

    T = dataloader.T_gt[start:end]
    pq = pqs_from_transforms(T)

    # Plot visual settings
    ax_scale = np.max(np.abs(pq[:, :2]))
    z_center = np.average(pq[:, 2])
    n_frames = int(T.shape[0] / 200)

    ax = plot_trajectory(P=pq, n_frames=n_frames, show_direction=False, ax_s=ax_scale, s=0.8, lw=1.5, c='k')
    ax.set_zlim(z_center - ax_scale, z_center + ax_scale)
    plt.show()