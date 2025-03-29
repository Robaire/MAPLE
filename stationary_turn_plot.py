# Quick Plotter
import matplotlib.pyplot as plt
import numpy as np

gt_data = np.load("gt_data.npy")
est_data = np.load("est_data.npy")

# Create a figure with 4 subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 3, figsize=(10, 8))

axes[0, 0].plot(gt_data[0], label="gt_x", linestyle="--")
axes[0, 0].plot(est_data[0], label="est_x", linestyle=":")
axes[0, 0].set_title("x data")
axes[0, 0].legend()

axes[0, 1].plot(gt_data[1], label="gt_y", linestyle="--")
axes[0, 1].plot(est_data[1], label="est_y", linestyle=":")
axes[0, 1].set_title("y data")
axes[0, 1].legend()

axes[0, 2].plot(gt_data[2], label="gt_z", linestyle="--")
axes[0, 2].plot(est_data[2], label="est_z", linestyle=":")
axes[0, 2].set_title("z data")
axes[0, 2].legend()

axes[1, 0].plot(gt_data[3], label="gt_roll", linestyle="--")
axes[1, 0].plot(est_data[3], label="est_roll", linestyle=":")
axes[1, 0].set_title("roll data")
axes[1, 0].legend()

axes[1, 1].plot(gt_data[4], label="gt_pitch", linestyle="--")
axes[1, 1].plot(est_data[4], label="est_pitch", linestyle=":")
axes[1, 1].set_title("pitch data")
axes[1, 1].legend()

axes[1, 2].plot(gt_data[4], label="gt_yaw", linestyle="--")
axes[1, 2].plot(est_data[4], label="est_yaw", linestyle=":")
axes[1, 2].set_title("yaw data")
axes[1, 2].legend()
plt.tight_layout()
plt.savefig("stationary_turn_results.png")
