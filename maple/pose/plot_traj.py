import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv

def read_estimated_trajectory(txt_path):
    """
    Reads estimated trajectory from a space-separated text file.
    Returns (times, x_vals, y_vals, z_vals) as numpy arrays.
    """
    times, x_vals, y_vals, z_vals = [], [], [], []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Expecting at least: time, x, y, z
            if len(parts) < 4:
                continue
            try:
                t = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError:
                # Skip lines that don’t parse
                continue
            
            times.append(t)
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
    
    return (np.array(times), np.array(x_vals), np.array(y_vals), np.array(z_vals))


def read_ground_truth(csv_path):
    """
    Reads ground-truth positions from a comma-separated CSV file.
    Returns (times, x_vals, y_vals, z_vals) as numpy arrays.
    Assumes columns are like: time, x, y, z, ...
    """
    times, x_vals, y_vals, z_vals = [], [], [], []
    with open(csv_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        
        # If there's a header, uncomment the next line to skip it:
        # header = next(csv_reader, None)

        for row in csv_reader:
            if len(row) < 4:
                continue
            try:
                t = float(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
            except ValueError:
                # Skip lines that don’t parse
                continue
            
            times.append(t)
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
    
    return (np.array(times), np.array(x_vals), np.array(y_vals), np.array(z_vals))


def plot_est_and_gt(txt_path, csv_path):
    # Read estimated data from the TXT file
    t_est, x_est, y_est, z_est = read_estimated_trajectory(txt_path)
    # Read ground truth data from the CSV file
    t_gt, x_gt, y_gt, z_gt = read_ground_truth(csv_path)
    
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the estimated trajectory
    ax.plot(x_est, y_est, z_est, label='Estimated Trajectory')
    
    # Plot the ground truth trajectory
    ax.plot(x_gt, y_gt, z_gt, label='Ground Truth')
    
    ax.set_title('Estimated vs. Ground Truth 3D Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Save the figure (optional)
    plt.savefig('est_vs_gt_trajectory.png', dpi=300)
    # Or show it interactively:
    # plt.show()

if __name__ == "__main__":
    txt_file = "/home/annikat/LAC_data/live_trajectory_LAC.txt"
    csv_file = "/home/annikat/MAPLE/data/orb_02/imu_data.csv"
    
    plot_est_and_gt(txt_file, csv_file)
