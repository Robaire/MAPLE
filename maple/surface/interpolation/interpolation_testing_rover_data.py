import numpy as np
import matplotlib.pyplot as plt
from surface_interpolation import (linear_interpolation, nearest_interpolation,
                                regular_grid_interpolation, bspline_interpolation)
from sklearn.metrics import r2_score
import os
from datetime import datetime
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import convolve

def load_ground_truth(file_name):
    """Load the ground truth .dat file."""
    data = np.load(file_name, allow_pickle=True)
    print(f"Ground truth data shape: {data.shape}")
    
    # Extract coordinates and heights
    x = data[:, :, 0].flatten(order='F')
    y = data[:, :, 1].flatten(order='F')
    z = data[:, :, 2].flatten(order='F')
    boulder_presence = data[:, :, 3].flatten(order='F')
    
    # Remove any NaN values
    valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    boulder_presence = boulder_presence[valid_mask]
    
    return x, y, z, boulder_presence

def load_rover_samples(csv_file):
    """Load the sampled points from the rover's CSV file."""
    df = pd.read_csv(csv_file)
    x = df['0'].values  # First column
    y = df['1'].values  # Second column
    z = df['2'].values  # Third column
    return x, y, z

def calculate_metrics(z_true, z_pred):
    """Calculate comprehensive error metrics."""
    # Remove NaN values for metric calculation
    mask = ~np.isnan(z_pred) & ~np.isnan(z_true)
    z_true = z_true[mask]
    z_pred = z_pred[mask]
    
    if len(z_true) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'max_error': np.nan,
            'std_error': np.nan,
            'score': np.nan
        }
    
    mae = np.mean(np.abs(z_true - z_pred))
    rmse = np.sqrt(np.mean((z_true - z_pred)**2))
    r2 = r2_score(z_true, z_pred)
    max_error = np.max(np.abs(z_true - z_pred))
    std_error = np.std(z_true - z_pred)
    # Score based on points with absolute error < 0.05
    absolute_error = np.abs(z_true - z_pred)
    score = np.sum(absolute_error < 0.05)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'max_error': max_error,
        'std_error': std_error,
        'score': score
    }

def smoothing_filter(zi, filter_size=3):
    """Apply a smoothing filter to the height data."""
    if filter_size % 2 == 0:
        raise ValueError("Filter size must be an odd integer.")

    kernel = np.ones((filter_size, filter_size)) / (filter_size ** 2)
    smoothed_zi = convolve(zi, kernel, mode='nearest')
    return smoothed_zi

def plot_height_comparisons(xi_grid, yi_grid, zi_grid, zi_true, sources, output_dir=None, save=False):
    """Plot the error between true and interpolated height data."""
    n_plots = len(zi_grid) + 1  # +1 for ground truth
    n_rows = (n_plots + 2) // 3  # Ensure we have enough rows
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    fig.suptitle("Height Comparison", fontsize=16)
    fig.subplots_adjust(hspace=0.3)

    if n_rows == 1:
        axes = [axes]
    
    # Plot ground truth first
    ax = axes[0][0]
    im = ax.imshow(zi_true, extent=[xi_grid[0].min(), xi_grid[0].max(), 
                                   yi_grid[0].min(), yi_grid[0].max()],
                   origin='lower', aspect='equal', cmap='terrain')
    ax.set_title('Ground Truth')
    plt.colorbar(im, ax=ax, label='Height')

    # Plot interpolation results and errors
    for i, (xi, yi, zi, source) in enumerate(zip(xi_grid, yi_grid, zi_grid, sources)):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row][col]

        # Plot absolute error
        diff = zi - zi_true
        abs_diff = np.abs(diff)
        
        im = ax.imshow(abs_diff, extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                    origin='lower', aspect='equal', cmap='viridis')
        
        results = calculate_metrics(zi_true.flatten(), zi.flatten())
        score = results['score']
        normalized_score = score / (np.shape(zi)[0]*np.shape(zi)[1]) * 300
        ax.set_title(f'Error ({source})\nScore: {score:.0f} points\nNormalized: {normalized_score:.2f}')
        plt.colorbar(im, ax=ax, label='Height Difference')

    # Remove any empty subplots
    for i in range(n_plots, n_rows * n_cols):
        row = i // 3
        col = i % 3
        fig.delaxes(axes[row][col])

    if save and output_dir:
        filename = f'rover_height_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(os.path.join(output_dir, filename))
    plt.show()

    # Add a separate figure for the interpolated surfaces
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    fig.suptitle("Interpolated Surfaces", fontsize=16)
    fig.subplots_adjust(hspace=0.3)

    if n_rows == 1:
        axes = [axes]
    
    # Plot ground truth first
    ax = axes[0][0]
    im = ax.imshow(zi_true, extent=[xi_grid[0].min(), xi_grid[0].max(), 
                                   yi_grid[0].min(), yi_grid[0].max()],
                   origin='lower', aspect='equal', cmap='terrain')
    ax.set_title('Ground Truth')
    plt.colorbar(im, ax=ax, label='Height')

    # Plot interpolated surfaces
    for i, (xi, yi, zi, source) in enumerate(zip(xi_grid, yi_grid, zi_grid, sources)):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row][col]

        im = ax.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                    origin='lower', aspect='equal', cmap='terrain')
        ax.set_title(f'Interpolated Surface ({source})')
        plt.colorbar(im, ax=ax, label='Height')

    # Remove any empty subplots
    for i in range(n_plots, n_rows * n_cols):
        row = i // 3
        col = i % 3
        fig.delaxes(axes[row][col])

    if save and output_dir:
        filename = f'rover_interpolated_surfaces_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(os.path.join(output_dir, filename))
    plt.show()

def evaluate_interpolation(x_train, y_train, z_train, x_test, y_test, z_test):
    """Evaluate different interpolation methods."""
    # Create regular grid for interpolation
    grid_size = 100
    x_min, x_max = min(x_test.min(), x_train.min()), max(x_test.max(), x_train.max())
    y_min, y_max = min(y_test.min(), y_train.min()), max(y_test.max(), y_train.max())
    
    # Ground truth values on the grid
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    zi_true = griddata((x_test, y_test), z_test, (xi_grid, yi_grid), method='cubic')

    # Different interpolation methods
    methods = {
        'Linear': linear_interpolation,
        'Nearest': nearest_interpolation,
        'RBF': regular_grid_interpolation,
        'B-Spline': bspline_interpolation
    }

    results = {}
    for name, method in methods.items():
        xi, yi, zi = method(x_train, y_train, z_train)  # Get interpolated values and grid
        zi_smoothed = smoothing_filter(zi)
        results[name] = {
            'xi': xi,
            'yi': yi,
            'zi': zi,
            'zi_smoothed': zi_smoothed,
            'metrics': calculate_metrics(zi_true.flatten(), zi.flatten()),
            'metrics_smoothed': calculate_metrics(zi_true.flatten(), zi_smoothed.flatten())
        }

    return results, zi_true

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Interpolation testing with rover data')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth .dat file')
    parser.add_argument('--rover-data', required=True, help='Path to rover samples CSV file')
    parser.add_argument('--output-dir', default='output', help='Directory for output plots')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    x_true, y_true, z_true, boulder_presence = load_ground_truth(args.ground_truth)
    x_rover, y_rover, z_rover = load_rover_samples(args.rover_data)

    # Evaluate interpolation methods
    results, zi_true = evaluate_interpolation(x_rover, y_rover, z_rover, x_true, y_true, z_true)

    # Prepare data for plotting
    xi_grid = []
    yi_grid = []
    zi_grid = []
    sources = []

    for method_name, method_results in results.items():
        # Regular interpolation
        xi_grid.append(method_results['xi'])
        yi_grid.append(method_results['yi'])
        zi_grid.append(method_results['zi'])
        sources.append(f'{method_name}')

        # Smoothed interpolation
        xi_grid.append(method_results['xi'])
        yi_grid.append(method_results['yi'])
        zi_grid.append(method_results['zi_smoothed'])
        sources.append(f'{method_name} (Smoothed)')

    # Plot results
    plot_height_comparisons(xi_grid, yi_grid, zi_grid, zi_true, sources, args.output_dir, save=True)

    # Print metrics
    print("\nInterpolation Metrics:")
    print("=" * 50)
    for method_name, method_results in results.items():
        print(f"\n{method_name}:")
        print("Regular interpolation:")
        for metric, value in method_results['metrics'].items():
            print(f"{metric}: {value:.4f}")
        print("\nSmoothed interpolation:")
        for metric, value in method_results['metrics_smoothed'].items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
