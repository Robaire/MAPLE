import numpy as np
import matplotlib.pyplot as plt
from surface_interpolation import (linear_interpolation, nearest_interpolation,
                                regular_grid_interpolation, bspline_interpolation,
                                plot_comparison)
import random
import argparse
from sklearn.metrics import r2_score
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

def load_and_prepare_data(file_name):
    """Load the .dat file and prepare data for interpolation."""
    data = np.load(f'{file_name}.dat', allow_pickle=True)
    print(f"Data shape: {data.shape}")
    
    # Extract coordinates and heights
    x = data[:, :, 0].flatten(order='F')  # Use Fortran order to maintain orientation
    y = data[:, :, 1].flatten(order='F')
    z = data[:, :, 2].flatten(order='F')
    boulder_presence = data[:, :, 3].flatten(order='F')
    
    # Remove any NaN values
    valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    boulder_presence = boulder_presence[valid_mask]
    
    # Print data ranges for debugging
    print(f"\nData ranges:")
    print(f"X: {x.min():.4f} to {x.max():.4f}")
    print(f"Y: {y.min():.4f} to {y.max():.4f}")
    print(f"Z: {z.min():.4f} to {z.max():.4f}")
    
    # Print some actual data points for verification
    n_rows = int(np.sqrt(len(x)))
    z_2d = z.reshape(n_rows, n_rows, order='F')  # Use Fortran order for reshaping
    print("\nOriginal data sample (z values):")
    print("Top-left corner:")
    print(z_2d[0:3, 0:3])
    print("\nBottom-left corner:")
    print(z_2d[-3:, 0:3])
    print("\nTop-right corner:")
    print(z_2d[0:3, -3:])
    print("\nBottom-right corner:")
    print(z_2d[-3:, -3:])
    
    return x, y, z, boulder_presence

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
    # Determine the score from the interpolation method
    # Find the absolute error for each point and check if it is less than 0.05
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

def generate_spiral_path(x, y, random_center=True, seed=None, fixed_spiral=False):
    """Generate a spiral path composed of straight line segments."""
    if seed is not None:
        np.random.seed(seed)
        
    # Calculate domain bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    domain_width = x_max - x_min
    domain_height = y_max - y_min
    
    # Generate center offset between -20% and 20% of the domain size if random_center is True
    if random_center:
        center_offset = (
            np.random.uniform(-0.2, 0.2),  # x offset
            np.random.uniform(-0.2, 0.2)   # y offset
        )
    else:
        center_offset = (0, 0)
    
    # Calculate center point with offset
    center_x = (x_max + x_min) / 2 + domain_width * center_offset[0]
    center_y = (y_max + y_min) / 2 + domain_height * center_offset[1]
    if fixed_spiral:
        center_x = 0.
        center_y = 0.
    
    print(f"Spiral center offset: ({center_offset[0]:.2f}, {center_offset[1]:.2f})")
    
    # Start with a smaller radius and gradually increase if needed
    max_radius = 0.7 * min(domain_width, domain_height) / 2  # Reduced from 0.8 to ensure better boundary behavior
    
    num_revolutions = 3
    num_segments = 32
    
    # Generate spiral points
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_segments)
    radius = np.linspace(max_radius * 0.1, max_radius, len(theta))
    
    # Calculate spiral coordinates
    spiral_x = center_x + radius * np.cos(theta)
    spiral_y = center_y + radius * np.sin(theta)
    
    # Check and adjust points that are out of bounds
    while True:
        # Find points that are within bounds
        within_bounds = (
            (spiral_x >= x_min) & (spiral_x <= x_max) &
            (spiral_y >= y_min) & (spiral_y <= y_max)
        )
        
        if np.all(within_bounds):
            break
            
        # If any points are out of bounds, reduce the radius by 5%
        max_radius *= 0.95
        radius = np.linspace(max_radius * 0.1, max_radius, len(theta))
        spiral_x = center_x + radius * np.cos(theta)
        spiral_y = center_y + radius * np.sin(theta)
    
    return spiral_x, spiral_y

def generate_line_segments(x, y, n_segments, min_length=0.2, max_length=0.5, center_bias=0.7, seed=None):
    """Generate random line segments within the data bounds, biased towards the center.
    
    Args:
        x, y: Coordinate arrays
        n_segments: Number of line segments to generate
        min_length, max_length: Length range as fraction of domain size
        center_bias: How much to bias towards center (0-1), higher means more central
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    domain_size = min(x_max - x_min, y_max - y_min)
    
    # Calculate center point
    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2
    
    segments = []
    for _ in range(n_segments):
        # Generate random start point with center bias
        t = np.random.beta(center_bias, center_bias)  # Beta distribution for center bias
        angle = np.random.uniform(0, 2 * np.pi)
        radius = t * domain_size * 0.5  # Reduce max radius to keep points more central
        
        start_x = center_x + radius * np.cos(angle)
        start_y = center_y + radius * np.sin(angle)
        
        # Generate random angle and length for the segment
        segment_angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(min_length, max_length) * domain_size
        
        # Calculate end point
        end_x = start_x + length * np.cos(segment_angle)
        end_y = start_y + length * np.sin(segment_angle)
        
        # Clip to bounds
        start_x = np.clip(start_x, x_min, x_max)
        start_y = np.clip(start_y, y_min, y_max)
        end_x = np.clip(end_x, x_min, x_max)
        end_y = np.clip(end_y, y_min, y_max)
        
        segments.append(((start_x, start_y), (end_x, end_y)))
    
    return segments

def sample_along_path(x, y, path_x, path_y, sample_rate, path_width=0.02, path_density=0.8):
    """Sample points along a path with higher density, with some points outside.
    
    Args:
        x, y: Coordinate arrays
        path_x, path_y: Path coordinates
        sample_rate: Total fraction of points to sample
        path_width: Width of sampling region around the path as fraction of domain size
        path_density: Fraction of points to sample along path vs random (0-1)
    """
    n_points = len(x)
    n_samples = int(n_points * sample_rate)
    
    # Calculate number of points for path and random sampling
    n_path_samples = int(n_samples * path_density)
    n_random = n_samples - n_path_samples
    
    # Create KD-tree for efficient nearest neighbor search
    xy_points = np.column_stack((x, y))
    tree = cKDTree(xy_points)
    
    # Create path segments
    path_points = np.column_stack((path_x, path_y))
    path_segments = np.diff(path_points, axis=0)
    segment_lengths = np.sqrt(np.sum(path_segments**2, axis=1))
    total_length = np.sum(segment_lengths)
    
    # Calculate path width in absolute units
    domain_size = min(x.max() - x.min(), y.max() - y.min())
    absolute_width = domain_size * path_width
    
    # Sample points along the path with higher density
    points_per_length = n_path_samples / total_length
    samples_per_segment = np.maximum(1, (segment_lengths * points_per_length).astype(int))
    
    # Generate points along each segment with controlled width
    path_sampled_points = []
    for i, n_samples in enumerate(samples_per_segment):
        if n_samples > 0:
            # Sample more densely along each segment
            t = np.linspace(0, 1, n_samples + 2)[1:-1]
            
            # Add controlled random offset perpendicular to path direction
            segment = path_segments[i]
            perpendicular = np.array([-segment[1], segment[0]])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            
            for t_val in t:
                base_point = path_points[i] + t_val * segment
                # Use triangular distribution to concentrate points closer to the path
                offset = np.random.triangular(-absolute_width, 0, absolute_width)
                point = base_point + offset * perpendicular
                path_sampled_points.append(point)
    
    path_sampled_points = np.array(path_sampled_points)
    
    # Find nearest actual data points to path points
    _, path_indices = tree.query(path_sampled_points)
    path_indices = np.unique(path_indices)  # Remove duplicates
    
    # Sample additional random points away from the path
    # First, create a mask of points that are far from the path
    all_distances, _ = tree.query(path_points)
    far_mask = all_distances > absolute_width * 2  # Points at least 2x the path width away
    far_indices = np.where(far_mask)[0]
    
    if len(far_indices) > 0:
        random_indices = np.random.choice(far_indices, size=min(n_random, len(far_indices)), replace=False)
    else:
        # Fallback if no points are far enough
        remaining_indices = np.setdiff1d(np.arange(n_points), path_indices)
        random_indices = np.random.choice(remaining_indices, size=n_random, replace=False)
    
    # Combine path and random indices
    sampled_indices = np.concatenate([path_indices, random_indices])
    
    return sampled_indices, path_x, path_y

def sample_clusters(x, y, sample_rate, cluster_radius=0.1, n_clusters=None, along_lines=False, 
                   n_line_segments=None, line_segment_length=0.3, cluster_density=0.85, center_bias=0.7, seed=None):
    """Sample points in clusters, optionally along line segments.
    
    Args:
        x, y: Coordinate arrays
        sample_rate: Fraction of points to sample
        cluster_radius: Radius of clusters as fraction of domain size
        n_clusters: Optional, explicit number of clusters. If None, calculated from sample_rate
        along_lines: If True, place clusters along random line segments
        n_line_segments: Number of line segments if along_lines is True
        line_segment_length: Length of line segments as fraction of domain size
        cluster_density: Fraction of points to place in clusters vs random sampling
        center_bias: How much to bias clusters towards center (0-1), higher means more central
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = len(x)
    n_samples = int(n_points * sample_rate)
    
    # Split samples between clusters and random points
    n_cluster_samples = int(n_samples * cluster_density)
    n_random_samples = n_samples - n_cluster_samples
    
    # Calculate domain size and absolute cluster radius
    domain_size = min(x.max() - x.min(), y.max() - y.min())
    absolute_radius = domain_size * cluster_radius
    
    # Determine number of clusters if not specified
    if n_clusters is None:
        points_per_cluster = int(np.sqrt(n_cluster_samples))
        n_clusters = max(1, n_cluster_samples // points_per_cluster)
    
    points_per_cluster = max(1, n_cluster_samples // n_clusters)
    
    # Generate cluster centers
    if along_lines:
        if n_line_segments is None:
            n_line_segments = max(1, n_clusters // 3)
        
        # Generate line segments
        segments = generate_line_segments(x, y, n_line_segments, 
                                       min_length=line_segment_length * 0.8,
                                       max_length=line_segment_length * 1.2,
                                       center_bias=center_bias,
                                       seed=seed)
        
        # Distribute clusters along segments
        clusters_per_segment = max(1, n_clusters // len(segments))
        centers_x = []
        centers_y = []
        
        for (start_x, start_y), (end_x, end_y) in segments:
            # Generate points along the segment
            t = np.linspace(0, 1, clusters_per_segment)
            seg_x = start_x + t * (end_x - start_x)
            seg_y = start_y + t * (end_y - start_y)
            
            # Add some random offset to avoid perfect alignment
            offset = absolute_radius * 0.2
            seg_x += np.random.uniform(-offset, offset, len(seg_x))
            seg_y += np.random.uniform(-offset, offset, len(seg_y))
            
            centers_x.extend(seg_x)
            centers_y.extend(seg_y)
    else:
        centers_x, centers_y = generate_cluster_centers(x, y, n_clusters, seed)
    
    # Create KD-tree for efficient nearest neighbor search
    xy_points = np.column_stack((x, y))
    tree = cKDTree(xy_points)
    
    # Sample points around each cluster center
    sampled_indices = []
    
    for i in range(len(centers_x)):
        center_x, center_y = centers_x[i], centers_y[i]
        
        # Generate points in a cluster using gaussian distribution
        cluster_size = points_per_cluster
        
        # Generate more points than needed to ensure we have enough after filtering
        n_candidates = cluster_size * 2
        # Increase standard deviation to spread points more
        dx = np.random.normal(0, absolute_radius/2, n_candidates)
        dy = np.random.normal(0, absolute_radius/2, n_candidates)
        
        candidate_x = center_x + dx
        candidate_y = center_y + dy
        
        # Filter points to keep only those within bounds
        valid_mask = (
            (candidate_x >= x.min()) & (candidate_x <= x.max()) &
            (candidate_y >= y.min()) & (candidate_y <= y.max())
        )
        
        valid_x = candidate_x[valid_mask][:cluster_size]
        valid_y = candidate_y[valid_mask][:cluster_size]
        
        # Find nearest actual data points
        cluster_points = np.column_stack((valid_x, valid_y))
        _, indices = tree.query(cluster_points)
        
        sampled_indices.extend(indices)
    
    # Remove duplicates from cluster points
    sampled_indices = np.unique(sampled_indices)
    
    # Add random points away from clusters
    # Create a mask for points that are far from any cluster center
    cluster_centers = np.column_stack((centers_x, centers_y))
    distances, _ = tree.query(cluster_centers)
    far_mask = distances > absolute_radius * 2  # Points at least 2x the cluster radius away
    far_indices = np.where(far_mask)[0]
    
    if len(far_indices) > 0:
        random_indices = np.random.choice(far_indices, size=min(n_random_samples, len(far_indices)), replace=False)
    else:
        # Fallback if no points are far enough
        remaining_indices = np.setdiff1d(np.arange(n_points), sampled_indices)
        random_indices = np.random.choice(remaining_indices, size=n_random_samples, replace=False)
    
    # Combine cluster and random indices
    sampled_indices = np.concatenate([sampled_indices, random_indices])
    
    # If we still have too many points, randomly subsample
    if len(sampled_indices) > n_samples:
        sampled_indices = np.random.choice(sampled_indices, size=n_samples, replace=False)
    
    return sampled_indices, None, None

def generate_cluster_centers(x, y, n_clusters, seed=None):
    """Generate random cluster centers within the data bounds."""
    if seed is not None:
        np.random.seed(seed)
        
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Generate random centers
    centers_x = np.random.uniform(x_min, x_max, n_clusters)
    centers_y = np.random.uniform(y_min, y_max, n_clusters)
    
    return centers_x, centers_y

def sample_data(x, y, sample_rate, use_spiral=False, path_width=0.02, spiral_seed=None, use_clusters=False, 
               cluster_radius=0.1, path_density=0.8, n_clusters=None, along_lines=False, 
               n_line_segments=None, line_segment_length=0.3, cluster_density=0.85, center_bias=0.7):
    """Sample data points using either random, spiral, or cluster sampling."""
    if use_spiral:
        # Generate spiral path with optional seed for consistency
        path_x, path_y = generate_spiral_path(x, y, random_center=(spiral_seed is None), seed=spiral_seed, fixed_spiral=True)
        # Sample along the path with specified width
        sampled_indices, path_x, path_y = sample_along_path(x, y, path_x, path_y, sample_rate, path_width, path_density)
        return sampled_indices, path_x, path_y
    elif use_clusters:
        # Use cluster sampling
        return sample_clusters(x, y, sample_rate, cluster_radius=cluster_radius, n_clusters=n_clusters, 
                              along_lines=along_lines, n_line_segments=n_line_segments, 
                              line_segment_length=line_segment_length, cluster_density=cluster_density, center_bias=center_bias, seed=spiral_seed)
    else:
        # Random sampling
        n_points = len(x)
        n_samples = int(n_points * sample_rate)
        sampled_indices = np.random.choice(n_points, size=n_samples, replace=False)
        return sampled_indices, None, None

def evaluate_interpolation(x_train, y_train, z_train, x_test, y_test, z_test):
    """Evaluate different interpolation methods."""
    methods = {
        'Linear': linear_interpolation,
        'Nearest': nearest_interpolation,
        'Regular Grid': regular_grid_interpolation,
        'B-Spline': bspline_interpolation
    }
    
    results = {}
    # Add points to the x and y boundary of the grid, with the z fixed to the mean of the training data
    x_train_new = np.concatenate([x_test, x_test, [min(x_test), max(x_train)] * len(y_test)])
    y_train_new = np.concatenate([[min(y_test), max(y_test)] * len(x_test), y_test, y_test])
    z_guess = np.mean(z_train)
    z_train_new = np.concatenate([np.full(len(x_test), z_guess), np.full(len(x_test), z_guess), z_test, z_test])
    x_train = np.concatenate([x_train, x_train_new])
    y_train = np.concatenate([y_train, y_train_new])
    z_train = np.concatenate([z_train, z_train_new])


    for name, method in methods.items():
        print(f"\nTesting {name} interpolation...")
        try:
            # Perform interpolation
            xi, yi, zi = method(x_train, y_train, z_train)
            
            # Interpolate at test points
            z_interp = griddata((xi.flatten(), yi.flatten()), 
                              zi.flatten(),
                              (x_test, y_test),
                              method='linear')
            
            # Calculate metrics
            metrics = calculate_metrics(z_test, z_interp)
            metrics['interpolated'] = (xi, yi, zi)
            
            results[name] = metrics
            print(f"Metrics for {name}:")
            for metric, value in metrics.items():
                if metric != 'interpolated':
                    print(f"{metric}: {value:.4f}")
                    
        except Exception as e:
            print(f"Error with {name} interpolation: {str(e)}")
            results[name] = {
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'max_error': np.nan,
                'std_error': np.nan,
                'interpolated': None
            }
    
    return results

def get_sampling_title(sampling_method, args):
    """Get the sampling method title with relevant parameters."""
    if sampling_method == "spiral":
        param_str = f"Path Width: {args.path_width:.3f}, Density: {args.path_density:.1f}"
        if args.batch_mode == 'path_widths':
            param_str = f"Path Density: {args.path_density:.1f}"
        elif args.batch_mode == 'path_densities':
            param_str = f"Path Width: {args.path_width:.3f}"
        return f"Spiral Path ({param_str})"
    elif sampling_method == "cluster":
        return f"Clusters (Radius: {args.cluster_radius:.2f}, Density: {args.cluster_density:.1f})"
    else:
        return "Random"

def plot_results(x, y, z, results, sample_rate, output_dir, args):
    """Plot original surface and interpolated results."""
    n_methods = len(results)
    
    # Calculate number of rows needed (2 rows minimum)
    n_rows = max(2, (n_methods + 4) // 3)  # +4 to include original surface and ensure space for future methods
    
    # Create figure with more height to accommodate titles
    fig = plt.figure(figsize=(20, 6 * n_rows + 2))
    
    # Create a grid with 3 columns and additional space at top
    gs = plt.GridSpec(n_rows, 3, figure=fig, height_ratios=[1] * n_rows)
    gs.update(top=0.85)  # Adjust top margin for main title
    
    # Get sampling method and parameters
    sampling_method = "spiral" if args.spiral_path else "cluster" if args.cluster else "random"
    params_str = get_sampling_title(sampling_method, args)
    
    # Set main title with sampling information
    title = f'Interpolation Comparison - {sampling_method} sampling\nSample Rate: {sample_rate:.1%}'
    if params_str:
        title += f'\n{params_str}'
    plt.suptitle(title, y=0.98, fontsize=16)
    
    # Plot original surface in first position
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
    ax.set_title('Original Surface', pad=20)
    plt.colorbar(scatter)
    
    # Plot interpolated surfaces
    for i, (name, result) in enumerate(results.items()):
        if result['interpolated'] is not None:
            xi, yi, zi = result['interpolated']
            row = (i + 1) // 3  # +1 because original surface takes first position
            col = (i + 1) % 3
            ax = fig.add_subplot(gs[row, col], projection='3d')
            surf = ax.plot_surface(xi, yi, zi, cmap='viridis')
            title = f'{name} Interpolation\n'
            title += f'MAE: {result["mae"]:.4f}, RMSE: {result["rmse"]:.4f}\n'
            title += f'R²: {result["r2"]:.4f}, Max Error: {result["max_error"]:.4f}\n'
            title += f'Score: {result["score"]:.4f}'
            ax.set_title(title, pad=20)
            plt.colorbar(surf)
    
    plt.tight_layout()
    
    # Save the plot with sampling method in filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'interpolation_comparison_{sampling_method}_{sample_rate:.1f}_{timestamp}.png'
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    return filename

def plot_2d_height_comparison(x, y, z, results, sample_rate, output_dir):
    """Create 2D height comparison plots."""
    # Get the grid dimensions from the original data shape
    n_rows = int(np.sqrt(len(x)))
    n_cols = n_rows
    
    # Reshape original data to 2D grid using Fortran order
    z_original = z.reshape(n_rows, n_cols, order='F')
    
    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Height Map Comparison (Sample Rate: {sample_rate:.1%})')
    
    # Plot original surface
    im0 = axes[0, 0].imshow(z_original, extent=[x.min(), x.max(), y.min(), y.max()],
                           origin='lower', aspect='equal')
    axes[0, 0].set_title('Original Surface')
    plt.colorbar(im0, ax=axes[0, 0], label='Height')
    
    # Plot sampled points on original surface
    axes[0, 1].imshow(z_original, extent=[x.min(), x.max(), y.min(), y.max()],
                     origin='lower', aspect='equal')
    axes[0, 1].set_title(f'Sampled Points ({sample_rate:.1%})')
    
    # Get sampled points
    n_samples = int(len(x) * sample_rate)
    indices = np.random.choice(len(x), size=n_samples, replace=False)
    axes[0, 1].scatter(x[indices], y[indices], c='red', s=1, alpha=0.5)
    plt.colorbar(im0, ax=axes[0, 1], label='Height')
    
    # Plot interpolated surfaces
    for idx, (name, result) in enumerate(results.items(), 1):
        if result['interpolated'] is not None:
            xi, yi, zi = result['interpolated']
            metrics = result.get('metrics', result)
            r2 = metrics['r2']
            rmse = metrics['rmse']
            mae = metrics['mae']
            score = metrics['score']
            
            row = (idx + 1) // 3
            col = (idx + 1) % 3
            
            im = axes[row, col].imshow(zi, extent=[x.min(), x.max(), y.min(), y.max()],
                                     origin='lower', aspect='equal')
            axes[row, col].set_title(f'{name}\nR²: {r2:.4f}, RMSE: {rmse:.4f}\nMAE: {mae:.4f}, Score: {score:.4f}')
            plt.colorbar(im, ax=axes[row, col], label='Height')
    
    # Remove empty subplot if any
    if len(results) + 2 < 6:  # +2 for original and sampled plots
        axes[1, 2].remove()
    
    # Set common labels and adjust layout
    for ax in axes.flat:
        if ax.get_subplotspec().get_geometry()[2] < len(results) + 2:  # Skip removed subplot
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
    
    plt.tight_layout()
    filename = f'height_comparison_{sample_rate:.1f}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    return filename

def create_interactive_plots(x, y, z, results, sample_rate, output_dir, args):
    """Create interactive 3D surface plots using plotly."""
    # Create interactive_graphs subfolder
    interactive_dir = os.path.join(output_dir, 'interactive_graphs')
    os.makedirs(interactive_dir, exist_ok=True)
    
    # Get sampling method and parameters
    sampling_method = "spiral" if args.spiral_path else "cluster" if args.cluster else "random"
    params_str = get_sampling_title(sampling_method, args)
    
    # Get the grid dimensions from the original data shape
    n_rows = int(np.sqrt(len(x)))
    n_cols = n_rows
    
    # Reshape original data to 2D grid
    z_original = z.reshape(n_rows, n_cols, order='F')
    x_grid, y_grid = np.meshgrid(np.unique(x), np.unique(y), indexing='ij')
    
    # Create separate HTML files for each interpolation method
    for name, result in results.items():
        if result['interpolated'] is not None:
            xi, yi, zi = result['interpolated']
            r2 = result['r2']
            rmse = result['rmse']
            mae = result['mae']
            
            # Interpolate zi to match original grid size
            x_interp = np.linspace(x.min(), x.max(), n_rows)
            y_interp = np.linspace(y.min(), y.max(), n_cols)
            x_mesh, y_mesh = np.meshgrid(x_interp, y_interp, indexing='ij')
            zi_regrid = griddata((xi.flatten(), yi.flatten()), zi.flatten(),
                               (x_mesh, y_mesh), method='linear')
            
            # Calculate differences
            diff = zi_regrid - z_original
            abs_diff = np.abs(diff)
            max_diff = np.max(abs_diff)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=1,
                specs=[[{'type': 'surface'}],
                       [{'type': 'heatmap'}]],
                subplot_titles=('Surface Comparison', 'Height Difference Map'),
                vertical_spacing=0.1,
                row_heights=[0.6, 0.4]
            )
            
            # Add original surface with higher opacity
            fig.add_trace(
                go.Surface(
                    x=x_grid,
                    y=y_grid,
                    z=z_original,
                    colorscale='Viridis',
                    name='Original',
                    showscale=True,
                    opacity=0.8,
                    colorbar=dict(
                        x=1.15,
                        y=0.8,
                        len=0.3,
                        title=dict(
                            text='Height',
                            side='right'
                        )
                    )
                ),
                row=1, col=1
            )
            
            # Add interpolated surface with lower opacity and difference-based coloring
            fig.add_trace(
                go.Surface(
                    x=x_grid,
                    y=y_grid,
                    z=zi_regrid,
                    surfacecolor=abs_diff,  # Color by absolute difference
                    colorscale='RdBu_r',    # Reverse RdBu to show larger differences in red
                    name='Interpolated',
                    showscale=True,
                    opacity=0.5,
                    colorbar=dict(
                        x=1.0,
                        y=0.8,
                        len=0.3,
                        title=dict(
                            text='|Difference|',
                            side='right'
                        )
                    ),
                    hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Height: %{z:.3f}<br>Difference: %{surfacecolor:.3f}<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Add difference heatmap
            fig.add_trace(
                go.Heatmap(
                    z=diff,
                    x=np.unique(x),
                    y=np.unique(y),
                    colorscale='RdBu',
                    zmid=0,  # Center the colorscale at 0
                    showscale=True,
                    colorbar=dict(
                        x=1.15,
                        y=0.3,
                        len=0.3,
                        title=dict(
                            text='Height Difference',
                            side='right'
                        ),
                        tickformat='.3f'
                    ),
                    hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Difference: %{z:.3f}<extra></extra>",
                    name='Difference'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'{name} Interpolation (Sample Rate: {sample_rate:.1%})<br>R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}',
                    x=0.5,
                    y=0.98,
                    font=dict(size=16)
                ),
                scene=dict(
                    aspectmode='cube',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Height'
                ),
                width=1200,
                height=1200,
                margin=dict(t=100, b=20, l=20, r=150),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.95,
                    xanchor="left",
                    x=0.1,
                    bgcolor='rgba(255,255,255,0.8)'
                )
            )
            
            # Update axes labels for heatmap
            fig.update_xaxes(title_text="X Position", row=2, col=1)
            fig.update_yaxes(title_text="Y Position", row=2, col=1)
            
            # Save the plot in interactive_graphs subfolder
            filename = f'interactive_{name.lower().replace(" ", "_")}_{sample_rate:.1f}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(os.path.join(interactive_dir, filename))
            print(f'Generated interactive plot for {name}: {filename}')
    
    print('Generated interactive plot: Interactive plots generated')
    return True

def plot_height_comparison(x, y, z, results, sample_rate, output_dir, sampled_indices=None, path_x=None, path_y=None, args=None):
    """Create a height comparison plot showing original surface and interpolation results."""
    n_rows = 2  # Fixed 2 rows
    n_cols = 3  # Fixed 3 columns
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
    
    # Get sampling method and parameters
    sampling_method = "spiral" if args.spiral_path else "cluster" if args.cluster else "random"
    params_str = get_sampling_title(sampling_method, args)
    
    # Set main title with sampling information
    title = f'Height Map Comparison - {sampling_method} Sampling\nSample Rate: {sample_rate:.1%}'
    if params_str:
        title += f'\n{params_str}'
    fig.suptitle(title, y=0.98, fontsize=16)
    
    # Reshape original data
    n_points = int(np.sqrt(len(x)))
    z_original = z.reshape(n_points, n_points)
    
    # Calculate point size based on data resolution
    dx = (x.max() - x.min()) / n_points
    dy = (y.max() - y.min()) / n_points
    point_size = 1  # Base size in points
    
    # Create meshgrid for plotting
    x_grid = np.linspace(x.min(), x.max(), n_points)
    y_grid = np.linspace(y.min(), y.max(), n_points)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Plot original surface without points
    im = axes[0, 0].imshow(z_original, origin='lower', aspect='equal', 
                          extent=[x.min(), x.max(), y.min(), y.max()])
    axes[0, 0].set_title('Original Surface')
    if path_x is not None and path_y is not None:
        axes[0, 0].plot(path_x, path_y, 'y-', linewidth=0.5, alpha=0.8, label='Sampling Path')
    plt.colorbar(im, ax=axes[0, 0], label='Height')
    
    # Plot original surface with sampled points
    im = axes[0, 1].imshow(z_original, origin='lower', aspect='equal', 
                          extent=[x.min(), x.max(), y.min(), y.max()])
    axes[0, 1].set_title(f'Sampled Points ({sample_rate:.1%})')
    
    # Overlay sampled points if provided
    if sampled_indices is not None:
        x_sampled = x[sampled_indices]
        y_sampled = y[sampled_indices]
        axes[0, 1].scatter(x_sampled, y_sampled, c='red', s=point_size, alpha=0.5)
        if path_x is not None and path_y is not None:
            axes[0, 1].plot(path_x, path_y, 'y-', linewidth=0.5, alpha=0.8)
    plt.colorbar(im, ax=axes[0, 1], label='Height')
    
    # Plot interpolation results
    plot_idx = 2  # Start from third plot
    for name, result in results.items():
        if result['interpolated'] is not None:
            xi, yi, zi = result['interpolated']
            metrics = result.get('metrics', result)
            r2 = metrics['r2']
            rmse = metrics['rmse']
            mae = metrics['mae']
            score = metrics['score']
            
            row = plot_idx // 3
            col = plot_idx % 3
            
            # Create interpolation grid
            xi_grid = np.linspace(x.min(), x.max(), n_points)
            yi_grid = np.linspace(y.min(), y.max(), n_points)
            Xi, Yi = np.meshgrid(xi_grid, yi_grid)
            Zi = griddata((xi.ravel(), yi.ravel()), zi.ravel(), (Xi, Yi), method='linear')
            
            im = axes[row, col].imshow(Zi, origin='lower', aspect='equal',
                                     extent=[x.min(), x.max(), y.min(), y.max()])
            axes[row, col].set_title(f'{name}\nR²: {r2:.4f}, RMSE: {rmse:.4f}\nMAE: {mae:.4f}, Score: {score:.4f}')
            plt.colorbar(im, ax=axes[row, col], label='Height')
            
            plot_idx += 1
    
    # Remove any remaining empty subplots
    for i in range(n_rows):
        for j in range(n_cols):
            if i * n_cols + j >= plot_idx:
                axes[i, j].remove()
    
    plt.tight_layout()
    
    # Save with sampling method in filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'height_comparison_{sampling_method}_{sample_rate:.1f}_{timestamp}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f'Generated height comparison: {filename}')
    plt.close()
    
    return filename

def plot_interactive_3d_multi(x, y, z, all_results, output_dir):
    """Create interactive 3D surface plots with multiple sample rates."""
    methods = list(next(iter(all_results.values())).keys())
    
    for method in methods:
        fig = go.Figure()
        
        # Add original surface
        z_grid = z.reshape(len(np.unique(x)), len(np.unique(y)), order='F')
        fig.add_trace(
            go.Surface(
                x=np.unique(x),
                y=np.unique(y),
                z=z_grid,
                colorscale='viridis',
                name='Original',
                showscale=False,
                opacity=0.7
            )
        )
        
        # Add interpolated surfaces for each sample rate
        for sample_rate, results in all_results.items():
            if results[method]['interpolated'] is not None:
                xi, yi, zi = results[method]['interpolated']
                fig.add_trace(
                    go.Surface(
                        x=xi[0],
                        y=yi[:, 0],
                        z=zi,
                        colorscale='viridis',
                        name=f'Sample Rate: {sample_rate:.1%}',
                        opacity=0.7,
                        visible='legendonly'  # Initially hide all interpolated surfaces
                    )
                )
        
        fig.update_layout(
            title=f'{method} Interpolation - Multiple Sample Rates',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Height'
            ),
            width=1200,
            height=800,
            showlegend=True
        )
        
        filename = f'interactive_multi_{method.lower().replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        fig.write_html(os.path.join(output_dir, filename))
        print(f"Generated multi-sample interactive plot for {method}: {filename}")

def plot_summary(all_results, output_dir, batch_mode):
    """Plot summary of metrics across different sample rates."""
    metrics = ['mae', 'rmse', 'r2', 'max_error', 'std_error']
    methods = list(next(iter(all_results.values())).keys())
    sample_rates = sorted(all_results.keys())
    
    # Get x-axis label and values based on batch mode
    if batch_mode == 'sample_rates':
        x_label = 'Sample Rate (%)'
        x_values = [x * 100 for x in sample_rates]
    elif batch_mode == 'path_widths':
        x_label = 'Path Width'
        x_values = sample_rates
    elif batch_mode == 'path_densities':
        x_label = 'Path Density (%)'
        x_values = [x * 100 for x in sample_rates]
    elif batch_mode == 'cluster_sizes':
        x_label = 'Cluster Radius'
        x_values = sample_rates
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for method in methods:
            values = [all_results[rate][method][metric] for rate in sample_rates]
            plt.plot(x_values, values, 'o-', label=method)
        
        plt.xlabel(x_label)
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} vs {x_label}')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'metric_summary_{metric}_{timestamp}.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Generated summary plot: {filename}")

def plot_error_comparison(all_results, output_dir, batch_mode):
    """Create plots comparing error metrics across different sampling rates."""
    # Get x-axis label and values based on batch mode
    if batch_mode == 'sample_rates':
        x_label = 'Sample Rate (%)'
        x_values = [x * 100 for x in sorted(all_results.keys())]
    elif batch_mode == 'path_widths':
        x_label = 'Path Width'
        x_values = sorted(all_results.keys())
    elif batch_mode == 'path_densities':
        x_label = 'Path Density (%)'
        x_values = [x * 100 for x in sorted(all_results.keys())]
    elif batch_mode == 'cluster_sizes':
        x_label = 'Cluster Radius'
        x_values = sorted(all_results.keys())

    metrics = ['r2', 'rmse', 'mae', 'max_error']
    titles = {'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE', 'max_error': 'Maximum Error'}
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for method in ['Linear', 'Nearest', 'Regular Grid', 'B-Spline']:
            y_values = [all_results[x][method][metric] for x in sorted(all_results.keys())]
            plt.plot(x_values, y_values, marker='o', label=method)
        
        plt.xlabel(x_label)
        plt.ylabel(titles[metric])
        plt.title(f'{titles[metric]} vs {x_label}')
        plt.grid(True)
        plt.legend()
        
        filename = f'error_comparison_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f'Generated error comparison plot: {filename}')
        plt.close()

def plot_batch_metrics(all_results, output_dir, batch_mode):
    """Plot batch metrics for different parameters."""
    # Generate summary plots for each metric
    plot_summary(all_results, output_dir, batch_mode)
    # Generate comparison plots
    plot_error_comparison(all_results, output_dir, batch_mode)

def process_and_plot(x, y, z, boulder_presence, sampled_indices, path_x, path_y, sample_rate, output_dir, args):
    """Process the sampled data and generate all plots."""
    # Prepare training and test data
    x_train = x[sampled_indices]
    y_train = y[sampled_indices]
    z_train = z[sampled_indices]
    x_test = np.delete(x, sampled_indices)
    y_test = np.delete(y, sampled_indices)
    z_test = np.delete(z, sampled_indices)
    
    # Test interpolation methods
    results = evaluate_interpolation(x_train, y_train, z_train, x_test, y_test, z_test)
    
    # Generate comparison plot
    plot_filename = plot_results(x, y, z, results, sample_rate, output_dir, args)
    print(f"Generated plot: {plot_filename}")
    
    # Generate interactive plots
    interactive_filename = create_interactive_plots(x, y, z, results, sample_rate, output_dir, args)
    print(f"Generated interactive plot: {interactive_filename}")
    
    # Generate height comparison
    height_filename = plot_height_comparison(x, y, z, results, sample_rate, output_dir, 
                                          sampled_indices, path_x, path_y, args)
    print(f"Generated height comparison: {height_filename}")
    
    return results

def main():
    """Main function for running interpolation tests."""
    parser = argparse.ArgumentParser(description='Test interpolation methods on lunar surface data.')
    parser.add_argument('filename', help='Input file name')
    parser.add_argument('--sample-rate', type=float, help='Single sample rate to test (0-1)')
    parser.add_argument('--batch', action='store_true', help='Run batch processing with predefined sample rates')
    parser.add_argument('--spiral-path', action='store_true', help='Use spiral path sampling')
    parser.add_argument('--cluster', action='store_true', help='Use cluster sampling')
    parser.add_argument('--path-width', type=float, default=0.02,
                       help='Width of sampling region around the path as fraction of domain size (default: 0.02)')
    parser.add_argument('--path-density', type=float, default=0.8,
                       help='Fraction of points to sample along path vs random (default: 0.8)')
    parser.add_argument('--cluster-radius', type=float, default=0.1,
                       help='Radius of clusters as fraction of domain size (default: 0.1)')
    parser.add_argument('--n-clusters', type=int, help='Number of clusters (default: auto)')
    parser.add_argument('--along-lines', action='store_true', help='Place clusters along line segments')
    parser.add_argument('--n-line-segments', type=int, help='Number of line segments for cluster placement')
    parser.add_argument('--line-length', type=float, default=0.3,
                       help='Length of line segments as fraction of domain size (default: 0.3)')
    parser.add_argument('--output-dir', help='Output directory', default='interpolation_results')
    parser.add_argument('--batch-mode', choices=['sample_rates', 'path_widths', 'cluster_sizes', 'path_densities'], 
                       help='Batch processing mode: vary sample rates, path widths, cluster sizes, or path densities')
    parser.add_argument('--cluster-density', type=float, default=0.85,
                       help='Fraction of points to place in clusters vs random sampling (default: 0.85)')
    parser.add_argument('--center-bias', type=float, default=0.7,
                       help='How much to bias clusters towards center (0-1) (default: 0.7)')
    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    x, y, z, boulder_presence = load_and_prepare_data(args.filename)
    
    # Set fixed random seed for consistent generation in batch mode
    sampling_seed = 42 if args.batch else None
    
    if args.batch:
        all_batch_results = {}  # Store results for batch comparison
        
        if args.batch_mode == 'sample_rates':
            # Fixed width/radius, varying sample rates
            SAMPLE_RATES = [0.05, 0.1, 0.2, 0.3, 0.5]
            
            for sample_rate in SAMPLE_RATES:
                print(f"\n=== Testing with sample rate: {sample_rate*100:.1f}% ===")
                
                sampled_indices, path_x, path_y = sample_data(x, y, sample_rate, 
                                                            use_spiral=args.spiral_path,
                                                            path_width=args.path_width,
                                                            spiral_seed=sampling_seed,
                                                            use_clusters=args.cluster,
                                                            cluster_radius=args.cluster_radius,
                                                            path_density=args.path_density,
                                                            n_clusters=args.n_clusters,
                                                            along_lines=args.along_lines,
                                                            n_line_segments=args.n_line_segments,
                                                            line_segment_length=args.line_length,
                                                            cluster_density=args.cluster_density,
                                                            center_bias=args.center_bias)
                
                results = process_and_plot(x, y, z, boulder_presence, sampled_indices, path_x, path_y,
                                         sample_rate, output_dir, args)
                all_batch_results[sample_rate] = results
                
        elif args.batch_mode == 'path_widths':
            # Fixed sample rate, varying path widths
            PATH_WIDTHS = [0.01, 0.02, 0.03, 0.04, 0.05]
            sample_rate = args.sample_rate or 0.2
            
            for path_width in PATH_WIDTHS:
                print(f"\n=== Testing with path width: {path_width:.3f} ===")
                
                sampled_indices, path_x, path_y = sample_data(x, y, sample_rate,
                                                            use_spiral=args.spiral_path,
                                                            path_width=path_width,
                                                            spiral_seed=sampling_seed,
                                                            path_density=args.path_density)
                
                results = process_and_plot(x, y, z, boulder_presence, sampled_indices, path_x, path_y,
                                         sample_rate, output_dir, args)
                all_batch_results[path_width] = results
                
        elif args.batch_mode == 'path_densities':
            # Fixed sample rate and path width, varying path densities
            PATH_DENSITIES = [0.2, 0.4, 0.6, 0.8, 0.95]
            sample_rate = args.sample_rate or 0.2
            
            for path_density in PATH_DENSITIES:
                print(f"\n=== Testing with path density: {path_density*100:.1f}% ===")
                
                sampled_indices, path_x, path_y = sample_data(x, y, sample_rate,
                                                            use_spiral=args.spiral_path,
                                                            path_width=args.path_width,
                                                            spiral_seed=sampling_seed,
                                                            path_density=path_density)
                
                results = process_and_plot(x, y, z, boulder_presence, sampled_indices, path_x, path_y,
                                         sample_rate, output_dir, args)
                all_batch_results[path_density] = results
        
        # Generate batch comparison plots
        plot_batch_metrics(all_batch_results, output_dir, args.batch_mode)
    else:
        # Single run mode
        sample_rate = args.sample_rate
        if sample_rate is None:
            print("Error: --sample-rate is required when not in batch mode")
            sys.exit(1)
            
        sampled_indices, path_x, path_y = sample_data(x, y, sample_rate,
                                                    use_spiral=args.spiral_path,
                                                    path_width=args.path_width,
                                                    spiral_seed=sampling_seed,
                                                    use_clusters=args.cluster,
                                                    cluster_radius=args.cluster_radius,
                                                    path_density=args.path_density,
                                                    n_clusters=args.n_clusters,
                                                    along_lines=args.along_lines,
                                                    n_line_segments=args.n_line_segments,
                                                    line_segment_length=args.line_length,
                                                    cluster_density=args.cluster_density,
                                                    center_bias=args.center_bias)
        process_and_plot(x, y, z, boulder_presence, sampled_indices, path_x, path_y,
                        sample_rate, output_dir, args)

if __name__ == "__main__":
    print("test")
    main()
