import numpy as np
from scipy.interpolate import (griddata, RegularGridInterpolator, bisplrep, bisplev,
                             RBFInterpolator, SmoothBivariateSpline)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_grid(x, y, grid_size=100):
    """Create a regular grid based on x, y scattered points."""
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    xi, yi = np.meshgrid(xi, yi, indexing='ij')  # Use 'ij' indexing for correct orientation
    return xi, yi

def linear_interpolation(x, y, z, grid_size=100):
    """Linear interpolation of scattered data."""
    xi, yi = create_grid(x, y, grid_size)
    zi = griddata((x, y), z, (xi, yi), method='linear')
    return xi, yi, zi

def nearest_interpolation(x, y, z, grid_size=100):
    """Nearest neighbor interpolation of scattered data."""
    xi, yi = create_grid(x, y, grid_size)
    zi = griddata((x, y), z, (xi, yi), method='nearest')
    return xi, yi, zi

def regular_grid_interpolation(x, y, z, grid_size=100):
    """Regular grid interpolation using cubic interpolation."""
    xi, yi = create_grid(x, y, grid_size)
    
    # Create regular grid for interpolation
    x_reg = np.linspace(x.min(), x.max(), int(np.sqrt(len(x))))
    y_reg = np.linspace(y.min(), y.max(), int(np.sqrt(len(y))))
    x_mesh, y_mesh = np.meshgrid(x_reg, y_reg, indexing='ij')
    
    # Initial interpolation to regular grid
    z_reg = griddata((x, y), z, (x_mesh, y_mesh), method='linear')
    
    # Handle NaN values
    if np.any(np.isnan(z_reg)):
        z_reg = griddata((x, y), z, (x_mesh, y_mesh), method='nearest')
    
    # Create interpolator
    interp = RegularGridInterpolator((x_reg, y_reg), z_reg,
                                   method='cubic',
                                   bounds_error=False,
                                   fill_value=None)
    
    # Interpolate to desired grid
    pts = np.array([(x, y) for x, y in zip(xi.flatten(), yi.flatten())])
    zi = interp(pts).reshape(grid_size, grid_size)
    
    return xi, yi, zi

def bspline_interpolation(x, y, z, grid_size=100):
    """B-spline interpolation of scattered data using RBF for stability."""
    xi, yi = create_grid(x, y, grid_size)
    
    try:
        # Sort input points
        sort_idx = np.lexsort((y, x))
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        z_sorted = z[sort_idx]
        
        # Remove duplicates
        unique_idx = np.unique(np.column_stack((x_sorted, y_sorted)), axis=0, return_index=True)[1]
        x_unique = x_sorted[unique_idx]
        y_unique = y_sorted[unique_idx]
        z_unique = z_sorted[unique_idx]
        
        # Use RBF interpolation with optimized parameters for better stability
        rbf = RBFInterpolator(np.column_stack((x_unique, y_unique)), z_unique,
                            kernel='quintic',  # Use quintic kernel for smoother interpolation
                            epsilon=0.05,      # Small epsilon for accuracy
                            neighbors=20)      # More neighbors for better local fitting
        
        # Create a fine grid for evaluation
        grid_points = np.column_stack((xi.flatten(), yi.flatten()))
        zi = rbf(grid_points).reshape(grid_size, grid_size)
        
        # Apply light smoothing if needed
        if np.any(np.abs(zi) > np.max(np.abs(z)) * 2):
            from scipy.ndimage import gaussian_filter
            zi = gaussian_filter(zi, sigma=0.5)
        
    except Exception as e:
        print(f"RBF interpolation failed: {str(e)}")
        # Fallback to linear interpolation
        print("Falling back to linear interpolation...")
        zi = griddata((x, y), z, (xi, yi), method='linear')
    
    return xi, yi, zi

def calculate_error(x, y, z, xi, yi, zi):
    """Calculate error metrics between original and interpolated data."""
    interp_z = griddata((xi.flatten(), yi.flatten()), zi.flatten(), (x, y), method='linear')
    mae = np.nanmean(np.abs(z - interp_z))
    rmse = np.sqrt(np.nanmean((z - interp_z)**2))
    return mae, rmse

def plot_comparison(x, y, z, xi, yi, zi, title, sample_rate=5):
    """Create side-by-side comparison of original and interpolated data."""
    fig = plt.figure(figsize=(20, 8))
    
    # Calculate error metrics
    mae, rmse = calculate_error(x, y, z, xi, yi, zi)
    
    # Original scattered points
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(x[::sample_rate], y[::sample_rate], z[::sample_rate],
                         c=z[::sample_rate], cmap='viridis', s=20)
    ax1.set_title(f'Original Points\n(Sampled: 1/{sample_rate})', fontsize=12)
    fig.colorbar(scatter, ax=ax1)
    
    # Interpolated surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(xi, yi, zi, cmap='viridis')
    ax2.set_title(f'{title}\nMAE: {mae:.4f}, RMSE: {rmse:.4f}', fontsize=12)
    fig.colorbar(surf, ax=ax2)
    
    # Set consistent view angles and limits
    for ax in [ax1, ax2]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    # 2D projection comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Original points projection
    scatter = ax1.scatter(x[::sample_rate], y[::sample_rate], 
                         c=z[::sample_rate], cmap='viridis', s=20)
    ax1.set_title(f'Original Points Projection\n(Sampled: 1/{sample_rate})')
    fig.colorbar(scatter, ax=ax1)
    
    # Interpolated surface projection
    im = ax2.contourf(xi, yi, zi, levels=20, cmap='viridis')
    ax2.set_title(f'{title} Projection\nMAE: {mae:.4f}, RMSE: {rmse:.4f}')
    fig.colorbar(im, ax=ax2)
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_all_projections(x, y, z, interpolations, sample_rate=5):
    """Plot all interpolation projections side by side for comparison."""
    n_methods = len(interpolations) - 1 
    fig = plt.figure(figsize=(20, 12))
    
    # Set consistent color range
    vmin = z.min()
    vmax = z.max()
    for method, (xi, yi, zi) in interpolations.items():
        if method != 'Original' and zi is not None:
            vmin = min(vmin, zi.min())
            vmax = max(vmax, zi.max())
    
    # First row: 3D plots
    for i, (method, (xi, yi, zi)) in enumerate(interpolations.items()):
        if method == 'Original':
            continue
            
        ax = fig.add_subplot(2, n_methods, i, projection='3d')
        
        if i == 1:  # First plot also shows original points
            scatter = ax.scatter(x[::sample_rate], y[::sample_rate], z[::sample_rate],
                               c=z[::sample_rate], cmap='viridis', s=10,
                               alpha=0.5, vmin=vmin, vmax=vmax)
            
        surf = ax.plot_surface(xi, yi, zi, cmap='viridis',
                             vmin=vmin, vmax=vmax, alpha=0.8)
        mae, rmse = calculate_error(x, y, z, xi, yi, zi)
        ax.set_title(f'{method}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}')
        
        # Set consistent view angle
        ax.view_init(elev=30, azim=45)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    # Second row: 2D projections
    for i, (method, (xi, yi, zi)) in enumerate(interpolations.items()):
        if method == 'Original':
            continue
            
        ax = fig.add_subplot(2, n_methods, n_methods + i)
        
        if i == 1:  # First plot also shows original points
            ax.scatter(x[::sample_rate], y[::sample_rate],
                      c=z[::sample_rate], cmap='viridis', s=10,
                      alpha=0.5, vmin=vmin, vmax=vmax)
            
        im = ax.contourf(xi, yi, zi, levels=20, cmap='viridis',
                       vmin=vmin, vmax=vmax, alpha=0.8)
        ax.set_title(f'{method} Projection')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
    
    # Add colorbar
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)
    cax = plt.axes([0.96, 0.15, 0.01, 0.7])
    plt.colorbar(im, cax=cax, label='Z Value')
    
    plt.show()

def generate_scattered_points(n_points=5000):
    """Generate scattered points with random peaks."""
    x_base = np.linspace(-9, 9, int(np.sqrt(n_points)))
    y_base = np.linspace(-9, 9, int(np.sqrt(n_points)))
    x_grid, y_grid = np.meshgrid(x_base, y_base)
    
    x = x_grid.flatten() + np.random.normal(0, 0.3, x_grid.size)
    y = y_grid.flatten() + np.random.normal(0, 0.3, y_grid.size)
    
    z_base = np.random.normal(0, 0.2, x.size)
    
    n_peaks = np.random.randint(4, 8)
    for _ in range(n_peaks):
        peak_x = np.random.uniform(-8, 8)
        peak_y = np.random.uniform(-8, 8)
        height = np.random.uniform(0.4, 0.8) * np.random.choice([-1, 1])
        width = np.random.uniform(1.0, 2.5)
        
        r_squared = (x - peak_x)**2 + (y - peak_y)**2
        z_base += height * np.exp(-r_squared / (2 * width**2))
    
    z = z_base + np.random.normal(0, 0.05, x.size)
    z = np.clip(z, -1, 1)
    
    return x, y, z

def main():
    # Generate test data
    x, y, z = generate_scattered_points(5000)
    
    # Set sampling rate for visualization
    sample_rate = 5
    
    # Store all interpolation results
    interpolations = {
        'Original': (None, None, None),
        'Linear': linear_interpolation(x, y, z),
        'Nearest': nearest_interpolation(x, y, z),
        'Regular Grid': regular_grid_interpolation(x, y, z),
        'B-spline': bspline_interpolation(x, y, z)
    }
    
    # Plot individual comparisons
    for method, (xi, yi, zi) in list(interpolations.items())[1:]:  # Skip original points
        plot_comparison(x, y, z, xi, yi, zi, method, sample_rate)
    
    # Plot final comparison of all methods
    plot_all_projections(x, y, z, interpolations, sample_rate)

if __name__ == "__main__":
    main()
