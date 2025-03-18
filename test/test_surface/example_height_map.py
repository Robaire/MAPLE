import numpy as np
from pytransform3d.transformations import transform_from
from maple.surface.map import SurfaceHeight
import matplotlib.pyplot as plt

# Create a mock geometric map (similar to the test class)
class MockGeometricMap:
    def __init__(self):
        self.size = 20  # Larger size for better visualization
        self.heights = np.full((self.size, self.size), np.NINF)
        
    def get_cell_number(self):
        return self.size
        
    def get_cell_indexes(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return int(x), int(y)
        return None
        
    def _is_cell_valid(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size
        
    def set_cell_height(self, x, y, height):
        if self._is_cell_valid(x, y):
            self.heights[x, y] = height

# Create sample data that simulates a hilly terrain
def create_sample_terrain():
    """
    Creates realistic lunar-like terrain samples with features including:
    - Impact craters of varying sizes
    - Highland regions
    - Mare (flat basaltic plains)
    - Small ridges and valleys
    """
    samples = []
    
    # Create a grid for generating the underlying terrain
    x_coords = np.linspace(0, 20, 100)
    y_coords = np.linspace(0, 20, 100)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = np.zeros_like(X)
    
    # Base elevation - gentle rolling highlands
    Z += 2 * np.sin(X/4) + 1.5 * np.cos(Y/3)
    
    def create_crater(x0, y0, radius, depth):
        """Creates a crater with raised rim at specified location"""
        R = np.sqrt((X-x0)**2 + (Y-y0)**2)
        rim_height = depth * 0.2  # Crater rim height
        crater = np.where(R < radius,
                         -depth * (1 - (R/radius)**2) + rim_height * np.exp(-(R-radius)**2/(radius*0.1)**2),
                         0)
        return crater
    
    # Add various craters
    Z += create_crater(8, 7, 3, 2)    # Large crater
    Z += create_crater(15, 12, 2, 1)  # Medium crater
    Z += create_crater(5, 15, 1.5, 0.8)  # Small crater
    Z += create_crater(12, 4, 1, 0.5)   # Tiny crater
    
    # Add mare (flat plain) region
    mare_mask = ((X-10)**2 + (Y-10)**2 < 25)  # Circular mare region
    Z = np.where(mare_mask, -1 + 0.1*np.random.rand(*Z.shape), Z)
    
    # Add small-scale roughness
    Z += 0.1 * np.random.rand(*Z.shape)
    
    # Add some ridges
    Z += 0.2 * np.sin(X/1.5 + Y/2)
    
    # Sample points from the surface (simulate sparse measurements)
    num_samples = 150  # Number of measurement points
    
    # Systematic sampling (like satellite tracks)
    for x in np.linspace(0, 19, 10):
        for y in np.linspace(0, 19, 15):
            # Add some randomness to sampling positions
            x_noise = np.random.normal(0, 0.2)
            y_noise = np.random.normal(0, 0.2)
            sample_x = min(max(x + x_noise, 0), 19)
            sample_y = min(max(y + y_noise, 0), 19)
            
            # Find closest point in our generated terrain
            x_idx = np.abs(x_coords - sample_x).argmin()
            y_idx = np.abs(y_coords - sample_y).argmin()
            height = Z[y_idx, x_idx]
            
            # Add measurement noise
            height += np.random.normal(0, 0.05)
            
            samples.append([sample_x, sample_y, height])
    
    # Add some random samples for additional coverage
    for _ in range(50):
        x = np.random.uniform(0, 19)
        y = np.random.uniform(0, 19)
        x_idx = np.abs(x_coords - x).argmin()
        y_idx = np.abs(y_coords - y).argmin()
        height = Z[y_idx, x_idx]
        height += np.random.normal(0, 0.05)  # Add measurement noise
        samples.append([x, y, height])
    
    return samples

def visualize_height_and_confidence(height_map, confidence_map, save_path='height_and_confidence.png'):
    """Visualize both height map and confidence values."""
    plt.figure(figsize=(15, 6))
    
    # Height Map
    plt.subplot(121)
    masked_height_map = np.ma.masked_where(height_map == np.NINF, height_map)
    im1 = plt.imshow(masked_height_map.T, origin='lower', cmap='gray')  # Changed to gray colormap for lunar-like appearance
    plt.colorbar(im1, label='Height (m)')
    plt.title('Lunar Surface Height Map')
    plt.xlabel('X Cell Index')
    plt.ylabel('Y Cell Index')
    
    # Confidence Map
    plt.subplot(122)
    im2 = plt.imshow(confidence_map.T, origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im2, label='Confidence')
    plt.title('Confidence Map')
    plt.xlabel('X Cell Index')
    plt.ylabel('Y Cell Index')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_accuracy(interpolated_map, samples, geometric_map):
    """Calculate RMSE and MAE between interpolated map and ground truth samples."""
    errors = []
    for sample in samples:
        x, y, z = sample
        cell_indexes = geometric_map.get_cell_indexes(x, y)
        if cell_indexes is not None:
            x_c, y_c = cell_indexes
            if geometric_map._is_cell_valid(x_c, y_c):
                interpolated_height = interpolated_map[x_c, y_c]
                if interpolated_height != np.NINF:
                    errors.append(interpolated_height - z)
    
    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    return rmse, mae

def main():
    # Create the geometric map and surface height objects
    geometric_map = MockGeometricMap()
    surface_height = SurfaceHeight(geometric_map)
    
    # Generate sample terrain data
    samples = create_sample_terrain()
    
    # First create the initial height map without interpolation
    size = geometric_map.get_cell_number()
    initial_height_map = np.full((size, size), np.NINF)
    cell_counts = np.zeros((size, size))

    # Fill in the sample points
    for sample in samples:
        x, y, z = sample
        cell_indexes = geometric_map.get_cell_indexes(x, y)
        if cell_indexes is not None:
            x_c, y_c = cell_indexes
            if geometric_map._is_cell_valid(x_c, y_c):
                if initial_height_map[x_c, y_c] == np.NINF:
                    initial_height_map[x_c, y_c] = 0
                initial_height_map[x_c, y_c] += z
                cell_counts[x_c, y_c] += 1

    nonzero_cells = cell_counts > 0
    initial_height_map[nonzero_cells] /= cell_counts[nonzero_cells]
    
    # Now run the post-processor on the initial height map
    from maple.surface.post_processing import PostProcessor
    post_processor = PostProcessor(initial_height_map)
    interpolated_map, confidence_map = post_processor.interpolate_with_confidence()
    
    # Print statistics
    print("\nHeight Map Statistics:")
    print(f"Shape: {interpolated_map.shape}")
    print(f"Min height (excluding NINF): {np.min(interpolated_map[interpolated_map != np.NINF]):.2f}")
    print(f"Max height: {np.max(interpolated_map):.2f}")
    print(f"Mean height (excluding NINF): {np.mean(interpolated_map[interpolated_map != np.NINF]):.2f}")
    print(f"Number of interpolated points: {np.sum(interpolated_map != np.NINF)}")
    
    print("\nConfidence Statistics:")
    print(f"Min confidence: {np.min(confidence_map):.2f}")
    print(f"Max confidence: {np.max(confidence_map):.2f}")
    print(f"Mean confidence: {np.mean(confidence_map):.2f}")
    print(f"Points with high confidence (>0.8): {np.sum(confidence_map > 0.8)}")
    print(f"Points with medium confidence (0.5-0.8): {np.sum((confidence_map > 0.5) & (confidence_map <= 0.8))}")
    print(f"Points with low confidence (<0.5): {np.sum(confidence_map < 0.5)}")
    
    # Calculate and print accuracy
    rmse, mae = calculate_accuracy(interpolated_map, samples, geometric_map)
    print("\nAccuracy Statistics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Save visualizations
    visualize_height_and_confidence(interpolated_map, confidence_map)
    print("\nVisualization saved as 'height_and_confidence.png'")

if __name__ == "__main__":
    main()
