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
    samples = []
    
    # Create a circular pattern of samples
    center_x, center_y = 10, 10
    for radius in range(1, 8, 2):
        for angle in np.linspace(0, 2*np.pi, 8*radius):
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            # Height varies with distance from center
            height = 5 * np.sin(radius) + np.random.normal(0, 0.2)
            samples.append([x, y, height])
    
    # Add some random samples
    for _ in range(20):
        x = np.random.uniform(0, 20)
        y = np.random.uniform(0, 20)
        height = 2 * np.sin(x/5) * np.cos(y/5) + np.random.normal(0, 0.2)
        samples.append([x, y, height])
    
    return samples

def visualize_height_and_confidence(height_map, confidence_map, save_path='height_and_confidence.png'):
    """Visualize both height map and confidence values."""
    plt.figure(figsize=(15, 6))
    
    # Height Map
    plt.subplot(121)
    masked_height_map = np.ma.masked_where(height_map == np.NINF, height_map)
    im1 = plt.imshow(masked_height_map.T, origin='lower', cmap='terrain')
    plt.colorbar(im1, label='Height')
    plt.title('Surface Height Map')
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
    
    # Save visualizations
    visualize_height_and_confidence(interpolated_map, confidence_map)
    print("\nVisualization saved as 'height_and_confidence.png'")

if __name__ == "__main__":
    main()
