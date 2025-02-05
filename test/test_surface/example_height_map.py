import numpy as np
from pytransform3d.transformations import transform_from
from maple.surface.map import SurfaceHeight

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

def main():
    # Create the geometric map and surface height objects
    geometric_map = MockGeometricMap()
    surface_height = SurfaceHeight(geometric_map)
    
    # Generate sample terrain data
    samples = create_sample_terrain()
    
    # Set the map and generate visualization
    surface_height.set_map(samples)
    
    # Print some statistics about the height map
    height_map = surface_height._last_height_map
    print("\nHeight Map Statistics:")
    print(f"Shape: {height_map.shape}")
    print(f"Min height (excluding NINF): {np.min(height_map[height_map != np.NINF]):.2f}")
    print(f"Max height: {np.max(height_map):.2f}")
    print(f"Mean height (excluding NINF): {np.mean(height_map[height_map != np.NINF]):.2f}")
    print(f"Number of interpolated points: {np.sum(height_map != np.NINF)}")
    
    # Save visualizations
    surface_height.visualize_height_map(save_path='height_map.png')
    print("\nVisualization saved as 'height_map.png'")

if __name__ == "__main__":
    main()
