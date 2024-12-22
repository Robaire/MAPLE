import numpy as np
from noise import pnoise2

def generate_ground(height, width, vertical_height, scale=10, octaves=4, persistence=0.5, lacunarity=2.0):
    """
    Generate a ground height map with smooth, hill-like features using Perlin noise.

    :param height: Number of rows in the array.
    :param width: Number of columns in the array.
    :param scale: Determines the size of features (larger = smoother terrain).
    :param octaves: Number of layers of noise to combine for more detail.
    :param persistence: Amplitude reduction factor for each octave (0.5 is typical).
    :param lacunarity: Frequency increase factor for each octave (2.0 is typical).
    :return: 2D numpy array of ground heights.
    """
    ground = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            ground[i][j] = pnoise2(i / scale, j / scale, octaves=octaves, 
                                   persistence=persistence, lacunarity=lacunarity, 
                                   repeatx=width, repeaty=height, base=42)
    # Normalize to 0 - 1
    ground = (ground - np.min(ground)) / (np.max(ground) - np.min(ground))

    # Normalize to go to vertical max height
    ground *= vertical_height

    return ground

if __name__ == '__main__':
    # Example usage
    height = 100
    width = 100
    vertical_height = 5
    ground = generate_ground(height, width, vertical_height)

    # Visualize the ground using matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(ground, cmap='terrain')
    plt.colorbar(label="Height")
    plt.title("Random Hill-Like Ground Heights")
    plt.show()
