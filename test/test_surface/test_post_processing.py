from maple.surface.post_processing import PostProcessor
import numpy as np
import matplotlib.pyplot as plt
from pytest import approx

def test_interpolate_blanks(create_plots=False):
    """
    Test if the post processor can succesfully fill in blank cells in the map using interpolation.
    """
    # Create a simple map based on a smooth function
    true_map = np.zeros((10, 10))
    est_map = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            true_map[i][j] = np.sin(i) + np.cos(j)
            if i+j % 3 == 0:
                est_map[i][j] = np.nan

    # Create the post processor
    pp = PostProcessor()
    pp.height_map = est_map
    est_map = pp.interpolate_blanks()

    if create_plots:
        # Plot the true map
        plt.figure()
        plt.imshow(true_map)
        plt.colorbar()
        plt.title("True map")

        # Plot the estimated map
        plt.figure()
        plt.imshow(est_map)
        plt.colorbar()
        plt.title("Estimated map")

        plt.show()

    # Check the result
    assert est_map == approx(true_map)

if __name__ == "__main__":
    test_interpolate_blanks(create_plots=True)