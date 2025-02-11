import numpy as np


def compute_blob_mean_and_covariance(binary_image):
    # Create a grid of pixel coordinates.
    y, x = np.indices(binary_image.shape)

    # Threshold the binary image to isolate the blob.
    blob_pixels = (binary_image > 0).astype(int)

    # Compute the mean of pixel coordinates.
    mean_x, mean_y = np.mean(x[blob_pixels == 1]), np.mean(y[blob_pixels == 1])
    mean = np.array([mean_x, mean_y])

    # Stack pixel coordinates to compute covariance using Scipy's cov function.
    pixel_coordinates = np.vstack((x[blob_pixels == 1], y[blob_pixels == 1]))

    # Compute the covariance matrix using Scipy's cov function.
    covariance_matrix = np.cov(pixel_coordinates)

    return mean, covariance_matrix


def plotErrorEllipse(
    ax,
    x,
    y,
    covariance,
    color=None,
    stdMultiplier=1,
    showMean=True,
    idText=None,
    marker=".",
):
    covariance = np.asarray(covariance)

    (lambdas, eigenvectors) = np.linalg.eig(covariance)

    t = np.linspace(0, np.pi * 2, 30)

    lambda1 = lambdas[0]
    lambda2 = lambdas[1]

    scaledEigenvalue1 = stdMultiplier * np.sqrt(lambda1) * np.cos(t)
    scaledEigenvalue2 = stdMultiplier * np.sqrt(lambda2) * np.sin(t)

    scaledEigenvalues = np.vstack((scaledEigenvalue1, scaledEigenvalue2))

    ellipseBorderCoords = eigenvectors @ scaledEigenvalues

    ellipseBorderCoords_x = x + ellipseBorderCoords[0, :]
    ellipseBorderCoords_y = y + ellipseBorderCoords[1, :]

    if color is not None:
        p = ax.plot(ellipseBorderCoords_x, ellipseBorderCoords_y, color=color)
    else:
        p = ax.plot(ellipseBorderCoords_x, ellipseBorderCoords_y)

    if showMean:
        ax.plot(x, y, marker, color=p[0].get_color())

    if idText is not None:
        ax.text(
            x,
            y,
            idText,
            bbox=dict(
                boxstyle="square, pad=-0.1",
                facecolor="white",
                alpha=0.5,
                edgecolor="none",
            ),
            fontsize=8,
        )
