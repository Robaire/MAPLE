import cv2
import numpy as np
from PIL import Image
from pytest import fixture, raises
from pytransform3d.transformations import transform_from, invert_transform, concat
from pytransform3d.rotations import matrix_from_euler

from maple.boulder.detector import BoulderDetector
from maple.utils import camera_parameters
from test.mocks import mock_agent


@fixture
def input_data():
    """A fixture for generating input_data"""

    front_left = np.array(
        Image.open("test/test_boulder/front_left_99.png").convert("L"),
        dtype=np.uint8,
    )

    front_right = np.array(
        Image.open("test/test_boulder/front_right_99.png").convert("L"),
        dtype=np.uint8,
    )

    rng = np.random.default_rng()
    random_image = rng.integers(0, 255, (720, 1280), dtype=np.uint8)

    return {
        "Grayscale": {
            "FrontLeft": front_left,
            "FrontRight": front_right,
            "BackLeft": random_image,  # For testing when no boulders are present
            "BackRight": None,  # For testing when no image is present
        }
    }


def test_boulder(mock_agent, input_data):
    """Test creating and running the mapper."""

    # Raise an error if required cameras aren't available
    with raises(ValueError):
        BoulderDetector(mock_agent, "FrontLeft", "Front")

    mapper = BoulderDetector(mock_agent, "FrontLeft", "FrontRight")

    # Raise an error if input_data is missing images
    with raises(ValueError):
        mapper(None)

    # Map boulders and check results
    boulders_rover = mapper(input_data)
    assert len(boulders_rover) == 13


def _test_visualize_boulders(mock_agent, input_data):
    """Test the results."""

    mapper = BoulderDetector(mock_agent, "FrontLeft", "FrontRight")

    # Get test images
    left_image = input_data["Grayscale"]["FrontLeft"]
    right_image = input_data["Grayscale"]["FrontRight"]

    # Generate artifacts for inspection
    centroids, _ = mapper._find_boulders(left_image)
    depth_map, _ = mapper._depth_map(left_image, right_image)
    boulders_camera = mapper._get_positions(depth_map, centroids)
    # boulders_rover = mapper(input_data)

    # Load the color image
    image = cv2.imread("./test/test_boulder/front_left_99.png")

    # Overlay the centroids on the image
    for centroid in centroids:
        point = (round(centroid[0]), round(centroid[1]))
        cv2.circle(
            image,
            point,
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )

    # Needed to convert from camera coordinates to image coordinates
    fl, _, cx, cy = camera_parameters(left_image.shape)
    camera_image = invert_transform(
        transform_from(
            matrix_from_euler([-np.pi / 2, 0, -np.pi / 2], 2, 1, 0, False),
            [0, 0, 0],
        )
    )

    # Plot boulders in the scene
    for boulder_camera in boulders_camera:
        # Find the image coordinates of each boulder in the scene
        boulder_image = concat(boulder_camera, camera_image)
        x, y, z = boulder_image[:3, 3]
        u = ((x * fl) / z) + cx
        v = ((y * fl) / z) + cy

        cv2.circle(
            image,
            (round(u), round(v)),
            radius=3,
            color=(255, 0, 0),
            thickness=-1,
        )

        # Plot the boulder's position in the camera frame
        x, y, z = boulder_camera[:3, 3]
        cv2.putText(
            image,
            f"({x:.2f}, {y:.2f}, {z:.2f})",
            (round(u) + 5, round(v) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 0, 0),
            thickness=1,
        )

    cv2.imshow("Boulders", image)
    cv2.waitKey()
