import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from pytest import fixture, raises
from pytransform3d.transformations import transform_from, invert_transform, concat
from pytransform3d.rotations import matrix_from_euler

from maple.boulder.detector import BoulderDetector
from maple.utils import camera_parameters, carla_to_pytransform
from test.mocks import mock_agent
from test.data_parser import CSVParser, CSVAgent


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


def generate_test_data(datadir, indices=None, save_images=False):
    """Generate test data for the boulder detector using stored data

    Args:
        datadir: Path to the data directory
        index: Index of the stereo image pair to process (default: 0)
        mock_agent: Mock agent to use for testing

    Returns:
        dict: Dictionary containing detected boulder positions and input images
    """
    all_data = CSVParser(datadir)
    boulders_rover_all = []

    if indices is None:
        indices = range(0, len(all_data), 25)

    for index in indices:
        # Get stereo images at specified index
        input_data = {
            "Grayscale": {
                "FrontLeft": np.array(all_data.cam("FrontLeft", index)),
                "FrontRight": np.array(all_data.cam("FrontRight", index)),
                "BackLeft": None,
                "BackRight": None,
            }
        }

        mock_agent = CSVAgent()

        # Create detector and process images
        detector = BoulderDetector(mock_agent, "FrontLeft", "FrontRight")
        boulders_rover = detector(input_data)

        if save_images:
            camera_rover = carla_to_pytransform(
                mock_agent.get_camera_position("FrontLeft")
            )
            boulders_camera = [
                concat(boulder_rover, invert_transform(camera_rover))
                for boulder_rover in boulders_rover
            ]
            _visualize_boulders(
                boulders_camera,
                boulders_camera,
                image=all_data.cam("FrontLeft", index, path=True),
            )

        boulders_rover_all.extend(boulders_rover)

    # Save the boulder data to a numpy file
    output_path = Path(datadir) / "boulder_positions.npy"
    np.save(output_path, np.array(boulders_rover_all))
    print(f"Saved boulder positions to {output_path}")

    return boulders_rover_all


def generate_test_data_semantic(datadir, indices=None, save_images=False):
    """Generate test data for the boulder detector using semantic images

    Args:
        datadir: Path to the data directory
        indices: List of indices to process (default: every 25th image)
        save_images: Whether to save visualization images

    Returns:
        list: List of detected boulder positions in rover frame
    """
    all_data = CSVParser(Path(datadir))
    boulders_rover_all = []

    if indices is None:
        indices = range(0, len(all_data), 25)

    # RGB value for boulders in semantic images
    BOULDER_COLOR = np.array([108, 59, 42])

    for index in indices:
        # Get left semantic image
        left_image = np.array(all_data.cam("FrontLeft", index, semantic=True))

        # Create boulder mask (True where pixel matches boulder color exactly)
        boulder_mask = np.all(left_image == BOULDER_COLOR, axis=2)

        # Find all connected components, no matter how small
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            boulder_mask.astype(np.uint8), connectivity=8
        )

        # Skip background component (index 0)
        centroids = centroids[1:]
        stats = stats[1:]

        # # Filter out very small components that might be noise
        # valid_indices = stats[:, cv2.CC_STAT_AREA] > 25
        # centroids = centroids[valid_indices]

        # Get corresponding grayscale images for depth calculation
        input_data = {
            "Grayscale": {
                "FrontLeft": np.array(all_data.cam("FrontLeft", index, semantic=True))
                .mean(axis=2)
                .astype(np.uint8),
                "FrontRight": np.array(all_data.cam("FrontRight", index, semantic=True))
                .mean(axis=2)
                .astype(np.uint8),
                "BackLeft": None,
                "BackRight": None,
            }
        }

        mock_agent = CSVAgent()
        detector = BoulderDetector(mock_agent, "FrontLeft", "FrontRight")

        # Calculate depth map using grayscale images
        depth_map, _ = detector._depth_map(
            input_data["Grayscale"]["FrontLeft"], input_data["Grayscale"]["FrontRight"]
        )

        # Get 3D positions using depth map and centroids
        boulders_camera = detector._get_positions(depth_map, centroids)

        # Transform to rover frame
        camera_rover = carla_to_pytransform(mock_agent.get_camera_position("FrontLeft"))
        # Calculate the boulder positions in the rover frame
        boulders_rover = [
            concat(boulder_camera, camera_rover) for boulder_camera in boulders_camera
        ]
        boulders_rover_all.extend(boulders_rover)

        if save_images:
            _visualize_boulders(
                centroids,
                boulders_camera,
                image=all_data.cam("FrontLeft", index, path=True, semantic=True),
            )

    # Save the boulder data to a numpy file
    output_path = Path(datadir) / "boulder_positions_semantic.npy"
    np.save(output_path, np.array(boulders_rover_all))
    print(f"Saved boulder positions to {output_path}")

    return boulders_rover_all


def test_boulder(mock_agent, input_data):
    """Test creating and running the mapper."""

    # Raise an error if required cameras aren't available
    with raises(ValueError):
        BoulderDetector(mock_agent, "FrontLeft", "Front")

    detector = BoulderDetector(mock_agent, "FrontLeft", "FrontRight")

    # Raise an error if input_data is missing images
    with raises(ValueError):
        detector(None)

    # Map boulders and check results
    boulders_rover = detector(input_data)
    assert len(boulders_rover) == 13


def _test_visualize_boulders(mock_agent, input_data):
    """Test the results."""

    detector = BoulderDetector(mock_agent, "FrontLeft", "FrontRight")

    # Get test images
    left_image = input_data["Grayscale"]["FrontLeft"]
    right_image = input_data["Grayscale"]["FrontRight"]

    # Generate artifacts for inspection
    centroids, _ = mapper._find_boulders(left_image)
    depth_map, _ = mapper._depth_map(left_image, right_image)
    boulders_camera = mapper._get_positions(depth_map, centroids)

    _visualize_boulders(centroids, boulders_camera, left_image)


def _visualize_boulders(centroids, boulders_camera, image):
    """
    Label the boulders' detected centroids in the image.
    """
    # Load the color image
    image_path = Path(image)
    image = cv2.imread(image_path)

    # Overlay the centroids on the image
    for centroid in centroids:
        if centroid.shape == (2,):
            point = (round(centroid[0]), round(centroid[1]))
            cv2.circle(
                image,
                point,
                radius=4,
                color=(0, 0, 255),
                thickness=-1,
            )

    # Needed to convert from camera coordinates to image coordinates
    fl, _, cx, cy = camera_parameters(image.shape)
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
            radius=2,
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

    # output_path = Path(
    #     f"/home/altair_above/Lunar_Autonomy_2025/MAPLE/test/test_boulder/{image_path.stem}_boulders.png"
    # )
    output_path = Path(f"test/test_boulder/{image_path.stem}_boulders.png")
    cv2.imwrite(output_path, image)
    print(f"Annotated boulders image saved to {output_path}")
