import numpy as np
from PIL import Image
from pytest import fixture, raises

from maple.boulder.mapper import BoulderMapper
from test.mock_agent import mock_agent


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
    """Test creating a boulder mapper."""

    # Raise an error if required cameras aren't available
    with raises(ValueError):
        BoulderMapper(mock_agent, "FrontLeft", "Front")

    mapper = BoulderMapper(mock_agent, "FrontLeft", "FrontRight")

    # Raise an error if input_data is missing images
    with raises(ValueError):
        mapper(None)

    # Map boulders and check results
    boulders_rover = mapper(input_data)
    assert len(boulders_rover) == 17
