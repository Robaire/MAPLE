from pyapriltags import Detector
from PIL import Image
import numpy as np


def test_noisy():
    """Create a noisy image and check for detections."""
    rng = np.random.default_rng()
    image = rng.integers(0, 255, (1024, 1024), dtype=np.uint8)
    detector = Detector()
    assert detector is not None

    # No detections should be found
    assert len(detector.detect(image)) == 0


def test_tag():
    """Load an image of a tag and check for detections."""

    # Create the detector
    detector = Detector()
    assert detector is not None

    # Load a sample image
    image = np.array(Image.open("test/tag36_11_00005.png").convert("L"), dtype=np.uint8)

    # Find detections
    detections = detector.detect(image)
    assert len(detections) == 1
    assert detections[0].tag_id == 5
