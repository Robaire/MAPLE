import numpy as np
import math

# from maple.boulder.map import BoulderMap
from maple.navigation.navigator import angle_helper


def test_pure_pursuit():

    test_cases = [
        {"name": "Facing correct direction", "start": (0, 0), "yaw": 0, "end": (1, 0), "expected": 0},
        {"name": "Slightly off", "start": (0, 0), "yaw": math.radians(10), "end": (1, 0), "expected_range": (-math.radians(10), 0)},
        {"name": "90 degrees off (left)", "start": (0, 0), "yaw": math.pi / 2, "end": (1, 0), "expected": -math.pi / 2},
        {"name": "90 degrees off (right)", "start": (0, 0), "yaw": -math.pi / 2, "end": (1, 0), "expected": math.pi / 2},
        {"name": "Completely wrong direction", "start": (0, 0), "yaw": math.pi, "end": (1, 0), "expected": -math.pi},
        {"name": "Slightly past 90 degrees", "start": (0, 0), "yaw": math.radians(100), "end": (1, 0), "expected_range": (-math.pi, -math.pi / 2)}
    ]

    for test in test_cases:
        start_x, start_y = test["start"]
        yaw = test["yaw"]
        end_x, end_y = test["end"]

        result = angle_helper(start_x, start_y, yaw, end_x, end_y)

        if "expected_range" in test:
            if test["expected_range"][0] <= result <= test["expected_range"][1]:
                print(f"✅ {test['name']} Passed: Output {result:.3f} is within {test['expected_range']}")
            else:
                print(f"❌ {test['name']} Failed: Output {result:.3f} not within {test['expected_range']}")
        else:
            if math.isclose(result, test["expected"], abs_tol=0.01):
                print(f"✅ {test['name']} Passed: Output {result:.3f} matches expected {test['expected']:.3f}")
            else:
                print(f"❌ {test['name']} Failed: Output {result:.3f} does not match expected {test['expected']:.3f}")
