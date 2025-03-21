import numpy as np
import math

from maple.navigation.drive_control import angle_helper


def test_pure_pursuit():

    test_cases = [
        {"name": "Facing correct direction", "start": (0, 0), "yaw": 0, "end": (1, 0), "expected": 0},
        {"name": "Slightly off", "start": (0, 0), "yaw": math.radians(10), "end": (1, 0), "expected": -math.radians(10)},
        {"name": "90 degrees off (left)", "start": (0, 0), "yaw": math.pi / 2, "end": (1, 0), "expected": -math.pi / 2},
        {"name": "90 degrees off (right)", "start": (0, 0), "yaw": -math.pi / 2, "end": (1, 0), "expected": math.pi / 2},
        {"name": "Completely wrong direction", "start": (0, 0), "yaw": math.pi, "end": (1, 0), "expected": -math.pi},
        {"name": "Wrong direction from the sim", "start": (-2.5666699409484863, 0.8248041272163391), "yaw": 1.8352155993164114, "end": (8.311442308832195, -2.5345112376503365), "expected_range": (-math.pi, -1.535)},
        {"name": "Slightly past 90 degrees", "start": (0, 0), "yaw": math.radians(100), "end": (1, 0), "expected_range": (-math.pi, -math.pi / 2)},
        {"name": "This should fail. this is wrong direction in sim", "start": (-1.0750133991241455, -7.091660499572754), "yaw": -1.41277899509559, "end": (0.008629743236936155, -0.005052477774770978), "expected": -0.006278516233543696 + math.pi}
    ]

    for test in test_cases:
        start_x, start_y = test["start"]
        yaw = test["yaw"]
        end_x, end_y = test["end"]

        result = angle_helper(start_x, start_y, yaw, end_x, end_y)

        if "expected_range" in test:
            if test["expected_range"][0] <= result <= test["expected_range"][1]:
                print(f"✅ {test['name']} Passed: Output {result:.3f} is within {test['expected_range']}")
                assert True
            else:
                print(f"❌ {test['name']} Failed: Output {result:.3f} not within {test['expected_range']}")
                assert False
        else:
            if math.isclose(result, test["expected"], abs_tol=0.01):
                print(f"✅ {test['name']} Passed: Output {result:.3f} matches expected {test['expected']:.3f}")
                assert True
            else:
                print(f"❌ {test['name']} Failed: Output {result:.3f} does not match expected {test['expected']:.3f}")
                assert False