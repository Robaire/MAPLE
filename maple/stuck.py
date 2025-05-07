from collections import deque

import numpy as np


class StuckDetector:
    unstuck_sequence = [
        {"lin_vel": -0.45, "ang_vel": 0, "frames": 200},  # Backward
        {"lin_vel": 0, "ang_vel": 4, "frames": 100},  # Rotate clockwise
        {"lin_vel": 0.45, "ang_vel": 0, "frames": 200},  # Forward
        {"lin_vel": 0, "ang_vel": -4, "frames": 100},  # Rotate counter-clockwise
    ]

    def __init__(self, stuck_frames=2000, stuck_threshold=2.0, unstuck_threshold=2.0):
        self.position_history = deque(maxlen=stuck_frames)
        self.stuck = False
        self.stuck_threshold = stuck_threshold
        self.unstuck_threshold = unstuck_threshold

        self.unstuck_phase = 0
        self.frames = 0

    def get_unstuck_control(self) -> tuple[float, float]:
        """Get the control input for the rover."""

        # Get the control input for the current phase
        linear_vel = self.unstuck_sequence[self.unstuck_phase]["lin_vel"]
        angular_vel = self.unstuck_sequence[self.unstuck_phase]["ang_vel"]
        frames = self.unstuck_sequence[self.unstuck_phase]["frames"]

        # Increment the frame counter
        self.frames += 1

        # Keep track of the number of frames in the current phase
        if self.frames > frames:
            # Add one to the phase and reset if necessary
            self.unstuck_phase += 1
            self.unstuck_phase = self.unstuck_phase % len(self.unstuck_sequence)
            self.frames = 0

        # Return the control input
        return (linear_vel, angular_vel)

    def is_stuck(self, rover_global) -> bool:
        """Check if the rover is stuck."""

        if rover_global is not None:
            self.position_history.append(rover_global[:2, 3])

        # If the rover is stuck, check if it has gotten unstuck
        if self.stuck:
            if self._check_if_unstuck():
                self.stuck = False
                return False

        # If the rover is not stuck, check if it has gotten stuck
        else:
            if self._check_if_stuck():
                self.stuck = True
                return True

        # Return the current stuck state
        return self.stuck

    def _distance_moved(self) -> float:
        """Get the distance moved by the rover between the most recent and oldest position."""

        if len(self.position_history) > 0:
            return np.linalg.norm(self.position_history[-1] - self.position_history[0])
        else:
            return 0

    def _check_if_stuck(self) -> bool:
        """Check if the rover has moved less than the stuck threshold."""

        # If we don't have enough history, return False
        if len(self.position_history) < self.stuck_frames:
            return False

        distance = self._distance_moved()
        if distance < self.stuck_threshold:
            print(
                f"Stuck detected! Moved {distance:.2f}m in the last {self.stuck_frames} estimates."
            )
            return True

        return False

    def _check_if_unstuck(self) -> bool:
        """Check if the rover has moved more than the unstuck threshold."""

        if self._distance_moved() > self.unstuck_threshold:
            self.stuck = False
            return True

        return False
