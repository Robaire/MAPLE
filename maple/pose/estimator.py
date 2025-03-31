from abc import ABC, abstractmethod
from numpy.typing import NDArray


class Estimator(ABC):
    @abstractmethod
    def estimate(self, input_data, use_imu_ang=False) -> NDArray:
        """
        All estimators must implement this function.
        This should return the current estimated pose (as a pytransform matrix) or None if it is not possible to do so.

        Args:
            input_data: The input_data dictionary this time step

        Returns:
            A pytransform representing the rover in the global frame.
        """
        pass

    def __call__(self, input_data, use_imu_ang=False) -> NDArray:
        """Equivalent to calling `estimate`."""
        return self.estimate(input_data)
