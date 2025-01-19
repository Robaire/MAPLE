from maple.utils import carla_to_pytransform

from pytransform3d.transformations import concat


class BoulderMapper:
    """Estimates the position of boulders around the rover."""

    agent: None
    left: None  # This is the carla.SensorPosition object
    right: None  # This is the carla.SensorPosition object

    def __init__(self, agent, left, right):
        """Intializer.

        Args:
            agent: The Agent instance
            left: The left camera instance (string or object)
            right: The right camera instance (string or object)
        """
        self.agent = agent

        # Look through all the camera objects in the agent's sensors and save the ones for the stereo pair
        # We do this since the objects themselves are used as keys in the input_data dict and to check they exist
        for key in agent.sensors().keys():
            if str(key) == str(left):
                self.left = key

            if str(key) == str(right):
                self.right = key

        # Confirm that the two required cameras exist
        if self.left is None or self.right is None:
            raise ValueError("Required cameras are not defined in the agent's sensors.")

    def map(self, input_data) -> list:
        """Estimates the position of boulders in the scene.,

        Args:
            input_data: The input data dictionary provided by the simulation

        Returns:
            A list of boulder positions in the rover frame.
        """

        # Get camera images
        try:
            left_image = input_data["Grayscale"][self.left]
            right_image = input_data["Grayscale"][self.right]
        except KeyError:
            raise ValueError("Required cameras have no data.")

        # Run the FastSAM pipeline to detect boulders
        # Run the stereo vision pipeline to get their positions

        # Get the camera positions
        camera_rover = carla_to_pytransform(self.agent.get_camera_position(self.left))

        # Calculator the boulder positions in the rover frame
        boulders_camera = []

        # Return the list of boulders in the rover frame
        boulders_rover = [
            concat(boulder_camera, camera_rover) for boulder_camera in boulders_camera
        ]
        return boulders_rover
