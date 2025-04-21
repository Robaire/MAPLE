import io
import os
import tarfile
from datetime import datetime
from PIL import Image

from maple.utils import carla_to_pytransform, pytransform_to_tuple


class Recorder:
    """Records data from a simulation run to an archive."""

    # To use this, initialize it in the agent setup, and then call it every run_step
    # When the agent is done, call finalize to save the data

    done: bool = False  # Whether the recording is done
    agent: None  # AutonomousAgent
    tar_path: str  # Output archive path
    tar_file: tarfile.TarFile  # Output archive

    def __init__(self, agent, output_file=None, max_size: float = 1):
        """Initialize the recorder.

        Args:
            agent: The agent to record data from
            output: Output file path (will be gzipped)
            max_size: Maximum size of the archive in GB
        """
        self.agent = agent

        # Create the archive file
        self.tar_path = self._parse_file_name(output_file)
        self.tar_file = tarfile.open(self.tar_path, "w:gz")

        # Record agent configuration and simulation start parameters
        use_fiducials = self.agent.use_fiducials()  # bool
        lander_initial_position = pytransform_to_tuple(
            carla_to_pytransform(self.agent.get_initial_lander_position())
        )  # [x, y, z, roll, pitch, yaw]
        rover_initial_position = pytransform_to_tuple(
            carla_to_pytransform(self.agent.get_initial_position())
        )  # [x, y, z, roll, pitch, yaw]

        # Record initial sensor configuration
        # TODO: convert keys to strings
        sensors = self.agent.sensors()  # Sensor dict (of carla.SensorPosition keys)
        # TODO: Record this data...

    def record_frame(self, frame: int, input_data: dict):
        """Record a frame of data from the simulation.

        Args:
            frame: The frame number
            input_data: The input data from the simulation
        """
        # Each frame write the images into the archive and add
        # numerical data into a buffer, the buffer will be written
        # to the archive when the recording is finished

        # The simulation may keep running after we are done recording so do nothing
        if self.done:
            return

        # Get agent data
        frame = frame  # So I don't forget to include this
        pose = pytransform_to_tuple(carla_to_pytransform(self.agent.get_transform()))
        imu_data = self.agent.get_imu_data()  # [ax, ay, az, gx, gy, gz]
        mission_time = self.agent.get_mission_time()  # float [s]
        power = self.agent.get_current_power()  # float [Wh]
        linear_speed = self.agent.get_linear_speed()  # float [m/s]
        angular_speed = self.agent.get_angular_speed()  # float [rad/s]
        cover_angle = self.agent.get_radiator_cover_angle()  # float [rad]

        # Iterate over items for each configured camera
        for camera in self.agent.sensors().keys():
            camera_state = self.agent.get_camera_state(camera)  # bool
            camera_position = pytransform_to_tuple(
                carla_to_pytransform(self.agent.get_camera_position(camera))
            )  # [x, y, z, roll, pitch, yaw]
            light_intensity = self.agent.get_light_state(camera)
            light_position = pytransform_to_tuple(
                carla_to_pytransform(self.agent.get_light_position(camera))
            )  # [x, y, z, roll, pitch, yaw]

            # TODO: Do we want to record the camera info for frames that don't have images?

            # Get the grayscale image if the camera is active
            grayscale = ""
            if camera_state:
                try:
                    image = input_data["Grayscale"][camera]
                    grayscale = self._archive_image(image, camera, frame, "grayscale")
                except Exception:
                    pass

            # Only attempt this if the camera has semantics enabled
            semantic = ""
            if self.agent.sensors()[camera]["use_semantic"] and camera_state:
                try:
                    image = input_data["Semantic"][camera]
                    semantic = self._archive_image(image, camera, frame, "semantic")
                except Exception:
                    pass

            # TODO: Log in a csv the name of the image for for the corresponding frame number

        """
        input_data is a dictionary that contains the sensors data:
        - Active sensors will have their data represented as a numpy array
        - Active sensors without any data in this tick will instead contain 'None'
        - Inactive sensors will not be present in the dictionary.

        Example:

        input_data = {
            'Grayscale': {
                carla.SensorPosition.FrontLeft:  np.array(...),
                carla.SensorPosition.FrontRight:  np.array(...),
            },
            'Semantic':{
                carla.SensorPosition.FrontLeft:  np.array(...),
            }
        }
        """

        # TODO: implement

        # Check file size and save if over the limit
        # TODO: Probably don't have to do this every frame
        # TODO: Sum the size of the tar and the numerical data buffers
        if os.path.getsize(self.tar_path) > self.max_size * 1024 * 1024 * 1024:
            self.save()

    def _archive_image(self, image, camera, frame: int, type: str) -> str:
        """Add an image in the archive."""

        # Convert the image to a PIL image
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        buffer.seek(0)

        # Determine the filepath of the image
        filepath = f"images/{str(camera)}/{type}/{str(camera)}_{type}_{str(frame)}.png"

        # Build a tarinfo object
        tar_info = tarfile.TarInfo(name=filepath)
        tar_info.size = len(buffer.getvalue())
        tar_info.mtime = int(datetime.now().timestamp())
        tar_info.mode = 0o644

        # Add the image to the archive
        self.tar_file.addfile(tar_info, buffer)

        return filepath

    def _parse_file_name(self, output_file):
        """Parse the output file name."""

        # If no output file is provided, use the current timestamp
        if output_file is None:
            output_file = f"run-{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.tar.gz"

        # Check if the file already has the .tar.gz extension
        if not output_file.endswith(".tar.gz"):
            output_file = f"{output_file}.tar.gz"

        return output_file

    def set_description(self, description: str):
        """Set a description for the run if desired."""

        if self.done:
            raise RuntimeError("Cannot set description after recording is done.")

        # TODO: Implement this
        pass

    def save(self):
        """Stop recording and save the archive."""

        # Write numerical data and description files into the archive

        self.tar_file.close()  # Flush the buffer to the file
        self.done = True  # Set the done flag
