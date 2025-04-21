import io
import os
import tarfile
from datetime import datetime

class Recorder:
    """Records data from a simulation run to an archive."""

    # To use this, initialize it in the agent setup, and then call it every run_step
    # When the agent is done, call finalize to save the data

    done: bool = False # Whether the recording is done
    agent: None # AutonomousAgent
    tar_path: str # Output archive path
    tar_file: tarfile.TarFile # Output archive

    def __init__(self, agent, output_file = None, max_size: float = 1):
        """Initialize the recorder.

        Args:
            agent: The agent to record data from
            output: Output file path (will be gzipped)
            max_size: Maximum size of the archive in GB
        """
        self.agent = agent

        # Create the archive file
        self.tar_path = self.parse_file_name(output_file)
        self.tar_file = tarfile.open(self.tar_path, "w:gz")

        # Record agent configuration and simulation start parameters
        use_fiducials = self.agent.use_fiducials()
        initial_lander_position = self.agent.get_initial_lander_position()
        initial_rover_position = self.agent.get_initial_position()

        # Record initial sensor configuration
        # TODO: convert keys to strings
        sensors = self.agent.sensors() # Sensor dict (of carla.SensorPosition keys)


        # TODO: Record this data...


    def record_frame(self, frame: int, input_data: dict):
        """Record a frame of data from the simulation.

        Args:
            frame: The frame number
            input_data: The input data from the simulation
        """

        # The simulation may keep running after we are done recording so do nothing
        if self.done:
            return

        # Get agent data 



        # Get input_data


        # Check file size and save if over the limit
        # TODO: Probably don't have to do this every frame
        if os.path.getsize(self.tar_path) > self.max_size * 1024 * 1024 * 1024:
            self.save()

    def parse_file_name(self, output_file):
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

        self.tar_file.close() # Flush the buffer to the file
        self.done = True # Set the done flag
