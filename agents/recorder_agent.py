from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from maple.data import Recorder


class DataRecorderAgent(AutonomousAgent):
    def __init__(self, agent):
        # TODO: Figure out if references to self will work in _agent...

        # The agent to record data from
        self._agent = agent

        # The Data Recorder
        self._recorder = Recorder(self, max_size=5)
        self._recorder_frame = 1

    def setup(self, path_to_conf_file):
        self._agent.setup(path_to_conf_file)

    def use_fiducials(self):
        return self._agent.use_fiducials()

    def sensors(self):
        return self._agent.sensors()

    def run_step(self, input_data):
        # Do data logging
        self._recorder(self._recorder_frame, input_data)
        self._recorder_frame += 1

        # Run the agent
        self._agent.run_step(input_data)

    def finalize(self):
        # Do data logging
        self._recorder.stop()

        self._agent.finalize()
