import carla
import numpy as np
from maple.utils import carla_to_pytransform
from leaderboard.leaderboard.autoagents.autonomous_agent import AutonomousAgent
from maple.navigation.navigator import Navigator

class TransformSaver:
    def __init__(self, agent: AutonomousAgent, save_path: str = 'rover_transforms.csv'):
        self.agent = agent
        self.save_path = save_path
        self.transforms = []

    def save_transform(self):
        """Save the current transform of the rover."""
        rover_global = carla_to_pytransform(self.agent.get_transform())
        x, y, z, roll, pitch, yaw = rover_global[:3, 3].tolist() + list(rover_global[:3, :3].flatten())
        self.transforms.append([x, y, z, roll, pitch, yaw])

    def write_to_file(self):
        """Write all saved transforms to a CSV file."""
        np.savetxt(self.save_path, self.transforms, delimiter=',', header='x,y,z,roll,pitch,yaw', comments='')

def main():
    agent = AutonomousAgent()
    navigator = Navigator(agent)
    transform_saver = TransformSaver(agent)

    # Simulate the rover path and save transforms
    while not agent.has_finished():
        transform_saver.save_transform()
        # Simulate a step in the rover's path
        # This would typically involve updating the rover's position using the PID controller
        # For example: agent.run_step(input_data)
        navigator.get_goal_loc()  # Update the goal location

    # Write the saved transforms to a file
    transform_saver.write_to_file()
    print(f"Transforms saved to {transform_saver.save_path}")

if __name__ == "__main__":
    main()
