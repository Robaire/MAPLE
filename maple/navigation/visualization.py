import matplotlib.pyplot as plt
import numpy as np
from maple.navigation.navigator import Navigator
from maple.utils import pytransform_to_tuple, carla_to_pytransform
from pytransform3d.transformations import concat, transform_from  # Add transform_from import
from leaderboard.leaderboard.autoagents.autonomous_agent import AutonomousAgent

def visualize_path(agent, actual_positions):
    navigator = Navigator(agent)
    intended_path = navigator.path.path

    # Extract intended path coordinates
    intended_x = [point[0] for point in intended_path]
    intended_y = [point[1] for point in intended_path]

    # Extract actual path coordinates
    actual_x = [pos[0] for pos in actual_positions]
    actual_y = [pos[1] for pos in actual_positions]

    plt.figure(figsize=(10, 8))
    plt.plot(intended_x, intended_y, label='Intended Path', linestyle='--', color='blue')
    plt.plot(actual_x, actual_y, label='Actual Path', linestyle='-', color='red')
    plt.scatter(intended_x, intended_y, color='blue', s=10)
    plt.scatter(actual_x, actual_y, color='red', s=10)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Rover Path Following')
    plt.legend()
    plt.grid(True)
    plt.show()

def simulate_rover_path(agent, steps=100):
    navigator = Navigator(agent)
    actual_positions = []

    # Initialize rover position
    rover_position = carla_to_pytransform(agent.get_initial_position())

    for _ in range(steps):
        lin_vel, ang_vel = navigator(rover_position)
        rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(rover_position)

        # Update rover position based on velocities
        rover_x += lin_vel * np.cos(rover_yaw) * 0.1  # Assuming DT = 0.1
        rover_y += lin_vel * np.sin(rover_yaw) * 0.1
        rover_yaw += ang_vel * 0.1

        # Update the rover position transform
        rover_position = transform_from(np.eye(3), [rover_x, rover_y, rover_yaw])
        actual_positions.append((rover_x, rover_y))

    return actual_positions

if __name__ == "__main__":
    agent = AutonomousAgent()
    # Set initial position and other necessary initializations for the agent
    # agent.set_initial_position(...)
    # agent.set_initial_lander_position(...)
    # agent.set_geometric_map(...)

    actual_positions = simulate_rover_path(agent)
    visualize_path(agent, actual_positions)
