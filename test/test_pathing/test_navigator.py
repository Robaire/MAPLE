from maple.navigation.navigator import Navigator, State
from maple.utils import tuple_to_pytransform
from pytransform3d.transformations import transform_from
import random

from test.mocks import mock_agent, Transform


# class MockTranform:
#     def __init__(self):
#         self.translation = [0, 0, 0]
#         self.rotation = [0, 0, 0]

# class MockAgent:
#     def get_initial_position(self):
#         # Simulate initial rover position
#         return transform_from([0, 0, 0], [0, 0, 0])

#     def get_initial_lander_position(self):
#         # Simulate initial lander position
#         return transform_from([5, 5, 0], [0, 0, 0])

def simulate_rover_movement(navigator, steps=20, obstacle_trigger_step=5):
    """Simulates the rover moving and hitting an obstacle that forces dynamic planning."""

    x, y, z = [5, 10, 15]
    roll, pitch, yaw = [0.2, 0.5, 1.0]

    rover_position = tuple_to_pytransform((x, y, z, roll, pitch, yaw)) # Just a random starting position
    path_states = []

    for step in range(steps):
        # At a certain step, inject an obstacle to force dynamic planning
        if step == obstacle_trigger_step:
            print("\n=== Obstacle Detected: Forcing Dynamic Path ===\n")
            # Add a large boulder right in the static path
            navigator.add_large_boulder_detection([(2, 2, 1)])

        lin_vel, ang_vel = navigator.get_lin_vel_ang_vel(rover_position)

        # Log current state and velocities
        path_states.append((navigator.state, lin_vel, ang_vel))
        print(f"Step {step}: State={navigator.state}, Lin_Vel={lin_vel:.2f}, Ang_Vel={ang_vel:.2f}")

        # Simulate basic rover forward movement (advance towards the goal)
        if navigator.goal_loc:
            goal_x, goal_y = navigator.goal_loc
            rover_position = tuple_to_pytransform((goal_x, goal_y, 0, 0, 0, 0))

    return path_states

def test_navigation_switching(mock_agent):
    navigator = Navigator(mock_agent)

    # Run simulation
    results = simulate_rover_movement(navigator)

    # Check if switching between static and dynamic happened
    dynamic_triggered = any(state == State.DYNAMIC_PATH for state, _, _ in results)
    static_restored = results[-1][0] == State.STATIC_PATH

    print("\nTest Result:")
    print(f"Dynamic Triggered: {'Yes' if dynamic_triggered else 'No'}")
    print(f"Returned to Static: {'Yes' if static_restored else 'No'}")

if __name__ == "__main__":
    test_navigation_switching()
