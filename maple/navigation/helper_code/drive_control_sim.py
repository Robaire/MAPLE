import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Arrow
import time
import math
from math import atan2

# Add constant values since the original import may not be available in the test environment
DT = 0.1  # Time step in seconds
goal_speed = 1.0  # Target forward speed in m/s
goal_hard_turn_speed = 0.5  # Slower speed when making sharp turns

class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0

    def update(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def angle_helper(start_x, start_y, yaw, end_x, end_y):
    """Given a a start location and yaw this will return the desired turning angle to point towards end"""
    # Do trig to find the angle between the goal location and rover location
    angle_of_triangle = atan2((end_y - start_y), (end_x - start_x))
    
    # Calculate goal angular velocity
    goal_ang = angle_of_triangle - yaw

    # Normalize the angle to be within [-pi, pi]
    while goal_ang > np.pi:
        goal_ang -= 2 * np.pi
    while goal_ang < -np.pi:
        goal_ang += 2 * np.pi

    return goal_ang

class DriveController:
    def __init__(self):
        self.linear_pid = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=goal_speed)
        self.angular_pid = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=0)
        self.prev_distance_to_goal = 0

    def get_lin_vel_ang_vel_drive_control(self, rover_x, rover_y, rover_yaw, goal_x, goal_y):
        """Get the linear and angular velocity to drive the rover to the goal location"""
        # Calculate the angle helper
        goal_ang = angle_helper(rover_x, rover_y, rover_yaw, goal_x, goal_y)

        # Negate the goal_ang to tell how far off we are from the measurement, where zero is towards the goal location
        measured_off_ang = -goal_ang

        # Calculate distance to the goal
        distance_to_goal = np.sqrt((goal_x - rover_x) ** 2 + (goal_y - rover_y) ** 2)

        # Calculate velocity from position estimates
        measured_velocity = (self.prev_distance_to_goal - distance_to_goal) / DT if self.prev_distance_to_goal != 0 else 0
        self.prev_distance_to_goal = distance_to_goal

        # Update PID controllers
        linear_velocity = self.linear_pid.update(measured_velocity, DT)
        angular_velocity = self.angular_pid.update(measured_off_ang, DT)

        # Check if we need to do a tight turn then override goal speed
        if abs(goal_ang) > 0.1:
            linear_velocity = goal_hard_turn_speed

        return linear_velocity, angular_velocity

class RoverSimulator:
    def __init__(self, start_x=0.0, start_y=0.0, start_yaw=0.0):
        self.x = start_x
        self.y = start_y
        self.yaw = start_yaw
        self.path_x = [start_x]
        self.path_y = [start_y]
        self.controller = DriveController()
        
    def update(self, goal_x, goal_y):
        # Get control commands
        lin_vel, ang_vel = self.controller.get_lin_vel_ang_vel_drive_control(
            self.x, self.y, self.yaw, goal_x, goal_y
        )
        
        # Update rover state
        self.yaw += ang_vel * DT
        self.x += lin_vel * np.cos(self.yaw) * DT
        self.y += lin_vel * np.sin(self.yaw) * DT
        
        # Normalize yaw angle
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi
        
        # Record path
        self.path_x.append(self.x)
        self.path_y.append(self.y)
        
        # Return current position and distance to goal
        distance = np.sqrt((goal_x - self.x) ** 2 + (goal_y - self.y) ** 2)
        return self.x, self.y, self.yaw, distance, lin_vel, ang_vel

def run_simulation(test_cases):
    """Run simulation on multiple test cases"""
    results = []
    
    for i, (start_pos, goal_pos) in enumerate(test_cases):
        print(f"Running test case {i+1}:")
        print(f"  Start: ({start_pos[0]}, {start_pos[1]}), yaw: {start_pos[2]}")
        print(f"  Goal: ({goal_pos[0]}, {goal_pos[1]})")
        
        # Initialize rover
        rover = RoverSimulator(start_pos[0], start_pos[1], start_pos[2])
        
        # Run simulation
        steps = 0
        max_steps = 1000
        goal_threshold = 0.1
        simulation_data = []
        
        while steps < max_steps:
            x, y, yaw, distance, lin_vel, ang_vel = rover.update(goal_pos[0], goal_pos[1])
            simulation_data.append((x, y, yaw, distance, lin_vel, ang_vel))
            steps += 1
            
            if distance < goal_threshold:
                print(f"  Goal reached in {steps} steps!")
                break
        
        if steps >= max_steps:
            print(f"  Failed to reach goal within {max_steps} steps")
            print(f"  Final position: ({x}, {y}), distance to goal: {distance}")
        
        results.append({
            'test_case': i+1,
            'path_x': rover.path_x,
            'path_y': rover.path_y,
            'steps': steps,
            'reached_goal': distance < goal_threshold,
            'final_distance': distance,
            'simulation_data': simulation_data,
        })
    
    return results

def visualize_simulation(results, test_cases):
    """Visualize simulation results"""
    num_tests = len(results)
    fig, axes = plt.subplots(1, num_tests, figsize=(5*num_tests, 5))
    
    # Handle single test case
    if num_tests == 1:
        axes = [axes]
    
    for i, (result, (start_pos, goal_pos)) in enumerate(zip(results, test_cases)):
        ax = axes[i]
        
        # Plot path
        ax.plot(result['path_x'], result['path_y'], 'b-', label='Path')
        
        # Plot start and goal positions
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
        ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')
        
        # Plot initial orientation
        arrow_length = 0.5
        dx = arrow_length * np.cos(start_pos[2])
        dy = arrow_length * np.sin(start_pos[2])
        ax.arrow(start_pos[0], start_pos[1], dx, dy, 
                head_width=0.2, head_length=0.3, fc='g', ec='g')
        
        # Add title and labels
        ax.set_title(f'Test Case {i+1}' + 
                    f'\nGoal {"Reached" if result["reached_goal"] else "Not Reached"}' +
                    f'\nSteps: {result["steps"]}')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('drive_controller_simulation.png')
    plt.show()

def create_animation(result, test_case, filename='rover_animation.gif'):
    """Create animation of rover movement"""
    start_pos, goal_pos = test_case
    path_x, path_y = result['path_x'], result['path_y']
    simulation_data = result['simulation_data']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set up the plot
    ax.grid(True)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Rover Navigation Simulation')
    
    # Add padding to axis limits
    max_x = max(max(path_x), goal_pos[0]) + 1
    min_x = min(min(path_x), goal_pos[0]) - 1
    max_y = max(max(path_y), goal_pos[1]) + 1
    min_y = min(min(path_y), goal_pos[1]) - 1
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Create a line for the path
    line, = ax.plot([], [], 'b-', label='Path')
    
    # Create arrow for rover orientation
    rover_arrow = Arrow(0, 0, 0, 0, width=0.4, color='green')
    ax.add_patch(rover_arrow)
    
    # Plot start and goal positions
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')
    
    # Text elements for velocity info
    lin_vel_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ang_vel_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    dist_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
    
    ax.legend()
    
    def init():
        line.set_data([], [])
        rover_arrow.remove()
        rover_arrow = Arrow(0, 0, 0, 0, width=0.4, color='green')
        ax.add_patch(rover_arrow)
        lin_vel_text.set_text('')
        ang_vel_text.set_text('')
        dist_text.set_text('')
        return line, rover_arrow, lin_vel_text, ang_vel_text, dist_text
    
    def update(frame):
        # Update path line
        line.set_data(path_x[:frame+1], path_y[:frame+1])
        
        # Update rover position and orientation
        x, y, yaw, distance, lin_vel, ang_vel = simulation_data[frame]
        
        # Remove old arrow and create new one
        rover_arrow.remove()
        arrow_length = 0.3
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        rover_arrow = Arrow(x, y, dx, dy, width=0.4, color='green')
        ax.add_patch(rover_arrow)
        
        # Update text elements
        lin_vel_text.set_text(f'Linear Velocity: {lin_vel:.2f} m/s')
        ang_vel_text.set_text(f'Angular Velocity: {ang_vel:.2f} rad/s')
        dist_text.set_text(f'Distance to Goal: {distance:.2f} m')
        
        return line, rover_arrow, lin_vel_text, ang_vel_text, dist_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(simulation_data),
                                   init_func=init, blit=True, interval=50)
    
    # Save animation
    anim.save(filename, writer='pillow', fps=20)
    
    plt.close()

def run_tests():
    """Run a series of test cases for the drive controller"""
    # Define test cases as (start_x, start_y, start_yaw), (goal_x, goal_y)
    test_cases = [
        # Simple straight line test
        ((0.0, 0.0, 0.0), (5.0, 0.0)),
        
        # 90 degree turn test
        ((0.0, 0.0, 0.0), (0.0, 5.0)),
        
        # Reverse direction test
        ((0.0, 0.0, 0.0), (-5.0, -5.0)),
        
        # Test with rover not facing goal initially
        ((0.0, 0.0, math.pi), (5.0, 0.0)),
    ]
    
    # Run simulation
    results = run_simulation(test_cases)
    
    # Visualize results
    visualize_simulation(results, test_cases)
    
    # Create animation for the first test case
    create_animation(results[0], test_cases[0], 'rover_animation_straight.gif')
    
    # Create animation for a more complex case
    create_animation(results[3], test_cases[3], 'rover_animation_complex.gif')
    
    return results

if __name__ == "__main__":
    print("Starting DriveController 2D simulator tests...")
    results = run_tests()
    print("Tests completed. Check the generated visualizations.")