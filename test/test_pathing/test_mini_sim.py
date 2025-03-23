import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
from enum import Enum
from maple.navigation.navigator import Navigator, State
from maple.utils import tuple_to_pytransform

from test.mocks import mock_agent, Transform

class SimulationState(Enum):
    RUNNING = 0
    OBSTACLE_HIT = 1
    GOAL_REACHED = 2
    PATH_BLOCKED = 3

class RoverSimulator:
    def __init__(self, navigator, timestep=0.1, collision_radius=0.5, visualization=True):
        """
        Mini simulator for testing rover navigation with obstacle detection
        
        Args:
            navigator: The Navigator instance controlling the rover
            timestep: Simulation time step in seconds
            collision_radius: Radius to check for collisions
            visualization: Whether to visualize the simulation
        """
        self.navigator = navigator
        self.timestep = timestep
        self.collision_radius = collision_radius
        self.visualization = visualization
        
        # Rover state
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        
        # Environment setup
        self.obstacles = []
        self.goal = None
        self.path_history = []
        self.state_history = []
        
        # Simulation state
        self.sim_state = SimulationState.RUNNING
        self.elapsed_time = 0.0
        
        # Visualization setup
        self.fig = None
        self.ax = None
        if visualization:
            self.setup_visualization()
    
    def setup_visualization(self):
        """Initialize the visualization plot"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('Rover Navigation Simulation')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        plt.ion()  # Interactive mode on
    
    def set_position(self, x, y, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        """Set the rover's position and orientation"""
        self.position = np.array([x, y, z])
        self.orientation = np.array([roll, pitch, yaw])
        self.path_history.append((x, y))
    
    def set_goal(self, x, y):
        """Set the goal location"""
        self.goal = (x, y)
        if self.navigator:
            self.navigator.goal_loc = (x, y)
    
    def add_obstacle(self, x, y, radius=1.0):
        """Add an obstacle to the environment"""
        self.obstacles.append((x, y, radius))
        if self.navigator:
            # Convert to the format expected by the navigator
            self.navigator.add_large_boulder_detection([(x, y, radius)])
    
    def clear_obstacles(self):
        """Clear all obstacles from the environment"""
        self.obstacles = []
    
    def get_transform(self):
        """Get the current transform matrix"""
        return tuple_to_pytransform((*self.position, *self.orientation))
    
    def move(self, lin_vel, ang_vel):
        """
        Move the rover based on linear and angular velocity commands
        
        Args:
            lin_vel: Linear velocity (m/s)
            ang_vel: Angular velocity (rad/s)
            
        Returns:
            SimulationState: Current simulation state
        """
        # Store current velocities
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel
        
        # Simple kinematics model - update position based on velocities
        yaw = self.orientation[2]
        
        # Calculate displacement in x and y
        dx = lin_vel * np.cos(yaw) * self.timestep
        dy = lin_vel * np.sin(yaw) * self.timestep
        
        # Update position and orientation
        self.position[0] += dx
        self.position[1] += dy
        self.orientation[2] += ang_vel * self.timestep
        
        # Normalize yaw to [-pi, pi]
        self.orientation[2] = np.arctan2(np.sin(self.orientation[2]), np.cos(self.orientation[2]))
        
        # Update path history
        self.path_history.append((self.position[0], self.position[1]))
        
        # Record navigator state if available
        if self.navigator:
            nav_state = self.navigator.state
            self.state_history.append((self.elapsed_time, nav_state))
        
        # Check for collisions or goal reached
        self.sim_state = self.check_simulation_state()
        
        # Update elapsed time
        self.elapsed_time += self.timestep
        
        # Update visualization if enabled
        if self.visualization:
            self.update_visualization()
        
        return self.sim_state
    
    def check_simulation_state(self):
        """Check if an obstacle was hit or goal was reached"""
        # Check for obstacle collisions
        for obs_x, obs_y, obs_radius in self.obstacles:
            distance = np.sqrt((self.position[0] - obs_x)**2 + (self.position[1] - obs_y)**2)
            if distance < (obs_radius + self.collision_radius):
                return SimulationState.OBSTACLE_HIT
        
        # Check if goal reached
        if self.goal:
            goal_x, goal_y = self.goal
            distance_to_goal = np.sqrt((self.position[0] - goal_x)**2 + (self.position[1] - goal_y)**2)
            if distance_to_goal < self.collision_radius:
                return SimulationState.GOAL_REACHED
        
        return SimulationState.RUNNING
    
    def update_visualization(self):
        """Update the visualization plot"""
        if not self.fig or not plt.fignum_exists(self.fig.number):
            self.setup_visualization()
        
        self.ax.clear()
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.grid(True)
        self.ax.set_title(f'Rover Navigation Simulation - t={self.elapsed_time:.1f}s')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        
        # Plot path history
        if len(self.path_history) > 1:
            path = np.array(self.path_history)
            self.ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.5)
        
        # Plot rover
        rover = Circle((self.position[0], self.position[1]), self.collision_radius, color='blue')
        self.ax.add_patch(rover)
        
        # Add direction indicator
        yaw = self.orientation[2]
        arrow_len = 1.0
        self.ax.arrow(self.position[0], self.position[1], 
                     arrow_len * np.cos(yaw), arrow_len * np.sin(yaw),
                     head_width=0.3, head_length=0.5, fc='blue', ec='blue')
        
        # Plot obstacles
        for obs_x, obs_y, obs_radius in self.obstacles:
            obstacle = Circle((obs_x, obs_y), obs_radius, color='red', alpha=0.7)
            self.ax.add_patch(obstacle)
        
        # Plot goal
        if self.goal:
            goal_x, goal_y = self.goal
            goal_marker = Circle((goal_x, goal_y), 0.5, color='green')
            self.ax.add_patch(goal_marker)
            self.ax.text(goal_x + 0.6, goal_y + 0.6, 'Goal', fontsize=10)
        
        # Add state and velocity info
        state_text = f"Lin Vel: {self.lin_vel:.2f} m/s, Ang Vel: {self.ang_vel:.2f} rad/s"
        if self.navigator:
            state_text += f"\nState: {self.navigator.state.name}"
        self.ax.text(-19, 19, state_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Show simulation state
        if self.sim_state != SimulationState.RUNNING:
            state_msg = f"Status: {self.sim_state.name}"
            self.ax.text(-5, 0, state_msg, fontsize=14, color='red',
                       bbox=dict(facecolor='white', alpha=0.7))
        
        plt.draw()
        plt.pause(0.01)
    
    def run_simulation(self, duration=None, max_steps=None):
        """
        Run the simulation with the navigator providing control inputs
        
        Args:
            duration: Maximum simulation duration in seconds (None for unlimited)
            max_steps: Maximum number of simulation steps (None for unlimited)
            
        Returns:
            List of (time, state, lin_vel, ang_vel) tuples
        """
        results = []
        step_count = 0
        
        while True:
            # Get velocity commands from navigator
            transform = self.get_transform()
            lin_vel, ang_vel = self.navigator.get_lin_vel_ang_vel(transform)
            
            # Move the rover
            sim_state = self.move(lin_vel, ang_vel)
            
            # Store results
            results.append((self.elapsed_time, self.navigator.state, lin_vel, ang_vel))
            
            # Check termination conditions
            if sim_state != SimulationState.RUNNING:
                print(f"\nSimulation ended: {sim_state.name}")
                break
                
            if duration is not None and self.elapsed_time >= duration:
                print(f"\nSimulation completed: {duration}s duration reached")
                break
                
            if max_steps is not None and step_count >= max_steps:
                print(f"\nSimulation completed: {max_steps} steps reached")
                break
            
            step_count += 1
        
        return results
    
    def run_scenario(self, scenario_name="obstacle_avoidance"):
        """
        Run a predefined test scenario
        
        Args:
            scenario_name: Name of the scenario to run
            
        Returns:
            List of (time, state, lin_vel, ang_vel) tuples
        """
        if scenario_name == "obstacle_avoidance":
            # Reset simulator
            self.clear_obstacles()
            self.path_history = []
            self.state_history = []
            self.elapsed_time = 0.0
            self.sim_state = SimulationState.RUNNING
            
            # Setup scenario
            self.set_position(0, 0, yaw=0)
            self.set_goal(15, 0)
            
            # Add an obstacle in the path
            self.add_obstacle(7, 0, radius=1.5)
            
            print("\n=== Running Obstacle Avoidance Scenario ===")
            return self.run_simulation(duration=30)
            
        elif scenario_name == "dynamic_replanning":
            # Reset simulator
            self.clear_obstacles()
            self.path_history = []
            self.state_history = []
            self.elapsed_time = 0.0
            self.sim_state = SimulationState.RUNNING
            
            # Setup scenario
            self.set_position(0, 0, yaw=0)
            self.set_goal(20, 0)
            
            print("\n=== Running Dynamic Replanning Scenario ===")
            
            # Define a step at which to dynamically add an obstacle
            obstacle_trigger_step = 50
            
            step_count = 0
            results = []
            
            while True:
                # Add obstacle at a specific step
                if step_count == obstacle_trigger_step:
                    print("\n=== Adding Dynamic Obstacle ===\n")
                    self.add_obstacle(15, 0, radius=2.0)
                
                # Get velocity commands from navigator
                transform = self.get_transform()
                lin_vel, ang_vel = self.navigator.get_lin_vel_ang_vel(transform)
                
                # Move the rover
                sim_state = self.move(lin_vel, ang_vel)
                
                # Store results
                results.append((self.elapsed_time, self.navigator.state, lin_vel, ang_vel))
                
                # Check termination conditions
                if sim_state != SimulationState.RUNNING:
                    print(f"\nSimulation ended: {sim_state.name}")
                    break
                    
                if self.elapsed_time >= 60:  # 60 second timeout
                    print(f"\nSimulation completed: timeout reached")
                    break
                
                step_count += 1
            
            return results
        
        else:
            print(f"Unknown scenario: {scenario_name}")
            return []
    
    def print_summary(self, results):
        """Print a summary of the simulation results"""
        print("\n=== Simulation Summary ===")
        print(f"Duration: {self.elapsed_time:.2f} seconds")
        print(f"Final State: {self.sim_state.name}")
        print(f"Final Position: ({self.position[0]:.2f}, {self.position[1]:.2f})")
        
        # Check if dynamic replanning happened
        nav_states = [state for _, state, _, _ in results]
        dynamic_triggered = State.DYNAMIC_PATH in nav_states
        static_restored = nav_states[-1] == State.STATIC_PATH if nav_states else False
        
        print(f"Dynamic Path Triggered: {'Yes' if dynamic_triggered else 'No'}")
        print(f"Returned to Static Path: {'Yes' if static_restored else 'No'}")
        
        # Display statistics if goal reached
        if self.sim_state == SimulationState.GOAL_REACHED and self.goal:
            path = np.array(self.path_history)
            path_length = 0.0
            for i in range(1, len(path)):
                path_length += np.sqrt(np.sum((path[i] - path[i-1])**2))
            
            direct_distance = np.sqrt(np.sum((path[-1] - path[0])**2))
            efficiency = (direct_distance / path_length) * 100 if path_length > 0 else 0
            
            print(f"Path Length: {path_length:.2f} m")
            print(f"Direct Distance: {direct_distance:.2f} m")
            print(f"Path Efficiency: {efficiency:.2f}%")
    
    def plot_state_history(self):
        """Plot the history of navigation states"""
        if not self.state_history:
            print("No state history available")
            return
        
        # Convert state history to numpy array for plotting
        times = [t for t, _ in self.state_history]
        states = [s.value for _, s in self.state_history]
        
        # Create a new figure
        plt.figure(figsize=(12, 6))
        plt.plot(times, states, 'b.-')
        
        # Add state labels to y-axis
        plt.yticks([s.value for s in State], [s.name for s in State])
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Navigation State')
        plt.title('Navigation State History')
        
        # Add annotations for state transitions
        last_state = None
        for t, s in self.state_history:
            if last_state != s:
                plt.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
                last_state = s
        
        plt.tight_layout()
        plt.show()


def test_obstacle_avoidance(mock_agent):
    """Test the rover's ability to avoid obstacles"""
    # Create navigator and simulator
    navigator = Navigator(mock_agent)
    simulator = RoverSimulator(navigator, visualization=True)
    
    # Run the obstacle avoidance scenario
    results = simulator.run_scenario("obstacle_avoidance")
    
    # Print summary and plot state history
    simulator.print_summary(results)
    simulator.plot_state_history()
    
    return results


def test_dynamic_replanning(mock_agent):
    """Test the rover's ability to replan when new obstacles appear"""
    # Create navigator and simulator
    navigator = Navigator(mock_agent)
    simulator = RoverSimulator(navigator, visualization=True)
    
    # Run the dynamic replanning scenario
    results = simulator.run_scenario("dynamic_replanning")

    # Print summary and plot state history
    simulator.print_summary(results)
    simulator.plot_state_history()
    
    return results


