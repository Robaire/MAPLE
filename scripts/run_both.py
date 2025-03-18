import subprocess
import os

def main():
    # Define the paths to the scripts
    run_agent_path = os.path.join(os.path.dirname(__file__), 'run_agent.py')
    save_transforms_path = os.path.join(os.path.dirname(__file__), '../maple/navigation/save_transforms.py')

    # Start the run_agent.py script
    run_agent_process = subprocess.Popen(['python', run_agent_path])

    # Start the save_transforms.py script
    save_transforms_process = subprocess.Popen(['python', save_transforms_path])

    # Wait for both processes to complete
    run_agent_process.wait()
    save_transforms_process.wait()

if __name__ == "__main__":
    main()
