# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

# Steps needed to build the docker container
# 1. Build MAPLE into a wheel
# uv build --wheel
# 2. Copy the maple.whl into a team_code folder
# 3. Copy the target agent into a team_code folder
# 4. Move into the docker folder
# 4. Call make_docker.sh to build the docker container

import argparse
import subprocess
import os
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Docker Builder",
        description="Builds a docker container for the Lunar Autonomy Challenge",
    )

    # Get the agent we want to build into the docker file
    parser.add_argument("agent", help="Path to the agent file", type=str)
    parser.add_argument(
        "-n", "--name", help="Docker container name", type=str, default="lac-user"
    )
    args = parser.parse_args()

    # Check that the agent file exists before typing to run anything
    agent_path = os.path.expanduser(args.agent)
    if not os.path.exists(agent_path):
        print(f"No file found at: {args.agent}")
        exit()

    # Make the team_code folder
    os.makedirs("docker/team_code", exist_ok=True)

    # Build MAPLE
    # subprocess.run(["uv", "build", "--wheel", "--out-dir=docker/team_code"])

    # Copy MAPLE source code to team_code
    shutil.copy("pyproject.toml", "docker/team_code/pyproject.toml")
    shutil.copytree("maple", "docker/team_code/maple")
    shutil.copytree("resources", "docker/team_code/resources")

    # Copy the target agent into the team_code folder and rename it
    shutil.copy(agent_path, "docker/team_code/mit_agent.py")

    # Change working directory and build the docker container
    subprocess.run(
        [
            "bash",
            "make_docker.sh",
            "--team-code",
            "team_code",
            "--target-name",
            f"{args.name}",
        ],
        cwd="docker",
    )

    # Clean the team_code folder
    shutil.rmtree("docker/team_code")
