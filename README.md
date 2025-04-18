# :maple_leaf: MIT Autonomous Pathfinding for Lunar Exploration :maple_leaf:

# Setup
This project uses [uv](https://docs.astral.sh/uv/) for package and environment management.
After cloning the repository and installing `uv` run the following commands:
- `uv sync`: Install dependencies
- `uvx pre-commit install`: Install [pre-commit](https://pre-commit.com/) git hooks

FastSAM requires model weights that are too large to be stored on GitHub. Download the weights using `uv run ./scripts/fastsam_checkpoint.py`. This will place the weights in the correct location (`resources/FastSAM-x.pt`).

ORBSLAM requires a vocabulary file that is too large to be stored on GitHub raw. Extract the vocabulary file using `uv run ./scripts/orbslam_vocab.py`. This will extract the vocabulary file to the correct location (`resoucres/ORBvoc.txt`).

# Optional Setup
If using [Visual Studio Code](https://code.visualstudio.com/) the following extensions are recommended:
[python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), 
[pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance),
[ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).

# Running Tests
- `uv run pytest`: Run tests
- `uv run pytest test/test_file.py`: Run a specific test file

# Running Agents
A python script is included for convenience to automatically start the simulator and run an agent.
Due to the large file size of the lunar simulator it is not included in this repository, instead manually copy the contents of `LunarSimulator` to `./simulator`.
Alternatively a file path to the `LunarSimulator` directory can be provided as an optional argument.
To start the simulator and evaluate an agent run `uv run ./scripts/run_agent.py path_to_agent` from the root of this repository.
If specifying an alternate location for the simulator use `uv run ./scripts/run_agent.py path_to_agent --sim="path_to_lunar_simulator"`.

# Preparing Docker Container
Building the docker container requires `Leaderboard` and `LunarSimulator` be present in the project directory.
To build a container with a specific agent, run `uv run ./scripts/build_docker.py ./agents/target_agent.py`.
This will build MAPLE and then build a docker container with the agent. 